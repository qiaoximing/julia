using MLDatasets
using LightGraphs
using ProgressMeter
using Plots
theme(:orange)

α = 1e-2 # learn rate
ζ = 6 # vision field
ξ = 4 # number of actions

mutable struct Agent
    g::SimpleDiGraph # the raw graph
    z::Array{Bool} # activity of nodes
    w::Array{Float32, 2} # output weights
end
function init!(ag::Agent)
    num_in = ζ ^ 2
    num_act = ξ
    ag.g = SimpleDiGraph(num_in + num_act + 1)
    # initialize input nodes
    # initialize action nodes
    ag.z = zeros(Bool, nv(ag.g))
    # initialize a always-on node
    ag.z[num_in + num_act + 1] = true
    # each output corresponds to a distribution over all nodes
    # outputs are not actural nodes (not in the graph)
    # num_in + num_act weights per node
    ag.w = zeros(Float32, num_in + num_act, nv(ag.g))
end
function predict(ag::Agent, soft::Bool=false)
    # predict with max or softmax weight
    idx = argmax(ag.w, dims=2) # Nx1 Cart idx array
    return ag.z[last.(Tuple.(idx))] # Bool array
end
function loss(ag::Agent, obs)
    num_in = ζ ^ 2
    # compute prediction error
    # ignore actions for now
    pred = Float32.(predict(ag))[1:num_in]
    obs = vec(obs .> 0.5)
    return sum(abs.(pred .- obs)) / length(obs)
end
function loss_center(ag::Agent, obs)
    h = w = ζ
    num_in = h * w
    # certer pixels only
    pred = Float32.(predict(ag))[1:num_in]
    pred = vec(reshape(pred, h, w)[2:(h - 1), 2:(w - 1)])
    obs = vec(obs[2:(h - 1), 2:(w - 1)] .> 0.5)
    return sum(abs.(pred .- obs)) / length(obs)
end
function explore!(ag::Agent)
    # add new nodes to graph ag.g
    if add_vertex!(ag.g)
        v_new = nv(ag.g)
        v = 1
        add_edge!(ag.g, v, v_new)
    else
        @warn "Add vertex failed."
    end
    # extend state vector ag.z
    push!(ag.z, false)
    # extend weight matrix ag.w
    w_new = maximum(ag.w, dims=2) .- 10α
    ag.w = [ag.w w_new]
end
function step!(ag::Agent, obs, act, train=true)
    num_in = ζ ^ 2
    num_act = ξ
    #===========================================
        Learn from new observation
    ===========================================#
    # Compare the previous prediction to new observations. Most predictions are
    # not novel, except the previous attention point
    # If the novel hypothesis is right, reinforce it
    obs = vec(obs)
    if train
        # for v in vertices(ag.g)
            # err = ag.z[v] .!= obs
            # ag.w[1:num_in, v] .-= α * Float32.(err)
        # end
        # ag.w .-= maximum(ag.w, dims=2)
    end
    # Whatever the prediction is, create a new node that memorize the
    # current-step transition and also multi-step transition if executing long
    # programs (high-level sequential actions)

    #===========================================
        Inference with new observation
    ===========================================#
    # decay and diffuse previous node activity

    # The wrong prediction causes bottom-up activity (if the prediction is
    # perfect, then observation causes no activity). When the activity is strong
    # enough, it may change the attention

    # feed in new observation and action
    ag.z[1:num_in] = obs .> 0.5
    ag.z[(num_in + 1):(num_in + num_act)] = act
    ag.z[num_in + num_act + 1] = true
    # run the network to get bottom-up activity
    converge = false
    while !converge
        converge = true
        for v in vertices(ag.g)
            z_in = ag.z[inneighbors(ag.g, v)]
            # fire when having two active inputs
            if sum(z_in) >= 2
                ag.z[v] = true
                converge = false
            end
        end
    end

    #===========================================
        Make prediction from memory or hypothesis
    ===========================================#
    # Try to predict all inputs, all actions

    # If the transition is known, directly retrieve the memory.     

    # Combinatorial generalization: the current state is not new but the
    # binding of the working memory is new
    # Binding 1: unordered set of past states (OR gate)
    # Binding 2: ordered pair of current and past (AND gate)
    # Combination of the above two gives generalization

    # Sequential generalization: try a random function that is most likely to be
    # useful. If lucky, the function will be a part of a useful decomposition.
    # The composite function will be memorized later. 

    # In either case, the prediction becomes the next attention, which will be
    # used in the next step. The new node will be memorized if it is useful


    # no need of ϵ-greedy exploration
    # ϵ = 1e-2; if rand() < ϵ explore!(ag) end

end

mutable struct Env
    x::Int64
    y::Int64
    h::Int64
    w::Int64
    H::Int64
    W::Int64
    data::Array{Float32,2}
    label::Int64
end
function init!(env::Env, data, label)
    h = w = ζ
    dataH, dataW = size(data)
    env.x, env.y = h + 1, w + 1
    env.h, env.w = h, w
    env.H, env.W = dataH + 2h, dataW + 2w
    env.data = zeros(env.H, env.W)
    env.data[env.x:(env.x + dataH - 1), 
             env.y:(env.y + dataW - 1)] = Float32.(data)
    env.label = Int64(label)
    env.x += (dataH - env.h) ÷ 2 # integer divide
    env.y += (dataW - env.w) ÷ 2
end
function observe(env::Env)
    obs = env.data[env.x:(env.x + env.h - 1), 
                   env.y:(env.y + env.w - 1)]
    return obs
end
function step!(env::Env, act)
    # act = 4x1 BitArray/Array{Bool}
    d = 1 # moving distance
    dx = d * (act[1] - act[2])
    dy = d * (act[3] - act[4])
    if env.x + dx >= 1 && env.x + dx + env.h - 1 <= env.H
        env.x += dx
    end
    if env.y + dy >= 1 && env.y + dy + env.w - 1 <= env.W
        env.y += dy
    end
end

agent = Agent(DiGraph(0), [false], zeros(2,2))
env = Env(0, 0, 0, 0, 0, 0, zeros(2,2), 0)
data, label = MNIST.testdata()
init!(agent)
num_img, num_step = 200, 20
log1 = zeros(num_step, num_img)
log2 = zeros(num_step, num_img)
# main loop
@showprogress for i in 1:num_img
    train = i <= num_img * 0.8 ? true : false
    rand_idx = rand(1:size(data)[3])
    init!(env, data[:, :, rand_idx], label[i])
    for j in 1:num_step
        obs = observe(env)
        if sum(obs) == 0 break end
        # use previous prediction to compute loss
        log1[j, i] = loss(agent, obs)
        log2[j, i] = loss_center(agent, obs)
        # act = rand(4) .> 0.5 # 2D move
        act = [rand(2); zeros(2)] .> 0.5 # 1D move
        step!(agent, obs, act, train)
        step!(env, act)
    end
end
@info "Number of nodes" nv(agent.g)
p1 = heatmap(clamp.(log1, 0, 1))
p2 = heatmap(clamp.(log2, 0, 1))
plot(p1, p2, layout=(2, 1), size=(700, 1000))