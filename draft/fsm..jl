using Printf

# generate data
function getdata(len::Int, mod::Int)
    # X = rand([0,1], len)
    X = rand([1], len)
    Y = zeros(Int, len)
    Y[1] = X[1]
    cnt = X[1]
    for i in 2:len
        cnt += X[i]
        Y[i] = cnt % mod == 1
    end
    return X, Y
end

mutable struct Model
    nodes::Int64 # number of nodes
    maxnodes::Int64 # maximum number of nodes
    act::Array{Float32} # activity
    W::Array{Float32, 2} # weight
    pred::Array{Float32, 2} # prediction
end
# initialize model
function init()
    # node 1 -> always on
    nodes = 1
    maxnodes = 4
    m = Model(
        nodes, 
        maxnodes,
        zeros(maxnodes), 
        zeros(maxnodes, maxnodes), 
        ones(2, maxnodes))
    m.act[1] = 1
    m.W[1,1] = 0.1
    return m
end

function pred(m, winner)
    return m.pred[1, winner] > m.pred[2, winner] ? 0 : 1
end

function learn!(m, x, y)
    # update state activity (forward inference)
    pre_act = copy(m.act)
    # m.act[1] = 1
    m.act = m.W * m.act
    # m.act[1] = 1
    m.act ./= sum(m.act)
    # m.act[2:end] ./= sum(m.act[2:end]) + 1e-9
    # println(m.act)
    # find the winning node
    winner = argmax(m.act)
    predict = pred(m, winner)
    pre_winner = winner
    if predict != y && m.nodes < m.maxnodes
        # generate a new node
        m.nodes += 1
        winner = m.nodes
        m.act[winner] = 100
        m.act ./= sum(m.act)
        # expand nodes
        # if m.nodes >= m.maxnodes
        #     m.act = vcat(m.act, zeros(m.maxnodes))
        #     m.W = vcat(hcat(m.W, zeros(m.maxnodes, m.maxnodes)),
        #                zeros(m.maxnodes, m.maxnodes * 2))
        #     m.pred = vcat(m.pred, zeros(m.maxnodes))
        #     m.maxnodes *= 2
        #     println("Expand to size ", m.maxnodes)
        # end
    else
        # backward inference
        m.act .*= m.pred[y + 1,:]
        m.act ./= sum(m.act)
        winner = argmax(m.act)
    end
    # update prediction
    lr = 1
    m.pred[y + 1, winner] += lr
    # m.pred[y + 1, :] .+= lr .* m.act
    # connect the winning/new node to active states
    # m.W[winner, pre_winner] += lr
    m.W .+= lr .* (pre_act * m.act')
    # -> in graphs, this updates the neighbors of attention
    # println("winner ", winner, " pred ", predict, " y ", y)
    # println("act ", pre_act, m.act)
    # println("pred ", m.pred)
    # println("W ", m.W, pre_act * m.act')
    return predict
end

# formattings
Base.show(io::IO, f::Float32) = @printf(io, "%1.2f", f)

model = init()
for epoch in 1:1
    println("Epoch: ", epoch)
    X, Y = getdata(100, 2)
    CM = zeros(Int, 2, 2) # confusion matrix
    for i in 1:length(X)
        predict = learn!(model, X[i], Y[i])
        CM[predict + 1, Y[i] + 1] += 1
    end
    println(CM)
end
# for epoch in 1:1
#     println("Epoch: ", epoch)
#     X, Y = getdata(1000, 3)
#     CM = zeros(Int, 2, 2) # confusion matrix
#     for i in 1:length(X)
#         predict = learn!(model, X[i], Y[i])
#         CM[predict + 1, Y[i] + 1] += 1
#     end
#     println(CM)
# end