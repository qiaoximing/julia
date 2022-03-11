inlcude("ParseBUHeuristics.jl")

"Particle state"
struct BUParticle
    w::Float32 # particle weight
    stack::LinkedList{BUState} # bottom-up parse stack
    values::Vector{Int} # values in BUState
    ances::Union{BUParticle, Nothing} # ancestor of the particle
end
Base.copy(p::BUParticle) = BUParticle(p.w, p.stack, copy(p.values), p.ances)

"Progress a particle forward to the next observation"
function forward(ptl::BUParticle, obs::Int, g::GrammarEx, max_iter::Int)
    iter = 1
    while !isa(ptl.stack, Nil) # stack not empty
        if iter > max_iter
            debugln("Too many iterations between observation")
            return BUParticle(0., ptl.stack, ptl.values, ptl)
        else
            iter += 1
        end
        # pop the top symbol
        state = head(ptl.stack)
        stack_new = tail(ptl.stack)
        # retrieve Ref contents
        sym, input = ptl.values[state.sym], ptl.values[state.input]
        type = g.type[sym]
        # decide an action
        p_shift = h.
        weights = (p_shift, p_reduce1, p_reduce2, p_reduce3)
        action, _ = sample(normalize(weights))
        if action == 1 # decide to shift
            # shift the observation
            # return the particle
            return
        elseif actions == 2 # reduce one symbol
            # type U
            # type (Ll, Lr, Pl, Pr) with left null
            # type (Ll, Lr, Pl, Pr) with right null
            # type (Llr, Lrl, Plr, Prl) with two left nulls
            # type (Llr, Lrl, Plr, Prl) with left and right nulls
            # type (Llr, Lrl, Plr, Prl) with two right nulls
        elseif actions == 3 # reduce two symbols
            # type (Ll, Lr, Pl, Pr)
            # type (Llr, Lrl, Plr, Prl) with left 
            # type (Llr, Lrl, Plr, Prl) with middle null
            # type (Llr, Lrl, Plr, Prl) with right null
        else # reduce three symbols
            # type (Llr, Lrl, Plr, Prl)
    end
end
function parse_bupf(str::String, g::GrammarEx, h::BUHeuristics, n_particle::Int=4, max_iter::Int=100, training::Bool=false)
    # compile function and rule probabilities
    if training
        p_fn = normalize(g.w_fn .+ g.alpha, dims=1)
        p_cn = normalize(g.w_cn .+ g.alpha, dims=1)
        p_left = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1 .+ g.alpha, dims=1)
        p_right = normalize(g.w_cm2 .+ g.alpha, dims=1)
    else
        p_fn = normalize(g.w_fn, dims=1)
        p_cn = normalize(g.w_cn, dims=1)
        p_left = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1, dims=1)
        p_right = normalize(g.w_cm2, dims=1)
    end
    g.h = (; p_fn, p_cn, p_left, p_right)
    # init particles
    state_init = BUState(1, 2, 3)
    obs_first = g.index[string(str[1])]
    particle_init = BUParticle(1., list(state_init), [obs_first, 0, 0], nothing)
    particles = Vector{BUParticle}()
    for i in 1:n_particle
        push!(particles, copy(particle_init))
    end
    logprob = 0. # log of marginal probability
    # start parsing
    for char in str[2:end]
        obs = g.index[string(char)]
        # progress the particle, and shift next observation
        for (i, p) in enumerate(particles)
            debugln("Simulating particle $i")
            particles[i] = forward(p, obs, g, max_iter)
        end
        # accumulate log prob
        total_weight = sum([p.w for p in particles])
        if total_weight <= 0.
            debugln("All particles fail")
            return Vector{BUParticle}(), NaN
        end
        logprob += log(total_weight / n_particle)
        # resample is effective sample size too small
        ess = Int(ceil(effective_size(particles)))
        if ess <= n_particle / 2
            particles = resample(particles, n_particle)
        end
    end
    # finalize the simulation after the last observation
    # have zero weight if the final merge fails
    for (i, p) in enumerate(particles)
        debugln("Simulating particle $i")
        particles[i] = forward(p, 0, g, max_iter)
    end
    return particles, logprob
end