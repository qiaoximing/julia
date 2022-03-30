inlcude("ParseBUHeuristics.jl")

"Particle state"
struct BUParticle
    w::Float32 # particle weight
    stack::LinkedList{BUState} # bottom-up parse stack
    values::Vector{Int} # values in BUState
    ances::Union{BUParticle, Nothing} # ancestor of the particle
end
Base.copy(p::BUParticle) = BUParticle(p.w, p.stack, copy(p.values), p.ances)

"Helper function to get sym/input/output values from a particle"
function get_sio(ptl, depth)
    stack = ptl.stack
    if depth > length(stack)
        return nothing, nothing, nothing
    else
        for i in 2:depth stack = tail(stack) end
        h = head(stack)
        return ptl.values[h.sym], ptl.values[h.input], ptl.values[h.output]
    end
end

"Helper function to compare two values"
match(a, b) = a == 0 || b == 0 || a == b

"Evaluate weights for action sampling"
function action_weights(ptl, h, g)
    # get the top symbols
    As, Ai, Ao = get_sio(ptl, 1)
    Bs, Bi, Bo = get_sio(ptl, 2)
    Cs, Ci, Co = get_sio(ptl, 3)
    # init weights
    w_sft = Float32(0)
    w_A = zeros(Float32, g.n_cm)
    w_Ae = zeros(Float32, g.n_cm)
    w_eA = zeros(Float32, g.n_cm)
    w_BA = zeros(Float32, g.n_cm)
    w_Aee = zeros(Float32, g.n_cm)
    w_eAe = zeros(Float32, g.n_cm)
    w_BAe = zeros(Float32, g.n_cm)
    w_eeA = zeros(Float32, g.n_cm)
    w_BeA = zeros(Float32, g.n_cm)
    w_eBA = zeros(Float32, g.n_cm)
    w_CBA = zeros(Float32, g.n_cm)
    # enumerate parent symbol (Ps) from all composite symbols
    for Ps in 1:g.n_cm
        Pt = g.type[Ps]
        if Pt == U
            w_A[Ps] += get_prob(h, Ps, Ai, Ao) * g.h.p_u[As, Ps]
        elseif Pt in (Ll, Lr, Pl, Pr)
            # A as left child, unknown right child
            Pi = Ai
            Po = Pt in (Ll, Pl) ? Ao : 0
            p = get_prob(h, Ps, Pi, Po)
            e = get_nul_left(h, g, Ps, As)
            w_Ae[Ps] += p * e # not observed
            w_sft += p * (1 - e) # shift if observed (Az)
            # A as right child, unknown null left child
            Pi = Pt in (Ll, Lr) ? 0 : Ai
            Po = Pt in (Ll, Pl) ? 0 : Ao
            p = get_prob(h, Ps, Pi, Po)
            e = get_nul_right(h, g, Ps, As)
            w_eA[Ps] += p * e
            # A as right child, B as left child
            if Bs !== nothing && (Pt in (Ll, Lr) ? match(Bo, Ai) : match(Bi, Ai))
                Pi = Pt in (Ll, Lr) ? Bi : max(Bi, Ai)
                Po = Pt in (Ll, Pl) ? Bo : Ao
                w_BA[Ps] += get_prob(h, Ps, Pi, Po) * g.h.p_cm2[As, Bs, Ps] 
            end
        elseif Pt in (Llr, Lrl, Plr, Prl)
            # A left, unknown right
            Pi = Ai
            Po = 0
            p = get_prob(h, Ps, Pi, Po)
            if Pt in (Llr, Plr) 
                e1 = get_nul_left(h, g, Ps, As) # null right
                e2 = Ao > 0 ? get_nul(h, Ao) : get_nul_out(h, As, Ai) # null dynam
                ee = e1 * e2 # assume independent right and dynam
            else # ?rl
                Di = Ao # dynam in
                ee = normalize(g.w_cm2[:, As, Ps])' * (h.nul .* get_nul_out_syms(h, Di))
            end
            w_Aee[Ps] += p * ee # both unobserved
            w_sft += p * (1 - ee) # any is observed (Aze, Aez, Azz)
            # A right, unknown null left
            Pi = Pt in (Llr, Lrl) ? 0 : Ai
            Po = 0
            p = get_prob(h, Ps, Pi, Po)
            if Pt in (Llr, Plr) 
                Di = Ao # dynam in
                ee = normalize(g.w_cm2[As, :, Ps])' * (h.nul .* get_nul_out_syms(h, Di))
                ez = normalize(g.w_cm2[As, :, Ps])' * (h.nul .* (1 .- get_nul_out_syms(h, Di)))
            else
                e1 = get_nul_right(h, g, Ps, As)
                e2 = Ao > 0 ? get_nul(h, Ao) : get_nul_out(h, As, Ai) # null dynam
                ee = e1 * e2 
                ez = e1 * (1 - e2)
            end
            w_eAe[Ps] += p * ee
            w_sft += p * ez # shift if (eAz)
            # A right, B left
            if Bs !== nothing && (Pt in (Llr, Lrl) ? match(Bo, Ai) : match(Bi, Ai))
                Pi = Pt in (Llr, Lrl) ? Bi : max(Bi, Ai)
                Po = 0
                p = get_prob(h, Ps, Pi, Po) * g.h.p_cm2[As, Bs, Ps] 
                if Pt in (Llr, Plr)
                    e = Bo > 0 : get_nul(h, Bo) : get_nul_out(h, Bs, Bi)
                else
                    e = Bo > 0 : get_nul(h, Ao) : get_nul_out(h, As, Ai)
                end
                w_BAe[Ps] += p * e
                w_sft += p * (1 - e) # shift if (BAz)
            end
            # A dynam, unknown left and right
            Pi = 0
            Po = Ao
            p = get_prob(h, Ps, Pi, Po)
            ee = h.nul' * g.h.p_cm2[:, :, Ps] * h.nul
            w_eeA[Ps] += p * e
            # A dynam, B left, unknown right
            if Bs !== nothing 
                Pi = Bi
                Po = Ao
                p = get_prob(h, Ps, Pi, Po)
                if Pt in (Llr, Plr) && match(Bo, As)
                    e = get_nul_left(h, g, Ps, Bs)
                    w_BeA[Ps] += p * e
                elseif Pt in (Lrl, Prl)
                    e = get_nul_left(h, g, Ps, Bs)
                    w_BeA[Ps] += p * e
                end
            end
            # A dynam, B right, unknown left
            if Bs !== nothing 
                Pi = Pt in (Llr, Lrl) ? 0 : Bi
                Po = Ao
                p = get_prob(h, Ps, Pi, Po)
                if Pt in (Llr, Plr)
                    e = get_nul_right(h, g, Ps, Bs)
                    w_eBA[Ps] += p * e
                elseif Pt in (Lrl, Prl) && match(Bo, As)
                    e = get_nul_right(h, g, Ps, Bs)
                    w_eBA[Ps] += p * e
                end
            end
            # A dynam, B right, C left
            if Bs !== nothing && Cs !== nothing
                if (Pt in (Llr, Lrl) ? match(Co, Bi) : match(Ci, Bi)) &&
                (Pt in (Llr, Plr) ? match(Co, As) && match(Bo, Ai) : 
                                    match(Bo, As) && match(Co, Ai))
                    Pi = Pt in (Llr, Lrl) ? Ci : max(Ci, Bi)
                    Po = A0
                    w_CBA[Ps] += get_prob(h, Ps, Pi, Po) * g.h.p_cm2[Bs, Cs, Ps]
                end
            end
        end
    end
    return ([w_sft], w_A, w_Ae, w_eA, w_BA, 
            w_Aee, w_eAe, w_BAe, w_eeA, w_BeA, w_eBA, w_CBA)
end

function perform_action!(ptl, action, action_Ps, obs)
    stack = ptl.stack
    function pushval!(x=0)
        push!(ptl.values, x) 
        return length(ptl.values)
    end
    function pushexp!(x...)
        push!(ptl.expans, x...) 
    end
    if action == :sft
        # new observation Z
        Zs = pushval!(obs)
        Zi = pushval!()
        Zo = pushval!()
        Z = BUState(Zs, Zi, Zo)
        # push Z
        stack_new = cons(Z, stack)
    else
        Ps = pushval!(action_Ps)
        Pt = g.type[values[Ps]]
        A = head(stack)
        As, Ai, Ao = A.sym, A.in, A.out
        if length(stack > 1)
            B = head(tail(stack))
            Bs, Bi, Bo = B.sym, B.in, B.out
        end
        if action == :A
            # pop A, push P
            Pi = Ai
            Po = Ao
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(stack))
        elseif action == :Ae
            # new null symbol E
            Es = pushval!()
            Ei = Pt in (Ll, Lr) ? Ao : Ai
            Eo = pushval!()
            E = BUState(Es, Ei, Eo)
            pushexp!(E)
            # pop A, push P
            Pi = Ai
            Po = Pt in (Ll, Pl) ? Ao : Eo
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(stack))
        elseif action == :eA
            # new null symbol E
            Es = pushval!()
            Ei = Pt in (Ll, Lr) ? pushval!() : Ai
            Eo = Pt in (Ll, Lr) ? Ai : pushval!()
            E = BUState(Es, Ei, Eo)
            pushexp!(E)
            # pop A, push P
            Pi = Ei
            Po = Pt in (Ll, Pl) ? Ao : Eo
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(stack))
        elseif action == :BA
            # merge values
            succ = Pt in (Ll, Lr) ? mergeval!(Bo, Ai) : mergeval!(Bi, Ai)
            succ || warning("Merge fail")
            # pop A, pop B, push P
            Pi = Bi
            Po = Pt in (Ll, Pl) ? Bo : Ao
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(tail(stack)))
        elseif action == :Aee
            # new null symbols E1 E2
            E1s = pushval!()
            E1i = Pt in (Llr, Lrl) ? Ao : Ai
            E1o = pushval!()
            E2s = Pt in (Llr, Plr) ? Ao : E1o
            E2i = Pt in (Llr, Plr) ? E1o : Ao
            E2o = pushval!()
            E1, E2 = BUState(E1s, E1i, E1o), BUState(E2s, E2i, E2o)
            pushexp!(E1, E2)
            # pop A, push P
            Pi = Ai
            Po = E2o
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(stack))
        elseif action == :eAe
            # new null symbols E1 E2
            E1s = pushval!()
            E1i = Pt in (Llr, Lrl) ? pushval!() : Ai
            E1o = Pt in (Llr, Lrl) ? Ai : pushval!()
            E2s = Pt in (Llr, Plr) ? E1o : Ao
            E2i = Pt in (Llr, Plr) ? Ao : E1o
            E2o = pushval!()
            E1, E2 = BUState(E1s, E1i, E1o), BUState(E2s, E2i, E2o)
            pushexp!(E1, E2)
            # pop A, push P
            Pi = E1i
            Po = E2o
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(stack))
        elseif action == :BAe
            # merge values
            succ = Pt in (Llr, Lrl) ? mergeval!(Bo, Ai) : mergeval!(Bi, Ai)
            succ || warning("Merge fail")
            # new null symbol E
            Es = Pt in (Llr, Plr) ? Bo : Ao
            Ei = Pt in (Llr, Plr) ? Ao : Bo
            Eo = pushval!()
            E = BUState(Es, Ei, Eo)
            pushexp!(E)
            # pop A, pop B, push P
            Pi = Bi
            Po = Pt in (Ll, Pl) ? Bo : Ao
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(tail(stack)))
        elseif action == :eeA
            # new null symbols E1 E2
            E1s = pushval!()
            E1i = pushval!()
            E1o = Pt in (Llr, Plr) ? As : Ai
            E2s = pushval!()
            E2i = Pt in (Llr, Lrl) ? E1o : E1i
            E2o = Pt in (Llr, Plr) ? Ai : As
            E1, E2 = BUState(E1s, E1i, E1o), BUState(E2s, E2i, E2o)
            pushexp!(E1, E2)
            # pop A, push P
            Pi = E1i
            Po = Ao
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(stack))
        elseif action == :BeA
            # merge values
            succ = Pt in (Llr, Plr) ? mergeval!(Bo, As) : mergeval!(Bo, Ai)
            succ || warning("Merge fail")
            # new null symbol E
            Es = pushval!()
            Ei = Pt in (Llr, Lrl) ? Bo : Bi
            Eo = Pt in (Llr, Plr) ? Ai : As
            E = BUState(Es, Ei, Eo)
            pushexp!(E)
            # pop A, pop B, push P
            Pi = Bi
            Po = Ao
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(tail(stack)))
        elseif action == :eBA
            # merge values
            succ = Pt in (Llr, Plr) ? mergeval!(Bo, Ai) : mergeval!(Bo, As)
            succ || warning("Merge fail")
            # new null symbol E
            Es = pushval!()
            Ei = Pt in (Llr, Lrl) ? pushval!() : Bi
            Eo = Pt in (Llr, Lrl) ? Bi : pushval!()
            E = BUState(Es, Ei, Eo)
            pushexp!(E)
            # pop A, pop B, push P
            Pi = Ei
            Po = Ao
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(tail(stack)))
        else # action == :CBA
            # merge values
            C = head(tail(tail(stack)))
            Cs, Ci, Co = C.sym, C.in, C.out
            succ = Pt in (Llr, Plr) ? mergeval!(Bo, As) : mergeval!(Bo, Ai)
            succ &= Pt in (Llr, Plr) ? mergeval!(Co, Ai) : mergeval!(Co, As)
            succ || warning("Merge fail")
            # pop A, pop B, pop C, push P
            Pi = Ci
            Po = Ao
            P = BUState(Ps, Pi, Po)
            stack_new = cons(P, tail(tail(tail(stack))))
        end
    end
    ptl.stack = stack_new
end

"Progress a particle forward to the next observation"
function forward(ptl::BUParticle, obs::Int, g::GrammarEx, h::BUHeuristics, max_iter::Int)
    iter = 1
    while !isa(ptl.stack, Nil) # stack not empty
        if iter > max_iter
            debugln("Too many iterations between observation")
            return BUParticle(0., ptl.stack, ptl.values, ptl)
        else
            iter += 1
        end
        # evaluate weights for action sampling
        w_act = vcat(action_weights(ptl, h, g)...)
        # evaluate weights for value sampling
        for idx in 1:length(ptl.values)
        # evaluate weights for expansion sampling

        # perform actions 
        perform_action!(ptl, action, action_Ps, obs)
        # sample value
        
        # perform expansion
    end
end
function parse_bupf(str::String, g::GrammarEx, h::BUHeuristics, n_particle::Int=4, max_iter::Int=100, training::Bool=false)
    # compile function and rule probabilities
    if training
        # p_fn = normalize(g.w_fn .+ g.alpha, dims=1)
        # p_cn = normalize(g.w_cn .+ g.alpha, dims=1)
        # p_left = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1 .+ g.alpha, dims=1)
        # p_right = normalize(g.w_cm2 .+ g.alpha, dims=1)
    else
        p_fn = normalize(g.w_fn, dims=1)
        p_cn = normalize(g.w_cn, dims=1)
        p_u = normalize(g.w_cm1, dims=1)
        p_l = normalize(sum(g.w_cm2, dims=1)[1, :, :], dims=1)
        p_r = normalize(sum(g.w_cm2, dims=2)[:, 1, :], dims=1)
        p_cm2 = normalize(g.w_cm2, dims=(1,2))
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
            particles[i] = forward(p, obs, g, h, max_iter)
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
        particles[i] = forward(p, 0, g, h, max_iter)
    end
    return particles, logprob
end