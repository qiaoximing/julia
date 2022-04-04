inlcude("ParseBUHeuristics.jl")

"Factor for value sampling"
struct Factor 
    state::BUState
    relation::Symbol # one of (:in, :out, :left, :right, :dynam)
    enable::Bool # true if enabled
end
"Return a disabled copy of Factor"
disable(f::Factor) = Factor(f.state, f.relation, false)

"Merge two vectors of factors"
function union(f1::Vector{Factor}, f2::Vector{Factor})
    # TODO: efficient implementation by linked list with tail pointer
    return vcat(f1, f2)
end

"Particle state"
struct BUParticle
    w::Float32 # particle weight
    bustack::LinkedList{BUState} # bottom-up parse stack
    tdstack::LinkedList{TDState} # top-down parse stack
    values::Vector{Int} # values in BUState
    dset::IntDisjointSet # disjoint set for value indexing
    factors::Vector{Vector{Factor}} # related factors for each value
    tosample::Vector{Bool} # flag of values to sample
    ances::Union{BUParticle, Nothing} # ancestor of the particle
end
Base.copy(p::BUParticle) = BUParticle(p.w, p.bustack, p.tdstack, 
    copy(p.values), deepcopy(p.dset), copy(p.factors), copy(tosample), p.ances)

"Get value through root of disjoint set"
function getval(ptl, x)
    return ptl.values[find_root!(ptl.dset, x)]
end

"Check if a value is sampled"
function tosample(ptl, x)
    return ptl.tosample[find_root!(ptl.dset, x)]
end

"Helper function to get sym/input/output values from a particle"
function get_sio(ptl, depth)
    stack = ptl.bustack
    if depth > length(stack)
        return nothing, nothing, nothing
    else
        for i in 2:depth stack = tail(stack) end
        h = head(stack)
        return getval(ptl, h.sym), getval(ptl, h.input), getval(ptl, h.output)
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
            w_A[Ps] += get_prob(h, Ps, Ai, Ao) * g.h.p_l[As, Ps]
        elseif Pt in (Ll, Lr, Pl, Pr)
            # A as left child, unknown right child
            Pi = Ai
            Po = Pt in (Ll, Pl) ? Ao : 0
            p = get_prob(h, Ps, Pi, Po)
            e = get_nul_right(h, g, Ps, As)
            w_Ae[Ps] += p * e # not observed
            w_sft += p * (1 - e) # shift if observed (Az)
            # A as right child, unknown null left child
            Pi = Pt in (Ll, Lr) ? 0 : Ai
            Po = Pt in (Ll, Pl) ? 0 : Ao
            p = get_prob(h, Ps, Pi, Po)
            e = get_nul_left(h, g, Ps, As)
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
                e1 = get_nul_right(h, g, Ps, As) # null right
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
                e1 = get_nul_left(h, g, Ps, As)
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
                    e = get_nul_right(h, g, Ps, Bs)
                    w_BeA[Ps] += p * e
                elseif Pt in (Lrl, Prl)
                    e = get_nul_right(h, g, Ps, Bs)
                    w_BeA[Ps] += p * e
                end
            end
            # A dynam, B right, unknown left
            if Bs !== nothing 
                Pi = Pt in (Llr, Lrl) ? 0 : Bi
                Po = Ao
                p = get_prob(h, Ps, Pi, Po)
                if Pt in (Llr, Plr)
                    e = get_nul_left(h, g, Ps, Bs)
                    w_eBA[Ps] += p * e
                elseif Pt in (Lrl, Prl) && match(Bo, As)
                    e = get_nul_left(h, g, Ps, Bs)
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

function decode_action(x, n)
    if x == 1
        return :sft, 0
    else
        div, rem = divrem(x - 1, n)
        action = (:A, :Ae, :eA, :BA, :Aee, :eAe, :BAe, :eeA, :BeA, :eBA, :CBA)[div + 1]
        action_Ps = rem + 1
        return action, action_Ps
    end
end

function mergeval!(ptl, x, y)
    rx, ry = find_root(ptl.dset, x), find_root(ptl.dset, y)
    if rx == ry
        # already merged
    elseif ptl.tosample[rx] && ptl.tosample[ry]
        # merge dset and factors
        union!(ptl.dset, x, y)
        r_new = find_root(ptl.dset, x)
        r_other = r_new == rx ? ry : rx
        ptl.factors[r_new] = union(ptl.factors[rx], ptl.factors[ry])
        ptl.factors[r_other] = []
        ptl.tosample[r_other] = false
    elseif !ptl.tosample[rx] && !ptl.tosample[ry]
        # check equality
        ptl.values[rx] == ptl.values[ry] || return false
        union!(ptl.dset, x, y)
    else # set the sampled value to the unsampled one
        idx = ptl.tosample[rx] ? rx : ry
        val = ptl.values[ptl.tosample[rx] ? ry : rx]
        setval!(ptl, g, idx, val)
        union!(ptl.dset, x, y)
    end
    return true
end

function pushval!(ptl, x=0)
    push!(ptl.values, x) 
    push!(ptl.tosample, x == 0 ? true : false)
    push!(ptl.dset)
    # should always have length(ptl.dset) == length(ptl.values)
    return length(ptl.values)
end
function addfactor!(ptl, x, state, relation)
    rx = find_root(ptl.dset, x)
    if ptl.tosample[rx]
        push!(ptl.factors[rx], Factor(state, relation, true))
    end
end
function removefactor!(ptl, x, state)
    rx = find_root(ptl.dset, x)
    for (idx, factor) in enumerate(ptl.factors[rx])
        if factor.state === state
            ptl.factors[rx][idx] = disable(factor)
        end
    end
end
function addleaf!(ptl, sym, in, out)
    state = BUState(sym, in, out)
    addfactor!(ptl, in, state, :in)
    addfactor!(ptl, out, state, :out)
    return state
end
function setroot!(ptl, state, left=nothing, right=nothing, dynam=nothing)
    isnothing(left) || addfactor!(ptl, left, state, :left)
    isnothing(right) || addfactor!(ptl, right, state, :right)
    isnothing(dynam) || addfactor!(ptl, dynam, state, :dynam)
end
function addroot!(ptl, sym, in, out, left=nothing, right=nothing, dynam=nothing)
    state = BUState(sym, in, out)
    setroot!(ptl, state, left, right, dynam)
    return state
end

function perform_action!(ptl, g, action, action_Ps, obs)
    bustack = ptl.bustack
    if action == :sft
        # new observation Z
        Zs = pushval!(ptl, obs)
        Zi = pushval!(ptl)
        Zo = Zi
        Z = BUState(Zs, Zi, Zo)
        # push Z
        bustack_new = cons(Z, bustack)
    else
        Ps = pushval!(ptl, action_Ps)
        Pt = g.type[getval(ptl, Ps)]
        A = head(bustack)
        As, Ai, Ao = A.sym, A.in, A.out
        if length(bustack > 1)
            B = head(tail(bustack))
            Bs, Bi, Bo = B.sym, B.in, B.out
        end
        if action == :A
            # pop A, push P
            Pi = Ai
            Po = Ao
            P = addroot!(ptl, Ps, Pi, Po)
            bustack_new = cons(P, tail(bustack))
        elseif action == :Ae
            # new null symbol E
            Es = pushval!(ptl)
            Ei = Pt in (Ll, Lr) ? Ao : Ai
            Eo = pushval!(ptl)
            E = addleaf!(ptl, Es, Ei, Eo)
            # pop A, push P
            Pi = Ai
            Po = Pt in (Ll, Pl) ? Ao : Eo
            P = addroot!(ptl, Ps, Pi, Po, nothing, Es)
            bustack_new = cons(P, tail(bustack))
        elseif action == :eA
            # new null symbol E
            Es = pushval!(ptl)
            Ei = Pt in (Ll, Lr) ? pushval!(ptl) : Ai
            Eo = Pt in (Ll, Lr) ? Ai : pushval!(ptl)
            E = addleaf!(ptl, Es, Ei, Eo)
            # pop A, push P
            Pi = Ei
            Po = Pt in (Ll, Pl) ? Ao : Eo
            P = addroot!(ptl, Ps, Pi, Po, Es)
            bustack_new = cons(P, tail(bustack))
        elseif action == :BA
            # merge values
            succ = Pt in (Ll, Lr) ? mergeval!(ptl, Bo, Ai) : mergeval!(ptl, Bi, Ai)
            succ || warning("Merge fail")
            # pop A, pop B, push P
            Pi = Bi
            Po = Pt in (Ll, Pl) ? Bo : Ao
            P = addroot!(ptl, Ps, Pi, Po)
            bustack_new = cons(P, tail(tail(bustack)))
        elseif action == :Aee
            # new null symbols E1 E2
            E1s = pushval!(ptl)
            E1i = Pt in (Llr, Lrl) ? Ao : Ai
            E1o = pushval!(ptl)
            E2s = Pt in (Llr, Plr) ? Ao : E1o
            E2i = Pt in (Llr, Plr) ? E1o : Ao
            E2o = pushval!(ptl)
            E1, E2 = addleaf!(ptl, E1s, E1i, E1o), addleaf!(ptl, E2s, E2i, E2o)
            # pop A, push P
            Pi = Ai
            Po = E2o
            P = addroot!(ptl, Ps, Pi, Po, nothing, E1s, E2s)
            bustack_new = cons(P, tail(bustack))
        elseif action == :eAe
            # new null symbols E1 E2
            E1s = pushval!(ptl)
            E1i = Pt in (Llr, Lrl) ? pushval!(ptl) : Ai
            E1o = Pt in (Llr, Lrl) ? Ai : pushval!(ptl)
            E2s = Pt in (Llr, Plr) ? E1o : Ao
            E2i = Pt in (Llr, Plr) ? Ao : E1o
            E2o = pushval!(ptl)
            E1, E2 = addleaf!(ptl, E1s, E1i, E1o), addleaf!(ptl, E2s, E2i, E2o)
            # pop A, push P
            Pi = E1i
            Po = E2o
            P = addroot!(ptl, Ps, Pi, Po, E1s, nothing, E2s)
            bustack_new = cons(P, tail(bustack))
        elseif action == :BAe
            # merge values
            succ = Pt in (Llr, Lrl) ? mergeval!(ptl, Bo, Ai) : mergeval!(ptl, Bi, Ai)
            succ || warning("Merge fail")
            # new null symbol E
            Es = Pt in (Llr, Plr) ? Bo : Ao
            Ei = Pt in (Llr, Plr) ? Ao : Bo
            Eo = pushval!(ptl)
            E = addleaf!(ptl, Es, Ei, Eo)
            # pop A, pop B, push P
            Pi = Bi
            Po = Pt in (Ll, Pl) ? Bo : Ao
            P = BUState(Ps, Pi, Po, nothing, nothing, Es)
            bustack_new = cons(P, tail(tail(bustack)))
        elseif action == :eeA
            # new null symbols E1 E2
            E1s = pushval!(ptl)
            E1i = pushval!(ptl)
            E1o = Pt in (Llr, Plr) ? As : Ai
            E2s = pushval!(ptl)
            E2i = Pt in (Llr, Lrl) ? E1o : E1i
            E2o = Pt in (Llr, Plr) ? Ai : As
            E1, E2 = addleaf!(ptl, E1s, E1i, E1o), addleaf!(ptl, E2s, E2i, E2o)
            # pop A, push P
            Pi = E1i
            Po = Ao
            P = addroot!(ptl, Ps, Pi, Po, E1s, E2s)
            bustack_new = cons(P, tail(bustack))
        elseif action == :BeA
            # merge values
            succ = Pt in (Llr, Plr) ? mergeval!(ptl, Bo, As) : mergeval!(ptl, Bo, Ai)
            succ || warning("Merge fail")
            # new null symbol E
            Es = pushval!(ptl)
            Ei = Pt in (Llr, Lrl) ? Bo : Bi
            Eo = Pt in (Llr, Plr) ? Ai : As
            E = addleaf!(ptl, Es, Ei, Eo)
            # pop A, pop B, push P
            Pi = Bi
            Po = Ao
            P = addroot!(ptl, Ps, Pi, Po, nothing, Es)
            bustack_new = cons(P, tail(tail(bustack)))
        elseif action == :eBA
            # merge values
            succ = Pt in (Llr, Plr) ? mergeval!(ptl, Bo, Ai) : mergeval!(ptl, Bo, As)
            succ || warning("Merge fail")
            # new null symbol E
            Es = pushval!(ptl)
            Ei = Pt in (Llr, Lrl) ? pushval!(ptl) : Bi
            Eo = Pt in (Llr, Lrl) ? Bi : pushval!(ptl)
            E = addleaf!(ptl, Es, Ei, Eo)
            # pop A, pop B, push P
            Pi = Ei
            Po = Ao
            P = addroot!(ptl, Ps, Pi, Po, Es)
            bustack_new = cons(P, tail(tail(bustack)))
        else # action == :CBA
            # merge values
            C = head(tail(tail(bustack)))
            Cs, Ci, Co = C.sym, C.in, C.out
            succ = Pt in (Llr, Plr) ? mergeval!(ptl, Bo, As) : mergeval!(ptl, Bo, Ai)
            succ &= Pt in (Llr, Plr) ? mergeval!(ptl, Co, Ai) : mergeval!(ptl, Co, As)
            succ || warning("Merge fail")
            # pop A, pop B, pop C, push P
            Pi = Ci
            Po = Ao
            P = addroot!(ptl, Ps, Pi, Po)
            bustack_new = cons(P, tail(tail(tail(bustack))))
        end
    end
    ptl.bustack = bustack_new
end

function setval!(ptl, g, idx, val)
    # update value
    ptl.values[idx] = val
    ptl.tosample[idx] = false
    # perform top-down expansion
    for factor in ptl.factors[idx]
        if factor.relation in (:left, :right, :dynam)
            type = g.type[val]
            if type == Tr
                warning("Expanding an observable")
            elseif type == Id
                mergeval!(ptl, factor.state.in, factor.state.out)
            elseif type in (Fn, Cn)
                # do nothing
            else
                P = factor.state
                Ps, Pi, Po = P.sym, P.in, P.out
                # remove factors related to P
                removefactor!(ptl, Pi, P)
                removefactor!(ptl, Po, P)
                # expand and add new factors
                if type == U 
                    Es = pushval!(ptl)
                    Ei = Pi
                    Eo = Po
                    E = addleaf!(ptl, Es, Ei, Eo)
                    setroot!(ptl, P, E)
                elseif type in (Ll, Lr, Pl, Pr)
                    E1s = pushval!(ptl)
                    E1i = Pi
                    E1o = type in (Ll, Pl) ? Po : pushval!(ptl)
                    E2s = pushval!(ptl)
                    E2i = type in (Ll, Lr) ? E1o : Pi
                    E2o = type in (Ll, Pl) ? pushval!(ptl) : Po
                    E1 = addleaf!(ptl, E1s, E1i, E1o)
                    E2 = addleaf!(ptl, E2s, E2i, E2o)
                    setroot!(ptl, P, E1, E2)
                else # type in (Llr, Lrl, Plr, Prl)
                    E1s = pushval!(ptl)
                    E1i = Pi
                    E1o = pushval!(ptl)
                    E2s = pushval!(ptl)
                    E2i = type in (Llr, Lrl) ? E1o : Pi
                    E2o = pushval!(ptl)
                    E3s = type in (Llr, Plr) ? E1o : E2o
                    E3i = type in (Llr, Plr) ? E2o : E1o
                    E3o = Po
                    E1 = addleaf!(ptl, E1s, E1i, E1o)
                    E2 = addleaf!(ptl, E2s, E2i, E2o)
                    E3 = addleaf!(ptl, E3s, E3i, E3o)
                    setroot!(ptl, P, E1, E2, E3)
                end
            end
        end
    end
    # cleanup
    ptl.factors[idx] = []
end

"Return unnormalized weights of a factor distribution"
function eval(ptl::BUParticle, factor::Factor, g, h)
    # uniform weights by default
    w_def = ones(Float32, g.size)
    # disabled factor
    factor.enable || return w_def
    # unsampled symbol
    state = factor.state
    sym, in, out = map(x->getval(ptl, x), (state.sym, state.in, state.out))
    sym == 0 && return w_def
    rel = factor.relation
    type = g.type[sym]
    if type in (Id, Tr)
        # onehot_(x) = x > 0 ? onehot(g.size, x) : w_def
        warning("Factor on Id or Tr")
    elseif type == Cn
        rel == :out && return h.out[:, sym]
        # rel == :out && return g.h.p_cn[:, offset(sym, g)]
    elseif type == Fn
        rel == :out && in > 0 && return h.oi[:, in, sym]
        rel == :in && out > 0 && return h.oi[out, :, sym]
        # rel == :out && in > 0 && return g.h.p_fn[:, in, offset(sym, g)] 
        # rel == :in && out > 0 && return g.h.p_fn[out, :, offset(sym, g)] 
    else
        rel == :out && in > 0 && return h.oi[:, in, sym] 
        rel == :in && out > 0 && return h.oi[out, :, sym] 
        # TODO: get left and right
        rel == :left && return get_nul_left(h, g, sym, right)
        rel == :right && return get_nul_right(h, g, sym, left)
        # do nothing for :dynam
    end
    return w_def
end

"Progress a particle forward to the next observation"
function forward(ptl::BUParticle, obs::Int, g::GrammarEx, h::BUHeuristics, max_iter::Int)
    iter = 1
    while !isa(ptl.bustack, Nil) # stack not empty
        if iter > max_iter
            debugln("Too many iterations between observation")
            return BUParticle(0., ptl.bustack, ptl.values, ptl)
        else
            iter += 1
        end
        max_memory = 10
        if sum(ptl.tosample) > max_memory
            debugln("Too many items in working memory")
            return BUParticle(0., ptl.bustack, ptl.values, ptl)
        end
        # evaluate weights for action sampling
        weight = vcat(action_weights(ptl, h, g)...)
        distr = normalize(weight)
        choice = (:action, distr)
        min_enp = entropy(distr)
        # evaluate weights for value sampling
        # note: this includes symbols and I/Os
        for idx in 1:length(ptl.values)
            ptl.tosample[idx] || continue
            # evaluate each factor and get the element-wise product
            weight = reduce((.*), eval.(ptl.factors[idx]))
            distr = normalize(weight)
            enp = entropy(distr_tmp)
            if enp < min_enp
                choice = (idx, normalize(weight))
                min_enp = enp
            end
        end
        # choose the distribution with lowest entropy
        if choice[1] == :action
            # perform actions 
            tmp, _ = sample(choice[2])
            action, action_Ps = decode_action(tmp, g.n_cm)
            ptl = perform_action(ptl, g, action, action_Ps, obs)
        else
            # sample value
            idx = choice[1]
            val, _ = sample(choice[2])
            setval!(ptl, g, idx, val)
        end
    end
    return ptl
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
        p_l = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1, dims=1)
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