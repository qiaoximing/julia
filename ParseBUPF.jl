inlcude("ParseBUHeuristics.jl")

"Factor for value sampling"
struct Factor 
    state::BUState
    relation::Symbol # one of (:in, :out)
end
"An array of factors"
mutable struct Factors
    f::Vector{Factor} # the factors
    ni::Int # number of :in factors
    no::Int # number of :out factors
end
Factors() = Factors([], 0, 0)

"Merge two vectors of factors"
function union(fs1::Factors, fs2::Factors)
    # Optim: efficient implementation by linked list with tail pointer
    return Factors(vcat(fs1.f, fs2.f), fs1.ni + fs2.ni, fs1.no + fs2.no)
end

"Particle state"
struct BUParticle
    w::Float32 # particle weight
    stack::LinkedList{BUState} # bottom-up parse stack
    values::Vector{Int} # values in BUState
    dset::IntDisjointSet # disjoint set for value indexing
    factors::Vector{Factors} # related factors for each value
    ances::Union{BUParticle, Nothing} # ancestor of the particle
end
Base.copy(p::BUParticle) = BUParticle(p.w, p.stack, 
    copy(p.values), deepcopy(p.dset), copy(p.factors), p.ances)

"Get value through root of disjoint set"
function getval(ptl, x)
    return ptl.values[find_root!(ptl.dset, x)]
end

"Helper function to get sym/input/output values from a particle"
function get_sio(ptl, depth)
    stack = ptl.stack
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
        action = (:A, :Ae, :eA, :BA, :Aee, :eAe, :BAe, 
                  :eeA, :BeA, :eBA, :CBA)[div + 1]
        action_Ps = rem + 1
        return action, action_Ps
    end
end

function mergeval!(ptl, x, y)
    rx, ry = find_root(ptl.dset, x), find_root(ptl.dset, y)
    vx, vy = ptl.values[rx], ptl.values[ry]
    if rx == ry
        # already merged
    elseif vx == 0 && vy == 0
        # merge dset and factors
        union!(ptl.dset, x, y)
        r_new = find_root(ptl.dset, x)
        r_other = r_new == rx ? ry : rx
        ptl.factors[r_new] = union(ptl.factors[rx], ptl.factors[ry])
        ptl.factors[r_other] = Factors()
    elseif vx > 0 && vy > 0
        # check equality
        vx == vy || return false
        union!(ptl.dset, x, y)
    else # set the sampled value to the unsampled one
        idx = vx == 0 ? rx : ry
        val = max(vx, vy)
        setval!(ptl, g, idx, val)
        union!(ptl.dset, x, y)
    end
    return true
end

function pushval!(ptl, x=0)
    push!(ptl.values, x) 
    push!(ptl.dset)
    # should always have length(ptl.dset) == length(ptl.values)
    return length(ptl.values)
end

function setval!(ptl, g, idx, val)
    # update value
    ptl.values[idx] = val
    ptl.factors[idx] = Factors()
end

function addfactor!(ptl, x, state, relation)
    rx = find_root(ptl.dset, x)
    if ptl.values[rx] == 0
        push!(ptl.factors[rx].f, Factor(state, relation))
        if relation == :in
            ptl.factors[rx].ni += 1
        else # relation == :out 
            ptl.factors[rx].no += 1
        end
    end
end

"Return a factor distribution or likelihood vector"
function eval(ptl::BUParticle, factor::Factor, g, h)
    # uniform weights by default
    w_def = ones(Float32, g.size)
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
    else # type in Cm
        rel == :out && in > 0 && return h.oi[:, in, sym] 
        rel == :in && out > 0 && return h.oi[out, :, sym] 
    end
    return w_def
end

function addleaf!(ptl, g; sym, in, out)
    state = BUState(sym, in, out)
    type = g.type[getval(ptl, sym)]
    if type == Tr
        warning("Expanding observations")
    elseif type == Id
        mergeval!(ptl, in, out)
    else
        addfactor!(ptl, in, state, :in)
        addfactor!(ptl, out, state, :out)
    end
    return state
end

"Perform reduction, and sample values that are ready"
function addroot!(ptl, g; sym, in, out, left=nothing, right=nothing, dynam=nothing)
    P = BUState(sym, in, out)
    type = g.type[getval(ptl, sym)]
    if type == Lr
        sampleval!(ptl, left.out)
    elseif type in (Llr, Lrl, Plr, Prl)
        sampleval!(ptl, left.out)
        sampleval!(ptl, right.out)
    end
    return P
end

function sampleval!(ptl, x)
    rx = find_root(ptl.dset, x)
    ptl.values[rx] > 0 && return
    # Must include 1 :out factor and >=1 :in factor
    # (Root I/O automatically fail this test)
    fs = ptl.factors[rx]
    if fs.no == 1 && fs.ni >= 1
        weight = reduce((.*), eval.(fs.f))
        distr = normalize(weight)
        val, _ = sample(distr)
        setval!(ptl, g, rx, val)
    end
end

function samplesym(g, h, action, Pv, Av, Bv=0)
    if action == :Ae
        left = Av
    elseif action == :eA
        right = Av
    elseif action == :Aee
        left = Av
    elseif action == :eAe
        right = Av
    elseif action == :eeA

    elseif action == :BeA
        left = Bv
    elseif action == :eBA
        right = Bv
    end
end

function perform_action!(ptl, g, action, action_Ps, obs)
    stack = ptl.stack
    if action == :sft
        # new observation Z
        Zs = pushval!(ptl, obs)
        Zi = pushval!(ptl)
        Zo = Zi
        Z = BUState(Zs, Zi, Zo)
        # push Z
        stack_new = cons(Z, stack)
    else
        # new root symbol
        Ps = pushval!(ptl, action_Ps)
        Pv = getval(ptl, Ps)
        Pt = g.type[Pv]
        # get stack top
        A = head(stack)
        Av = getval(ptl, A.sym)
        # get second stack top
        if length(stack > 1)
            B = head(tail(stack))
            Bv = getval(ptl, Bs)
        end
        if action == :A # one child (left)
            left = A
            P = addroot!(ptl, g, 
                sym=Ps, 
                in=left.in, 
                out=left.out, 
                left=left)
            stack_new = cons(P, tail(stack))
        elseif action in (:Ae, :eA, :BA) # two children (left, right)
            # prepare children
            if action == :Ae
                v = samplesym(g, h, action, Pv, Av)
                left = A
                right = addleaf!(ptl, g, 
                    sym=pushval(ptl, v), 
                    in=Pt in (Ll, Lr) ? left.out : left.in, 
                    out=pushval!(ptl))
            elseif action == :eA
                v = samplesym(g, h, action, Pv, Av)
                right = A
                left = addleaf!(ptl, g, 
                    sym=pushval!(ptl, v), 
                    in=Pt in (Ll, Lr) ? pushval!(ptl) : right.in,
                    out=Pt in (Ll, Lr) ? right.in : pushval!(ptl))
            elseif action == :BA
                left = B
                right = A
                mergeval!(ptl, 
                    Pt in (Ll, Lr) ? left.out : left.in, 
                    right.in) || warning("Merge fail")
            end
            # add parent
            P = addroot!(ptl, g, 
                sym=Ps, 
                in=left.in, 
                out=Pt in (Ll, Pl) ? left.out : right.out, 
                left=left, right=right)
            stack_new = cons(P, action == :BA ? tail(tail(stack)) : 
                                tail(stack))
        else # three children (left, right, dynam)
            if action in (:Aee, :eAe, :BAe)
                if action == :Aee
                    v = samplesym(g, h, action, Pv, Av)
                    left = A
                    right = addleaf!(ptl, g,
                        sym=pushval!(ptl, v),
                        in=Pt in (Llr, Lrl) ? left.out : left.in,
                        out=pushval!(ptl))
                elseif action == :eAe
                    v = samplesym(g, h, action, Pv, Av)
                    right = A
                    left = addleaf!(ptl, g,
                        sym=pushval!(ptl, v),
                        in=Pt in (Llr, Lrl) ? pushval!(ptl) : right.in,
                        out=Pt in (Llr, Lrl) ? right.in : pushval!(ptl))
                else # action == :BAe
                    left = B
                    right = A
                    mergeval!(ptl, 
                        Pt in (Llr, Lrl) ? left.out : left.in, 
                        right.in) || warning("Merge fail")
                end
                dynam = addleaf!(ptl, g,
                    sym=Pt in (Llr, Plr) ? left.out : right.out,
                    in=Pt in (Llr, Plr) ? right.out : left.out,
                    out=pushval!(ptl))
            else
                if action == :eeA
                    v1, v2 = samplesym(g, h, action, Pv, Av)
                    dynam = A
                    left = addleaf!(ptl, g,
                        sym=pushval!(ptl, v1),
                        in=pushval!(ptl),
                        out=Pt in (Llr, Plr) ? dynam.sym : dynam.in)
                    right = addleaf!(ptl, g,
                        sym=pushval!(ptl, v2),
                        in=Pt in (Llr, Lrl) ? left.out : left.in,
                        out=Pt in (Llr, Plr) ? dynam.in : dynam.sym)
                elseif action == :BeA
                    v = samplesym(g, h, action, Pv, Av, Bv)
                    left = B
                    dynam = A
                    mergeval!(ptl, 
                        Pt in (Llr, Plr) ? dynam.sym : dynam.in,
                        left.out) || warning("Merge fail")
                    right = addleaf!(ptl, g,
                        sym=pushval!(ptl, v),
                        in=Pt in (Llr, Lrl) ? left.out : left.in,
                        out=Pt in (Llr, Plr) ? dynam.in : dynam.sym)
                elseif action == :eBA
                    v = samplesym(g, h, action, Pv, Av, Bv)
                    right = B
                    dynam = A
                    mergeval!(ptl, 
                        Pt in (Llr, Plr) ? dynam.in : dynam.sym,
                        right.out) || warning("Merge fail")
                    left = addleaf!(ptl, g,
                        sym=pushval!(ptl, v),
                        in=Pt in (Llr, Lrl) ? pushval!(ptl) : right.in,
                        out=Pt in (Llr, Plr) ? dynam.sym : dynam.in)
                else # action == :CBA
                    left = head(tail(tail(stack)))
                    right = B
                    dynam = A
                    mergeval!(ptl, 
                        Pt in (Llr, Lrl) ? left.out : left.in, 
                        right.in) || warning("Merge fail")
                    mergeval!(ptl, 
                        Pt in (Llr, Plr) ? dynam.sym : dynam.in,
                        left.out) || warning("Merge fail")
                    mergeval!(ptl, 
                        Pt in (Llr, Plr) ? dynam.in : dynam.sym,
                        right.out) || warning("Merge fail")
                end
            end
            P = addroot!(ptl, g, 
                sym=Ps,
                in=left.in, 
                out=dynam.out, 
                left=left, right=right, dynam=dynam)
            stack_new = cons(P, action == :CBA ? tail(tail(tail(stack))) : 
                                action in (eBA, Bea) ? tail(tail(stack)) :
                                tail(stack))
        end
    end
    return BUParticle(ptl.w, stack_new, ptl.values, ptl.dset, ptl.factors, ptl)
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
        weight = vcat(action_weights(ptl, h, g)...)
        distr = normalize(weight)
        tmp, _ = sample(distr)
        action, action_Ps = decode_action(tmp, g.n_cm)
        ptl = perform_action(ptl, g, action, action_Ps, obs)
    end
    return ptl
end

"Finish the particle simulation and update weight"
function finalize(ptl, g, h, max_iter)
    w = ptl.w
    # check stack size
    if length(ptl.stack) != 1
        w = 0
    # check root symbol (should be 1)
    elseif getval(ptl, head(ptl.stack).sym) != 1
        w = 0
    # perform TD expansions and update heuristics
    else
        # TODO
    end
    ptl = BUParticle(w, ptl.stack, ptl.values, ptl.dset, ptl.factors, ptl)
    return ptl
end

"Prepare function and rule probabilities, with training/testing mode"
function parse_prepare!(g, training)
    w_extra = training ? g.alpha : 0.
    p_fn = normalize(g.w_fn .+ w_extra, dims=1)
    p_cn = normalize(g.w_cn .+ w_extra, dims=1)
    p_l = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1 .+ w_extra, dims=1)
    p_r = normalize(sum(g.w_cm2, dims=2)[:, 1, :] .+ w_extra, dims=1)
    p_cm2 = normalize(g.w_cm2 .+ w_extra, dims=(1,2))
    g.h = (; p_fn, p_cn, p_l, p_r, p_cm2)
end

function get_obs(g, char)
    return g.index[string(char)]
end

function buparticle_init(obs, n_particle)
    values = [obs, 0, 0]
    dset = IntDisjointSets(3)
    stack = list(BUState(1, 2, 3))
    factors = [Factors(), Factors(), Factors()]
    ptl = BUParticle(1., stack, values, dset, factors, nothing)
    particles = Vector{BUParticle}()
    for i in 1:n_particle
        push!(particles, copy(ptl))
    end
    return particles
end

function forward_all!(particles, args...)
    for (i, ptl) in enumerate(particles)
        debugln("Simulating particle $i")
        particles[i] = forward(ptl, args...)
    end
end

function finalize_all!(particles, args...)
    for (i, ptl) in enumerate(particles)
        debugln("Finishing particle $i")
        particles[i] = finalize(ptl, args...)
    end
end

"Effective sample size (ESS) of a particle system"
function effective_size(particles::Vector{BUParticle})
    weights = [p.w for p in particles]
    return sum(weights) ^ 2 / sum(weights .^ 2)
end

"Resample the particle system to the target number"
function resample!(particles::Vector{BUParticle}, target_num::Int)
    weights = [p.w for p in particles]
    samples = first.(sample_n(normalize(weights), target_num, :systematic))
    debugln("Resampling: ancestor indices are $samples")
    particles_new = Vector{BUParticle}()
    for i in samples
        push!(particles_new, copy(particles[i]))
    end
    particles = particles_new
end

"Check is all particles have zero weight"
function is_all_fail(particles)
    total_weight = sum([ptl.w for ptl in particles])
    return total_weight <= 0.
end

function parse_bupf(str::String, g::GrammarEx, h::BUHeuristics, n_particle::Int=4, max_iter::Int=100, training::Bool=false)
    parse_prepare!(g, trailing_ones)
    # init
    obs_first = get_obs(g, str[1])
    particles = buparticle_init(obs_first, n_particle)
    logprob = 0. # log of marginal probability
    # start parsing
    for char in str[2:end]
        # progress the particle, and shift next observation
        obs = get_obs(g, char)
        forward_all!(particles, obs, g, h, max_iter)
        # check failure
        if is_all_fail(particles)
            debugln("All particles fail")
            return Vector{BUParticle}(), NaN
        end
        # TODO: fix this for non-binary weights
        logprob += log(total_weight / n_particle)
        # resample is effective sample size too small
        if effective_size(particles) <= n_particle / 2
            resample!(particles, n_particle)
        end
    end
    # finalize the simulation after the last observation
    finalize_all!(particles, g, h, max_iter)
    return particles, logprob
end