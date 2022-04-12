include("ParseBUHeuristics.jl")

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
    dset::IntDisjointSets # disjoint set for value indexing
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
        return getval(ptl, h.sym), getval(ptl, h.in), getval(ptl, h.out)
    end
end

"Helper function to compare two values"
match(a, b) = a == 0 || b == 0 || a == b

"Evaluate weights for action sampling"
function action_weights_old(ptl, h, g)
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
            p = get_prob(h, Ps, Pi, Po) * g.h.p_l[As, Ps]
            if p > 0
                e = get_nul_right(h, g, Ps, As)
                w_Ae[Ps] += p * e # not observed
                w_sft += p * (1 - e) # shift if observed (Az)
            end
            # A as right child, unknown null left child
            Pi = Pt in (Ll, Lr) ? 0 : Ai
            Po = Pt in (Ll, Pl) ? 0 : Ao
            p = get_prob(h, Ps, Pi, Po) * g.h.p_r[As, Ps]
            if p > 0 
                e = get_nul_left(h, g, Ps, As)
                w_eA[Ps] += p * e
            end
            # A as right child, B as left child
            # TODO: need to factor in the match likelihood
            if Bs !== nothing && (Pt in (Ll, Lr) ? match(Bo, Ai) : match(Bi, Ai))
                Pi = Pt in (Ll, Lr) ? Bi : max(Bi, Ai)
                Po = Pt in (Ll, Pl) ? Bo : Ao
                w_BA[Ps] += get_prob(h, Ps, Pi, Po) * g.h.p_cm2[As, Bs, Ps] 
            end
        elseif Pt in (Llr, Lrl, Plr, Prl)
            # A left, unknown right
            Pi = Ai
            Po = 0
            p = get_prob(h, Ps, Pi, Po) * g.h.p_l[As, Ps]
            if p > 0
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
            end
            # A right, unknown null left
            Pi = Pt in (Llr, Lrl) ? 0 : Ai
            Po = 0
            p = get_prob(h, Ps, Pi, Po) * g.h.p_r[As, Ps]
            if p > 0 
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
            end
            # A right, B left
            if Bs !== nothing && (Pt in (Llr, Lrl) ? match(Bo, Ai) : match(Bi, Ai))
                Pi = Pt in (Llr, Lrl) ? Bi : max(Bi, Ai)
                Po = 0
                p = get_prob(h, Ps, Pi, Po) * g.h.p_cm2[As, Bs, Ps] 
                if Pt in (Llr, Plr)
                    e = Bo > 0 ? get_nul(h, Bo) : get_nul_out(h, Bs, Bi)
                else
                    e = Bo > 0 ? get_nul(h, Ao) : get_nul_out(h, As, Ai)
                end
                w_BAe[Ps] += p * e
                w_sft += p * (1 - e) # shift if (BAz)
            end
            # A dynam, unknown left and right
            Pi = 0
            Po = Ao
            p = get_prob(h, Ps, Pi, Po)
            ee = h.nul' * g.h.p_cm2[:, :, Ps] * h.nul
            w_eeA[Ps] += p * ee
            # A dynam, B left, unknown right
            if Bs !== nothing 
                Pi = Bi
                Po = Ao
                p = get_prob(h, Ps, Pi, Po) * g.h.p_l[Bs, Ps]
                if p > 0 
                    if Pt in (Llr, Plr) && match(Bo, As)
                        e = get_nul_right(h, g, Ps, Bs)
                        w_BeA[Ps] += p * e
                    elseif Pt in (Lrl, Prl)
                        e = get_nul_right(h, g, Ps, Bs)
                        w_BeA[Ps] += p * e
                    end
                end
            end
            # A dynam, B right, unknown left
            if Bs !== nothing 
                Pi = Pt in (Llr, Lrl) ? 0 : Bi
                Po = Ao
                p = get_prob(h, Ps, Pi, Po) * g.h.p_r[Bs, Ps]
                if p > 0 
                    if Pt in (Llr, Plr)
                        e = get_nul_left(h, g, Ps, Bs)
                        w_eBA[Ps] += p * e
                    elseif Pt in (Lrl, Prl) && match(Bo, As)
                        e = get_nul_left(h, g, Ps, Bs)
                        w_eBA[Ps] += p * e
                    end
                end
            end
            # A dynam, B right, C left
            if Bs !== nothing && Cs !== nothing
                if (Pt in (Llr, Lrl) ? match(Co, Bi) : match(Ci, Bi)) &&
                   (Pt in (Llr, Plr) ? match(Co, As) && match(Bo, Ai) : 
                                       match(Bo, As) && match(Co, Ai))
                    Pi = Pt in (Llr, Lrl) ? Ci : max(Ci, Bi)
                    Po = Ao
                    w_CBA[Ps] += get_prob(h, Ps, Pi, Po) * g.h.p_cm2[Bs, Cs, Ps]
                end
            end
        end
    end
    debugln("A: $As, $Ai, $Ao")
    debugln("B: $Bs, $Bi, $Bo")
    debugln("C: $Cs, $Ci, $Co")
    # debugln("sft: ", w_sft)
    # debugln("A: ", w_A)
    # debugln("Ae: ", w_Ae)
    # debugln("eA: ", w_eA)
    # debugln("BA: ", w_BA)
    # debugln("Aee: ", w_Aee)
    # debugln("eAe: ", w_eAe)
    # debugln("BAe: ", w_BAe)
    # debugln("eeA: ", w_eeA)
    # debugln("BeA: ", w_BeA)
    # debugln("eBA: ", w_eBA)
    # debugln("CBA: ", w_CBA)
    return ([Float32(w_sft)], w_A, w_Ae, w_eA, w_BA, 
            w_Aee, w_eAe, w_BAe, w_eeA, w_BeA, w_eBA, w_CBA)
end

function decode_action(x, n)
    if x == 1
        return :sft, 0
    else
        x -= 1
        action_idx, action_Ps = divrem(x - 1, n) .+ 1
        action = (:A, :Ae, :eA, :BA, :Aee, :eAe, :BAe, 
                  :eeA, :BeA, :eBA, :CBA)[action_idx]
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
    push!(ptl.factors, Factors())
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
        # onehot_(x) = x > 0 ? onehot(size(g), x) : w_def
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
        mergeval!(ptl, in, out) || warning("Merge fail")
    else
        addfactor!(ptl, in, state, :in)
        addfactor!(ptl, out, state, :out)
    end
    return state
end

"Perform reduction, and sample values that are ready"
function addroot!(ptl, g; sym, in, out, left=nothing, right=nothing, dynam=nothing)
    P = BUState(sym, in, out)
    Pt = g.type[getval(ptl, sym)]
    if Pt == Lr
        sampleval!(ptl, left.out)
    end
    if Pt == Llr || Pt == Lrl || Pt == Plr || Pt == Prl
    # if Pt in (Llr, Lrl, Plr, Prl)
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

"A shorter & slower version of perform_action"
function perform_action_short!(ptl, g, action, action_Ps, obs)
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
        # get stack tops
        A = head(stack)
        Av = getval(ptl, A.sym)
        B = length(stack) > 1 ? head(tail(stack)) : nothing
        Bv = length(stack) > 1 ? getval(ptl, B.sym) : 0
        C = length(stack) > 2 ? head(tail(tail(stack))) : nothing
        # sample children
        if action in (:Ae, :Aee, :BeA)
            rightv = samplesym(g, h, action, Pv, Av, Bv)
        elseif action in (:eA, :eAe, :eBA)
            leftv = samplesym(g, h, action, Pv, Av, Bv)
        elseif action == :eeA
            leftv, rightv = samplesym(g, h, action, Pv, Av, Bv)
        end
        # set children
        left = action in (:A, :Ae, :Aee) ? A :
               action in (:BA, :BAe) ? B :
               action in (:CBA) ? C : 
               addleaf!(ptl, g, 
                   sym=pushval(ptl, leftv), 
                   in=pushval(ptl),
                   out=pushval!(ptl))
        right = action in (:A) ? nothing :
                action in (:eA, :BA, :eAe, :BAe) ? A :
                action in (:eBA, :CBA) ? B : 
                addleaf!(ptl, g, 
                    sym=pushval(ptl, rightv), 
                    in=pushval(ptl),
                    out=pushval!(ptl))
        dynam = action in (:A, :Ae, :eA, :BA) ? nothing :
                action in (:eeA, :BeA, :eBA, :CBA) ? A : 
                addleaf!(ptl, g, 
                    sym=pushval(ptl), 
                    in=pushval(ptl),
                    out=pushval!(ptl))
        # Merge values
        if right !== nothing
            mergeval!(ptl, 
                Pt in (Ll, Lr, Llr, Lrl) ? left.out : left.in, 
                right.in) || warning("Merge fail")
        end
        if dynam !== nothing
            mergeval!(ptl, 
                Pt in (Llr, Plr) ? dynam.sym : dynam.in,
                left.out) || warning("Merge fail")
            mergeval!(ptl, 
                Pt in (Llr, Plr) ? dynam.in : dynam.sym,
                right.out) || warning("Merge fail")
        end
        # add root
        P = addroot!(ptl, g, 
            sym=Ps, 
            in=left.in, 
            out=Pt in (U, Ll, Pl) ? left.out :
                Pt in (Lr, Pr) ? right.out : 
                dynam.out, 
            left=left, right=right, dynam=dynam)
        stack_new = cons(P, action == :CBA ? tail(tail(tail(stack))) : 
                            action in (:BA, :eBA, :Bea) ? tail(tail(stack)) :
                            tail(stack))
    end
    return BUParticle(ptl.w, stack_new, ptl.values, ptl.dset, ptl.factors, ptl)
end

function eval_state(ptl::BUParticle, g, h, state)
    isnothing(state) && return nothing, nothing, nothing
    ret = [getval(ptl, state.sym)]
    for x in (state.in, state.out)
        rx = find_root(ptl.dset, x)
        val = ptl.values[rx]
        factors = ptl.factors[rx]
        if val == 0
            push!(ret, onehot(g.n_id, val))
        else
            push!(ret, reduce((.*), eval.(factors.f)))
    end
    return tuple(ret...)
end

function action_weight(ptl, g, h)
    # get the top symbols
    stack = ptl.stack
    A = head(stack)
    # eval returns a vector
    As, Ai, Ao, Aoi = eval_state(ptl, g, h, A)
    B = length(stack) > 1 ? head(tail(stack)) : nothing
    Bs, Bi, Bo, Boi = eval_state(ptl, g, h, B)
    C = length(stack) > 2 ? head(tail(tail(stack))) : nothing
    Cs, Ci, Co, Coi = eval_state(ptl, g, h, C)
    w_sft = 0
    # case A
    w_UA = h.p[:UA][:, As]' * Ai
    # case Ae
    w_LlAe = sum(h.p[:LlAe][:, :, As] .* Aoi)
    w_LrAe = sum(h.p[:LrAe][:, :, As] .* Aoi)
    w_PlAe = h.p[:PlAe][:, As]' * Ai
    w_PrAe = h.p[:PrAe][:, As]' * Ai
    # case Az
    w_sft += sum(h.p[:LlAz][:, :, As] .* Aoi)
    w_sft += sum(h.p[:LrAz][:, :, As] .* Aoi)
    w_sft += h.p[:PlAz][:, As]' * Ai
    w_sft += h.p[:PrAz][:, As]' * Ai
    # case eA
    w_LleA = h.p[:LleA][:, As]' * Ai
    w_LreA = h.p[:LreA][:, As]' * Ai
    w_PleA = h.p[:PleA][:, As]' * Ai
    w_PreA = h.p[:PreA][:, As]' * Ai
    # case BA
    w_LlBA = Ai' * Boi * h.p[:LlBA][:, As, Bs]
    w_LrBA = Ai' * Boi * h.p[:LrBA][:, As, Bs]
    w_PlBA = h.p[:PlBA][:, As, Bs]' * (Bi .* Ai)
    w_PrBA = h.p[:PrBA][:, As, Bs]' * (Bi .* Ai)
    # case Aee
    w_LlrAee = sum(h.p[:LlrAee][:, :, As] .* Aoi)
    w_LrlAee = sum(h.p[:LrlAee][:, :, As] .* Aoi)
    w_PlrAee = sum(h.p[:PlrAee][:, :, As] .* Aoi)
    w_PrlAee = sum(h.p[:PrlAee][:, :, As] .* Aoi)
    # case Aez
    w_sft += sum(h.p[:LlrAez][:, :, As] .* Aoi)
    w_sft += sum(h.p[:LrlAez][:, :, As] .* Aoi)
    w_sft += sum(h.p[:PlrAez][:, :, As] .* Aoi)
    w_sft += sum(h.p[:PrlAez][:, :, As] .* Aoi)
    # case Azx
    w_sft += sum(h.p[:LlrAzx][:, :, As] .* Aoi)
    w_sft += sum(h.p[:LrlAzx][:, :, As] .* Aoi)
    w_sft += sum(h.p[:PlrAzx][:, :, As] .* Aoi)
    w_sft += sum(h.p[:PrlAzx][:, :, As] .* Aoi)
    # case eAe
    w_LlreAe = sum(h.p[:LlreAe][:, :, As] .* Aoi)
    w_LrleAe = sum(h.p[:LrleAe][:, :, As] .* Aoi)
    w_PlreAe = sum(h.p[:PlreAe][:, :, As] .* Aoi)
    w_PrleAe = sum(h.p[:PrleAe][:, :, As] .* Aoi)
    # case eAz
    w_sft += sum(h.p[:LlreAz][:, :, As] .* Aoi)
    w_sft += sum(h.p[:LrleAz][:, :, As] .* Aoi)
    w_sft += sum(h.p[:PlreAz][:, :, As] .* Aoi)
    w_sft += sum(h.p[:PrleAz][:, :, As] .* Aoi)
    # case BAe
    # w_LlrBAe = sum(Aoi .* h.e, dims=1) * (Boi * h.p[:LlrBA][:, As, Bs])
    # w_LrlBAe = sum(Aoi .* h.e', dims=1) * (Boi * h.p[:LlrBA][:, As, Bs])
    # w_PlrBAe = (Aoi * h.p[:PlrBA][:, As, Bs])' * h.e * (Boi * h.p[:PlrBA][:, As, Bs])
    # w_PrlBAe = (Aoi * h.p[:PrlBA][:, As, Bs])' * h.e' * (Boi * h.p[:PrlBA][:, As, Bs])
    # case BAz
    # w_sft += sum(Aoi .* h.z, dims=1) * (Boi * h.p[:LlrBA][:, As, Bs])
    # w_sft += sum(Aoi .* h.z', dims=1) * (Boi * h.p[:LlrBA][:, As, Bs])
    # w_sft += (Aoi * h.p[:PlrBA][:, As, Bs])' * h.z * (Boi * h.p[:PlrBA][:, As, Bs])
    # w_sft += (Aoi * h.p[:PrlBA][:, As, Bs])' * h.z' * (Boi * h.p[:PrlBA][:, As, Bs])
    # case eeA
    w_LlreeA = h.p[:LlreeA][:, As]' * Ai
    w_LrleeA = h.p[:LrleeA][:, As]' * Ai
    w_PlreeA = h.p[:PlreeA][:, As]' * Ai
    w_PrleeA = h.p[:PrleeA][:, As]' * Ai
    # case BeA
    # w_LlrBeA = Boi[As, :]' * h.p[:LlrBA][:, :, Bs] * (h.e[As, :] .* (Ai' * h.f[As, :, :]))
    # w_LrlBeA = sum(Ai' * (Boi * h.p[:LrlBA][:, :, Bs] .* h.e .* h.f[As, :, :])) # warning: O(N^3)
    # w_PlrBeA = sum(Boi[As, :]' * (h.p[:PlrBA][:, :, Bs] .* h.e .* sum(Ai .* h.f, dims=1))) # O(N^3)
    # w_PrlBeA = sum(Ai' * Boi * (h.p[:PrlBA][:, :, Bs] .* h.e .* h.f[As, :, :])) 
    w_LlrBeA = Ai' * h.p[LlrBeA][:, :, As, Bs] * Boi[As, :] 
    w_LrlBeA = Ai' * (h.p[LrlBeA][:, :, As, Bs] .* Boi)
    w_PlrBeA = Ai' * h.p[PlrBeA][:, :, As, Bs] * Boi[As, :]
    w_PrlBeA = Ai' * Boi * h.p[PrlBeA][:, As, Bs]
    # case eBA
    # w_LlreBA = sum(h.p[:LlrBA][:, Bs, :] .* h.e .* h.f[As, :, :]) * (Ai' * Boi[:, As])
    # w_LrleBA = sum(h.p[:LrlBA][:, Bs, :] .* h.e .* sum(Ai .* Boi[As, :] .* h.f, dims=1)) # O(N^3)
    # w_PlreBA = sum(Ai' * Boi * (h.p[:PlrBA][:, Bs, :] .* h.e .* h.f[As, :, :]))
    # w_PrleBA = sum(Boi[As, :]' * (h.p[:LrlBA][:, Bs, :] .* h.e .* sum(Ai .* h.f, dims=1))) # O(N^3)
    w_LlrBeA = h.p[LlrBeA][As, Bs] * (Ai' * Boi[:, As])
    w_LrlBeA = h.p[LrlBeA][:, As, Bs]' * (Ai .* Boi[As, :])
    w_PlrBeA = Ai' * Boi * h.p[PlrBeA][:, As, Bs]
    w_PrlBeA = Ai' * h.p[PrlBeA][:, :, As, Bs] * Boi[As, :]
    # case CBA
    w_LlrCBA = (Ai' * Boi[:, As]) * Coi[A, :]' * h.d * h.p2[Bs, Cs, :]
    w_LrlCBA = (Ai .* Boi[As, :])' * Coi * h.d * h.p2[Bs, Cs, :]
    w_PlrCBA = ((Ai' * Boi) .* Coi[A, :])' * h.d * h.p2[Bs, Cs, :]
    w_PrlCBA = ((Ai' * Coi) .* Boi[A, :])' * h.d * h.p2[Bs, Cs, :]
end

function sample_action(ptl, g, h, is_last_step)
    # evaluate weights for action sampling
    weight = action_weight(ptl, g, h)
    # prevent shift in the last step
    if is_last_step weight[1] = 0 end
    distr = normalize(weight)
    action_idx, prob = sample(distr)
    return decode_action(action_idx, g.n_cm), prob
end

function sample_io(ptl, g, h)
    
end

function sample_children(ptl, g, h, action)
    # may call sample_io for certain cases
end

function sample_parent(ptl, g, h, action, children)
    
end

function shift(ptl, g, obs)
    stack = ptl.stack
    # new observation Z
    Zs = pushval!(ptl, obs)
    Zi = pushval!(ptl)
    Zo = Zi
    Z = BUState(Zs, Zi, Zo)
    # push Z
    stack_new = cons(Z, stack)
    w_new = ptl.w
    return BUParticle(w_new, stack_new, ptl.values, ptl.dset, ptl.factors, ptl)
end

function reduce(ptl, g, action, children, parent)
    stack = ptl.stack
    # new root symbol
    Ps = pushval!(ptl, parent)
    Pv = getval(ptl, Ps)
    Pt = g.type[Pv]
    # get stack top
    A = head(stack)
    Av = getval(ptl, A.sym)
    # get second stack top
    if length(stack) > 1
        B = head(tail(stack))
        Bv = getval(ptl, B.sym)
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
            v = children[1]
            left = A
            right = addleaf!(ptl, g, 
                sym=pushval(ptl, v), 
                in=Pt in (Ll, Lr) ? left.out : left.in, 
                out=pushval!(ptl))
        elseif action == :eA
            v = children[1]
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
                v = children[1]
                left = A
                right = addleaf!(ptl, g,
                    sym=pushval!(ptl, v),
                    in=Pt in (Llr, Lrl) ? left.out : left.in,
                    out=pushval!(ptl))
            elseif action == :eAe
                v = children[1]
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
                v1, v2 = children[1], children[2]
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
                v = children[1]
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
                v = children[1]
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
                            action in (:eBA, :BeA) ? tail(tail(stack)) :
                            tail(stack))
    end
    return BUParticle(ptl.w, stack_new, ptl.values, ptl.dset, ptl.factors, ptl)
end

"Progress a particle forward to the next observation"
function forward(ptl::BUParticle, obs::Int, g::GrammarEx, h::BUHeuristics, max_iter::Int)
    iter = 1
    is_last_step = obs == 0
    while !isa(ptl.stack, Nil) # stack not empty
        if iter > max_iter
            debugln("Too many iterations between observation")
            return BUParticle(0., ptl.stack, ptl.values, ptl.dset, ptl.factors, ptl)
        else
            iter += 1
        end
        action, prob_act = sample_action(ptl, g, h, is_last_step)
        if action == :sft
            ptl = shift(ptl, g, obs)
            break
        else
            children, prob_chl = sample_children(ptl, g, h, action)
            parent, prob_par = sample_parent(ptl, g, h, action, children)
            ptl = reduce(ptl, g, action, children, parent)
            if is_last_step && length(ptl.stack) == 1 &&
                getval(ptl, head(ptl.stack).sym) == 1
                # TODO: perform TD expansions and update heuristics
                break
            end
        end
    end
    return ptl
end

"Prepare function and rule probabilities, with training/testing mode"
function parse_prepare!(g::GrammarEx, h::BUHeuristics, training::Bool)
    w_extra = training ? g.alpha : 0.
    h.p_fn = normalize(g.w_fn .+ w_extra, dims=1)
    h.p_cn = normalize(g.w_cn .+ w_extra, dims=1)
    h.p_l = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1 .+ w_extra, dims=1)
    h.p_r = normalize(sum(g.w_cm2, dims=2)[:, 1, :] .+ w_extra, dims=1)
    h.p_cm2 = normalize(g.w_cm2 .+ w_extra, dims=(1,2))
end

function get_obs(g, char)
    return g.index[string(char)]
end

function buparticle_init(obs, n_particle)
    values = [obs, 0]
    dset = IntDisjointSets(2)
    stack = list(BUState(1, 2, 2))
    factors = [Factors(), Factors()]
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

function parse_bupf(str::String, g::GrammarEx, h::BUHeuristics, n_particle::Int=4, max_iter::Int=10, training::Bool=false)
    parse_prepare!(g, h, training)
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
        # logprob += log(total_weight / n_particle)
        # resample is effective sample size too small
        if effective_size(particles) <= n_particle / 2
            resample!(particles, n_particle)
        end
    end
    # finalize the simulation after the last observation
    forward_all!(particles, 0, g, h, max_iter)
    # TODO: increase diversity by backward simulation
    return particles, logprob
end

g = test_grammar1()
h = learn_from_grammar(g, 10000)
data = "xzx"
# ps, _ = parse_tdpf(data, g, 10)
# tr = sample_tdpf(ps)
# tr isa Tree && print_tree(tr, g)
ps, _ = parse_bupf(data, g, h, 1)
println("Finish")