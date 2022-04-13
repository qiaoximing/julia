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

# TODO: return probability, prob=0 when fail, prob=1 when doing nothing
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
function eval_factor(ptl::BUParticle, factor::Factor, g, h)
    # uniform weights by default
    w_def = ones(Float32, g.n_id)
    # unsampled symbol
    state = factor.state
    sym, in, out = map(x->getval(ptl, x), (state.sym, state.in, state.out))
    sym == 0 && return w_def
    rel = factor.relation
    type = g.type[sym]
    if type == Id || type == Tr
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
        # TODO: return weight=0
    elseif type == Id
        mergeval!(ptl, in, out) || warning("Merge fail")
        # TODO: return weight of mergeval
    else
        addfactor!(ptl, in, state, :in)
        addfactor!(ptl, out, state, :out)
        # TODO: return weight=1
    end
    return state
end

"Perform reduction, and sample values that are ready"
function addroot!(ptl, g, h; sym, in, out, left=nothing, right=nothing, dynam=nothing)
    P = BUState(sym, in, out, left, right, dynam)
    Pt = g.type[getval(ptl, sym)]
    if Pt == Lr
        sampleval!(ptl, left.out, g, h)
    end
    if Pt == Llr || Pt == Lrl || Pt == Plr || Pt == Prl
    # if Pt in (Llr, Lrl, Plr, Prl)
        sampleval!(ptl, left.out, g, h)
        sampleval!(ptl, right.out, g, h)
    end
    return P
end

function sampleval!(ptl, x, g, h)
    rx = find_root(ptl.dset, x)
    ptl.values[rx] > 0 && return
    # Must include 1 :out factor and >=1 :in factor
    # (Root I/O automatically fail this test)
    fs = ptl.factors[rx]
    if fs.no == 1 && fs.ni >= 1
        weight = reduce((.*), map(x->eval_factor(ptl, x, g, h), fs.f))
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
        P = addroot!(ptl, g, h,
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

function get_top_syms(ptl)
    stack = ptl.stack
    A = head(stack)
    As = getval(ptl, A.sym)
    B = length(stack) > 1 ? head(tail(stack)) : nothing
    Bs = length(stack) > 1 ? getval(ptl, B.sym) : 0
    C = length(stack) > 2 ? head(tail(tail(stack))) : nothing
    Cs = length(stack) > 2 ? getval(ptl, C.sym) : 0
    return As, Bs, Cs
end

function sample_action(ptl, g, h, is_last_step)
    # evaluate weights for action sampling
    As, Bs, Cs = get_top_syms(ptl)
    action_list = (:sft, :A, :Ae, :eA, :Aee, :eAe, :eeA, :BA, :BAe, :BeA, :eBA, :CBA)
    weight = [getaction(h, action, As, Bs, Cs) for action in action_list]
    # prevent shift in the last step
    if is_last_step weight[1] = 0 end
    distr = normalize(weight)
    idx, prob = sample(distr)
    return action_list[idx], prob
end

function sample_parent(ptl, g, h, action)
    As, Bs, Cs = get_top_syms(ptl)
    distr = getparent(h, action, As, Bs, Cs)
    parent, prob = sample(distr)
    return parent, prob
end

function sample_children(ptl, g, h, action, parent)
    if action in (:A, :BA, :CBA)
        return [], 1.
    end
    As, Bs, _ = get_top_syms(ptl)
    distr1 = getchild1(h, action, parent, As, Bs)
    child1, prob1 = sample(distr1)
    if action in (:Aee, :eAe, :eeA)
        distr2 = getchild2(h, action, child1, parent, As)
        child2, prob2 = sample(distr2)
        children = [child1, child2]
        prob = prob1 * prob2
    else
        children = [child1]
        prob = prob1
    end
    return children, prob
end

function parse_shift(ptl, g, obs)
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

function parse_reduce(ptl, g, action, children, parent)
    stack = ptl.stack
    # new root symbol
    Ps = pushval!(ptl, parent)
    Pt = g.type[parent]
    # get stack top
    A = head(stack)
    B = length(stack) > 1 ? head(tail(stack)) : nothing
    if action == :A # one child (left)
        left = A
        P = addroot!(ptl, g, h, 
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
                sym=pushval!(ptl, v), 
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
        P = addroot!(ptl, g, h, 
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
            dynam_s = Pt in (Llr, Plr) ? left.out : right.out
            # TODO: maybe dependent on existing factors in dynam_s
            dynam_v = action == :BAe ? children[1] : children[2]
            # TODO: adjust ptl weight
            setval!(ptl, g, find_root(ptl.dset, dynam_s), dynam_v)
            dynam = addleaf!(ptl, g,
                sym=dynam_s,
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
        P = addroot!(ptl, g, h, 
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
        println("Action: $action $prob_act")
        if action == :sft
            ptl = parse_shift(ptl, g, obs)
            break
        else
            parent, prob_par = sample_parent(ptl, g, h, action)
            println("Parent: $parent $prob_par")
            children, prob_chl = sample_children(ptl, g, h, action, parent)
            length(children) > 0 && println("Children: $children $prob_chl")
            ptl = parse_reduce(ptl, g, action, children, parent)
            if is_last_step && length(ptl.stack) == 1 &&
                getval(ptl, head(ptl.stack).sym) == 1
                # set root input = 1
                root = head(ptl.stack)
                idx = find_root(ptl.dset, root.in)
                setval!(ptl, g, idx, 1)
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
        print_trees(sample_bupf(particles), g)
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

function sample_bupf(particles, temp=1.)
    if length(particles) == 0 || effective_size(particles) == 0
        debugln("No particles to sample")
        return
    end
    weights = [p.w ^ (1 / Float32(temp)) for p in particles]
    idx, _ = sample(normalize(weights)) 
    ptl = particles[idx]
    # may have multimple roots is parse not finished
    roots = reverse(collect(ptl.stack))
    return [fill_value(root, ptl) for root in roots]
end

# Random.seed!(1)
# g = test_grammar1()
# h = learn_from_grammar(g, 10000)
# data = "xzx"
g = test_grammar2()
h = learn_from_grammar(g, 10000)
data = "1+0=1"
# ps, _ = parse_tdpf(data, g, 10)
# tr = sample_tdpf(ps)
# tr isa Tree && print_tree(tr, g)
ps, _ = parse_bupf(data, g, h, 1)
print_trees(sample_bupf(ps), g)
println("Finish")