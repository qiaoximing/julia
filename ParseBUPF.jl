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
    obs_idx::Int # index of current observation
    finish::Bool # if the parsing finishes
    stack::LinkedList{BUState} # bottom-up parse stack
    values::Vector{Int} # values in BUState
    dset::IntDisjointSets # disjoint set for value indexing
    factors::Vector{Factors} # related factors for each value
    ances::Union{BUParticle, Nothing} # ancestor of the particle
end
Base.copy(p::BUParticle) = BUParticle(p.w, p.obs_idx, p.finish, p.stack, 
    copy(p.values), deepcopy(p.dset), deepcopy(p.factors), p.ances)
reset_weight_copy(p::BUParticle) = BUParticle(1., p.obs_idx, p.finish, p.stack, 
    copy(p.values), deepcopy(p.dset), deepcopy(p.factors), p.ances)

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

# return probability, prob=0 when fail, prob=1 when doing nothing
function mergeval!(ptl, x, y)
    rx, ry = find_root(ptl.dset, x), find_root(ptl.dset, y)
    vx, vy = ptl.values[rx], ptl.values[ry]
    if rx == ry
        # already merged
        p_merge = 1.
    elseif vx == 0 && vy == 0
        # merge dset and factors
        union!(ptl.dset, x, y)
        r_new = find_root(ptl.dset, x)
        r_other = r_new == rx ? ry : rx
        ptl.factors[r_new] = union(ptl.factors[rx], ptl.factors[ry])
        ptl.factors[r_other] = Factors()
        p_merge = 1.
    elseif vx > 0 && vy > 0
        # check equality
        if vx == vy
            union!(ptl.dset, x, y)
            p_merge = 1.
        else
            p_merge = 0.
        end
    else # set the sampled value to the unsampled one
        idx = vx == 0 ? rx : ry
        val = max(vx, vy)
        p_set = setval!(ptl, g, h, idx, val)
        union!(ptl.dset, x, y)
        p_merge = p_set
    end
    return p_merge
end

function pushval!(ptl, x=0)
    push!(ptl.values, x) 
    push!(ptl.dset)
    push!(ptl.factors, Factors())
    if length(ptl.dset) != length(ptl.values)
        warning("wrong dset size")
    end
    return length(ptl.values)
end

function setval!(ptl, g, h, idx, val)
    # get probability
    fs = ptl.factors[idx]
    p_set = reduce((*), map(x->evalfactor(ptl, x, g, h)[val], fs.f))
    # update value
    ptl.values[idx] = val
    ptl.factors[idx] = Factors()
    return p_set
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
function evalfactor(ptl::BUParticle, factor::Factor, g, h)
    # uniform weights by default
    w_def = ones(Float32, g.n_id)
    # unsampled symbol
    state = factor.state
    sym = getval(ptl, state.sym)
    sym == 0 && return w_def
    in = getval(ptl, state.in)
    out = getval(ptl, state.out)
    rel = factor.relation
    type = g.type[sym]
    if type == Id || type == Tr
        # onehot_(x) = x > 0 ? onehot(size(g), x) : w_def
        warning("Factor on $type")
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
        # warning("Expanding observations")
        p_leaf = 0.
    elseif type == Id
        p_merge = mergeval!(ptl, in, out)
        p_leaf = p_merge
    else
        addfactor!(ptl, in, state, :in)
        addfactor!(ptl, out, state, :out)
        p_leaf = 1.
    end
    return state, p_leaf
end

"Perform reduction, and sample values that are ready"
function addroot!(ptl, g, h; sym, in, out, left=nothing, right=nothing, dynam=nothing)
    P = BUState(sym, in, out, left, right, dynam)
    Pt = g.type[getval(ptl, sym)]
    if Pt == Fn || Pt == Cn || Pt == Tr || Pt == Id
        return P, 0.
    end
    p_root = 1.
    # TODO: add g.alpha when training
    if Pt != U
        p_root *= g.h.w_l[getval(ptl, left.sym), getval(ptl, sym)] / 
                  g.h.w_sym[getval(ptl, sym)]
    else
        p_root *= g.w_cm2[getval(ptl, right.sym), getval(ptl, left.sym), getval(ptl, sym)] /
                  g.h.w_sym[getval(ptl, sym)]
    end
    if Pt == Lr
        p_root *= sampleval!(ptl, left.out, g, h)
    elseif Pt == Llr || Pt == Lrl || Pt == Plr || Pt == Prl
        p_root *= sampleval!(ptl, left.out, g, h)
        p_root *= sampleval!(ptl, right.out, g, h)
    end
    return P, p_root
end

function sampleval!(ptl, x, g, h)
    rx = find_root(ptl.dset, x)
    if ptl.values[rx] > 0 
        return 1. # already sampled
    end
    # Must include 1 :out factor and >=1 :in factor
    # (Root I/O automatically fail this test)
    fs = ptl.factors[rx]
    if fs.no == 1 && fs.ni >= 1
        weight = reduce((.*), map(x->evalfactor(ptl, x, g, h), fs.f))
        distr = normalize(weight)
        val, p_sample = sample(distr)
        p_set = setval!(ptl, g, h, rx, val)
        # TODO: this should == sum(weight)
        return p_set / p_sample
    else
        return 1. # not ready to sample
    end
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
    # in the first step, force shift
    if ptl.obs_idx == 1
        return :sft, 1.
    else
        # evaluate weights for action sampling
        As, Bs, Cs = get_top_syms(ptl)
        action_list = (:sft, :A, :Ae, :eA, :Aee, :eAe, :eeA, 
                       :BA, :BAe, :BeA, :eBA, :CBA)
        weight = [getaction(h, action, As, Bs, Cs) for action in action_list]
        # prevent shift in the last step
        if is_last_step 
            weight[1] = 0 
        end
        distr = normalize(weight)
        idx, prob = sample(distr)
        return action_list[idx], prob
    end
end

function sample_parent(ptl, g, h, action)
    As, Bs, Cs = get_top_syms(ptl)
    weight = getparent(h, action, As, Bs, Cs)
    # TODO: add exploration
    # weight[1:g.n_cm] .+= g.alpha / g.n_cm
    # remove all non-Cm symbols
    # weight[g.n_cm + 1:end] .= 0.
    distr = normalize(weight)
    parent, prob = sample(distr)
    return parent, prob
end

function sample_children(ptl, g, h, action, parent)
    if action in (:A, :BA, :CBA)
        return [], 1.
    end
    As, Bs, _ = get_top_syms(ptl)
    weight1 = getchild1(h, action, parent, As, Bs)
    # remove all Tr symbols
    # weight1[range(Tr, g)] .= 0.
    distr1 = normalize(weight1)
    child1, prob1 = sample(distr1)
    if action in (:Aee, :eAe, :eeA)
        weight2 = getchild2(h, action, child1, parent, As)
        # remove all Tr symbols
        # weight2[range(Tr, g)] .= 0.
        distr2 = normalize(weight2)
        child2, prob2 = sample(distr2)
        children = [child1, child2]
        prob = prob1 * prob2
    else
        children = [child1]
        prob = prob1
    end
    return children, prob
end

function parse_shift(ptl, g, h, obs, p_sample)
    stack = ptl.stack
    # new observation Z
    Zs = pushval!(ptl, obs)
    Zi = pushval!(ptl)
    Zo = Zi
    Z = BUState(Zs, Zi, Zo)
    # push Z
    stack_new = cons(Z, stack)
    # weight update
    p_target = getsym(h, obs)
    w_new = ptl.w * p_target / p_sample
    return BUParticle(w_new, ptl.obs_idx + 1, false, stack_new, 
                      ptl.values, ptl.dset, ptl.factors, ptl)
end

function match_num_children(action, type)
    n1 = action == :A ? 1 :
         action in (:Ae, :eA, :BA) ? 2 : 3
    n2 = type == U ? 1 :
         type in (Ll, Lr, Pl, Pr) ? 2 : 3
    return n1 == n2
end

function parse_reduce(ptl, g, h, action, children, parent, p_sample)
    stack = ptl.stack
    # new root symbol
    Ps = pushval!(ptl, parent)
    Pt = g.type[parent]
    # get stack top
    A = head(stack)
    B = length(stack) > 1 ? head(tail(stack)) : nothing
    C = length(stack) > 2 ? head(tail(tail(stack))) : nothing
    p_target = 1.
    stack_new = stack
    p_root = 0.
    if !match_num_children(action, Pt)
        # do nothing
    elseif action == :A # one child (left)
        left = A
        P, p_root = addroot!(ptl, g, h,
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
            right, p_leaf = addleaf!(ptl, g, 
                sym=pushval!(ptl, v), 
                in=Pt in (Ll, Lr) ? left.out : left.in, 
                out=pushval!(ptl))
            p_target *= p_leaf
        elseif action == :eA
            v = children[1]
            right = A
            left, p_leaf = addleaf!(ptl, g, 
                sym=pushval!(ptl, v), 
                in=Pt in (Ll, Lr) ? pushval!(ptl) : right.in,
                out=Pt in (Ll, Lr) ? right.in : pushval!(ptl))
            p_target *= p_leaf
        elseif action == :BA
            left = B
            right = A
            p_merge = mergeval!(ptl, 
                Pt in (Ll, Lr) ? left.out : left.in, 
                right.in)
            p_target *= p_merge
        end
        # add parent
        if p_target > 0.
            P, p_root = addroot!(ptl, g, h, 
                sym=Ps, 
                in=left.in, 
                out=Pt in (Ll, Pl) ? left.out : right.out, 
                left=left, right=right)
            stack_new = cons(P, action == :BA ? tail(tail(stack)) : 
                                tail(stack))
        end
    else # three children (left, right, dynam)
        if action in (:Aee, :eAe, :BAe)
            if action == :Aee
                v = children[1]
                left = A
                right, p_leaf = addleaf!(ptl, g,
                    sym=pushval!(ptl, v),
                    in=Pt in (Llr, Lrl) ? left.out : left.in,
                    out=pushval!(ptl))
                p_target *= p_leaf
            elseif action == :eAe
                v = children[1]
                right = A
                left, p_leaf = addleaf!(ptl, g,
                    sym=pushval!(ptl, v),
                    in=Pt in (Llr, Lrl) ? pushval!(ptl) : right.in,
                    out=Pt in (Llr, Lrl) ? right.in : pushval!(ptl))
                p_target *= p_leaf
            else # action == :BAe
                left = B
                right = A
                p_merge = mergeval!(ptl, 
                    Pt in (Llr, Lrl) ? left.out : left.in, 
                    right.in)
                p_target *= p_merge 
            end
            dynam_s = Pt in (Llr, Plr) ? left.out : right.out
            # TODO: maybe dependent on existing factors in dynam_s
            dynam_v = action == :BAe ? children[1] : children[2]
            p_set = setval!(ptl, g, h, find_root(ptl.dset, dynam_s), dynam_v)
            dynam, p_leaf = addleaf!(ptl, g,
                sym=dynam_s,
                in=Pt in (Llr, Plr) ? right.out : left.out,
                out=pushval!(ptl))
            p_target *= p_set * p_leaf
        else
            if action == :eeA
                v1, v2 = children[1], children[2]
                dynam = A
                left, p_leaf1 = addleaf!(ptl, g,
                    sym=pushval!(ptl, v1),
                    in=pushval!(ptl),
                    out=Pt in (Llr, Plr) ? dynam.sym : dynam.in)
                right, p_leaf2 = addleaf!(ptl, g,
                    sym=pushval!(ptl, v2),
                    in=Pt in (Llr, Lrl) ? left.out : left.in,
                    out=Pt in (Llr, Plr) ? dynam.in : dynam.sym)
                p_target *= p_leaf1 * p_leaf2
            elseif action == :BeA
                v = children[1]
                left = B
                dynam = A
                p_merge = mergeval!(ptl, 
                    Pt in (Llr, Plr) ? dynam.sym : dynam.in,
                    left.out)
                right, p_leaf = addleaf!(ptl, g,
                    sym=pushval!(ptl, v),
                    in=Pt in (Llr, Lrl) ? left.out : left.in,
                    out=Pt in (Llr, Plr) ? dynam.in : dynam.sym)
                p_target *= p_merge * p_leaf
            elseif action == :eBA
                v = children[1]
                right = B
                dynam = A
                p_merge = mergeval!(ptl, 
                    Pt in (Llr, Plr) ? dynam.in : dynam.sym,
                    right.out)
                left, p_leaf = addleaf!(ptl, g,
                    sym=pushval!(ptl, v),
                    in=Pt in (Llr, Lrl) ? pushval!(ptl) : right.in,
                    out=Pt in (Llr, Plr) ? dynam.sym : dynam.in)
                p_target *= p_merge * p_leaf
            else # action == :CBA
                left = C
                right = B
                dynam = A
                p_merge1 = mergeval!(ptl, 
                    Pt in (Llr, Lrl) ? left.out : left.in, 
                    right.in)
                p_merge2 = mergeval!(ptl, 
                    Pt in (Llr, Plr) ? dynam.sym : dynam.in,
                    left.out)
                p_merge3 = mergeval!(ptl, 
                    Pt in (Llr, Plr) ? dynam.in : dynam.sym,
                    right.out)
                p_target *= p_merge1 * p_merge2 * p_merge3
            end
        end
        if p_target > 0.
            P, p_root = addroot!(ptl, g, h, 
                sym=Ps,
                in=left.in, 
                out=dynam.out, 
                left=left, right=right, dynam=dynam)
            stack_new = cons(P, action == :CBA ? tail(tail(tail(stack))) : 
                                action in (:eBA, :BeA, :BAe) ? tail(tail(stack)) :
                                tail(stack))
        end
    end
    p_target *= p_root
    if p_target > 0.
        # adjust weight by getsym
        p_target *= getsym(h, parent) / getsym(h, getval(ptl, A.sym))
        if action in (:BA, :eBA, :BeA, :BAe, :CBA)
            p_target /= getsym(h, getval(ptl, B.sym))
        end
        if action == :CBA
            p_target /= getsym(h, getval(ptl, C.sym))
        end
    end
    w_new = ptl.w * p_target / p_sample
    return BUParticle(w_new, ptl.obs_idx, false, stack_new, 
                      ptl.values, ptl.dset, ptl.factors, ptl)
end

function parse_finish(ptl, g, success)
    if !success
        w_new = 0.
    else
        # set root input = 1
        root = head(ptl.stack)
        idx = find_root(ptl.dset, root.in)
        p_set = setval!(ptl, g, h, idx, 1)
        w_new = ptl.w * p_set
        # TODO: perform TD expansions and update heuristics
    end
    return BUParticle(w_new, ptl.obs_idx, true, ptl.stack, 
                      ptl.values, ptl.dset, ptl.factors, ptl)
end

"Progress a particle forward to the next observation"
function forward(ptl::BUParticle, str::String, g::GrammarEx, h::BUHeuristics, max_iter::Int)
    is_last_step = ptl.obs_idx > length(str)
    action, prob_act = sample_action(ptl, g, h, is_last_step)
    debugln("Action: $action $prob_act")
    if action == :sft
        # shift the next observation
        obs = get_obs(g, str[ptl.obs_idx])
        p_sample = prob_act
        ptl = parse_shift(ptl, g, h, obs, p_sample)
    else
        # sample parent and childre, then reduce
        parent, prob_par = sample_parent(ptl, g, h, action)
        debugln("Parent: $parent $prob_par")
        children, prob_chl = sample_children(ptl, g, h, action, parent)
        length(children) > 0 && debugln("Children: $children $prob_chl")
        p_sample = prob_act * prob_par * prob_chl
        ptl = parse_reduce(ptl, g, h, action, children, parent, p_sample)
        # finish when reaching symbol S (id=1)
        if getval(ptl, head(ptl.stack).sym) == 1
            success = is_last_step && length(ptl.stack) == 1
            ptl = parse_finish(ptl, g, success)
        end
    end
    return ptl
end

function get_obs(g, char)
    return g.index[string(char)]
end

function buparticle_init(obs, n_particle)
    values = Vector{Int}()
    dset = IntDisjointSets(0)
    stack = nil(BUState)
    factors = Vector{Factors}()
    ptl = BUParticle(1., 1, false, stack, values, dset, factors, nothing)
    particles = Vector{BUParticle}()
    for i in 1:n_particle
        push!(particles, copy(ptl))
    end
    return particles
end

function forward_all!(particles, args...)
    for (i, ptl) in enumerate(particles)
        if !ptl.finish
            debugln("Simulating particle $i")
            particles[i] = forward(ptl, args...)
            debugln("Weight: $(particles[i].w)")
            # print_trees(get_trees(particles[i]), g)
        end
    end
end

"Effective sample size (ESS) of a particle system"
function effective_size(particles::Vector{BUParticle})
    weights = [p.w for p in particles]
    ess = sum(weights) ^ 2 / sum(weights .^ 2)
    return isnan(ess) ? 0. : ess
end

"Resample the particle system to the target number"
function resample(particles::Vector{BUParticle}, target_num::Int)
    weights = [p.w for p in particles]
    debugln(weights)
    samples = first.(sample_n(normalize(weights), target_num, :systematic))
    debugln("Resampling: ancestor indices are $samples")
    particles_new = Vector{BUParticle}()
    for i in samples
        push!(particles_new, reset_weight_copy(particles[i]))
    end
    return particles_new
end

"Check if all particles have zero weight"
function is_all_fail(particles)
    total_weight = sum([ptl.w for ptl in particles])
    return total_weight <= 0.
end

"Check if all particle finish"
is_all_finish(particles) = reduce((&), [p.finish for p in particles])

"Check if a particle finished successfully"
is_success(ptl::BUParticle) = ptl.finish && ptl.w > 0

function parse_bupf(str::String, g::GrammarEx, h::BUHeuristics, n_particle::Int=4, max_iter::Int=10)
    # init
    w_l = sum(g.w_cm2, dims=1)[1,:,:] .+ g.w_cm1
    w_sym = sum(w_l, dims=1)[1,:]
    g.h = (; w_l, w_sym)
    obs_first = get_obs(g, str[1])
    particles = buparticle_init(obs_first, n_particle)
    logprob = 0. # log of marginal probability
    # start parsing
    # for char in str[2:end]
    max_step = 17
    for step in 1:max_step
        # progress the particle
        forward_all!(particles, str, g, h, max_iter)
        # check failure
        if is_all_fail(particles)
            debugln("All particles fail")
            return Vector{BUParticle}(), NaN
        end
        # check all finish
        if is_all_finish(particles)
            debugln("All particles finish")
            break
        end
        # resample is effective sample size too small
        if effective_size(particles) <= n_particle / 2
            weight_before = sum([p.w for p in particles])
            debugln(weight_before)
            particles = resample(particles, n_particle)
            weight_after = sum([p.w for p in particles])
            debugln(weight_after)
            logprob += log(weight_before / weight_after)
        end
    end
    particles = filter(is_success, particles)
    weight = sum([p.w for p in particles])
    logprob += log(weight)
    # TODO: increase diversity by backward simulation
    return particles, logprob
end

function get_trees(ptl::BUParticle)
    # may have multimple roots is parse not finished
    roots = reverse(collect(ptl.stack))
    return [fill_value(root, ptl) for root in roots]
end

function sample_bupf(particles, temp=1.)
    if length(particles) == 0 || effective_size(particles) == 0
        debugln("No particles to sample")
        return Vector{BUState}()
    end
    weights = [p.w ^ (1 / Float32(temp)) for p in particles]
    idx, _ = sample(normalize(weights)) 
    ptl = particles[idx]
    return get_trees(ptl)
end

# Random.seed!(12)
g = test_grammar1()
h = learn_from_grammar(g, 1000)
data = "xzx"
# g = test_grammar2()
# h = learn_from_grammar(g, 1000)
# data = "1+0=1"
# ps, _ = parse_tdpf(data, g, 10)
# tr = sample_tdpf(ps)
# tr isa Tree && print_tree(tr, g)
ps, _ = parse_bupf(data, g, h, 100)
print_trees(sample_bupf(ps), g)
println("Finish")