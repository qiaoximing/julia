# start enum from 1
@enum Etype B=1 D
@enum Axes Edge=1 Node Left Right Input Output

abstract type AbstractTree end # avoids circular reference

mutable struct Group{T}
    factors::Vector{T} # factor distributions
    enables::Vector{Bool} # enable/disable flags of factors
    prod::T # product of enabled factors
    score::Float64 # score of prod
end

mutable struct Distr{T<:AbstractTree}
    node::T # point to Tree to update
    group::Union{Group{Distr}, Nothing} # point to Group to update
    enable::Bool # enable/disable flag in Group
    targ::Axes
    lrp::Bool # when Axes is Node, use 0/1 to differentiate l/rparent
    prob::Vector{Float64}
end

safediv(x::T, y::T) where T <: Number = y != 0 ? x / y : zero(T)
normalize(x) = safediv.(x, sum(x))
Base.:(*)(x::Distr, y::Distr) = Distr(x.node, x.group, x.enable, x.targ, x.lrp, normalize(x.prob .* y.prob))
Base.:(/)(x::Distr, y::Distr) = Distr(x.node, x.group, x.enable, x.targ, x.lrp, normalize(safediv.(x.prob, y.prob)))
score(x::Distr) = begin p = filter(t->t>0, x.prob); return sum(p .* log.(p)) end
# Optimize: for a fixed distribution, use binary search to speed up
function sample(x::Vector{Float64})
    r = rand() # random float in [0, 1)
    i = 0
    while i < length(x) && r >= 0
        i += 1
        r -= x[i]
    end
    # note: concentrate to length(x) if sum(x) < 1
    return i, x[i]
end
sample(x::Distr) = sample(x.prob)

function distr_init(node::AbstractTree, targ::Axes, size::Int, lrp::Bool=false)
    prob = zeros(Float64, size)
    # new distrs are always enabled
    return Distr(node, nothing, true, targ, lrp, prob)
end
function group_init(x::Distr)
    prod = Distr(x.node, nothing, true, x.targ, x.lrp, copy(x.prob))
    group = Group{Distr}([x], [true], prod, score(x))
    x.group = group
    return group
end
function union!(x::Group{Distr}, y::Group{Distr})
    append!(x.factors, y.factors) # factors are always disjoint
    append!(x.enables, y.enables)
    x.prod *= y.prod
    x.score = score(x.prod)
    # update back-links
    for d in y.factors
        d.group = x
    end
end
# disable a Distr and from its Group
# TODO: optimize with type dispatch
function disable!(d)
    if d isa Distr && d.enable == true
        d.enable = false
        g = d.group
        idx = findfirst(x->x===d, g.factors)
        if idx===nothing
            warning("Distr not found in its Group")
        end
        g.enables[idx] = false
        g.prod /= d
        g.score = score(g.prod)
    end
end

mutable struct Tree <: AbstractTree
    edge::Union{Distr, Etype, Nothing} # init to Distr, sample to Etype
    # TODO: rename node to id to avoid confusion
    node::Int # node symbol as Int, always sampled before expansion
    left::Union{Distr, Tree, Nothing} # init to Distr, sample to Tree
    right::Union{Distr, Tree, Nothing}
    input::Union{Distr, Int} # init to Distr, sample to Int
    output::Union{Distr, Int}
    lparent::Union{Distr, Tree, Nothing} # init both to Distr
    rparent::Union{Distr, Tree, Nothing} # sample one to Tree and the other to nothing
    root::Union{Tree, Nothing} # root of the tree
    conds::Vector{Tuple{Axes, Int}} # Axes that are sampled
end

function add_condition!(d::Distr, net::Net, conds)
    vals = getval(net, d.targ, conds)
    d.prob = safediv.(vals, sum(vals))
end

function tree_init(node::Int, net::Net, type=:normal)
    # possible node types: :normal, :root, :leaf
    # first create a placeholder for the tree
    tree = Tree(Etype(1), node, nothing, nothing, 0, 0, nothing, nothing, nothing, [])
    tree.root = tree # singleton tree's root is itself
    n_edgetypes, n_nodes = net.size[1], net.size[2]
    # update conditioning
    push!(tree.conds, (Node, node))
    # init Distrs with links back to the tree
    if type == :leaf
        tree.edge = nothing # no downward expansion
        tree.left = nothing
        tree.right = nothing
        tree.input = node # constant input
        push!(tree.conds, (Input, tree.input))
        tree.output = node # constant output
        push!(tree.conds, (Output, tree.output))
        tree.lparent = distr_init(tree, Node, n_nodes, false)
        add_condition!(tree.lparent, net, [(Right, node)])
        tree.rparent = distr_init(tree, Node, n_nodes, true)
        add_condition!(tree.rparent, net, [(Left, node)])
        distrs = (tree.lparent, tree.rparent)
    elseif type == :root
        tree.input = net.init # constant input
        push!(tree.conds, (Input, tree.input))
        tree.edge = distr_init(tree, Edge, n_edgetypes)
        add_condition!(tree.edge, net, tree.conds)
        tree.output = distr_init(tree, Output, n_nodes)
        add_condition!(tree.output, net, tree.conds)
        tree.left = distr_init(tree, Left, n_nodes)
        add_condition!(tree.left, net, tree.conds)
        tree.right = distr_init(tree, Right, n_nodes)
        add_condition!(tree.right, net, tree.conds)
        tree.lparent = nothing # no upward expansion
        tree.rparent = nothing
        distrs = (tree.edge, tree.output, tree.left, tree.right)
    else
        tree.edge = distr_init(tree, Edge, n_edgetypes)
        add_condition!(tree.edge, net, tree.conds)
        tree.input = distr_init(tree, Input, n_nodes)
        add_condition!(tree.input, net, tree.conds)
        tree.output = distr_init(tree, Output, n_nodes)
        add_condition!(tree.output, net, tree.conds)
        tree.left = distr_init(tree, Left, n_nodes)
        add_condition!(tree.left, net, tree.conds)
        tree.right = distr_init(tree, Right, n_nodes)
        add_condition!(tree.right, net, tree.conds)
        tree.lparent = distr_init(tree, Node, n_nodes, false)
        add_condition!(tree.lparent, net, [(Right, node)])
        tree.rparent = distr_init(tree, Node, n_nodes, true)
        add_condition!(tree.rparent, net, [(Left, node)])
        distrs = (tree.edge, tree.input, tree.output, tree.left, tree.right,
                  tree.lparent, tree.rparent)
    end
    return tree, distrs
end

mutable struct State
    trees::Vector{Tree}
    expans::Vector{Vector{Group{Distr}}} # expansions
    lroots::Vector{Union{Group{Distr}, Nothing}} # left roots
    rroots::Vector{Union{Group{Distr}, Nothing}} # right roots
    lslots::Vector{Vector{Group{Distr}}} # left merge slots
    rslots::Vector{Vector{Group{Distr}}} # right merge slots
    success::Bool # terminal the parse when fail
end

mutable struct Option
    dgroups::Vector{Group{Distr}} # length = 1~3 for expand, merge-2, merge-3
    prod::Distr # product of products of dgroups
    score::Float64 # score of product
end
function Option(dgroups::Vector{Group{Distr}}, bias::Number)
    opt_prod = prod([dg.prod for dg in dgroups])
    if sum(opt_prod.prob) == 0
        return nothing
    else
        opt_score = score(opt_prod) + bias * (length(dgroups) - 1)
        return Option(dgroups, opt_prod, opt_score)
    end
end

# evaluate all options and return a Vector{Option}
function get_all_options(state::State, bias::Number)::Vector{Option}
    safepush!(x, i::Nothing) = x
    safepush!(x, i::Option) = push!(x, i)
    # expansions
    allexpans = vcat(state.expans...)
    options = Vector{Option}()
    for dgroup in allexpans
        # skip top-down expansions when I/O are both sampled
        # merging to these points are still allowed
        # if dgroup.prod.targ in (Left, Right)
        #     node = dgroup.prod.node
        #     if node.input isa Int && node.output isa Int
        #         continue
        #     end
        # end
        safepush!(options, Option([dgroup], bias))
    end
    # merge two trees
    for i in 1:length(state.trees)-1
        # root-root
        root1, root2 = state.rroots[i], state.lroots[i+1]
        if root1!==nothing && root2!==nothing
            safepush!(options, Option([root1, root2], bias))
        end
        # leaf-root
        for leaf in state.rslots[i]
            for root in (state.lroots[i+1], state.rroots[i+1])
                if root!==nothing
                    safepush!(options, Option([leaf, root], bias))
        end end end
        # root-leaf
        for root in (state.lroots[i], state.rroots[i])
            if root!==nothing
                for leaf in state.lslots[i+1]
                    safepush!(options, Option([root, leaf], bias))
        end end end
    end
    # merge three trees
    for i in 2:length(state.trees)-1
        # leaf-root-root
        for leaf in state.rslots[i-1]
            root1, root2 = state.rroots[i], state.lroots[i+1]
            if root1!==nothing && root2!==nothing
                safepush!(options, Option([leaf, root1, root2], bias))
        end end
        # root-root-leaf
        root1, root2 = state.rroots[i-1], state.lroots[i]
        if root1!==nothing && root2!==nothing
            for leaf in state.rslots[i+1]
                safepush!(options, Option([root1, root2, leaf], bias))
        end end
    end
    return options
end

function sample(options::Vector{Option}, temp::Number)
    scores = [option.score for option in options]
    # linear probability
    # prob = normalize(scores)
    # softmax score
    softmax(x, t) = normalize(exp.(t * x))
    prob = softmax(scores, temp)
    return sample(prob)
end

function traverse(f::Function, t::Tree)
    f(t)
    if t.left isa Tree traverse(f, t.left) end
    if t.right isa Tree traverse(f, t.right) end
end

function leaves(t)
    if t isa Tree
        return [leaves(t.l); leaves(t.r)]
    else
        return [t]
    end
end