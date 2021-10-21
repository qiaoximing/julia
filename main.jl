include("net.jl")
include("tree.jl")
include("utility.jl")

function state_init(net::Net)
    tree, distrs = tree_init(net.root, net, :root)
    # Temporary fix: remove zero distributions
    # Remove this after implementing automatic exploration
    dgroups = [group_init(d) for d in distrs if sum(d.prob)>0]
    trees = [tree]
    expans = [dgroups]
    lroots = [nothing] # no root expansions
    rroots = [nothing]
    lslots = [[]] # no lslot for root
    # get all leaves for rslot
    rslots = [[dg for dg in dgroups if dg.factors[1].targ in (Left, Right)]]
    return State(trees, lroots, rroots, expans, lslots, rslots)
end

function shift!(state::State, net::Net, item)
    tree, distrs = tree_init(net.prints[item], net, :leaf)
    # Temporary fix: remove zero distributions
    # Remove this after implementing automatic exploration
    dgroups = [group_init(d) for d in distrs if sum(d.prob)>0]
    # state update
    push!(state.trees, tree)
    push!(state.expans, dgroups)
    # add roots (might be nothing)
    push!(state.lroots, getfirst(x->x.factors[1]===tree.lparent, dgroups))
    push!(state.rroots, getfirst(x->x.factors[1]===tree.rparent, dgroups))
    # leaf has no slots
    push!(state.lslots, [])
    push!(state.rslots, [])
end

function add_condition!(node::Tree, axis::Axes, result::Int, net::Net)
    push!(node.conds, (axis, result))
    # add conditioning to related Distrs, this will not affect l/rparent
    for d in (node.edge, node.left, node.right, node.input, node.output)
        if d isa Distr
            if d.enable # update the related group
                d.group.prod /= d
            end
            add_condition!(d, net, node.conds)
            if d.enable
                d.group.prod *= d
                d.group.score = score(d.group.prod)
            end
        end
    end
end

# update all inputs and/or outputs in a dgroup, and clean up state
function update_io!(dgroup::Group{Distr}, result::Int, state::State, net::Net)
    node = distr.node
    for distr in dgroup.factors # include both enabled and disabled
        axis = distr.targ
        if axis == Input
            if node.input isa Int
                warning("update an existing value")
            end
            node.input = result
        elseif axis == Output
            if node.output isa Int
                warning("update an existing value")
            end
            node.output = result
        end
        add_condition!(node, axis, result, net)
    end
    tree_idx = findfirst(x->x===node.root, state.trees)
    deletefirst!(x->x===dgroup, state.expans[tree_idx])
end

# link two I/O dgroups or values, clean up state
function link_io!(node::Tree, d1, d2, state::State)
    if d1 isa Distr && d2 isa Distr
        g1, g2 = d1.group, d2.group
        if g1 !== g2
            union!(g1, g2)
            # d1 d2 must belong to the same tree
            tree_idx = findfirst(x->x===node.root, state.trees)
            deletefirst!(x->x===g2, state.expans[tree_idx])
        end
    elseif d1 isa Distr && d2 isa Int
        update_io!(d1.group, d2, state, net)
    elseif d2 isa Distr && d1 isa Int
        update_io!(d2.group, d1, state, net)
    elseif d1 !== d2 # both are Ints
        warning("attempt to link two different values")
    end
end

# update node.edge, link relevant input/outputs, and clean up state
function update_edge!(dgroup::Group{Distr}, result::Int, state::State, net::Net)
    node = dgroup.prod.node
    if node.edge isa Etype
        warning("update an existing edge")
    end
    node.edge = Etype(result) 
    add_condition!(node, Edge, result, net)
    tree_idx = findfirst(x->x===node.root, state.trees)
    deletefirst!(x->x===dgroup, state.expans[tree_idx])
    if node.edge == B # composition
        if node.left isa Tree
            disable!(node.input)
            link_io!(node, node.input, node.left.input, state)
        end
        if node.right isa Tree
            disable!(node.output)
            link_io!(node, node.output, node.right.output, state)
        end
        if node.left isa Tree && node.right isa Tree
            link_io!(node, node.left.output, node.right.input, state)
        end
    elseif node.edge == D # duplicate
        if node.left isa Tree
            disable!(node.input)
            link_io!(node, node.input, node.left.input, state)
        end
        if node.right isa Tree
            disable!(node.input)
            link_io!(node, node.input, node.right.input, state)
            disable!(node.output)
            link_io!(node, node.output, node.right.output, state)
        end
    end
end

# create a new node at idx and return it, update state
# TODO: optimize by replacing Array with Dict
function create_node_at!(node::Tree, result::Int, state::State, net::Net)
    idx = findfirst(x->x===node.root, state.trees)
    tree, distrs = tree_init(result, net.size) 
    dgroups = [group_init(d) for d in distrs if sum(d.prob)>0]
    # state update
    insert!(state.trees, idx, tree)
    insert!(state.expans, idx, dgroups)
    # add roots (might be nothing)
    insert!(state.lroots, idx, getfirst(x->x.factors[1]===tree.lparent, dgroups))
    insert!(state.rroots, idx, getfirst(x->x.factors[1]===tree.rparent, dgroups))
    # allow overlapping l/r slots for unobserved nodes
    insert!(state.lslots, idx, [[dg for dg in dgroups if dg.factors[1].targ in (Left, Right)]])
    insert!(state.rslots, idx, [[dg for dg in dgroups if dg.factors[1].targ in (Left, Right)]])
    return tree
end

# connect node1's left/right (given by axis) to node2's r/lparent
# update the root field of every nodes in t2, then update state
# slot update depends on whether node2 is observed
function connect!(node1::Tree, node2::Tree, axis::Axes, state::State, net::Net, observed::Bool=true)
    t1, t2 = node1.root, node2.root
    if t2!==node2 warning("connect to non-root node.") end
    # connect trees
    if axis == Left
        node1.left = t2
        t2.rparent = node1
        t2.lparent = nothing
    elseif axis == Right
        node1.right = t2
        t2.lparent = node1
        t2.rparent = nothing
    end
    # update conditional distributions
    add_condition!(node1, axis, t2.node, net)
    # update root in t2
    traverse(x->x.root=t1, t2)
    # remove expans and rearrange
    t1_idx = findfirst(x->x===t1, state.trees)
    t2_idx = findfirst(x->x===t2, state.trees)
    deletefirst!(x->(x.prod.node===node1 && 
                     x.prod.targ==axis), state.expans[t1_idx])
    deleteall!(x->(x.prod.node===node2 && 
                   x.prod.targ==Node), state.expans[t2_idx])
    append!(state.expans[t1_idx], state.expans[t2_idx])
    deleteat!(state.expans, t2_idx)
    # remove roots
    deleteat!(state.lroots, t2_idx)
    deleteat!(state.rroots, t2_idx)
    # remove slots and rearrange
    for (slots1, slots2) in ((state.lslots[t1_idx], state.lslots[t2_idx]),
                                (state.rslots[t1_idx], state.rslots[t2_idx]))
        slot_idx = findfirst(x->(x.prod.node===node1 && x.prod.targ==axis), slots)
        if slot_idx!==nothing
            if observed == false # replace the used slot with slots2
                slots1 = [slots1[1:slot_idx-1]; slots2; slots1[slot_idx+1:end]]
            elseif axis == Left # lslots1 to the left then lslots2
                slots1 = [slots1[1:slot_idx-1]; slots2]
            elseif axis == Right # rslots2 then rslots1 to the right
                slots1 = [slots2; slots2[slot_idx+1:end]]
            end
        end
    end
    deleteat!(state.lslots, t2_idx)
    deleteat!(state.rslots, t2_idx)
    # remove trees
    deleteat!(state.trees, t2_idx)
end

function run_option!(option::Option, result::Int, state::State, net::Net)
    lrpLR(lrp::Bool) = lrp ? Left : Right
    if length(option.dgroups == 1)
        dg = option.dgroups[1]
        node, targ = dg.prod.node, dg.prod.targ
        if targ == Edge
            update_edge!(dg, result, state, net)
        elseif targ in (Input, Output)
            update_io!(dg, result, state, net)
        else
            # create new node and add it to state
            new_node = create_node_at!(node, result, state, net)
            if targ in (Left, Right)
                # merge new node to node. Note the new node is unobserved
                connect!(node, new_node, targ, state, net, false)
            elseif targ == Node
                # merge node to new node
                connect!(new_node, node, lrpLR(dg.prod.lrp), state, net)
            end
        end
    elseif length(option.dgroups == 2)
        dg1, dg2 = option.dgroups
        node1, targ1 = dg1.prod.node, dg1.prod.targ
        node2, targ2 = dg2.prod.node, dg2.prod.targ
        new_node = create_node_at!(node1, result, state, net)
        if targ1 == Node && targ2 == Node
            connect!(new_node, node1, Left, state, net)
            connect!(new_node, node2, Right, state, net)
        elseif targ1 == Node && targ2 in (Left, Right) 
            connect!(new_node, node1, lrpLR(dg1.prod.lrp), state, net)
            connect!(node2, new_node, targ2, state, net)
        elseif targ1 in (Left, Right) && targ2 == Node 
            connect!(new_node, node2, lrpLR(dg2.prod.lrp), state, net)
            connect!(node1, new_node, targ1, state, net)
        else
            warning("invalid merge")
        end
    elseif length(option.dgroups == 3)
        dg1, dg2, dg3 = option.dgroups
        node1, targ1 = dg1.prod.node, dg1.prod.targ
        node2, targ2 = dg2.prod.node, dg2.prod.targ
        node3, targ3 = dg3.prod.node, dg3.prod.targ
        new_node = create_node_at!(node1, result, state, net)
        if targ1 in (Left, Right) && targ2 == Node && targ3 == Node
            connect!(new_node, node2, Left, state, net)
            connect!(new_node, node3, Right, state, net)
            connect!(node1, new_node, targ1, state, net)
        elseif targ1 == Node && targ2 == Node && targ3 in (Left, Right)
            connect!(new_node, node1, Left, state, net)
            connect!(new_node, node2, Right, state, net)
            connect!(node3, new_node, targ3, state, net)
        else
            warning("invalid merge")
        end
    end
end

function parse(net, data)
    state = state_init(net)
    prob = 0.0
    # first shift in all data
    for item in data
        shift!(state, net, item)
    end
    println(state)
    # repeat until all merged and sampled
    max_steps = 3
    success = false
    for step in 1:max_steps
        options = get_all_options(state)
        if length(options) == 0
            success = length(state.trees) == 1
            break 
        end
        option_idx = sample(options) # select an option
        option = options[option_idx]
        result = sample(option.prod) # sample a result from the option
        run_option!(option, result, state, net)
        println("step $step:")
        println(state)
    end
    if !success prob = 0.0 end
    return state, prob
end

function learn_step!(net, node, prob)
    idx = (Int(node.edge), node.node, node.left, node.right, 
           node.input, node.output)
    if idx isa NTuple{6, Int}
        add!(net, idx, prob)
    end
end

function learn!(net, state, prob, decay=1.0)
    # traverse the tree and update net by prob
    for tree in state.trees
        traverse(node -> learn_step!(net, node, prob), tree)
    end
    if decay < 1.0
        for array in values(net.data)
            array .*= decay
        end
    end
    # return net
end

function main()
    net = bmm_net_init()
    bmm_dataset = ["01"]
    # bmm_dataset = ["00", "01", "10", "11"]
    datasampler(dataset) = rand(dataset)
    num_data = 1
    for i in 1:num_data
        data = datasampler(bmm_dataset)
        state, prob = parse(net, data)
        println(data,':', prob)
        if prob > 0
            learn!(net, state, prob)
        end
    end
end
main()