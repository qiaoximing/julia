const DEBUG = false
using Printf
using Random
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
    return State(trees, expans, lroots, rroots, lslots, rslots, true)
end

function shift!(state::State, net::Net, item::Char)
    tree, distrs = tree_init(net.alphabet[item], net, :leaf)
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
            if d.enable && d.group!==nothing # update the related group
                d.group.prod /= d
            end
            add_condition!(d, net, node.conds)
            if d.enable && d.group!==nothing
                d.group.prod *= d
                d.group.score = score(d.group.prod)
            end
        end
    end
end

# update all inputs and/or outputs in a dgroup, and clean up state
function update_io!(dgroup::Group{Distr}, result::Int, state::State, net::Net)
    for distr in dgroup.factors # include both enabled and disabled
        node = distr.node
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
    tree = dgroup.factors[1].node.root
    tree_idx = findfirst(x->x===tree, state.trees)
    deletefirst!(x->x===dgroup, state.expans[tree_idx])
end

# link two I/O dgroups or values, clean up state
function link_io!(node::Tree, d1, d2, state::State, net::Net)
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
        # warning("attempt to link two different values")
        state.success = false
    end
end

# given an edge, link relevant input/outputs, and clean up state
function trigger_edge!(node::Tree, state::State, net::Net)
    if node.edge isa Etype
        if node.edge == B # composition
            if node.left isa Tree
                disable!(node.input)
                link_io!(node, node.input, node.left.input, state, net)
            end
            if node.right isa Tree
                disable!(node.output)
                link_io!(node, node.output, node.right.output, state, net)
            end
            if node.left isa Tree && node.right isa Tree
                link_io!(node, node.left.output, node.right.input, state, net)
            end
        elseif node.edge == D # duplicate
            if node.left isa Tree
                disable!(node.input)
                link_io!(node, node.input, node.left.input, state, net)
            end
            if node.right isa Tree
                disable!(node.input)
                link_io!(node, node.input, node.right.input, state, net)
                disable!(node.output)
                link_io!(node, node.output, node.right.output, state, net)
            end
        end
    end
end

# update node.edge, 
function update_edge!(dgroup::Group{Distr}, result::Int, state::State, net::Net)
    node = dgroup.prod.node
    if node.edge isa Etype
        warning("update an existing edge")
    end
    node.edge = Etype(result) 
    add_condition!(node, Edge, result, net)
    tree_idx = findfirst(x->x===node.root, state.trees)
    deletefirst!(x->x===dgroup, state.expans[tree_idx])
    trigger_edge!(node, state, net)
end

# create a new node at idx and return it, update state
# TODO: optimize by replacing Array with Dict
function create_node_at!(node::Tree, result::Int, state::State, net::Net)
    if result in values(net.alphabet)
        state.success = false
        return nothing
    end
    idx = findfirst(x->x===node.root, state.trees)
    tree, distrs = tree_init(result, net, :normal) 
    dgroups = [group_init(d) for d in distrs if sum(d.prob)>0]
    # state update
    insert!(state.trees, idx, tree)
    insert!(state.expans, idx, dgroups)
    # add roots (might be nothing)
    insert!(state.lroots, idx, getfirst(x->x.factors[1]===tree.lparent, dgroups))
    insert!(state.rroots, idx, getfirst(x->x.factors[1]===tree.rparent, dgroups))
    # allow overlapping l/r slots for unobserved nodes
    insert!(state.lslots, idx, [dg for dg in dgroups if dg.factors[1].targ in (Left, Right)])
    insert!(state.rslots, idx, [dg for dg in dgroups if dg.factors[1].targ in (Left, Right)])
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
    ls, rs = state.lslots, state.rslots
    ls1, ls2, rs1, rs2 = ls[t1_idx], ls[t2_idx], rs[t1_idx], rs[t2_idx]
    ls1_idx = findfirst(x->(x.prod.node===node1 && x.prod.targ==axis), ls1)
    rs1_idx = findfirst(x->(x.prod.node===node1 && x.prod.targ==axis), rs1)
    if ls1_idx!==nothing
            if observed == false # replace the used slot with slots2
            ls[t1_idx] = [ls1[1:ls1_idx-1]; ls2; ls1[ls1_idx+1:end]]
        else # the left of lslots1, then lslots2
            ls[t1_idx] = [ls1[1:ls1_idx-1]; ls2]
            end
    end
    if rs1_idx!==nothing
        if observed == false # replace the used slot with slots2
            rs[t1_idx] = [rs1[1:rs1_idx-1]; rs2; rs1[rs1_idx+1:end]]
        else # rslots2, then the right of rslots1 
            rs[t1_idx] = [rs2; rs1[rs1_idx+1:end]]
        end
    end
    deleteat!(state.lslots, t2_idx)
    deleteat!(state.rslots, t2_idx)
    # remove trees
    deleteat!(state.trees, t2_idx)
    # try to merge or propagate I/O dgroups
    # note: must run this step after the state update
    trigger_edge!(node1, state, net)
end

function run_option!(option::Option, result::Int, state::State, net::Net)
    lrpLR(lrp::Bool) = lrp ? Left : Right
    if length(option.dgroups) == 1
        dg = option.dgroups[1]
        node, targ = dg.prod.node, dg.prod.targ
        if targ == Edge
            update_edge!(dg, result, state, net)
        elseif targ in (Input, Output)
            update_io!(dg, result, state, net)
        else
            # create new node and add it to state
            new_node = create_node_at!(node, result, state, net)
            if new_node===nothing return end
            if targ in (Left, Right)
                # merge new node to node. Note the new node is unobserved
                connect!(node, new_node, targ, state, net, false)
            elseif targ == Node
                # merge node to new node
                connect!(new_node, node, lrpLR(dg.prod.lrp), state, net)
            end
        end
    elseif length(option.dgroups) == 2
        dg1, dg2 = option.dgroups
        node1, targ1 = dg1.prod.node, dg1.prod.targ
        node2, targ2 = dg2.prod.node, dg2.prod.targ
        new_node = create_node_at!(node1, result, state, net)
        if new_node===nothing return end
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
    elseif length(option.dgroups) == 3
        dg1, dg2, dg3 = option.dgroups
        node1, targ1 = dg1.prod.node, dg1.prod.targ
        node2, targ2 = dg2.prod.node, dg2.prod.targ
        node3, targ3 = dg3.prod.node, dg3.prod.targ
        new_node = create_node_at!(node1, result, state, net)
        if new_node===nothing return end
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

# evaluate the posterior probability of a tree
function eval_logp(node::Tree, net::Net, leftmost::Bool)
    logp_left, logp_right, logp_node = 0.0, 0.0, 0.0
    # leaf node
    if !(node.left isa Tree || node.right isa Tree)
        arrayidx = (0, 1, 0, 0, 1, 1)
        idx = (node.node, node.input, node.output)
        val = net.data[arrayidx][idx...]
        if leftmost
            arrayidx = (0, 1, 0, 0, 0, 0)
            idx = (node.node)
        else
            arrayidx = (0, 1, 0, 0, 1, 0)
            idx = (node.node, node.input)
        end
        Z = net.data[arrayidx][idx...]
        logp_node = log(val / Z)
    # middle node
    elseif node.left isa Tree && node.right isa Tree
        if node.edge isa Etype
            arrayidx = (1, 1, 1, 1, 0, 0)
            idx = (Int(node.edge), node.node, node.left.node, node.right.node)
            val = net.data[arrayidx][idx...]
            arrayidx = (1, 1, 0, 0, 0, 0)
            idx = (Int(node.edge), node.node)
            Z = net.data[arrayidx][idx...]
        else
            # warning("evaluating an unsampled edge")
            arrayidx = (0, 1, 1, 1, 0, 0)
            idx = (node.node, node.left.node, node.right.node)
            val = net.data[arrayidx][idx...]
            arrayidx = (0, 1, 0, 0, 0, 0)
            idx = (node.node)
            Z = net.data[arrayidx][idx...]
        end
        logp_node = log(val / Z)
        logp_left = eval_logp(node.left, net, leftmost)
        logp_right = eval_logp(node.right, net, false)
    else
        warning("evaluating an incomplete tree")
    end
    return logp_left + logp_right + logp_node
end

function parse(net::Net, data::String, max_steps::Int, 
               temp::Number, bias::Number)
    state = state_init(net)
    logq = 0.0 # proposal probability
    # first shift in all data
    for item in data
        shift!(state, net, item)
    end
    # repeat until all merged and sampled
    state.success = true
    state_repr = "[]"
    for step in 1:max_steps
        options = get_all_options(state, bias)
        if length(options) == 0
            state.success &= length(state.trees) == 1
            break 
        end
        option_idx, prob = sample(options, temp) # select an option
        # logq += log(prob)
        option = options[option_idx]
        result, prob = sample(option.prod) # sample a result from the option
        logq += log(prob)
        if DEBUG
            println("Step $step: ")
            # println("Slots: ", state.rslots[1])
            println("Options: ", options)
            product = option.dgroups[1].prod
            println("Selected: ", option, "->", product, "->", result)
        end
        run_option!(option, result, state, net)
        if DEBUG
            state_repr_ = repr(state.trees)
            println_diff(state_repr_, state_repr)
            state_repr = state_repr_
        end
        if !state.success break end
    end
    state.success &= length(get_all_options(state, bias)) == 0
    logp = state.success ? eval_logp(state.trees[1], net, true) : -Inf 
    return state, logp, logq
end

# return all rules as 6-tuples from a state
function extract_rules(state::State)
    push_rule! = rules -> node -> begin
        edge = (node.edge isa Etype) ? Int(node.edge) : 0
        left = (node.left isa Tree) ? node.left.node : 0
        right = (node.right isa Tree) ? node.right.node : 0
        input = (node.input isa Int) ? node.input : 0
        output = (node.output isa Int) ? node.output : 0
        rule = (edge, node.node, left, right, input, output)
        push!(rules, rule)
    end
    # only care about the first tree
    tree = state.trees[1]
    rules = []
    traverse(push_rule!(rules), tree)
    return rules
end

function learn!(net::Net, results, decay::Float64=1.0)
    # collect rules and weights from parse trees
    updates = nothing
    total_weight = 0.0
    # reduce numerical error
    bias = max([logp - logq for (_, logp, logq) in results]...)
    for (state, logp, logq) in results
        weight = exp(logp - logq - bias)
        for rule in extract_rules(state)
            if updates===nothing
                updates = Dict(rule => 0.0)
            elseif !haskey(updates, rule)
                updates[rule] = 0.0
            end
            updates[rule] += weight
        end
        total_weight += weight
    end
    if total_weight > 0
        for item in updates
            if item.second > 0
                add!(net, item.first, item.second / total_weight)
            end
        end
        if decay < 1.0
            for array in values(net.data)
                for i in keys(array.data)
                    array.data[i] *= decay
                end
            end
        end
    end
end

function main(seed::Int, max_step::Int, temp::Number, bias::Number)
    # Random.seed!(seed)
    net = bp_net_init()
    if DEBUG
        bp_dataset = (["00T"], [1])
        # println("Before training: ")
        # for data in bp_dataset[1]
        #     state, logp, logq = parse(net, data, max_step, temp, bias)
        #     println("Result: ", data, ": ", logp, ", ", logq, ", ", state)
        # end
        num_data, num_repeat = 1, 1
    else
        bp_dataset = (
            ["000T", "001T", "110T", "111T"],
            [9, 1, 1, 9])
        num_data, num_repeat = 100, 5
    end
    for i in 1:num_data
        data = datasampler(bp_dataset)
        results = []
        for j in 1:num_repeat
            state, logp, logq = parse(net, data, max_step, temp, bias)
            if DEBUG
                println("-----------", i, '-', j, "-----------")
                println("Result: ", data, ": ", logp, ", ", logq)
            end
            push!(results, (state, logp, logq))
            i % 10 == 0 && println(logp, ' ', logq)
        end
        i % 10 == 0 && println()
        learn!(net, results, 1.)
    end
    println("After training: ")
    for data in bp_dataset[1]
        success = false
        cnt = 0
        while !success && cnt < 10
            state, logp, logq = parse(net, data, max_step, temp, bias)
            cnt += 1
            if state.success
                success = true
                println("Result: ", data, ": ", logp, ", ", logq, ", ", state)
            end
        end
        if !success
            println("Result: ", data, ": fail.")
        end
    end
    println("Learned net: ", net)
    println(getval(net, Output, [(Node, 1)]))
    println(getval(net, Output, [(Node, 7)]))
    println(getval(net, Output, [(Node, 11), (Input, 13)]))
    println(getval(net, Output, [(Node, 11), (Input, 14)]))
end
main(1, 35, 5, 5)