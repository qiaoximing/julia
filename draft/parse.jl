include("net.jl")
include("tree.jl")

function shift!(state::State, g::Graph, obs::Char)
    p = g.prints[obs - '0']
    push!(state.trees, Tree(p, PointDistr(p), PointDistr(p)))
end

function sample(scores::Array)
    # linear probability
    normalize(x) = x / sum(x)
    prob = normalize(scores)
    # (softmax probability)
    # softmax(x, t=1) = normalize(exp.(t * x))
    # prob = softmax(scores)
    r = rand() # random float in [0, 1)
    i = 0
    while i < length(prob) && r >= 0
        i += 1
        r -= prob[i]
    end
    return i
end

function perform(st::State, g::Graph, op)
    if op.type == :expand
    else # merge
    end
end

function update!(state::State, g::Graph)
    options = []
    scores = []
    trs = state.trees
    # gather all options and their scores
    for i in 1:length(trs)
        exp_op, exp_sc = possible_expands(trs[i], g)
        append!(options, exp_op)
        append!(scores, exp_sc)
        if i < length(trs)
            mrg_op, mrg_sc = possible_merges(trs[i], trs[i+1], g)
            append!(options, mrg_op)
            append!(scores, mrg_sc)
    end
    # select an option by score
    scores = eval_score.(state.points)
    idx = sample(scores)
    # perform the option and update the state
    perform!(state, g, options[idx])
end

function eval_prob(state::State, g::Graph)
end

function bmm_graph_init()
    g = Graph()
    # hidden states
    v0 = add_vertex!(g) # initial state
    v1 = add_vertex!(g) 
    v2 = add_vertex!(g) 
    set_init!(v0)
    # printing states
    p0 = add_vertex!(g, P0) # print 0
    p1 = add_vertex!(g, P1) # print 1
    set_prints!(g, [p0, p1])
    # functions
    f1 = add_vertex!(g) 
    add_edge!(g, f1, v0, v1, F, 100)
    add_edge!(g, f1, v0, v2, F, 100)
    f2 = add_vertex!(g) 
    add_edge!(g, f2, v1, p0, F, 100)
    add_edge!(g, f2, v2, p1, F, 100)
    f3 = add_vertex!(g) 
    add_edge!(g, f3, v1, p1, F, 100)
    add_edge!(g, f3, v2, p0, F, 100)
    # nonterminals
    n1 = add_vertex!(g) 
    add_edge!(g, n1, f2, f3, B, 100)
    nr = add_vertex!(g) # root node
    add_edge!(g, nr, f1, n1, L, 100)
    set_root!(g, nr)
    return g
end

function bmm_state_init(g)
    root_tree = Tree(g.root, PointDistr(g.init), OneDistr())
    sample_points = [root_tree.l, root_tree.r] # exclude the parent tree
    st = State([root_tree], sample_points)
    return st
end

bmm_dataset = ["00", "01", "10", "11"]
datasampler(dataset) = rand(dataset)

function bmm_main()
    graph = bmm_graph_init()
    num_data = 2
    for i in 1:num_data
        data = datasampler(bmm_dataset)
        # Parse
        state = bmm_state_init(graph)
        for item in data
            shift!(state, graph, item)
        end
        success = true
        while success # update until fail
            success = update!(state, graph)
        end
        prob = eval_prob(state, graph)
        print(data, state, prob)
        # Learn
    end
end
bmm_main()
