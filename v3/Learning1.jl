using Plots
include("ParseTDPF.jl")
include("ParseTDSS.jl")

function init_grammar(alpha=1.0)
    cm, fn, cn, tr, id = split.((
    "S A B C",
    "",
    "",
    "x y z",
    "I"), " ", keepempty=false)
    n_all = cumsum(length.((cm, fn, cn, tr, id)))
    n_cm, n_fn, n_cn, n_tr, n_id = n_all
    label = [cm; fn; cn; tr; id]
    index = Dict(reverse.(enumerate(label)))
    w_all = (zeros(Float32, n_id, n_cm),
             zeros(Float32, n_id, n_id, n_cm),
             zeros(Float32, n_id, n_id, n_fn - n_cm),
             zeros(Float32, n_id, n_cn - n_fn))
    type = [repeat([Lr], n_cm); 
            repeat([Fn], n_fn - n_cm); repeat([Cn], n_cn - n_fn);
            repeat([Tr], n_tr - n_cn); repeat([Id], n_id - n_tr)]
    return GrammarEx(n_all..., w_all..., type, label, index, alpha, (;))
end

function grammar_update!(g::GrammarEx, t::Tree, w)
    sym, input, output = t.state.sym, t.state.input, t.state.output
    # println("Tree:")
    # print_tree(t, g)
    type = g.type[sym]
    if type == U
        g.w_cm1[t.left.state.sym, sym] += w
    elseif type in Cm
        g.w_cm2[t.right.state.sym, t.left.state.sym, sym] += w
    elseif type == Fn
        g.w_fn[output, input, offset(sym, g)] += w
    elseif type == Cn
        g.w_cn[output, offset(sym, g)] += w
    end
    t.left isa Tree && grammar_update!(g, t.left, w)
    t.right isa Tree && grammar_update!(g, t.right, w)
    t.dynam isa Tree && grammar_update!(g, t.dynam, w)
end

function grammar_decay!(g::GrammarEx, rate)
    for w in (g.w_cm1, g.w_cm2, g.w_fn, g.w_cn) 
        w .*= rate
    end
end

function get_weighted_trees(particles::Vector{TDParticle})
    w_total = sum([p.w for p in particles])
    weighted_trees = []
    for p in particles    
        if p.w > 0
            t = sample_tdpf([p])
            w = p.w / w_total
            push!(weighted_trees, (t, w))
        end
    end
    return weighted_trees
end

function get_weighted_trees(items::Vector{Item}, logprob)
    weighted_trees = []
    for item in items    
        t = get_tree(item)
        w = exp(item.w - logprob)
        push!(weighted_trees, (t, w))
    end
    return weighted_trees
end

"Learn with TDPF"
function test1()
    dict = ["xx", "yy", "z"]
    dataset(n) = [reduce(*, [rand(dict) for j in 1:2]) for i in 1:n]
    # Random.seed!(0)
    g = init_grammar(0.1)
    cnt = 0
    println(dataset(10))
    perplexity = []
    for data in dataset(1000)
        ps, logprob = parse_tdpf(data, g, 100, 10, true)
        grammar_decay!(g, 0.99)
        isnan(logprob) && continue
        push!(perplexity, logprob / length(data))
        # t = sample_tdpf(ps)
        # t isa Tree && grammar_update!(g, t, 1.)
        weighted_trees = get_weighted_trees(ps)
        cnt += 1
        for (t, w) in weighted_trees
            grammar_update!(g, t, w)
        end
    end
    for data in dataset(5)
        ps, logprob = parse_tdpf(data, g, 100, 10, false)
        t = sample_tdpf(ps, 0.01)
        println(data, " ", logprob / length(data))
        t isa Tree && print_tree(t, g)
    end
    display(plot(smooth(perplexity)))
    # generate_dataset(g, 20)
    # println("Count: $cnt")
    println(maximum(g.w_cm2))
end

"Learn with TDSS"
function test2()
    dict = ["xx", "yy", "z"]
    dataset(n) = [reduce(*, [rand(dict) for j in 1:2]) for i in 1:n]
    # Random.seed!(0)
    g = init_grammar(0.1)
    cnt = 0
    println(dataset(10))
    perplexity = []
    for data in dataset(1000)
        res, logprob = parse_tdss(data, g, 10, 10, true)
        grammar_decay!(g, 0.99)
        isnan(logprob) && continue
        push!(perplexity, logprob / length(data))
        weighted_trees = get_weighted_trees(res, logprob)
        cnt += 1
        for (t, w) in weighted_trees
            grammar_update!(g, t, w)
        end
    end
    for data in dataset(5)
        res, logprob = parse_tdss(data, g, 10, 10, false)
        println(data, " ", logprob / length(data))
        # length(res) > 0 && print_tree(get_tree(res[1]), g)
        count = 1
        for (t, w) in get_weighted_trees(res, logprob)
            println("Probability $w")
            print_tree(t, g)
            count += 1
            count > 1 && break
        end
    end
    display(plot(smooth(perplexity)))
    # generate_dataset(g, 20)
    println("Count: $cnt")
    println(maximum(g.w_cm2))
end

"Learn with TDPF on finite data"
function test3()
    # Random.seed!(0)
    dict = ["xx", "yy", "z"]
    num_data = 100
    dataset = [reduce(*, [rand(dict) for j in 1:2]) for i in 1:num_data]
    g = init_grammar(0.1)
    cnt = 0
    println(dataset[1:10])
    perplexity = []
    weighted_trees_prev = [[] for i in 1:num_data]
    for epoch in 1:10
        for (data_idx, data) in enumerate(dataset)
            for (t, w) in weighted_trees_prev[data_idx]
                grammar_update!(g, t, -w)
            end
            ps, logprob = parse_tdpf(data, g, 100, 10, true)
            isnan(logprob) || push!(perplexity, logprob / length(data))
            weighted_trees = get_weighted_trees(ps)
            for (t, w) in weighted_trees
                grammar_update!(g, t, w)
            end
            weighted_trees_prev[data_idx] = weighted_trees
        end
    end
    for data in dataset[1:5]
        res, logprob = parse_tdss(data, g, 10, 10, false)
        println(data, " ", logprob / length(data))
        count = 1
        for (t, w) in get_weighted_trees(res, logprob)
            println("Probability $w")
            print_tree(t, g)
            count += 1
            count > 1 && break
        end
    end
    display(plot(smooth(perplexity)))
    # generate_dataset(g, 20)
    # println("Count: $cnt")
    println(maximum(g.w_cm2))
end

"Learn with TDSS on finite data"
function test4()
    # Random.seed!(0)
    dict = ["xx", "yy", "z"]
    num_data = 100
    dataset = [reduce(*, [rand(dict) for j in 1:2]) for i in 1:num_data]
    g = init_grammar(0.1)
    cnt = 0
    println(dataset[1:10])
    perplexity = []
    weighted_trees_prev = [[] for i in 1:num_data]
    for epoch in 1:10
        for (data_idx, data) in enumerate(dataset)
            for (t, w) in weighted_trees_prev[data_idx]
                grammar_update!(g, t, -w)
            end
            res, logprob = parse_tdss(data, g, 10, 10, true)
            isnan(logprob) || push!(perplexity, logprob / length(data))
            weighted_trees = get_weighted_trees(res, logprob)
            for (t, w) in weighted_trees
                grammar_update!(g, t, w)
            end
            weighted_trees_prev[data_idx] = weighted_trees
        end
    end
    for data in dataset[1:5]
        res, logprob = parse_tdss(data, g, 10, 10, false)
        println(data, " ", logprob / length(data))
        count = 1
        for (t, w) in get_weighted_trees(res, logprob)
            println("Probability $w")
            print_tree(t, g)
            count += 1
            count > 1 && break
        end
    end
    display(plot(smooth(perplexity)))
    # generate_dataset(g, 20)
    # println("Count: $cnt")
    println(maximum(g.w_cm2))
end
# test1()
# test2()
# test3()
# test4()