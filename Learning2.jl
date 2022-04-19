using Plots
include("ParseTDPF.jl")
include("ParseTDSS.jl")

function init_grammar(alpha=1.0)
    cm, fn, cn, tr, id = split.((
    "S A B C D E F G H I J K",
    "",
    "c d e f g",
    "u v w x y z",
    "Id"), " ", keepempty=false)
    n_all = cumsum(length.((cm, fn, cn, tr, id)))
    n_cm, n_fn, n_cn, n_tr, n_id = n_all
    label = [cm; fn; cn; tr; id]
    index = Dict(reverse.(enumerate(label)))
    w_all = (zeros(Float32, n_id, n_cm),
             zeros(Float32, n_id, n_id, n_cm),
             zeros(Float32, n_id, n_id, n_fn - n_cm),
             zeros(Float32, n_id, n_cn - n_fn))
    # w_all[2][2, n_cn, 1] = 100.
    # w_all[2][n_cm, 3, 2] = 100.
    # w_all[2][n_id, n_id, n_cm] = 100.
    n_plr = 5
    type = [repeat([Lr], n_cm - n_plr); repeat([Plr], n_plr)
            repeat([Fn], n_fn - n_cm); repeat([Cn], n_cn - n_fn);
            repeat([Tr], n_tr - n_cn); repeat([Id], n_id - n_tr)]
    return GrammarEx(n_all..., w_all..., type, label, index, alpha, (;))
end

function grammar_update!(g::GrammarEx, t::Tree, w)
    sym, input, output = t.state.sym, t.state.input, t.state.output
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

function generate_sentence()
    dict = ["u", "v", "w", "x", "y", "z"]
    word1 = rand(dict)
    word2 = rand(dict)
    word3 = word1
    return word1 * word2 * word3
end

function test1()
    Random.seed!(1)
    dataset(n) = [generate_sentence() for i in 1:n]
    g = init_grammar(0.1)
    cnt = 0
    println(dataset(10))
    perplexity = []
    for data in dataset(3000)
        ps, logprob = parse_tdpf(data, g, 500, 20, true)
        isnan(logprob) && continue
        push!(perplexity, logprob / length(data))
        weighted_trees = get_weighted_trees(ps)
        cnt += 1
        for (t, w) in weighted_trees
            grammar_update!(g, t, w)
        end
        grammar_decay!(g, 0.99)
    end
    for data in dataset(5)
        res, logprob = parse_tdss(data, g, 20, 20, false)
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
test1()