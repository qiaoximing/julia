include("ParseBUPF.jl")

# focus on learning CFG
function init_grammar(alpha)
    cm, fn, cn, tr, id = split.((
    "S A B C",
    # "S A",
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
    # partial sums
    w_l = zeros(Float32, n_id, n_cm)
    w_sym = zeros(Float32, n_cm)
    h = (; w_l, w_sym)
    return GrammarEx(n_all..., w_all..., type, label, index, alpha, h)
end

"Update grammar and related heuristics from a BUState tree"
function grammar_update!(g::GrammarEx, h::BUHeuristics, s::BUState, w)
    sym, input, output = s.sym, s.in, s.out
    left, right, dynam = s.left, s.right, s.dynam
    type = g.type[sym]
    if type == U
        # rule weight
        g.w_cm1[left.sym, sym] += w
        # partial sums
        g.h.w_l[left.sym, sym] += w
        g.h.w_sym[sym] += w
        # heuristics
        count_A!(h, sym, left.sym, left.isnul, w)
    elseif type in Cm
        # rule weight
        g.w_cm2[right.sym, left.sym, sym] += w
        # partial sums
        g.h.w_l[left.sym, sym] += w
        g.h.w_sym[sym] += w
        # heuristics
        if type in (Ll, Lr, Pl, Pr)
            count_BA!(h, sym, left.sym, left.isnul, right.sym, right.isnul, w)
        else
            count_CBA!(h, sym, left.sym, left.isnul, right.sym, right.isnul, 
                       dynam.sym, dynam.isnul, w)
        end
    elseif type == Fn && output > 0 && input > 0
        # rule weight
        g.w_fn[output, input, offset(sym, g)] += w
    elseif type == Cn && output > 0
        # rule weight
        g.w_cn[output, offset(sym, g)] += w
    end
    h.sym1[sym] += w
    h.tot += w
    if output > 0 
        h.out2[output, sym] += w
        h.out1[sym] += w
        if input > 0
            h.oi3[output, input, sym] += w
            h.oi2a[input, sym] += w
            h.oi2b[output, sym] += w
            if s.isnul
                h.nul_oi3[output, input, sym] += w
                h.nul_oi2a[input, sym] += w
                h.nul_oi2b[output, sym] += w
            end
        end
    end
    if type == Tr
        h.obs += w
    end
    grammar_update!(g, h, left, w)
    grammar_update!(g, h, right, w)
    grammar_update!(g, h, dynam, w)
end
function grammar_update!(g::GrammarEx, h::BUHeuristics, s::Nothing, w)
    return
end

"Decay grammar and related heuristics"
function grammar_decay!(g::GrammarEx, h::BUHeuristics, decay)
    # rule weight
    for w in (g.w_cm1, g.w_cm2, g.w_fn, g.w_cn) 
        w .*= 1 - decay
    end
    # partial sums
    for w in (g.h.w_l, g.h.w_sym)
        w .*= 1 - decay
    end
    # heuristics
    h.tot *= 1 - decay
    h.obs *= 1 - decay
    for w in (h.sym1, h.out2, h.out1, h.oi3, h.oi2a, h.oi2b, h.nul_oi3, h.nul_oi2a, h.nul_oi2b)
        w .*= 1 - decay
    end
    for dict in (h.w1, h.w2, h.w3, h.w4)
        for w in values(dict)
            w .*= 1 - decay
        end
    end
end

"Extract BUState trees from a particle system"
function get_weighted_trees(particles::Vector{BUParticle})
    w_total = sum([p.w for p in particles])
    weighted_trees = []
    for p in particles    
        if p.w > 0
            ts = get_trees(p)
            if length(ts) > 1
                warning("Unfinished parsing")
            end
            t = ts[1]
            w = p.w / w_total
            push!(weighted_trees, (t, w))
        end
    end
    return weighted_trees
end

function test1()
    # Random.seed!(0)
    dict = ["xx", "yy", "z"]
    dataset(n) = [reduce(*, [rand(dict) for j in 1:2]) for i in 1:n]
    alpha, decay = 0.1, 0.01
    g = init_grammar(alpha)
    h = init_buheuristics(g, decay)
    cnt = 0
    perplexity = []
    for (step, data) in enumerate(dataset(100))
        println("-------------Step: $step, $data---------------")
        ps, logprob = parse_bupf(data, g, h, 100, 10)
        grammar_decay!(g, h, decay)
        isnan(logprob) && continue
        push!(perplexity, logprob / length(data))
        cnt += 1
        for (t, w) in get_weighted_trees(ps)
            grammar_update!(g, h, t, w)
        end
    end
    for data in dataset(5)
        g.alpha = 0.01 # reduce exploration
        ps, logprob = parse_bupf(data, g, h, 100, 10)
        t = sample_bupf(ps, 0.01) # sample with low temperature
        println(data, " ", logprob / length(data))
        length(t) == 1 && print_tree(t[1], g)
    end
    # display(plot(smooth(perplexity)))
    return g, h
end
g, h = test1()
println("Fin")