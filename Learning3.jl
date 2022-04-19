"""
Grammar learning with BU parsing
"""
include("ParseBUPF.jl")

# focus on learning CFG
function init_grammar(alpha)
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
    # partial sums
    w_l = zeros(Float32, n_id, n_cm)
    w_sym = zeros(Float32, n_cm)
    h = (; w_l, w_sym)
    return GrammarEx(n_all..., w_all..., type, label, index, alpha, h)
end

"Update grammar and related heuristics from a BUState tree"
function grammar_update!(g::GrammarEx, h::BUHeuristics, s::BUState, w)
    sym, input, output = s.sym, s.in, s.out
    type = g.type[sym]
    if type == U
        # rule weight
        g.w_cm1[s.left.sym, sym] += w
        # partial sums
        g.h.w_l[s.left.sym, sym] += w
        g.h.w_sym[sym] += w
        # heuristics
    elseif type in Cm
        # rule weight
        g.w_cm2[s.right.sym, s.left.sym, sym] += w
        # partial sums
        g.h.w_l[s.left.sym, sym] += w
        g.h.w_sym[sym] += w
        # heuristics
    elseif type == Fn
        # rule weight
        g.w_fn[output, input, offset(sym, g)] += w
        # heuristics
    elseif type == Cn
        # rule weight
        g.w_cn[output, offset(sym, g)] += w
        # heuristics
    end
    grammar_update!(g, h, s.left, w)
    grammar_update!(g, h, s.right, w)
    grammar_update!(g, h, s.dynam, w)
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
end

"Extract BUState trees from a particle system"
function get_weighted_trees(particles::Vector{BUParticle})
    w_total = sum([p.w for p in particles])
    weighted_trees = []
    for p in particles    
        if p.w > 0
            t = get_trees(p)
            w = p.w / w_total
            push!(weighted_trees, (t, w))
        end
    end
    return weighted_trees
end

# Random.seed!(0)
dict = ["xx", "yy", "z"]
dataset(n) = [reduce(*, [rand(dict) for j in 1:2]) for i in 1:n]
println(dataset(10))
alpha, decay = 0.1, 0.01
g = init_grammar(alpha)
h = init_buheuristics(g, decay)
cnt = 0
perplexity = []
for data in dataset(1000)
    ps, logprob = parse_bupf(data, g, 100, 10)
    grammar_decay!(g, h, decay)
    isnan(logprob) && continue
    push!(perplexity, logprob / length(data))
    cnt += 1
    for (t, w) in get_weighted_trees(ps)
        grammar_update!(g, h, t, w)
    end
end
for data in dataset(5)
    g.alpha = 0. # disable exploration
    ps, logprob = parse_bupf(data, g, 100, 10)
    t = sample_bupf(ps, 0.01) # sample with low temperature
    println(data, " ", logprob / length(data))
    isnothing(t) || print_tree(t, g)
end
display(plot(smooth(perplexity)))