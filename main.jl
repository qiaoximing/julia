push!(LOAD_PATH, pwd())
using Utility, MyGrammar, Parse

function test1(n::Int)
    data = "abz"
    grammar = test_grammar()
    heuristic = compile_lc_heuristic(grammar)
    println(grammar)
    n_succ_td, n_succ_slc = 0, 0
    for i in 1:n
        p, q = parse_td(data, grammar)
        # println("TD $(i): $p, $q")
        n_succ_td += p > 0
        p, q = parse_slc(data, grammar, heuristic)
        # println("SLC $(i): $p, $q")
        n_succ_slc += p > 0
    end
    println("Top-down success count: ", n_succ_td)
    println("Static LC success count: ", n_succ_slc)
end
# test1(100)

function test2(n::Int)
    grammar = test_grammar()
    dataset = generate_dataset(grammar, n)
    for data in dataset
        println("Parse $data:")
        parse_cyk(data, grammar)
    end
end
# test2(10)

function decay!(g::Grammar, rate::Float64)
    for r in g.rules
        for (i, (rhs, weight)) in enumerate(r)
            r[i] = (rhs, weight * rate)
        end
    end
    g.norms .*= rate
end

function grammar_induction(n::Int, m::Int)
    dataset = generate_dataset(test_grammar(), n)
    grammar = init_grammar()
    decay!(grammar, 10.0)
    for epoch in 1:m
        # println("Epoch $epoch:")
        for (i, data) in enumerate(dataset)
            parse_cyk!(data, grammar)
            if i % 5 == 0
                decay!(grammar, 0.95)
            end
        end
        # println(grammar.norms)
    end
    prune(x) = filter(x->x[2]>1, x)
    println("S: ", prune(grammar.rules[8]))
    println("A: ", prune(grammar.rules[9]))
    println("B: ", prune(grammar.rules[10]))
    for data in dataset[1:10]
        println("Parse $data:")
        parse_cyk(data, grammar)
    end
end
grammar_induction(1000, 100)