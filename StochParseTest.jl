push!(LOAD_PATH, pwd())
using Random
using Utility, SparseGrammar, StochParse

# Random.seed!(2)

function test1(n::Int)
    data = "ab"
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

function grammar_induction_cky(n::Int, m::Int)
    dataset = generate_dataset(test_grammar(), n)
    grammar = init_grammar()
    decay!(grammar, 10.0)
    for epoch in 1:m
        # println("Epoch $epoch:")
        for (i, data) in enumerate(dataset)
            p, q, update = parse_cyk(data, grammar)
            update!(grammar, update)
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
    for data in dataset[1:5]
        println("Parse $data:")
        parse_cyk(data, grammar, true)
    end
end
# grammar_induction_cky(1000, 10)

function grammar_induction_lc(n::Int, m::Int)
    dataset = generate_dataset(test_grammar(), n)
    grammar = init_grammar()
    decay!(grammar, 10.0)
    for epoch in 1:m
        success = 0
        for (i, data) in enumerate(dataset)
            h = compile_lc_heuristic(grammar)
            p, q, update = parse_slc(data, grammar, h)
            if p > 0
                update!(grammar, update) 
                success += 1
                if success % 5 == 0
                    decay!(grammar, 0.95)
                end
            end
        end
        println("Epoch $epoch: success $success/$n")
    end
    prune(x) = filter(x->x[2]>1, x)
    println("S0: ", prune(grammar.rules[7]))
    println("S: ", prune(grammar.rules[8]))
    println("A: ", prune(grammar.rules[9]))
    println("B: ", prune(grammar.rules[10]))
    for data in dataset[1:5]
        println("Parse $data:")
        h = compile_lc_heuristic(grammar)
        parse_slc(data, grammar, h, true)
    end
end
# grammar_induction_lc(1000, 10)

g = test_grammar()
h = compile_bu(g)
for i in 1:10
    str = sample_bu(g, h)
    println(str)
end