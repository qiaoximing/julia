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
test1(100)

function test2(n::Int)
    grammar = test_grammar()
    dataset = generate_dataset(grammar, n)
    for data in dataset
        println("Parse $data:")
        parse_cyk(data, grammar)
    end
end
test2(10)