using LinearAlgebra
include("utility.jl")
include("probability.jl")
include("grammar.jl")
include("topdown.jl")
include("leftcorner.jl")
include("cyk.jl")

data = "abz"
grammar = test_grammar()
heuristic = compile_heuristic(grammar)
println(grammar)
function main(n::Int)
    parse_cyk(data, grammar)
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
main(100)