include("utility.jl")
include("probability.jl")
include("grammar.jl")
# include("topdown.jl")
include("leftcorner.jl")

data = "abT"
grammar = test_grammar()
println(grammar)
function main(n::Int)
    n_succ = 0
    for i in 1:n
        # println("Parse $(i):")
        n_succ += parse(data, grammar)
    end
    println(n_succ)
end
main(10000)