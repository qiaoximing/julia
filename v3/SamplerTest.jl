using Random
using StatsBase
using Distributions
using StatsPlots
using DataStructures

function huffman_sample(weights::Vector, m::Int, n::Int)
    s = zeros(length(weights))
    for i in 1:n
        heap = BinaryMinHeap(reverse.(enumerate(weights)))
        while length(heap) > m
            w1, i1 = pop!(heap)
            w2, i2 = pop!(heap)
            i_new = rand() < w1 / (w1 + w2) ? i1 : i2
            push!(heap, (w1 + w2, i_new))
        end
        s0 = zeros(length(s))
        for (w, i) in extract_all!(heap)
            s0[i] = w
        end
        s .+= s0 #./ sum(s0)
    end
    return s ./ n
end

function gumbel_sample(x::Vector, m::Int, n::Int)
    s = zeros(length(x))
    for i in 1:n
        u = rand(length(x)) 
        k = x ./ u
        S = partialsortperm(k, 1:m+1, rev=true)
        s0 = zeros(length(s))
        for i in S[1:m]
            s0[i] = max(x[i], k[S[end]])
        end
        s .+= s0 #./ sum(s0)
    end
    return s ./ n
end

function gumbeln_sample(x::Vector, m::Int, n::Int)
    s = zeros(length(x))
    for i in 1:n
        u = rand(length(x)) 
        k = x ./ u
        S = partialsortperm(k, 1:m+1, rev=true)
        s0 = zeros(length(s))
        for i in S[1:m]
            s0[i] = max(x[i], k[S[end]])
        end
        s .+= s0 ./ sum(s0)
    end
    return s ./ n
end

u, v = 30, 10000
m, n = 5, 10
weights = rand(Dirichlet(u,0.3))
# weights = [pdf(Geometric(0.5), i) for i in 0:u-1]
# weights = [pdf(Poisson(0.5), i) for i in 0:u-1]
huff = [huffman_sample(weights, m, n) for i in 1:v]
gumb = [gumbel_sample(weights, m, n) for i in 1:v]
gumbn = [gumbeln_sample(weights, m, n) for i in 1:v]
# violin(repeat(1:2:2u, v), vcat(huff...))
# violin!(repeat(2:2:2u, v), vcat(gumb...))
# plot!(repeat(weights, inner=2))
println("MSE: $(mean(std(huff))), $(mean(std(gumb))), $(mean(std(gumbn)))")
println("Bias: $(mean(abs.(mean(huff).-weights))), $(mean(abs.(mean(gumb).-weights))), $(mean(abs.(mean(gumbn).-weights)))")
p = plot(weights)
plot!(mean(huff), ribbon=std(huff), fillalpha=0.2)
plot!(mean(gumb), ribbon=std(gumb), fillalpha=0.2)
plot!(mean(gumbn), ribbon=std(gumb), fillalpha=0.2)
ylims!(-0.05,0.1)
