using DataStructures

function warning(msg::String)
    printstyled("WARNING: ", bold=true, color=:yellow)
    println(msg)
end

function relative_error(target, estimate)
    return maximum(abs.(target - estimate)) / maximum(abs.(target))
end

normalize(x; dims=1) = x ./ sum(x, dims=dims)

"suppress NaN for 0/0"
div0(x::Float32, y::Float32)::Float32 = x == 0. ? 0. : x / y
normalize0(x; dims=1) = div0.(x, sum(x, dims=dims))

"stochastic quantization"
function quantize(x::Float32)
    a, b = modf(x)
    return b + (rand(Float32) < a ? 1.0 : 0.0)
end

"sample from a probability mass function"
function sample(x::Array{Float32})
    r = rand(Float32) # random float in [0, 1)
    for i in eachindex(x)
        r -= x[i]
        if r < 0
            return i, x[i]
        end
    end
    return 0, 1 - sum(x)
end

"sample n times from a probability mass function"
function sample_n(x::Array{Float32}, n::Int, method::Symbol=:systematic)
    if method == :multinomial
        return [sample(x) for i in 1:n]
    elseif method == :systematic
        r = rand(Float32) / n
        result = []
        for i in eachindex(x)
            r -= x[i]
            while r < 0
                push!(result, (i, x[i]))
                r += 1 / n
            end
        end
        while length(result) < n
            push!(result, (0, 1-sum(x)))
        end
        return result
    else
        warning("Method $method not supported")
    end
end