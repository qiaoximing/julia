module Utility

import LinearAlgebra
export warning, relative_error, div0, normalize, quantize, sample
export I

function warning(msg::String)
    printstyled("WARNING: ", bold=true, color=:yellow)
    println(msg)
end

function relative_error(target, estimate)
    return maximum(abs.(target - estimate)) / maximum(abs.(target))
end

div0(x::Float64, y::Float64)::Float64 = x == 0. ? 0. : x / y

normalize(x) = x ./ sum(x)

function quantize(x::Float64)
    a, b = modf(x)
    return b + (rand() < a ? 1.0 : 0.0)
end

function sample(x::Array{Float64})
    r = rand() # random float in [0, 1)
    for i in eachindex(x)
        r -= x[i]
        if r < 0
            return i, x[i]
        end
    end
    return 0, 1 - sum(x)
end

I = LinearAlgebra.I

end