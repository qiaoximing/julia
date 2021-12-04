module Utility

import LinearAlgebra
export warning, relative_error, normalize, sample
export I

function warning(msg::String)
    printstyled("WARNING: ", bold=true, color=:yellow)
    println(msg)
end

function relative_error(target, estimate)
    return maximum(abs.(target - estimate)) / maximum(abs.(target))
end

normalize(x) = x ./ sum(x)

function sample(x::Vector{Float64})
    r = rand() # random float in [0, 1)
    i = 0
    while i < length(x) && r >= 0
        i += 1
        r -= x[i]
    end
    if r < 0
        return i, x[i]
    else
        return 0, 1 - sum(x)
    end
end

I = LinearAlgebra.I

end