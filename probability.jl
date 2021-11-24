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