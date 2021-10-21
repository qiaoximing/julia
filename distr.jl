abstract type AbstractDistr end

# zero probability on all possible values (ill defined)
# used to stop the computation
struct ZeroDistr <: AbstractDistr end

# uniform probability on all possible values
# used for initialization
struct OneDistr <: AbstractDistr end

# only returned by sample()
mutable struct PointDistr <: AbstractDistr
    val::Int64 # the point mass with 100% probability
end

mutable struct SparseDistr <: AbstractDistr
    vals::Vector{Int64} # a sorted list of nodes 
    prob::Vector{Float64} # their probabilities (>0 and sum=1)
end

mutable struct DenseDistr <: AbstractDistr
    prob::Vector{Float64} # probabilities of all nodes (>=0 and sum=1)
end

Base.copy(d::PointDistr) = PointDistr(d.val)
Base.copy(d::SparseDistr) = SparseDistr(copy(d.vals), copy(d.prob))
Base.copy(d::DenseDistr) = DenseDistr(copy(d.prob))

function sample(d::SparseDistr)
    r = rand() # random float in [0, 1)
    i = 0
    while i < length(d.prob) && r >= 0
        i += 1
        r -= d.prob[i]
    end
    return PointDistr(d.vals[i])
end

function multiply(d1::SparseDistr, d2::SparseDistr)
    if length(d1.vals) > length(d2.vals)
        d1, d2 = d2, d1 # swap to save memory
    end
    vals = similar(d1.vals)
    prob = similar(d1.prob)
    i, j, k = 1, 1, 1
    while i <= length(d1.vals) && j <= length(d2.vals)
        v1, v2 = d1.vals[i], d2.vals[j]
        if v1 < v2
            i += 1
        elseif v1 > v2
            j += 1
        else # v1 == v2
            vals[k] = v1
            prob[k] = d1.prob[i] * d2.prob[j]
            # prob[k] > 0 should always happen
            i += 1; j += 1; k += 1
        end
    end
    if k > 1
        resize!(vals, k - 1) # need GC?
        resize!(prob, k - 1)
        prob ./= sum(prob)
        return SparseDistr(vals, prob)
    else
        # Zero if no overlap
        return ZeroDistr()
    end
end

# use copy to avoid unwanted sharing. remove if we use immutable struct
multiply(d1::PointDistr, d2::PointDistr) = d1.val == d2.val ? copy(d1) : ZeroDistr()
multiply(d1::PointDistr, d2::SparseDistr) = d1.val ∈ d2.vals ? copy(d1) : ZeroDistr()
multiply(d1::SparseDistr, d2::PointDistr) = multiply(d2, d1)

# One as the multiplicative identity
multiply(d1::OneDistr, d2::OneDistr) = OneDistr()
multiply(d1::OneDistr, d2::AbstractDistr) = copy(d2)
multiply(d1::AbstractDistr, d2::OneDistr) = copy(d1)

# overload *
Base.:*(d1::AbstractDistr, d2::AbstractDistr) = multiply(d1, d2)

# entropy of Zero and One are undefined
entropy(d::PointDistr) = 0 # certainty
entropy(d::SparseDistr) = -sum(d.prob .* log.(d.prob))
entropy(d::DenseDistr) = -sum(d.prob .* log.(d.prob))
const 𝒽 = entropy