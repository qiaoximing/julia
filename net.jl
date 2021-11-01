include("SimpleSparseArrays.jl")
using .SimpleSparseArrays

mutable struct Val
    val::Float64
end
Val() = Val(0)
Base.zero(::Type{Val}) = Val(0)
Base.:(==)(x::Val, y::Val) = x.val == y.val
Base.:(+)(x::Val, y::Val) = Val(x.val + y.val)
function add!(v::Val, x)
    v.val += x 
end

# For N-D net, create:
# 1 N-D array,
# N (N-1)-D arrays for 1D sums,
# C(N, 2) (N-2)-D arrays fro 2D sums,
# ...
# N 1-D arrays for (N-1)-D sums,
# 1 scalar for the total sum.
# For each update, update all the above sums.

struct Net{T,N}
    data::Dict{NTuple{N,Int},SimpleSparseArray{T}}
    size::NTuple{N,Int}
    b2i::Dict{NTuple{N,Int},Int} # binary to int
    i2b::Dict{Int,NTuple{N,Int}} # input to binary
    root::Int # root node
    init::Int # input of root
    prints::Dict{Char,Int} # alphabet of observations
end

function int_bit_converter(ndim::Int)
    b2i, i2b = [], []
    for i in 0:(2^ndim - 1)
        bits = bitstring(i)[end-ndim+1:end]
        bits = Tuple([bit-'0' for bit in bits])
        push!(b2i, (bits, i))
        push!(i2b, (i, bits))
    end
    return Dict(b2i), Dict(i2b)
end

function net_init(n_nodes::Int, alphabet::String, n_edgetypes::Int=2)
    size = (n_edgetypes, n_nodes, n_nodes, n_nodes, n_nodes, n_nodes)
    ndim = length(size)
    b2i, i2b = int_bit_converter(ndim)
    # init all arrays
    dict = []
    for b in keys(b2i)
        subsize = Tuple([size[i] for i in 1:ndim if b[i]==1])
        array = SimpleSparseArray{Float64}(subsize...)
        push!(dict, (b, array))
    end
    # annotate special nodes
    n = 2
    root, init = 1:n
    prints = Dict([(alphabet[i], n+i) for i in 1:length(alphabet)])
    if n + length(alphabet) > n_nodes
        warning("not enough nodes")
    end
    net = Net{Float64,eval(ndim)}(Dict(dict), size, b2i, i2b, root, init, prints)
    return net
end

# add value to Net, update all partial sums
# if a certain index=0, then only add to partial sums
# note that in this case, partial sums might be inconsistent
# this is equivalent to having an "unknown" value in each dimension
function add!(net::Net{Float64}, idx_raw, x)
    function toInt(i)::Int
        if i isa Etype
            return Int(i)
        elseif i isa Char
            return net.prints[i]
        elseif i isa Int
            return i
        else
            warning("wrong input")
            return 0
        end
    end
    idx = [toInt(i) for i in idx_raw]
    ndim = length(net.size)
    for b in keys(net.b2i)
        subidx = Tuple([idx[i] for i in 1:ndim if b[i]==1])
        if !(0 in subidx)
            net.data[b][subidx...] += x
        end
    end
end

# TODO: (Optimization)
# we can directly return the sum and avoid extra computation
function getval(net::Net{Float64}, targ, cond=[])
    # get raw counts from net
    ndim = length(net.size)
    idx = zeros(Int, ndim)
    targ = Int(targ) # Int or enum type start from 1
    targ_len = net.size[targ]
    vals = zeros(Float64, targ_len)
    for (ax, i) in cond
        idx[Int(ax)] = i
    end
    idx[targ] = 1
    arrayidx = Tuple([i==0 ? 0 : 1 for i in idx])
    array = net.data[arrayidx]
    for i in 1:targ_len
        idx[targ] = i
        subidx = Tuple([i for i in idx if i>0])
        vals[i] = array[subidx...]
    end
    return vals
end

# Bernoulli mixture model with shared emission 
function bmm_net_init()
    n_nodes = 10
    net = net_init(n_nodes, "01T") # inputs always terminate with 'T'
    # Etype: L D
    # Axes: Edge Node Left Right Input Output
    R, I = net.root, net.init
    p0, p1, T = '0', '1', 'T'
    A = net.prints['T'] + 1
    S, s0, s1 = A+1:A+3
    if s1 > n_nodes warning("need more nodes") end
    println("Net config: 1=R 2=I 3=p0 4=p1 5=T $A=A $S=S $s0=s0 $s1=s1")
    add!(net, (B, R, S, A, I, T), 100) 
    add!(net, (D, A, p0, A, s0, T), 45)
    add!(net, (D, A, p0, A, s1, T), 5)
    add!(net, (D, A, p1, A, s0, T), 5)
    add!(net, (D, A, p1, A, s1, T), 45)
    add!(net, (D, A, p0, T, s0, T), 45)
    add!(net, (D, A, p0, T, s1, T), 5)
    add!(net, (D, A, p1, T, s0, T), 5)
    add!(net, (D, A, p1, T, s1, T), 45)
    add!(net, (0, S, 0, 0, I, s0), 50) 
    add!(net, (0, S, 0, 0, I, s1), 50)
    add!(net, (0, p0, 0, 0, s0, p0), 90)
    add!(net, (0, p0, 0, 0, s1, p0), 10)
    add!(net, (0, p1, 0, 0, s0, p1), 10)
    add!(net, (0, p1, 0, 0, s1, p1), 90)
    add!(net, (0, T, 0, 0, s0, T), 50)
    add!(net, (0, T, 0, 0, s1, T), 50)
    return net
end