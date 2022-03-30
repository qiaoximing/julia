# Bottom-up parsing with particle filters
include("Utility.jl")
include("GrammarEx.jl")
include("ParseUtil.jl")
include("ParseTDPF.jl")
include("Learning1.jl")

mutable struct BUHeuristics
    decay::Float32 # decay rate of moving average
    tot::Float32 # total symbol count
    sym::Vector{Float32} # symbol prob
    nul::Vector{Float32} # null prob per symbol
    in::Matrix{Float32} # input prob per symbol
    out::Matrix{Float32} # output prob per symbol
    oi::Array{Float32, 3} # output prob per symbol-input
    io::Array{Float32, 3} # input prob per symbol-output
end
BUHeuristics(decay, n) = BUHeuristics(decay, 1, 
    ones(Float32, n), 
    ones(Float32, n), 
    ones(Float32, n, n), 
    ones(Float32, n, n), 
    ones(Float32, n, n, n), 
    ones(Float32, n, n, n))

function count_sym!(h::BUHeuristics, sym, w=1)
    h.sym .*= 1 - h.decay * w
    h.sym[sym] += h.decay * w
end

function count_in!(h::BUHeuristics, in, sym, w=1)
    h.in[:, sym] .*= 1 - h.decay * w
    h.in[in, sym] += h.decay * w
end
function count_out!(h::BUHeuristics, out, sym, w=1)
    h.out[:, sym] .*= 1 - h.decay * w
    h.out[out, sym] += h.decay * w
end

function count_oi!(h::BUHeuristics, out, in, sym, w=1)
    h.oi[:, in, sym] .*= 1 - h.decay * w
    h.oi[out, in, sym] += h.decay * w
end
function count_io!(h::BUHeuristics, in, out, sym, w=1)
    h.io[:, out, sym] .*= 1 - h.decay * w
    h.io[in, out, sym] += h.decay * w
end

function count_nul!(h::BUHeuristics, isnul, sym, w=1)
    h.nul[sym] .*= 1 - h.decay * w
    isnul && h.nul[sym] += h.decay * w
end

function get_prob(h::BUHeuristics, sym, in=0, out=0)
    if in == 0 && out == 0
        return h.sym[sym]
    elseif out == 0
        return h.sym[sym] * h.in[in, sym]
    elseif in == 0
        return h.sym[sym] * h.out[out, sym]
    else
        return h.sym[sym] * h.in[in, sym], h.oi[out, in, sym]
    end
end

function get_nul(h::BUHeuristics, sym)
    return sym > 0 ? h.nul[sym] : h.sym' * h.nul
end
function get_nul_out(h::BUHeuristics, sym, in)
    return in > 0 ? h.oi[:, in, sym]' * h.nul : h.out[:, sym]' * h.nul
end
function get_nul_out_syms(h::BUHeuristics, in)
    return in > 0 ? h.oi[:, in, :]' * h.nul : h.out' * h.nul
end
function get_nul_left(h::BUHeuristics, g::GrammarEx, sym, left)
    return normalize(g.w_cm2[:, left, sym])' * h.nul
end
function get_nul_right(h::BUHeuristics, g::GrammarEx, sym, right)
    return normalize(g.w_cm2[right, :, sym])' * h.nul
end

function simulate!(h::BUHeuristics, sym::Int, input::Int, g::GrammarEx)
    type = g.type[sym] # symbol type
    h.tot += h.decay
    count_sym!(h, sym)
    count_in!(h, input, sym)
    if type in Cm
        left, _ = sample(g.h.p_left[:, sym])
        output_left, isnul_left = simulate!(h, left, input, g)
        if type == U
            output = output_left
            isnul = isnul_left
        else
            right, _ = sample(g.h.p_right[:, left, sym])
            if type in (Ll, Lr, Llr, Lrl)
                input_right = output_left
            else # type in (Pl, Pr, Plr, Prl)
                input_right = input
            end
            output_right, isnul_right = simulate!(h, right, input_right, g)
            if type in (Ll, Pl)
                output = output_left
                isnul = isnul_left && isnul_right
            elseif type in (Lr, Pr)
                output = output_right
                isnul = isnul_left && isnul_right
            elseif type in (Llr, Plr)
                output, isnul_dynam = simulate!(h, output_left, output_right, g)
                isnul = isnul_left && isnul_right && isnul_dynam
            else # type in (Lrl, Prl) 
                output, isnul_dynam = simulate!(h, output_right, output_left, g)
                isnul = isnul_left && isnul_right && isnul_dynam
            end
        end
    elseif type == Fn
        output, _ = sample(g.h.p_fn[:, input, offset(sym, g)])
        isnul == true
    elseif type == Cn
        output, _ = sample(g.h.p_cn[:, offset(sym, g)])
        isnul == true
    elseif type == Tr
        # write(io, g.label[sym])
        output = input
        isnul == false
    else # type == Id
        output = input
        isnul == true
    end
    count_out!(h, output, sym)
    count_oi!(h, output, input, sym)
    count_io!(h, input, output, sym)
    count_nul!(h, isnul, sym)
    return output, isnul
end

function learn_from_grammar(g::GrammarEx, n)
    h = BUHeuristics(0.01, size(g))
    generate_dataset(g, 0) # compile g.h
    for i in 1:n
        h.tot *= 1 - h.decay
        simulate!(h, 1, 1, g)
    end
    return h
end

function update!(h::BUHeuristics, t::Tree, w)
    sym, in, out = t.state.sym, t.state.input, t.state.output
    h.tot += h.decay * w
    count_sym!(h, sym, w)
    count_in!(h, in, sym, w)
    count_out!(h, out, sym, w)
    count_oi!(h, out, in, sym, w)
    count_io!(h, in, out, sym, w)
    for child in (t.left, t.right, t.dynam)
        if child !== nothing
            update!(h, child, w)
        end
    end
end

function learn_from_dataset(g::GrammarEx, n)
    h = BUHeuristics(0.01, size(g))
    for (data, _) in generate_dataset(g, n)
        ps, _ = parse_tdpf(data, g, 10, 10, false)
        weighted_trees = get_weighted_trees(ps)
        for (t, w) in weighted_trees
            h.tot *= 1 - h.decay * w
            update!(h, t, w)
        end
    end
    return h
end

using LinearAlgebra: norm
g = test_grammar3()
h0 = learn_from_grammar(g, 10000)
for n in (10, 100, 1000, 10000)
    println(n)
    h1 = learn_from_grammar(g, n)
    h2 = learn_from_dataset(g, n)
    for field in (:tot, :sym, :in, :out)
        v0 = getfield(h0, field)
        v1 = getfield(h1, field)
        v2 = getfield(h2, field)
        println("grammar ", string(field), " ", norm(v1 .- v0, 1) / norm(v0, 1))
        println("dataset ", string(field), " ", norm(v2 .- v0, 1) / norm(v2, 1))
    end
end