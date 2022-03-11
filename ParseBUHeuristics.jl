# Bottom-up parsing with particle filters
include("Utility.jl")
include("GrammarEx.jl")
include("ParseUtil.jl")
include("ParseTDPF.jl")
include("Learning1.jl")

mutable struct BUHeuristics
    decay::Float32 # decay rate of moving average
    tot::Float32 # total symbol count
    sym::Vector{Float32} # symbol count
    in::Matrix{Float32} # input count per symbol
    out::Array{Float32, 3} # output count per symbol-input
end
BUHeuristics(decay, n) = BUHeuristics(decay, 1, ones(Float32, n), 
    ones(Float32, n, n), ones(Float32, n, n, n))

function count_sym!(h::BUHeuristics, sym, w=1)
    h.sym .*= 1 - h.decay * w
    h.sym[sym] += h.decay * w
end

function count_in!(h::BUHeuristics, in, sym, w=1)
    h.in[:, sym] .*= 1 - h.decay * w
    h.in[in, sym] += h.decay * w
end

function count_out!(h::BUHeuristics, out, in, sym, w=1)
    h.out[:, in, sym] .*= 1 - h.decay * w
    h.out[out, in, sym] += h.decay * w
end

function simulate!(h::BUHeuristics, sym::Int, input::Int, g::GrammarEx)
    type = g.type[sym] # symbol type
    h.tot += h.decay
    count_sym!(h, sym)
    count_in!(h, input, sym)
    if type in Cm
        left, _ = sample(g.h.p_left[:, sym])
        output_left = simulate!(h, left, input, g)
        if type == U
            output = output_left
        else
            right, _ = sample(g.h.p_right[:, left, sym])
            if type in (Ll, Lr, Llr, Lrl)
                input_right = output_left
            else # type in (Pl, Pr, Plr, Prl)
                input_right = input
            end
            output_right = simulate!(h, right, input_right, g)
            if type in (Ll, Pl)
                output = output_left
            elseif type in (Lr, Pr)
                output = output_right
            elseif type in (Llr, Plr)
                output = simulate!(h, output_left, output_right, g)
            else # type in (Lrl, Prl) 
                output = simulate!(h, output_right, output_left, g)
            end
        end
    elseif type == Fn
        output, _ = sample(g.h.p_fn[:, input, offset(sym, g)])
    elseif type == Cn
        output, _ = sample(g.h.p_cn[:, offset(sym, g)])
    elseif type == Tr
        # write(io, g.label[sym])
        output = input
    else # type == Id
        output = input
    end
    count_out!(h, output, input, sym)
    return output
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
    count_out!(h, out, in, sym, w)
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