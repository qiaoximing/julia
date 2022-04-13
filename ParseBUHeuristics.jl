# Bottom-up parsing with particle filters
include("Utility.jl")
include("GrammarEx.jl")
include("ParseUtil.jl")
include("ParseTDPF.jl")
include("Learning1.jl")
using Einsum

mutable struct BUHeuristics
    decay::Float32 # decay rate of moving average
    tot::Float32 # total symbol count
    sym::Vector{Float32} # symbol prob
    nul::Vector{Float32} # null prob per symbol
    in::Matrix{Float32} # input prob per symbol
    out::Matrix{Float32} # output prob per symbol
    oi::Array{Float32, 3} # output prob per symbol-input
    io::Array{Float32, 3} # input prob per symbol-output
    w::Dict
    a::Dict
    p::Dict
    c1::Dict
    c2::Dict
end
BUHeuristics(decay, n) = BUHeuristics(decay, 1, 
    ones(Float32, n), 
    ones(Float32, n), 
    ones(Float32, n, n), 
    ones(Float32, n, n), 
    ones(Float32, n, n, n), 
    ones(Float32, n, n, n), 
    Dict(),
    Dict(),
    Dict(),
    Dict(),
    Dict()
    )
function init!(h::BUHeuristics, g)
    names1 = (:Ae, :Az, :eA, :BA)
    names2 = (:Aee, :Aez, :Aze, :Azz, 
              :eAe, :eAz, :BAe, :BAz, 
              :eeA, :BeA, :eBA, :CBA)
    # dimension organization:
    # unknown children from right to left
    # then parent 
    # then the known children from right to left
    h.w[:A] = ones(Float32, g.n_id, g.n_id)
    for name in names1
        h.w[name] = ones(Float32, g.n_id, g.n_id, g.n_id)
    end
    for name in names2
        h.w[name] = ones(Float32, g.n_id, g.n_id, g.n_id, g.n_id)
    end
end

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
    # TODO: add input dependency and output distribution nul_oi
    h.nul[sym] *= 1 - h.decay * w
    if isnul
        h.nul[sym] += h.decay * w
    end
end

function compile!(h, g)
    dsum(x; dims) = dropdims(sum(x; dims); dims)
    # w: original weight, a: action, p: parent, c1/2:children
    h.a[:sft1] = dsum(h.w[:Az], dims=(1,2)) +
                 dsum(h.w[:Aez], dims=(1,2,3)) +
                 dsum(h.w[:Aze], dims=(1,2,3)) +
                 dsum(h.w[:Azz], dims=(1,2,3)) +
                 dsum(h.w[:eAz], dims=(1,2,3))
    h.a[:sft2] = dsum(h.w[:BAz], dims=(1,2))
    h.a[:A]    = dsum(h.w[:A], dims=1)
    h.a[:Ae]   = dsum(h.w[:Ae], dims=(1,2))
    h.a[:eA]   = dsum(h.w[:eA], dims=(1,2))
    h.a[:Aee]  = dsum(h.w[:Aee], dims=(1,2,3))
    h.a[:eAe]  = dsum(h.w[:eAe], dims=(1,2,3))
    h.a[:eeA]  = dsum(h.w[:eeA], dims=(1,2,3))
    h.a[:BA]   = dsum(h.w[:BA], dims=1)
    h.a[:BAe]  = dsum(h.w[:BAe], dims=(1,2))
    h.a[:BeA]  = dsum(h.w[:BeA], dims=(1,2))
    h.a[:eBA]  = dsum(h.w[:eBA], dims=(1,2))
    h.a[:CBA]  = dsum(h.w[:CBA], dims=1)
    h.p[:A]    = normalize(h.w[:A])
    h.p[:Ae]   = normalize(dsum(h.w[:Ae], dims=1))
    h.p[:eA]   = normalize(dsum(h.w[:eA], dims=1))
    h.p[:Aee]  = normalize(dsum(h.w[:Aee], dims=(1,2)))
    h.p[:eAe]  = normalize(dsum(h.w[:eAe], dims=(1,2)))
    h.p[:eeA]  = normalize(dsum(h.w[:eeA], dims=(1,2)))
    h.p[:BA]   = normalize(h.w[:BA])
    h.p[:BAe]  = normalize(dsum(h.w[:BAe], dims=1))
    h.p[:BeA]  = normalize(dsum(h.w[:BeA], dims=1))
    h.p[:eBA]  = normalize(dsum(h.w[:eBA], dims=1))
    h.p[:CBA]  = normalize(h.w[:CBA])
    h.c1[:Ae]  = normalize(h.w[:Ae])
    h.c1[:eA]  = normalize(h.w[:eA])
    h.c1[:Aee] = normalize(dsum(h.w[:Aee], dims=1))
    h.c1[:eAe] = normalize(dsum(h.w[:eAe], dims=1))
    h.c1[:eeA] = normalize(dsum(h.w[:eeA], dims=1))
    h.c1[:BAe] = normalize(h.w[:BAe])
    h.c1[:BeA] = normalize(h.w[:BeA])
    h.c1[:eBA] = normalize(h.w[:eBA])
    h.c2[:Aee] = normalize(h.w[:Aee])
    h.c2[:eAe] = normalize(h.w[:eAe])
    h.c2[:eeA] = normalize(h.w[:eeA])
end

function getaction(h, action, As, Bs=0, Cs=0)
    if action == :sft
        return h.a[:sft1][As] + (Bs > 0 ? h.a[:sft2][As, Bs] : 0)
    elseif action in (:A, :Ae, :eA, :Aee, :eAe, :eeA)
        return h.a[action][As]
    elseif Bs > 0 && action in (:BA, :BAe, :BeA, :eBA)
        return h.a[action][As, Bs]
    elseif Bs > 0 && Cs > 0 && action == :CBA
        return h.a[action][As, Bs, Cs]
    else
        return Float32(0)
    end
end

function getparent(h, action, As, Bs=0, Cs=0)
    if action in (:A, :Ae, :eA, :Aee, :eAe, :eeA)
        return h.p[action][:, As]
    elseif Bs > 0 && action in (:BA, :BAe, :BeA, :eBA)
        return h.p[action][:, As, Bs]
    elseif Bs > 0 && Cs > 0 && action == :CBA
        return h.p[action][:, As, Bs, Cs]
    else
        warning("Parent don't exist")
    end
end

function getchild1(h, action, Ps, As, Bs=0)
    if action in (:Ae, :eA, :Aee, :eAe, :eeA)
        return h.c1[action][:, Ps, As]
    elseif Bs > 0 && action in (:BAe, :BeA, :eBA)
        return h.c1[action][:, Ps, As, Bs]
    else
        warning("Child1 don't exist")
    end
end

function getchild2(h, action, C1s, Ps, As)
    if action in (:Aee, :eAe, :eeA)
        return h.c2[action][:, C1s, Ps, As]
    else
        warning("Child2 don't exist")
    end
end

function get_prob(h::BUHeuristics, sym, in=0, out=0)
    if in == 0 && out == 0
        return h.sym[sym]
    elseif out == 0
        return h.sym[sym] * h.in[in, sym]
    elseif in == 0
        return h.sym[sym] * h.out[out, sym]
    else
        return h.sym[sym] * h.in[in, sym] * h.oi[out, in, sym]
    end
end

function get_nul(h::BUHeuristics, sym=0)
    return sym > 0 ? h.nul[sym] : h.sym' * h.nul
end
function get_nul_out(h::BUHeuristics, sym, in=0)
    return in > 0 ? h.oi[:, in, sym]' * h.nul : h.out[:, sym]' * h.nul
end
function get_nul_out_syms(h::BUHeuristics, in=0)
    return in > 0 ? h.oi[:, in, :]' * h.nul : h.out' * h.nul
end
function get_nul_right(h::BUHeuristics, g::GrammarEx, sym, left=0)
    return left > 0 ? normalize(g.w_cm2[:, left, sym])' * h.nul :
                      g.h.p_r[:, sym]' * h.nul
end
function get_nul_left(h::BUHeuristics, g::GrammarEx, sym, right=0)
    return right > 0 ? normalize(g.w_cm2[right, :, sym])' * h.nul :
                      g.h.p_l[:, sym]' * h.nul
end

function count_A!(h, p, l, nl)
    if !nl
        h.w[:A][p, l] += h.decay
    end
end

function count_BA!(h, p, l, nl, r, nr)
    if nl && !nr
        h.w[:eA][l, p, r] += h.decay
    elseif !nl && nr
        h.w[:Ae][r, p, l] += h.decay
    elseif !nl && !nr
        h.w[:BA][p, r, l] += h.decay
        h.w[:Az][r, p, l] += h.decay
    end
end

function count_CBA!(h, p, l, nl, r, nr, d, nd)
    if nl && nr && !nd
        h.w[:eeA][r, l, p, d] += h.decay
    elseif nl && !nr && nd
        h.w[:eAe][d, l, p, r] += h.decay
    elseif !nl && nr && nd
        h.w[:Aee][d, r, p, l] += h.decay
    elseif nl && !nr && !nd
        h.w[:eAz][d, l, p, r] += h.decay
        h.w[:eBA][l, p, d, r] += h.decay
    elseif !nl && nr && !nd
        h.w[:Aez][d, r, p, l] += h.decay
        h.w[:BeA][r, p, d, l] += h.decay
    elseif !nl && !nr && nd
        h.w[:Aze][d, r, p, l] += h.decay
        h.w[:BAe][d, p, r, l] += h.decay
    elseif !nl && !nr && !nd
        h.w[:Azz][d, r, p, l] += h.decay
        h.w[:BAz][d, p, r, l] += h.decay
        h.w[:CBA][p, d, r, l] += h.decay
    end
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
            count_A!(h, sym, left, isnul_left)
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
                dynam = output_left
                output, isnul_dynam = simulate!(h, output_left, output_right, g)
                isnul = isnul_left && isnul_right && isnul_dynam
            else # type in (Lrl, Prl) 
                dynam = output_right
                output, isnul_dynam = simulate!(h, output_right, output_left, g)
                isnul = isnul_left && isnul_right && isnul_dynam
            end
            if type in (Ll, Lr, Pl, Pr)
                count_BA!(h, sym, left, isnul_left, right, isnul_right)
            else
                count_CBA!(h, sym, left, isnul_left, right, isnul_right, dynam, isnul_dynam)
            end
        end
    elseif type == Fn
        output, _ = sample(g.h.p_fn[:, input, offset(sym, g)])
        isnul = true
    elseif type == Cn
        output, _ = sample(g.h.p_cn[:, offset(sym, g)])
        isnul = true
    elseif type == Tr
        # write(io, g.label[sym])
        output = input
        isnul = false
    else # type == Id
        output = input
        isnul = true
    end
    count_out!(h, output, sym)
    count_oi!(h, output, input, sym)
    count_io!(h, input, output, sym)
    count_nul!(h, isnul, sym)
    return output, isnul
end

function decay!(h)
    h.tot *= 1 - h.decay
    for k in keys(h.w)
        h.w[k] .*= 1 - h.decay
    end
end

function learn_from_grammar(g::GrammarEx, n)
    h = BUHeuristics(0.01, size(g))
    init!(h, g)
    generate_dataset(g, 0) # compile g.h
    for i in 1:n
        decay!(h)
        simulate!(h, 1, 1, g)
    end
    compile!(h, g)
    return h
end

# TODO: count_nul not implemented
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

# g = test_grammar1()
# h = learn_from_grammar(g, 10000)
# println("finish")

# using LinearAlgebra: norm
# g = test_grammar3()
# h0 = learn_from_grammar(g, 10000)
# for n in (10, 100, 1000, 10000)
#     println(n)
#     h1 = learn_from_grammar(g, n)
#     h2 = learn_from_dataset(g, n)
#     for field in (:tot, :sym, :in, :out)
#         v0 = getfield(h0, field)
#         v1 = getfield(h1, field)
#         v2 = getfield(h2, field)
#         println("grammar ", string(field), " ", norm(v1 .- v0, 1) / norm(v0, 1))
#         println("dataset ", string(field), " ", norm(v2 .- v0, 1) / norm(v2, 1))
#     end
# end