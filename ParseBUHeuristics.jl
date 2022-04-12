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
    p::Dict
end
BUHeuristics(decay, n) = BUHeuristics(decay, 1, 
    ones(Float32, n), 
    ones(Float32, n), 
    ones(Float32, n, n), 
    ones(Float32, n, n), 
    ones(Float32, n, n, n), 
    ones(Float32, n, n, n), 
    Dict()
    )

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

function compile(g, h)
    w_extra = 0
    fn = normalize(g.w_fn .+ w_extra, dims=1)
    cn = normalize(g.w_cn .+ w_extra, dims=1)
    pl = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1 .+ w_extra, dims=1)
    pr = normalize(sum(g.w_cm2, dims=2)[:, 1, :] .+ w_extra, dims=1)
    p2 = normalize(g.w_cm2 .+ w_extra, dims=(1,2))
    d = (h.tot * h.in .* h.sym')[:, 1:g.n_cm]
    e = h.nul 
    z = 1 .- h.nul
    f = h.nul_oi # output distribution when not observed
    u, ll, lr, pl, pr, llr, lrl, plr, prl = map(
        T -> Vector{Float32}([g.type[i] == T for i in 1:g.n_cm]),
        (U, Ll, Lr, Pl, Pr, Llr, Lrl, Plr, Prl))
    # case A
    @einsum UA[i, l] := d[i, p] * pl[l, p] * u[p]
    # case Ae
    @einsum LlAe[o, i, l] := d[i, p] * p2[r, l, p] * e[o, r] * ll[p]
    @einsum LrAe[o, i, l] := d[i, p] * p2[r, l, p] * e[o, r] * lr[p]
    @einsum PlAe[i, l] := d[i, p] * p2[r, l, p] * e[i, r] * pl[p]
    @einsum PrAe[i, l] := d[i, p] * p2[r, l, p] * e[i, r] * pr[p]
    # case Az
    @einsum LlAz[o, i, l] := d[i, p] * p2[r, l, p] * z[o, r] * ll[p]
    @einsum LrAz[o, i, l] := d[i, p] * p2[r, l, p] * z[o, r] * lr[p]
    @einsum PlAz[i, l] := d[i, p] * p2[r, l, p] * z[i, r] * pl[p]
    @einsum PrAz[i, l] := d[i, p] * p2[r, l, p] * z[i, r] * pr[p]
    # case eA
    @einsum LleA[i, r] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * ll[p]
    @einsum LreA[i, r] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * lr[p]
    @einsum PleA[i, r] := d[i, p] * p2[r, l, p] * e[i, l] * pl[p]
    @einsum PreA[i, r] := d[i, p] * p2[r, l, p] * e[i, l] * pr[p]
    # case BA
    @einsum LlBA[i, r, l] := d[i, p] * p2[r, l, p] * ll[p]
    @einsum LrBA[i, r, l] := d[i, p] * p2[r, l, p] * lr[p]
    @einsum PlBA[i, r, l] := d[i, p] * p2[r, l, p] * pl[p]
    @einsum PrBA[i, r, l] := d[i, p] * p2[r, l, p] * pr[p]
    # case Aee
    @einsum LlrAee[o, i, l] := d[i, p] * p2[r, l, p] * f[x, o, r] * e[o, r] * e[x, o] * llr[p]
    @einsum LrlAee[o, i, l] := d[i, p] * p2[r, l, p] * f[x, o, r] * e[o, r] * e[o, x] * lrl[p]
    @einsum PlrAee[o, i, l] := d[i, p] * p2[r, l, p] * f[x, i, r] * e[i, r] * e[x, o] * plr[p]
    @einsum PrlAee[o, i, l] := d[i, p] * p2[r, l, p] * f[x, i, r] * e[i, r] * e[o, x] * prl[p]
    # case Aez
    @einsum LlrAez[o, i, l] := d[i, p] * p2[r, l, p] * f[x, o, r] * e[o, r] * z[x, o] * llr[p]
    @einsum LrlAez[o, i, l] := d[i, p] * p2[r, l, p] * f[x, o, r] * e[o, r] * z[o, x] * lrl[p]
    @einsum PlrAez[o, i, l] := d[i, p] * p2[r, l, p] * f[x, i, r] * e[i, r] * z[x, o] * plr[p]
    @einsum PrlAez[o, i, l] := d[i, p] * p2[r, l, p] * f[x, i, r] * e[i, r] * z[o, x] * prl[p]
    # case Azx (including Aze and Azz)
    @einsum LlrAzx[o, i, l] := d[i, p] * p2[r, l, p] * z[o, r] * llr[p]
    @einsum LrlAzx[o, i, l] := d[i, p] * p2[r, l, p] * z[o, r] * lrl[p]
    @einsum PlrAzx[o, i, l] := d[i, p] * p2[r, l, p] * z[i, r] * plr[p]
    @einsum PrlAzx[o, i, l] := d[i, p] * p2[r, l, p] * z[i, r] * prl[p]
    # case eAe
    @einsum LlreAe[o, i, r] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * e[o, i] * llr[p]
    @einsum LrleAe[o, i, r] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * e[i, o] * lrl[p]
    @einsum PlreAe[o, i, r] := d[i, p] * p2[r, l, p] * f[x, i, l] * e[i, l] * e[o, x] * prl[p]
    @einsum PrleAe[o, i, r] := d[i, p] * p2[r, l, p] * f[x, i, l] * e[i, l] * e[x, o] * prl[p]
    # case eAz
    @einsum LlreAz[o, i, r] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * z[o, i] * llr[p]
    @einsum LrleAz[o, i, r] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * z[i, o] * lrl[p]
    @einsum PlreAz[o, i, r] := d[i, p] * p2[r, l, p] * f[x, i, l] * e[i, l] * z[o, x] * prl[p]
    @einsum PrleAz[o, i, r] := d[i, p] * p2[r, l, p] * f[x, i, l] * e[i, l] * z[x, o] * prl[p]
    # case BAx (including BAe and BAz)
    # @einsum LlrBA[i, r, l] := d[i, p] * p2[r, l, p] * llr[p]
    # @einsum LrlBA[i, r, l] := d[i, p] * p2[r, l, p] * lrl[p]
    # @einsum PlrBA[i, r, l] := d[i, p] * p2[r, l, p] * prl[p]
    # @einsum PrlBA[i, r, l] := d[i, p] * p2[r, l, p] * prl[p]
    @einsum LlrBAe[r, l] := d[x, p] * p2[r, l, p] *  * llr[p]
    # case eeA
    @einsum LlreeA[i, a] := d[x, p] * p2[r, l, p] * f[a, x, l] * e[x, l] * f[i, a, r] * e[a, r] * llr[p]
    @einsum LrleeA[i, a] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * f[a, i, r] * e[i, r] * lrl[p]
    @einsum PlreeA[i, a] := d[x, p] * p2[r, l, p] * f[a, x, l] * e[x, l] * f[i, x, r] * e[x, r] * plr[p]
    @einsum PrleeA[i, a] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * f[a, x, r] * e[x, r] * prl[p]
    # case BeA 
    @einsum LlrBeA[i, x, a, l] := d[x, p] * p2[r, l, p] * f[i, a, r] * e[a, r] * llr[p]
    @einsum LrlBeA[i, x, a, l] := d[x, p] * p2[r, l, p] * f[a, i, r] * e[i, r] * lrl[p]
    @einsum PlrBeA[i, x, a, l] := d[x, p] * p2[r, l, p] * f[i, x, r] * e[x, r] * plr[p]
    @einsum PrlBeA[x, a, l] := d[x, p] * p2[r, l, p] * f[a, x, r] * e[x, r] * prl[p]
    # case eBA
    @einsum LlreBA[a, r] := d[x, p] * p2[r, l, p] * f[a, x, l] * e[x, l] * llr[p]
    @einsum LrleBA[i, a, r] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * lrl[p]
    @einsum PlreBA[x, a, r] := d[x, p] * p2[r, l, p] * f[a, x, l] * e[x, l] * plr[p]
    @einsum PrleBA[i, x, a, r] := d[x, p] * p2[r, l, p] * f[i, x, l] * e[x, l] * prl[p]
    # case CBA: no need
    names = (:UA, :LlAe)
    vars = (UA, LlAe)
    for (name, var) in zip(names, vars)
        h.p[name] = var
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

function learn_from_grammar(g::GrammarEx, n)
    h = BUHeuristics(0.01, size(g))
    generate_dataset(g, 0) # compile g.h
    for i in 1:n
        h.tot *= 1 - h.decay
        simulate!(h, 1, 1, g)
    end
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

g = test_grammar1()
h = learn_from_grammar(g, 10000)
compile(g, h)
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