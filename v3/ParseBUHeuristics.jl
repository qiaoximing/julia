# Bottom-up parsing with particle filters
include("Utility.jl")
include("GrammarEx.jl")
include("ParseUtil.jl")
include("ParseTDPF.jl")
include("Learning1.jl")

mutable struct BUHeuristics
    alpha::Float32 # exploration rate = g.alpha
    decay::Float32 # decay rate of moving average
    tot::Float32 # total symbol count
    obs::Float32 # total observation count
    sym::Vector{Float32} # symbol prob
    sym1::Vector{Float32} # symbol count
    nul::Vector{Float32} # null prob per symbol
    in::Matrix{Float32} # input prob per symbol
    out::Matrix{Float32} # output prob per symbol
    out2::Matrix{Float32} # output count per symbol
    out1::Vector{Float32} # total count per symbol
    oi::Array{Float32, 3} # output prob per symbol-input
    oi3::Array{Float32, 3} # output count per symbol-input
    oi2a::Array{Float32, 2} # total count per symbol-input
    oi2b::Array{Float32, 2} # total count per symbol-input
    nul_oi::Array{Float32, 3} # output prob per symbol-input given not observations
    nul_oi3::Array{Float32, 3} # output prob per symbol-input given not observations
    nul_oi2a::Array{Float32, 2} # total prob per symbol-input given not observations
    nul_oi2b::Array{Float32, 2} # total prob per symbol-input given not observations
    w1::Dict
    w2::Dict
    w3::Dict
    w4::Dict
end

function init_buheuristics(g::GrammarEx, decay=0.01)
    n = size(g)
    h = BUHeuristics(g.alpha, decay, 0, 0,
        zeros(Float32, n), 
        zeros(Float32, n), 
        zeros(Float32, n), 
        zeros(Float32, n, n), 
        zeros(Float32, n, n), 
        zeros(Float32, n, n), 
        zeros(Float32, n), 
        zeros(Float32, n, n, n), 
        zeros(Float32, n, n, n), 
        zeros(Float32, n, n), 
        zeros(Float32, n, n), 
        zeros(Float32, n, n, n), 
        zeros(Float32, n, n, n), 
        zeros(Float32, n, n), 
        zeros(Float32, n, n), 
        Dict(),
        Dict(),
        Dict(),
        Dict())
    init!(h, g)
    return h
end
    
function init!(h::BUHeuristics, g)
    names1 = (:Ae, :Az, :eA, :BA)
    names2 = (:Aee, :Aez, :Aze, :Azz, 
              :eAe, :eAz, :BAe, :BAz, 
              :eeA, :BeA, :eBA, :CBA)
    # dimension organization:
    # unknown children from right to left
    # then parent 
    # then the known children from right to left
    h.w2[:A] = zeros(Float32, g.n_id, g.n_id)
    h.w1[:A] = zeros(Float32, g.n_id)
    for name in names1
        h.w3[name] = zeros(Float32, g.n_id, g.n_id, g.n_id)
        h.w2[name] = zeros(Float32, g.n_id, g.n_id)
        h.w1[name] = zeros(Float32, g.n_id)
    end
    for name in names2
        h.w4[name] = zeros(Float32, g.n_id, g.n_id, g.n_id, g.n_id)
        h.w3[name] = zeros(Float32, g.n_id, g.n_id, g.n_id)
        h.w2[name] = zeros(Float32, g.n_id, g.n_id)
        h.w1[name] = zeros(Float32, g.n_id)
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

function count_nul_oi!(h::BUHeuristics, isnul, out, in, sym, w=1)
    if isnul
        h.nul_oi[:, out, sym] .*= 1 - h.decay * w
        h.nul_oi[in, out, sym] += h.decay * w
    end
end

function count_nul!(h::BUHeuristics, isnul, sym, w=1)
    h.nul[sym] *= 1 - h.decay * w
    if isnul
        h.nul[sym] += h.decay * w
    end
end

function getaction(g, h, action, As, Bs=0, Cs=0)
    if action == :sft
        return (5 * g.alpha / g.n_id) + h.w1[:Az][As] + h.w1[:Aez][As] +
            h.w1[:Aze][As] + h.w1[:Azz][As] + h.w1[:eAz][As] + 
            (Bs > 0 ? h.w2[:BAz][As, Bs] + g.alpha / g.n_id : 0)
    elseif action in (:A, :Ae, :eA, :Aee, :eAe, :eeA)
        return h.w1[action][As] + g.alpha / g.n_id
    elseif Bs > 0 && action in (:BA, :BAe, :BeA, :eBA)
        # return h.w2[action][As, Bs] + g.alpha / g.n_id^2
        return h.w2[action][As, Bs] + g.alpha / g.n_id
    elseif Bs > 0 && Cs > 0 && action == :CBA
        return h.w3[action][As, Bs, Cs] + g.alpha / g.n_id^3
    else
        return Float32(0)
    end
end

function getparent(g, h, action, As, Bs=0, Cs=0)
    if action in (:A, :Ae, :eA, :Aee, :eAe, :eeA)
        return (h.w2[action][:, As] .+ g.alpha / g.n_id^2) / 
               (h.w1[action][As] + g.alpha / g.n_id)
    elseif Bs > 0 && action in (:BA, :BAe, :BeA, :eBA)
        return (h.w3[action][:, As, Bs] .+ g.alpha / g.n_id^3) / 
               (h.w2[action][As, Bs] + g.alpha / g.n_id^2)
    elseif Bs > 0 && Cs > 0 && action == :CBA
        return (h.w4[action][:, As, Bs, Cs] .+ g.alpha / g.n_id^4) / 
               (h.w3[action][As, Bs, Cs] + g.alpha / g.n_id^3)
    else
        warning("Parent don't exist")
    end
end

function getchild1(g, h, action, Ps, As, Bs=0)
    if action in (:Ae, :eA, :Aee, :eAe, :eeA)
        return (h.w3[action][:, Ps, As] .+ g.alpha / g.n_id^3) / 
               (h.w2[action][Ps, As] + g.alpha / g.n_id^2)
    elseif Bs > 0 && action in (:BAe, :BeA, :eBA)
        return (h.w4[action][:, Ps, As, Bs] .+ g.alpha / g.n_id^4) / 
               (h.w3[action][Ps, As, Bs] + g.alpha / g.n_id^3)
    else
        warning("Child1 don't exist")
    end
end

function getchild2(g, h, action, C1s, Ps, As)
    if action in (:Aee, :eAe, :eeA)
        return (h.w4[action][:, C1s, Ps, As] .+ g.alpha / g.n_id^4) / 
               (h.w3[action][C1s, Ps, As] + g.alpha / g.n_id^3)
    else
        warning("Child2 don't exist")
    end
end

function getsym(g, h::BUHeuristics, sym)
    # return h.sym[sym] * h.tot / h.obs
    return (h.sym1[sym] + g.alpha / g.n_id) / (h.obs + g.alpha / g.n_id * (g.n_tr - g.n_cn))
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

function count!(h, name, idx, w)
    if length(idx) == 1
        h.w1[name][idx[1]] += w
        # h.w0[name] += w
    elseif length(idx) == 2
        h.w2[name][idx...] += w
        h.w1[name][idx[2]] += w
        # h.w0[name] += w
    elseif length(idx) == 3
        h.w3[name][idx...] += w
        h.w2[name][idx[2], idx[3]] += w
        h.w1[name][idx[3]] += w
        # h.w0[name] += w
    elseif length(idx) == 4
        h.w4[name][idx...] += w
        h.w3[name][idx[2], idx[3], idx[4]] += w
        h.w2[name][idx[3], idx[4]] += w
        h.w1[name][idx[4]] += w
        # h.w0[name] += w
    end
end

function count_A!(h, p, l, nl, w)
    if !nl
        count!(h, :A, (p, l), w)
    end
end

function count_BA!(h, p, l, nl, r, nr, w)
    if nl && !nr
        count!(h, :eA, (l, p, r), w)
    elseif !nl && nr
        count!(h, :Ae, (r, p, l), w)
    elseif !nl && !nr
        count!(h, :BA, (p, r, l), w)
        count!(h, :Az, (r, p, l), w)
    end
end

function count_CBA!(h, p, l, nl, r, nr, d, nd, w)
    if nl && nr && !nd
        count!(h, :eeA, (r, l, p, d), w)
    elseif nl && !nr && nd
        count!(h, :eAe, (d, l, p, r), w)
    elseif !nl && nr && nd
        count!(h, :Aee, (d, r, p, l), w)
    elseif nl && !nr && !nd
        count!(h, :eAz, (d, l, p, r), w)
        count!(h, :eBA, (l, p, d, r), w)
    elseif !nl && nr && !nd
        count!(h, :Aez, (d, r, p, l), w)
        count!(h, :BeA, (r, p, d, l), w)
    elseif !nl && !nr && nd
        count!(h, :Aze, (d, r, p, l), w)
        count!(h, :BAe, (d, p, r, l), w)
    elseif !nl && !nr && !nd
        count!(h, :Azz, (d, r, p, l), w)
        count!(h, :BAz, (d, p, r, l), w)
        count!(h, :CBA, (p, d, r, l), w)
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
            count_A!(h, sym, left, isnul_left, h.decay)
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
                count_BA!(h, sym, left, isnul_left, right, isnul_right, h.decay)
            else
                count_CBA!(h, sym, left, isnul_left, right, isnul_right, dynam, isnul_dynam, h.decay)
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
        h.obs += h.decay
        output = input
        isnul = false
    else # type == Id
        output = input
        isnul = true
    end
    count_out!(h, output, sym)
    count_oi!(h, output, input, sym)
    count_nul_oi!(h, isnul, output, input, sym)
    count_nul!(h, isnul, sym)
    return output, isnul
end

function decay!(h)
    h.tot *= 1 - h.decay
    h.obs *= 1 - h.decay
    for k in keys(h.w)
        h.w[k] .*= 1 - h.decay
    end
end



function learn_from_grammar(g::GrammarEx, n)
    h = init_buheuristics(g, 0.01)
    generate_dataset(g, 0) # compile g.h
    for i in 1:n
        decay!(h)
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
    h = init_buheuristics(g, 0.01)
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