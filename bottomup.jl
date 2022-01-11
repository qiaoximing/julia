function compile_bu(g::Grammar)
    s = g.size
    p = zeros(Float64, s, s)
    for lhs in 1:s
        for (rhs, weight) in g.rules[lhs]
            if weight > 0
                for r in rhs
                    p[r, lhs] += weight / g.norms[lhs]
                end
            end
        end
    end
    b = zeros(Float64, s)
    b[g.index["S0"]] = 1.
    d = (I - p) \ b
    dd = zeros(Float64, s, s)
    for lhs in 1:s
        for (rhs, weight) in g.rules[lhs]
            if length(rhs) == 2 && weight > 0
                dd[rhs[1], rhs[2]] += d[lhs] * weight / g.norms[lhs]
            end
        end
    end
    dl = sum(dd, dims=2)
    dr = sum(dd, dims=1)
    pa = dd ./ (dl * dr)
    pa[isnan.(pa)] .= 0
    pa ./= maximum(pa)
    return (;p, d, dd, dl, dr, pa)
end

struct StateBU
    s::Int # symbol, 0 means stack bottom
    l::Int # terminal length
    p::Float64 # target probability
    q::Float64 # path probability
end

# sample a string from the language of the grammar
function sample_bu(g::Grammar, h::NamedTuple, debug=false)
    # assume all terminals are before S0
    S0 = g.index["S0"]
    stack = [StateBU(0, 0, 1, 1)] 
    str = Vector{Int}()
    rej_count = 0
    while length(stack) > 0
        s = stack[end]
        # decide to shift or reduce
        if s.s == 0 # stack bottom
            ps = 1. # always shift
        elseif s.s < S0 # terminal
            ps = 0. # always reduce (for CNF grammar)
        else # nonterminal
            ps = h.dl[s.s] / h.d[s.s]
            if h.dl[s.s] + h.dr[s.s] - h.d[s.s] > 1e-5
                warning("Shift reduct mismatch")
            end
        end
        idx, q_sr = sample([ps, 1-ps])
        if idx == 1 # shift x
            weights = h.d[1:S0-1] # only shift terminals
            x, q_x = sample(normalize(weights))
            push!(str, x)
            i = j = length(str)
            push!(stack, StateBU(x, 1, s.p, s.q * q_sr * q_x))
        else # reduce
            if s.s < g.index["S0"] # rule A->x
                pop!(stack)
                rule_prob = h.p[s.s, :] # assuming CNF
                weights = rule_prob .* h.d
                lhs, q_rule = sample(normalize(weights))
                p = h.p[s.s, lhs]
                push!(stack, StateBU(lhs, s.l, s.p * p, s.q * q_sr * q_rule))
            else # rule A->BC
                # decide to accept or reject
                t = stack[end-1]
                pa = t.s > 0 ? h.pa[t.s, s.s] : 0.
                idx, q_acc = sample([pa, 1-pa])
                if idx == 1 # accept BC, sample A
                    rule_prob = zeros(g.size)
                    for lhs in 1:g.size
                        for (rhs, weight) in g.rules[lhs]
                            if length(rhs) == 2 && rhs[1] == t.s && rhs[2] == s.s
                                rule_prob[lhs] = weight / g.norms[lhs]
                            end
                        end
                    end
                    weights = rule_prob .* h.d
                    lhs, q_rule = sample(normalize(weights))
                    p = rule_prob[lhs]
                    pop!(stack)
                    t = pop!(stack)
                    push!(stack, StateBU(lhs, s.l + t.l, s.p * p, s.q * q_sr * q_acc * q_rule))
                else # reject stack top
                    for i in 1:s.l
                        pop!(str)
                    end
                    pop!(stack)
                    rej_count += 1
                    if rej_count > 20
                        warning("Too many rejects")
                        break
                    end
                end
            end
        end
    end
    return reduce(*, map(x->x[1], g.label[str]))
end