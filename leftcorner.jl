struct StateLC
    rule::Tuple{Int, Vector{Int}}
    dot::Int # dotted rule
    lc::Int # left corner (0 means nothing)
    p::Float64 # target probability
    q::Float64 # sample probability
end

# Parsing knowledge that doesn't belong to Grammar
struct Heuristic
    p::Matrix{Float64} # LC -> LHS -> sum of rule probabilities
    r::Matrix{Float64} # LC -> LHS -> probabilistic LC relation
end

# Statically compile heuristic from grammar
function compile_lc_heuristic(g::Grammar)
    s = g.size
    p = zeros(Float64, s, s)
    for lhs in 1:s
        for (rhs, weight) in g.rules[lhs]
            lc = rhs[1]
            p[lc, lhs] += weight
        end
    end
    r = Matrix{Float64}(I, s, s)
    max_iter, max_err = 100, 0.01
    for i in 1:max_iter
        r_old = r
        r = I + p * r
        err = relative_error(r, r_old)
        if err < max_err || i == max_iter
            println("Heuristic complied in $(i) iters with $(err) error.")
            break
        end
    end
    println("Error to groundtrugh is: ", relative_error(inv(I - p), r))
    return Heuristic(p, r)
end

# Static LC parser
function parse_slc(str::String, g::Grammar, h::Heuristic)
    S0 = g.index["S0"]
    toprule = (S0, g.rules[S0][1][1])
    loc = 1 # scan location
    stack = [StateLC(toprule, 1, 0, 1, 1)]
    while length(stack) > 0
        s = stack[end]
        # println(s)
        if s.rule[1] == S0 && s.dot == 2 && s.lc == 0 && loc > length(str) # finish
            return s.p, s.q
        elseif s.dot > length(s.rule[2]) # complete
            lc = s.rule[1] # lhs becomes new lc
            pop!(stack)
            t = pop!(stack) # move dot of the new stack top
            push!(stack, StateLC(t.rule, t.dot, lc, s.p, s.q))
        else
            next_sym = s.rule[2][s.dot]
            if s.lc == 0 # scan if lc is nothing
                if loc <= length(str) # scan success
                    lc = g.index[string(str[loc])]
                    pop!(stack) # add lc
                    push!(stack, StateLC(s.rule, s.dot, lc, s.p, s.q))
                    loc += 1 # move loc
                else # scan fail
                    return 0., s.q
                end
            else # project or move
                # create proposal distribution
                rule_prob = h.p[s.lc, :] # slightly inefficient
                lc_relation = h.r[:, next_sym]
                weights = rule_prob .* lc_relation
                push!(weights, next_sym == s.lc ? 1. : 0.)
                if sum(weights) == 0 return 0., s.q end # sample fail
                idx, q = sample(normalize(weights))
                if idx > g.size # move
                    s = pop!(stack) # move dot and remove lc
                    push!(stack, StateLC(s.rule, s.dot + 1, 0, s.p, s.q * q))
                else # project
                    lhs = idx # lhs of the new rule
                    # sample from rules with matched lc
                    weights = map(x->(x[1][1]==s.lc ? x[2] : 0.), g.rules[lhs])
                    if sum(weights) == 0 return 0., s.q end # sample fail
                    idx, q_rule = sample(normalize(weights))
                    rhs, p = g.rules[lhs][idx]
                    # move dot and remove lc in the new state
                    push!(stack, StateLC((lhs, rhs), 2, 0, s.p * p, s.q * q * q_rule))
                end
            end
        end
    end
end