struct StateTD 
    rule::Tuple{Int, Vector{Int}}
    dot::Int # dotted rule
    p::Float64 # target probability
    q::Float64 # sample probability
end

function parse_td(str::String, g::Grammar)
    S0 = g.index["S0"]
    toprule = (S0, g.rules[S0][1][1])
    loc = 1 # scan location
    stack = [StateTD(toprule, 1, 1, 1)]
    while length(stack) > 0
        s = stack[end]
        # println(s)
        if s.rule[1] == S0 && s.dot == 2 && loc > length(str) # finish
            return s.p, s.q
        elseif s.dot > length(s.rule[2]) # complete
            pop!(stack)
            t = pop!(stack) # move dot of the new stack top
            push!(stack, StateTD(t.rule, t.dot + 1, s.p, s.q))
        else
            next_sym = s.rule[2][s.dot]
            if g.label[next_sym][2] # scan if next_sym is a terminal
                if loc <= length(str) && next_sym == g.index[string(str[loc])] # scan success
                    pop!(stack) # move dot
                    push!(stack, StateTD(s.rule, s.dot + 1, s.p, s.q))
                    loc += 1 # move loc
                else # scan fail
                    return 0., s.q
                end
            else # predict if next_sym is a nonterminal
                # use rule probability as proposal distribution
                weights = map(x->x[2], g.rules[next_sym])
                idx, q = sample(normalize(weights))
                if sum(weights) == 0 return 0., s.q end # sample fail
                rhs, p = g.rules[next_sym][idx]
                push!(stack, StateTD((next_sym, rhs), 1, s.p * p, s.q * q))
            end
        end
    end
end