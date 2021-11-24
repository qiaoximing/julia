struct State 
    rule::Tuple{Int, Vector{Int}}
    dot::Int # dotted rule
    p::Float64 # target probability
    q::Float64 # sample probability
end

function parse(str::String, g::Grammar)
    S0 = g.index["S0"]
    toprule = (S0, g.rules[S0][1][1])
    loc = 1 # scan location
    stack = [State(toprule, 1, 1, 1)]
    while length(stack) > 0
        s = stack[end]
        println(length(stack), s, loc)
        if s.dot > length(s.rule[2]) # complete
            pop!(stack)
            if length(stack) == 0 break end
            s = pop!(stack) # move dot of the new stack top
            push!(stack, State(s.rule, s.dot + 1, s.p, s.q))
        else
            next_sym = s.rule[2][s.dot]
            if g.label[next_sym][2] # scan if next_sym is a terminal
                if loc <= length(str) && next_sym == g.index[string(str[loc])] # scan success
                    pop!(stack) # move dot
                    push!(stack, State(s.rule, s.dot + 1, s.p, s.q))
                    loc += 1 # move loc
                else break end
            else # predict if next_sym is a nonterminal
                # use rule probability as proposal distribution
                distr = map(x->x[2], g.rules[next_sym])
                idx, q = sample(distr)
                if idx == 0 break end # distr should be automatically normalized
                rhs, p = g.rules[next_sym][idx]
                push!(stack, State((next_sym, rhs), 1, s.p * p, s.q * q))
            end
        end
    end
    return length(stack) == 0
end