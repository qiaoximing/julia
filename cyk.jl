function compile_cyk_chart(str::String, g::Grammar)
    for lhs in 1:g.size
        for (rhs, _) in g.rules[lhs]
            if length(rhs) > 2
                warning("CYK fail: RHS longer than 2")
                return nothing
            end
        end
    end
    n = length(str)
    chart = zeros(Float64, g.size, n, n + 1)
    for s in 1:n
        for lhs in 1:g.size
            for (rhs, weight) in g.rules[lhs]
                if length(rhs) == 1 && rhs[1] == g.index[string(str[s])]
                    chart[lhs, s, s + 1] = weight
                end
            end
        end
    end
    for l in 2:n
        for s in 1:(n - l + 1)
            for t in 1:(l - 1)
                for lhs in 1:g.size
                    for (rhs, weight) in g.rules[lhs]
                        if length(rhs) == 2
                            chart[lhs, s, s + l] += (
                                chart[rhs[1], s, s + t] *
                                chart[rhs[2], s + t, s + l] *
                                weight)
                        end
                    end
                end
            end
        end
    end
    return chart
end

function sample_cyk(lhs::Int64, s::Int64, l::Int64, g::Grammar, chart::Array)
    if l == 1
        return chart[lhs, s, s + l], 1.
    else
        weights = zeros(l - 1, length(g.rules[lhs]))
        for t in 1:(l - 1)
            for (i, (rhs, weight)) in enumerate(g.rules[lhs])
                if length(rhs) == 2
                    weights[t, i] = (
                        chart[rhs[1], s, s + t] *
                        chart[rhs[2], s + t, s + l] *
                        weight)
                else
                    weights[t, i] = 0
                end
            end
        end
        distr = reduce(vcat, weights) ./ chart[lhs, s, s + l]
        idx, q = sample(distr)
        t = mod(idx - 1, l - 1) + 1
        rhs, p = g.rules[lhs][div(idx - 1, l - 1) + 1]
        println("$(g.label[lhs][1]) -> $(s) $(g.label[rhs[1]][1]) $(s+t) $(g.label[rhs[2]][1]) $(s+l) (p=$p, q=$q)")
        left_p, left_q = sample_cyk(rhs[1], s, t, g, chart)
        right_p, right_q = sample_cyk(rhs[2], s + t, l - t, g, chart)
        # we should always have p * left_p * right_p == chart[lhs, s, s + l]
        return p * left_p * right_p, q * left_q * right_q
    end
end

function parse_cyk(str::String, g::Grammar)
    chart = compile_cyk_chart(str, g) 
    # for lhs in g.index["S"]:g.size
    #     println(g.label[lhs][1], ": ", chart[lhs, :, :])
    # end
    if chart !== nothing
        # use S, not S0
        return sample_cyk(g.index["S"], 1, length(str), g, chart)
    else
        return 0., 0.
    end
end