export parse_cyk

function sample_cyk(lhs::Int, s::Int, l::Int, g::Grammar, input::Array, chart::Array)
    if l == 1
        w = input[s]
        return Tree(lhs, g.ur[w, lhs] / g.c.tw[lhs], Tree(
            w, 1., nothing, nothing), nothing)
    else
        weights = zeros(g.ns, g.ns, l - 1)
        for t in 1:l-1
            weights[:, :, t] = (
                (chart[:, s+t, s+l] * chart[:, s, s+t]') .* 
                g.br[1:g.ns, 1:g.ns, lhs]  ./ g.c.tw[lhs]) 
        end
        idx, _ = sample(normalize(weights))
        if idx == 0
            warning("Bad grammar at lhs=$lhs, s=$s, l=$l")
            return Tree(lhs, 0., nothing, nothing)
        end
        rhs2, rhs1, t = Tuple(CartesianIndices(weights)[idx])
        tree1 = sample_cyk(rhs1, s, t, g, input, chart)
        tree2 = sample_cyk(rhs2, s+t, l-t, g, input, chart)
        return Tree(lhs, g.br[rhs2, rhs1, lhs] / g.c.tw[lhs] * tree1.p * tree2.p, 
            tree1, tree2)
    end
end

function parse_cyk(str::String, g::Grammar)
    n = length(str)
    input = [g.c.c2i[chr] for chr in str]
    chart = zeros(Float64, g.ns, n, n + 1)
    for s in 1:n
        w = input[s]
        chart[:, s, s + 1] = g.ur[w, :] ./ g.c.tw[1, :]
    end
    for l in 2:n
        for s in 1:(n - l + 1)
            for t in 1:(l - 1)
                for lhs in 1:g.ns
                    chart[lhs, s, s+l] += (
                        chart[:, s+t, s+l]' * 
                        g.br[1:g.ns, 1:g.ns, lhs] * 
                        chart[:, s, s+t]
                    ) / g.c.tw[lhs]
                end
            end
        end
    end
    if chart[1, 1, 1 + n] <= 0 
        return [], chart
    else
        t = sample_cyk(1, 1, n, g, input, chart)
        return [Item(PStack(1, t, nothing), n, 1., nothing)], chart
    end
end