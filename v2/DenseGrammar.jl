module DenseGrammar

using Utility
export Grammar
export test_grammar, init_grammar, generate_dataset, decay!

mutable struct Grammar
    n::Int64 # total size
    ns::Int64 # number of symbols (nonterminals)
    br::Array{Float64, 3} # binary rules
    ur::Array{Float64, 2} # unary rules
    label::String # label of symbols and terminals
    c::NamedTuple # cache from rules
end
Grammar(n, ns, br, ur, lab) = Grammar(n, ns, br, ur, lab, (;))

function compile_rules(label, ns, rules)
    n = length(label)
    index = Dict(reverse.(enumerate(label)))
    br = zeros(n, n, ns)
    ur = zeros(n, ns)
    for rule in rules
        lhs, rhs_list = split(rule, " -> ")
        lhs_token, = split(lhs, " ")
        lhs_idx = index[lhs_token[1]]
        for rhs in split(rhs_list, " | ")
            rhs_tokens = split(rhs, " ")
            rhs_idxs = map(x->index[x[1]], rhs_tokens[1:end-1])
            weight = Base.parse(Float64, rhs_tokens[end])
            if length(rhs_idxs) == 1
                ur[rhs_idxs[1], lhs_idx] = weight
            else
                br[rhs_idxs[2], rhs_idxs[1], lhs_idx] = weight
            end
        end
    end
    return br, ur
end

function test_grammar()
    symbols = "SAB"
    terminals = "ab_z"
    label = symbols * terminals
    br, ur = compile_rules(label, length(symbols), [
        # "S -> A S 2 | z 1",
        # "A -> B A 2 | _ 1",
        # "B -> a 1 | b 1"
        "S -> A a 2 | a 1",
        "A -> S b 2 | b 1"
    ])
    return Grammar(length(label), length(symbols), br, ur, label)
end

function init_grammar()
    # symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # terminals = "abcdefghijklmnopqrstuvwxyz_."
    symbols = "SAB"
    terminals = "ab_z"
    label = symbols * terminals
    ns, n = length(symbols), length(label)
    # br, ur = zeros(n, n, ns), zeros(n, ns)
    # br[1:ns, 1:ns, :] .= 1. # restrict to CNF
    # ur[ns+1:end, :] .= 30. # restrict to CNF
    br, ur = ones(n, n, ns), ones(n, ns)
    br .*= 10
    ur .*= 10
    ur[ns+1:end, :] .*= 2 * ns / (n-ns)
    # for i in 1:ns
    #     br[i, :, i] .= 0
    #     br[:, i, i] .= 0
    #     ur[i, i] = 0
    # end
    return Grammar(n, ns, br, ur, label)
end

function decay!(x::Float64, g::Grammar)
    g.br .-= (g.br .- 0.001) .* x
    g.ur .-= (g.ur .- 0.001) .* x
    # g.ur[1:g.ns, :] .-= (g.ur[1:g.ns, :] .- 10) .* x
    # g.ur[g.ns+1:end, :] .-= (g.ur[g.ns+1:end, :] .- (2 * g.ns / (g.n-g.ns))) .* x
end

function simulate!(io::IOBuffer, lhs::Int64, g::Grammar)
    if lhs > g.ns # lhs is a terminal
        write(io, g.label[lhs])
    else
        left, _ = sample(normalize(g.c.lr[:, lhs] + g.ur[:, lhs]))
        simulate!(io, left, g)
        right, _ = sample(normalize([g.br[:, left, lhs]; g.ur[left, lhs]]))
        if right <= g.n
            simulate!(io, right, g)
        end
    end
end

function generate_sentence(g::Grammar)
    io = IOBuffer()
    simulate!(io, 1, g)
    return String(take!(io))
end

function generate_dataset(g::Grammar, n::Int64)
    if !haskey(g.c, :lr) # left symbol of rules
        lr = sum(g.br, dims=1)[1, :, :]
        g.c = (g.c..., lr=lr) 
    end
    return [generate_sentence(g) for i in 1:n]
end

end