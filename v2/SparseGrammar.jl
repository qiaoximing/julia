module SparseGrammar

using Utility
export Grammar, decay!, update!
export test_grammar, init_grammar, generate_dataset

struct Grammar
    size::Int64
    index::Dict{String, Int64}
    label::Vector{Tuple{String, Bool}} # 0->nonterminals, 1->terminals 
    rules::Vector{Vector{Tuple{Vector{Int}, Float64}}} # list of rules
    norms::Vector{Float64} # normalization of rules weights
end

function compile_index(terminals, nonterminals)
    index = Dict(reverse.(enumerate([terminals; nonterminals])))
    if length(index) < length(terminals) + length(nonterminals)
        warning("grammar includes repeating symbols")
    end
    return index
end

function compile_label(terminals, nonterminals)
    return [map(x->(x,true), terminals); 
            map(x->(x,false), nonterminals)]
end

function compile_rules(index, label, strs)
    s = length(index)
    rules = Vector{Vector{Tuple{Vector{Int64}, Float64}}}(undef, s)
    for i in 1:s rules[i] = [] end
    for str in strs
        lhs, rhs_many = split(str, " -> ")
        lhs_token, = split(lhs, " ")
        lhs_idx = index[lhs_token]
        if label[lhs_idx][2] == false
            for rhs in split(rhs_many, " | ")
                rhs_tokens = split(rhs, " ")
                rhs_idxs = map(x->index[x], rhs_tokens[1:end-1])
                weight = Base.parse(Float64, rhs_tokens[end])
                push!(rules[lhs_idx], (rhs_idxs, weight))
            end
        else
            warning("terminals in LHS $(lhs)")
        end
    end
    return rules
end

function compile_norm(rules)
    return [sum([y[2] for y in x]) for x in rules]
end

function decay!(g::Grammar, rate::Float64)
    for r in g.rules
        for (i, (rhs, weight)) in enumerate(r)
            r[i] = (rhs, quantize(weight * rate))
        end
    end
    g.norms .= compile_norm(g.rules)
end

function update!(g::Grammar, update::Vector)
    for (lhs, i) in update
        rhs, p = g.rules[lhs][i]
        g.rules[lhs][i] = (rhs, p + 1.0)
        g.norms[lhs] += 1.0
    end
end

function test_grammar()
    terminals = ["a","b","c","d","z","\0"]
    nonterminals = ["S0","S","A","B"]
    index = compile_index(terminals, nonterminals)
    label = compile_label(terminals, nonterminals)
    rules = compile_rules(index, label, [
        "S0 -> S \0 1",
        "S -> A S 1 | A B 1",
        # "S -> S A 1 | A A 1",
        "A -> a 1 | b 1 | c 1 | d 1",
        # "A -> a 1.0"
        "B -> z 1"
    ])
    norms = compile_norm(rules)
    return Grammar(length(index), index, label, rules, norms)
end

function init_grammar()
    terminals = ["a","b","c","d","z","\0"]
    nonterminals = ["S0","S","A","B"]
    index = compile_index(terminals, nonterminals)
    label = compile_label(terminals, nonterminals)
    rules = compile_rules(index, label, [
        "S0 -> S \0 1",
        "S -> S S 1 | S A 1 | S B 1",
        "S -> A S 1 | A A 1 | A B 1",
        "S -> B S 1 | B A 1 | B B 1",
        "S -> a 1 | b 1 | c 1 | d 1 | z 1",
        "A -> S S 1 | S A 1 | S B 1",
        "A -> A S 1 | A A 1 | A B 1",
        "A -> B S 1 | B A 1 | B B 1",
        "A -> a 1 | b 1 | c 1 | d 1 | z 1",
        "B -> S S 1 | S A 1 | S B 1",
        "B -> A S 1 | A A 1 | A B 1",
        "B -> B S 1 | B A 1 | B B 1",
        "B -> a 1 | b 1 | c 1 | d 1 | z 1",
    ])
    norms = compile_norm(rules)
    return Grammar(length(index), index, label, rules, norms)
end

function expand!(io::IOBuffer, lhs::Int64, g::Grammar)
    if g.label[lhs][2]
        write(io, g.label[lhs][1])
    else
        weights = map(x->x[2], g.rules[lhs])
        idx, _ = sample(normalize(weights))
        rhs, _ = g.rules[lhs][idx]
        for sym in rhs
            expand!(io, sym, g)
        end
    end
end

function generate_sentence(g::Grammar)
    io = IOBuffer()
    expand!(io, g.index["S"], g)
    return String(take!(io))
end

function generate_dataset(g::Grammar, n::Int64)
    return [generate_sentence(g) for i in 1:n]
end

end