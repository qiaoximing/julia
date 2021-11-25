struct Grammar
    size::Int64
    index::Dict{String, Int64}
    label::Vector{Tuple{String, Bool}} # 0->nonterminals, 1->terminals 
    rules::Vector{Vector{Tuple{Vector{Int}, Float64}}} # list of rules
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

function test_grammar()
    terminals = ["a","b","c","d","z","\0"]
    nonterminals = ["S0","S","A","B"]
    index = compile_index(terminals, nonterminals)
    label = compile_label(terminals, nonterminals)
    rules = compile_rules(index, label, [
        "S0 -> S \0 1.0",
        "S -> A S 0.5 | B 0.5",
        # "S -> S A 0.5 | A 0.5",
        # "S -> A 1.0",
        "A -> a 0.25 | b 0.25 | c 0.25 | d 0.25",
        # "A -> a 1.0"
        "B -> z 1.0"
    ])
    return Grammar(length(index), index, label, rules)
end