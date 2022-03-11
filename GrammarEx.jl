#= 
Extended grammar with functional semantics
=#
include("Utility.jl")

# 5 types of symbols ordered by
# - CoMposite, FuNction, CoNstant, TeRminal, IDentity
# - Subtypes for composite: U, Ll, Lr, Llr, Lrl, Pl, Pr, Plr, Prl
@enum NodeType U Ll Lr Llr Lrl Pl Pr Plr Prl Fn Cn Tr Id
const Cm = (U, Ll, Lr, Llr, Lrl, Pl, Pr, Plr, Prl)

mutable struct GrammarEx
    # accumulated number of symbols
    n_cm::Int # composite symbols
    n_fn::Int # function symbols
    n_cn::Int # constant symbols
    n_tr::Int # terminal symbols
    n_id::Int # identity symbol
    # rule weights
    w_cm1::Array{Float32, 2} # unary production rules
    w_cm2::Array{Float32, 3} # binary production rules
    w_fn::Array{Float32, 3} # function I/O's
    w_cn::Array{Float32, 2} # constant outputs
    # others
    type::Vector{NodeType} # type of nodes
    label::Vector{String} # label of all symbols
    index::Dict{String, Int} # index of all labels
    alpha::Float32 # hyperparameter for Dirichlet distribution prior
    h::NamedTuple # compiled heuristics for sampling, parsing, etc.
end

"Total number of symbols"
Base.size(g::GrammarEx) = g.n_id

"Index range of a node type.
 Note: any type in Cm gets the full Cm range"
function range(type::NodeType, g)
    if type == Fn
        return (g.n_cm + 1):g.n_fn
    elseif type == Cn
        return (g.n_fn + 1):g.n_cn
    elseif type == Tr
        return (g.n_cn + 1):g.n_tr
    elseif type == Id
        return (g.n_tr + 1):g.n_id
    else # type in Cm
        return 1:g.n_cm
    end
end

"Offset index for rule weights"
function offset(i::Int, g::GrammarEx)
    type = g.type[i]
    if type == Fn
        return i - g.n_cm
    elseif type == Cn
        return i - g.n_fn
    elseif type == Tr
        return i - g.n_cn
    elseif type == Id
        return i - g.n_tr
    else # type in Cm
        return i
    end
end

"Translate literal rule descriptions to weights"
function compile_rules(n_all, index, rules)
    n_cm, n_fn, n_cn, n_tr, n_id = n_all
    w_cm1 = zeros(Float32, n_id, n_cm)
    w_cm2 = zeros(Float32, n_id, n_id, n_cm)
    w_fn = zeros(Float32, n_id, n_id, n_fn - n_cm)
    w_cn = zeros(Float32, n_id, n_cn - n_fn)
    type = NodeType.(zeros(Int, n_id))
    g = (;n_cm, n_fn, n_cn, n_tr, n_id) # hack to get range
    for t in [Fn, Cn, Tr, Id]
        type[range(t, g)] .= t
    end
    for rule in split(rules, "\n")
        if rule == "" continue end
        lhs, rhs_list = split(rule, " -> ")
        lhs_label, lhs_type_name = split(lhs, " ")
        lhs_idx = index[lhs_label]
        lhs_type = eval(Meta.parse(lhs_type_name))
        for rhs in split(rhs_list, " | ")
            rhs_labels_and_weight = split(rhs, " ")
            # println(rhs_labels_and_weight)
            rhs_idxs = [index[i] for i in rhs_labels_and_weight[1:end-1]]
            weight = Base.parse(Float32, rhs_labels_and_weight[end])
            if lhs_type in Cm # Cm index starts from 1
                if length(rhs_idxs) == 1
                    w_cm1[rhs_idxs[1], lhs_idx] = weight
                else
                    w_cm2[rhs_idxs[2], rhs_idxs[1], lhs_idx] = weight
                end
                type[lhs_idx] = lhs_type
            elseif lhs_type == Fn # Fn index starts from n_cm + 1
                w_fn[rhs_idxs[2], rhs_idxs[1], lhs_idx - n_cm] = weight
            elseif lhs_type == Cn # Cn index starts from n_fn + 1
                w_cn[rhs_idxs[1], lhs_idx - n_fn] = weight
            end
        end
    end
    return (w_cm1, w_cm2, w_fn, w_cn), type
end

"Repeating element"
function test_grammar1()
    cm, fn, cn, tr, id = split.((
    "S A B C",
    "f",
    "c",
    "x y z",
    "I"), " ")
    n_all = cumsum(length.((cm, fn, cn, tr, id)))
    label = [cm; fn; cn; tr; id]
    index = Dict(reverse.(enumerate(label)))
    rules = """
    S Lr -> c A 1
    A Lr -> B C 1
    B Plr -> I I 1
    C Plr -> z I 1
    c Cn -> x 1 | y 1
    """
    w_all, type = compile_rules(n_all, index, rules)
    return GrammarEx(n_all..., w_all..., type, label, index, 1., (;))
end

"Repeating element (alternativ)"
function test_grammar1x()
    cm, fn, cn, tr, id = split.((
    "S A B C D",
    "f",
    "c",
    "x y z",
    "I"), " ")
    n_all = cumsum(length.((cm, fn, cn, tr, id)))
    label = [cm; fn; cn; tr; id]
    index = Dict(reverse.(enumerate(label)))
    rules = """
    S Lr -> C A 1
    A Lr -> B B 1
    B Plr -> I I 1
    C Lr -> c B 1
    D U -> x 1 | y 1
    c Cn -> D 1
    """
    w_all, type = compile_rules(n_all, index, rules)
    return GrammarEx(n_all..., w_all..., type, label, index, 1., (;))
end

"Basic logic"
function test_grammar2()
    cm, fn, cn, tr, id = split.((
    "S E F E1 And Or X P",
    "and or",
    "c0 c1 cx cop",
    "0 1 + * =",
    "I"), " ")
    n_all = cumsum(length.((cm, fn, cn, tr, id)))
    label = [cm; fn; cn; tr; id]
    index = Dict(reverse.(enumerate(label)))
    rules = """
    S Lr -> E F 1
    F Lr -> = P 1
    E Plr -> E1 X 1
    E1 Prl -> X cop 1
    X Lr -> cx P 1
    P Plr -> I I 1
    c0 Cn -> 0 1
    c1 Cn -> 1 1
    cx Cn -> 0 1 | 1 1
    cop Cn -> And 1 | Or 1
    And Lr -> and * 1
    Or Lr -> or + 1
    and Fn -> 0 c0 1 | 1 I 1
    or Fn -> 0 I 1 | 1 c1 1
    """
    w_all, type = compile_rules(n_all, index, rules)
    return GrammarEx(n_all..., w_all..., type, label, index, 1., (;))
end

"Basic recursion"
function test_grammar3()
    cm, fn, cn, tr, id = split.((
    "S E F G H P Rec",
    "isone prev fRec",
    "cx",
    "0 1 2 3 4 5 x =",
    "I true false"), " ")
    n_all = cumsum(length.((cm, fn, cn, tr, id)))
    label = [cm; fn; cn; tr; id]
    index = Dict(reverse.(enumerate(label)))
    rules = """
    S Lr -> E F 1
    E Lr -> cx P 1
    F Lr -> = Rec 1
    Rec Plr -> G H 1
    G Lr -> isone fRec 1
    H Lr -> x prev 1
    P Plr -> I I 1
    isone Fn -> 1 true 1 | 2 false 1 | 3 false 1 | 4 false 1
    prev Fn -> 1 0 1 | 2 1 1 | 3 2 1 | 4 3 1
    fRec Fn -> true I 1 | false Rec 1
    cx Cn -> 1 1 | 2 1 | 3 1 | 4 1
    """
    w_all, type = compile_rules(n_all, index, rules)
    return GrammarEx(n_all..., w_all..., type, label, index, 1., (;))
end

"""
Forward simulation of a symbol.
- Update IOBuffer inplace, and return the output value and tree probability
"""
function simulate!(io::IOBuffer, sym::Int, input::Int, g::GrammarEx)
    type = g.type[sym] # symbol type
    prob = 1. # initialize probability
    if type in Cm
        # first simulate left child
        left, p_rule_left = sample(g.h.p_left[:, sym])
        output_left, p_left = simulate!(io, left, input, g)
        prob *= p_rule_left * p_left
        # next simulate right child if necessary
        if type == U
            output = output_left
        else
            right, p_rule_right = sample(g.h.p_right[:, left, sym])
            # decide input of the right child
            if type in (Ll, Lr, Llr, Lrl)
                input_right = output_left
            else # type in (Pl, Pr, Plr, Prl)
                input_right = input
            end
            output_right, p_right = simulate!(io, right, input_right, g)
            prob *= p_rule_right * p_right
            # decide final output. simulate the dynamic sub-tree if necessary
            if type in (Ll, Pl)
                output = output_left
            elseif type in (Lr, Pr)
                output = output_right
            elseif type in (Llr, Plr)
                output, p_dynam = simulate!(io, output_left, output_right, g)
                prob *= p_dynam
            else # type in (Lrl, Prl) 
                output, p_dynam = simulate!(io, output_right, output_left, g)
                prob *= p_dynam
            end
        end
    elseif type == Fn
        output, p_fn = sample(g.h.p_fn[:, input, offset(sym, g)])
        prob *= p_fn
    elseif type == Cn
        output, p_cn = sample(g.h.p_cn[:, offset(sym, g)])
        prob *= p_cn
    elseif type == Tr
        write(io, g.label[sym])
        output = input
    else # type == Id
        output = input
    end
    return output, prob
end

"""
Sample a sentence from root node 1 with input value 1
- Return the sentence and its probability
"""
function generate_sentence(g::GrammarEx)
    io = IOBuffer()
    _, prob = simulate!(io, 1, 1, g)
    return String(take!(io)), prob
end

function generate_dataset(g::GrammarEx, n::Int=10)
    # compile heuristics
    p_fn = normalize(g.w_fn, dims=1)
    p_cn = normalize(g.w_cn, dims=1)
    p_left = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1, dims=1)
    p_right = normalize(g.w_cm2, dims=1)
    g.h = (; p_fn, p_cn, p_left, p_right)
    return [generate_sentence(g) for i in 1:n]
end

# testing:
# g = test_grammar2()
# g = test_grammar3()
# g = test_grammar1()
# generate_dataset(g)