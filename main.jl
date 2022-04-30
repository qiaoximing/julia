#===
This will be the main program
Our current goal is to implement PGAS learning with TD parser
=#
using DataStructures
using Plots

"Set to Float32 or Float64"
const Float = Float32

"Observations must be Int"
const Observation = Int

"Data must be string"
const Data = String

"Dataset as a vector of Data
Change it later to support curriculum."
const Dataset = Vector{Data}

"A random Int for observation sampling"
const ObX = 2814746717

"Types of nodes in DFGrammar"
@enum NodeType Sl Sr Pl Pr Slr Srl Plr Prl Fn Cn Id Ob

"Cm refers to all composite types"
const Cm = (Sl, Sr, Pl, Pr, Slr, Srl, Plr, Prl)

"Dynamic Functional Grammar.
Weight array organization:
- Rule weights, Fn weights, Cn weights share the array"
mutable struct DFGrammar
    nodenum::NTuple # number of nodes in type Cm, Fn, Cn, Id, Ob
    nodecumsum::NTuple # cumsum of nodenum (for indexing)
    nodetype::Vector{NodeType} # type of nodes
    label::Vector{String} # label of all nodes (for printing)
    labelindex::Dict{String, Int} # index of all labels
    alpha::Float # hyperparameter for weight smoothing
    weight3d::Array{Float, 3} # 3D weight array
    weight2d::Matrix{Float} # weight3d sum over dim 1
    weight1d::Vector{Float} # weight2d sum over dim 1
end

"number of nodes in a node type.
Any type in Cm will return all Cm"
function Base.size(gm::DFGrammar, type::NodeType)
    idx = max(Int(type) - length(Cm) + 1, 0)
    return gm.nodenum[idx + 1]
end

"number of nodes in all node types"
Base.size(gm::DFGrammar) = gm.nodecumsum[end]

"range of index for a node type.
Any type in Cm will return all Cm"
function Base.range(gm::DFGrammar, type::NodeType)
    idx = max(Int(type) - length(Cm) + 1, 0)
    first = idx > 0 ? gm.nodecumsum[idx] + 1 : 1
    last = gm.nodecumsum[idx + 1]
    return first:last
end

" Forward simulation of a symbol.
- Update IOBuffer inplace, and return the output value and tree probability "
function simulate!(io::IOBuffer, gm::DFGrammar, sym::Int, input::Int)
    type = getnodetype(gm, sym) # symbol type
    if type == Fn
        output = sample(distr_fn(gm, sym, input))
    elseif type == Cn
        output = sample(distr_cn(gm, sym))
    elseif type == Id
        output = input
    elseif type == Ob
        write(io, gm.label[sym])
        output = input
    else 
        # first simulate left child
        left = sample(distr_left(gm, sym))
        left_out = simulate!(io, gm, left, input)
        # next simulate right child
        right = sample(distr_right(gm, sym, left))
        right_in = type in (Sl, Sr, Slr, Srl) ? left_out : input
        right_out = simulate!(io, gm, right, right_in)
        # optionally simulate the dynamic child
        output = type in (Sl, Pl) ? left_out :
                 type in (Sr, Pr) ? right_out : 
                 type in (Slr, Plr) ? simulate!(io, gm, left_out, right_out) :
                                      simulate!(io, gm, right_out, left_out)
    end
    return output
end

" Sample a sentence from root node 1 with input value 1
- Return the sentence and its probability "
function generate_sentence(gm::DFGrammar)
    io = IOBuffer()
    simulate!(io, gm, 1, 1)
    return String(take!(io))
end

"Translate literal rule descriptions to weights.
Conventions:
- LHS of the first rule will be root
- LHS should only appear once
- Cm nodes before Fn nodes before Cn nodes before Id nodes
- Names not appeared on the left are Ob, and must have length 1"
function init_grammar(rules::AbstractString, alpha=0)
    # parse rules as a list of tuples, set up label, nodetype and labelindex
    label = String[]
    label_extra = String[]
    nodetype = NodeType[]
    parsedrules = Tuple[]
    for rule in split(rules, "\n")
        if rule == "" continue end
        splits = split(rule, " -> ")
        lhs = splits[1]
        lhs_label, lhs_type_name = split(lhs, " ")
        push!(label, lhs_label)
        lhs_type = eval(Meta.parse(lhs_type_name))
        push!(nodetype, lhs_type)
        if length(splits) > 1
            rhs_list = splits[2]
            for rhs in split(rhs_list, " | ")
                rhs_labels_and_weight = split(rhs, " ")
                rhs_labels = rhs_labels_and_weight[1:end-1]
                push!(label_extra, rhs_labels...)
                weight = Base.parse(Float, rhs_labels_and_weight[end])
                push!(parsedrules, (lhs_label, rhs_labels, weight))
            end
        end
    end
    for rhs_label in label_extra
        if !(rhs_label in label)
            push!(label, rhs_label)
            push!(nodetype, Ob)
        end
    end
    labelindex = Dict(reverse.(enumerate(label)))
    # count node cumsums and numbers
    csum = [[findfirst(x->x==type, nodetype) for type in (Fn, Cn, Id, Ob)]; length(nodetype) + 1]
    csum[2] = isnothing(csum[2]) ? csum[3] : csum[2]
    csum[1] = isnothing(csum[1]) ? csum[2] : csum[1]
    csum .-= 1
    nodenum = [csum[1]; csum[2:end] .- csum[1:end-1]]
    # set up weight array
    weight3d = zeros(Float, csum[end], csum[end], csum[2])
    weight2d = zeros(Float, csum[end], csum[3])
    weight1d = zeros(Float, csum[3])
    for (lhs_label, rhs_labels, weight) in parsedrules
        lhs_idx = labelindex[lhs_label]
        lhs_type = nodetype[lhs_idx]
        if lhs_type == Cn
            rhs_idx = labelindex[rhs_labels[1]]
            weight2d[rhs_idx, lhs_idx] += weight
            weight1d[lhs_idx] += weight
        else
            rhs_idx1 = labelindex[rhs_labels[1]]
            rhs_idx2 = labelindex[rhs_labels[2]]
            weight3d[rhs_idx2, rhs_idx1, lhs_idx] += weight
            weight2d[rhs_idx1, lhs_idx] += weight
            weight1d[lhs_idx] += weight
        end
    end
    return DFGrammar(tuple(nodenum...), tuple(csum...), nodetype, label, labelindex, 
                     alpha, weight3d, weight2d, weight1d)
end

"Initialize empty grammar given number of nodes in each type,
alphabet of observations, and grammar hyperparameter.
Nodenum format: Sl Sr Pl Pr Slr Srl Plr Prl Fn Cn Id"
function init_grammar(nums::Tuple, alphabet::String, alpha)
    if length(nums) != 11 return nothing end
    nodenum = (sum(nums[1:length(Cm)]), nums[length(Cm)+1:length(Cm)+3]..., length(alphabet))
    csum = cumsum(nodenum)
    nodetype = zeros(Int, csum[end])
    for n in cumsum(nums) nodetype[n+1:end] .+= 1 end
    nodetype = NodeType.(nodetype)
    label_initials = ["S", "F", "C", "I"]
    label = [label_initials[i] * string(j) for i in 1:4 for j in 1:nodenum[i]]
    append!(label, [string(char) for char in alphabet])
    labelindex = Dict(reverse.(enumerate(label)))
    weight3d = zeros(Float, csum[end], csum[end], csum[2])
    weight2d = zeros(Float, csum[end], csum[3])
    weight1d = zeros(Float, csum[3])
    return DFGrammar(nodenum, csum, nodetype, label, labelindex, alpha,
                     weight3d, weight2d, weight1d)
end

"Get node type from grammar. 
Always return Ob given the special number ObX"
function getnodetype(gm::DFGrammar, sym::Int)
    return sym == ObX ? Ob : gm.nodetype[sym]
end

"Fn output given symbol and input."
function distr_fn(gm::DFGrammar, sym, input)
    distr = Distribution(view(gm.weight3d, :, input, sym), gm.weight2d[input, sym], gm.alpha)
    return distr
end

"Cn output given symbol."
function distr_cn(gm::DFGrammar, sym)
    distr = Distribution(view(gm.weight2d, :, sym), gm.weight1d[sym], gm.alpha)
    return distr
end

"left child given symbol."
function distr_left(gm::DFGrammar, sym)
    bias = gm.alpha #* size(gm)
    distr = Distribution(view(gm.weight2d, :, sym), gm.weight1d[sym], bias)
    return distr
end

"right child given symbol and left."
function distr_right(gm::DFGrammar, sym, left)
    distr = Distribution(view(gm.weight3d, :, left, sym), gm.weight2d[left, sym], gm.alpha)
    return distr
end

"Unnormalized probabilistic distribution with optional smoothing"
struct Distribution{T<:Real}
    weight::AbstractArray{T} # a view or an array
    sum::T # sum of weights excluding bias
    bias::T # add bias to weights for smoothing
    totalsum::T # sum of weights and bias
end

"Initialize a distribution from weights and calculate sum if not provided.
Note: cannot use Distribution(weight, sum). Use Distribution(weight, sum, 0) instead"
function Distribution(weight::AbstractArray{T}, sum::T, bias::T) where {T}
    return Distribution(weight, sum, bias, sum + length(weight) * bias)
end
Distribution(weight::AbstractArray{T}, bias::T) where {T} = Distribution(weight, sum(weight), bias)
Distribution(weight::AbstractArray{T}) where {T} = Distribution(weight, sum(weight), T(0))

"Return a zero distribution"
zerodistr(T, size, bias=0) = Distribution(zeros(T, size), T(0), T(bias))

"Return a one-hot distribution with 1 at idx"
function onehotdistr(T, size, idx, bias=0)
    weight = zeros(T, size)
    weight[idx] = 1
    return Distribution(weight, T(1), T(bias))
end

"Get the normalized probability at location idx.
Might get NaN"
function getprobability(distr::Distribution, idx)
    return (distr.weight[idx] + distr.bias) / distr.totalsum
end

"Sample from a distribution and return index.
Return zero for zero distribution."
function sample(distr::Distribution)
    if distr.totalsum == 0
        # warning("sample from a zero distribution")
        return 0
    end
    r = rand() * distr.totalsum
    for idx in eachindex(distr.weight)
        r -= (distr.weight[idx] + distr.bias)
        if r < 0
            return idx
        end
    end
    # this should occur with probability zero
    return 1
end

#=======
Parser
=======#

"Index to values in Parser.value"
const ValId = Int

"A node of parse tree. 
Stores value indices, not values themselves"
struct Node
    sym::ValId
    input::ValId 
    output::ValId
end

"Stack of TD parsing"
const ParseStack = LinkedList{Node}

"Create a new stack"
newstack(pstate::Node) = list(pstate)
newstack() = nil(Node)

"Things can be used in a particle filter"
abstract type ParticleItem end

"TD Parser.
We consider the grammar as parameters of the parser, and put it inside."
struct Parser <: ParticleItem
    stack::ParseStack
    value::Vector{Int}
    distr::Vector{Union{Distribution, Nothing}}
    grammar::DFGrammar
    max_iter::Int # restrict the number of parsing actions performed in one simulation
    prev::Union{Parser, Nothing} # track the parse history. used to recover parse tree, not related to Particle.ancestor
end

"Set up an empty parser for configuration"
Parser(gm::DFGrammar, max_iter::Int) = Parser(newstack(), [], [], gm, max_iter, nothing)

"Get value from a parser through value id"
getvalue(prs::Parser, id::ValId) = prs.value[id]

"Progress the parser to next observation.
If observation == 0, check if the parser succeeds.
Return a new parser and the weight update.
Note that the new parser will create a copy of the mutable Parser.value"
function simulate(prs::Parser, obs::Observation)
    weight_update = Float(1)
    # create a new copy
    prs = Parser(prs.stack, copy(prs.value), copy(prs.distr), prs.grammar, prs.max_iter, prs.prev)
    for iter in 1:prs.max_iter
        # check empty stack
        if isempty(prs.stack)
            if obs > 0 # unfinished observation, parse fail
                return prs, Float(0) 
            else # all finished, parse success
                return prs, Float(1)
            end
        end
        stacktop = head(prs.stack)
        # one step of parsing
        prs, finish, step_weight_update = parse_step!(prs, stacktop, obs)
        weight_update *= step_weight_update
        if finish # return if finish
            return prs, weight_update
        end
    end
    # exceeds max number of iterations. Parse fail
    return prs, Float(0)
end

"Part of a parse step that handles the observation.
Return finish and weight update"
function parse_obs!(prs::Parser, root::Node, obs::Observation)
    sym, input = getvalue(prs, root.sym), getvalue(prs, root.input)
    input == ObX && sampleobs!(prs, root.input)
    propagate!(prs, root.input, root.output)
    if obs == 0 # try to read a non-existing observation, parse fail
        weight_update = 0
    elseif sym == ObX # observation not sampled
        probability = setvalue!(prs, root.sym, obs)
        weight_update = probability / obsprobability(prs, root.sym)
    else # observation already sampled
        probability = obs == sym ? 1 : 0
        weight_update = probability
    end
    finish = true
    return finish, weight_update
end

"One step of TD expansion of the current root.
The finish condition depends on whether obs == 0.
Retrive node type from prs.grammar, then update stack and value.
Note the value is mutable and will be changed.
Return a new parser with prev to track the history."
function parse_step!(prs::Parser, root::Node, obs::Observation)
    sym, input = getvalue(prs, root.sym), getvalue(prs, root.input)
    # sample the delayed right child
    if sym == 0
        sym = samplevalue!(prs, root.sym)
    end
    type = getnodetype(prs.grammar, sym)
    finish = false # true only if parse_obs!
    left_child_processed = false
    weight_update = Float(1)
    new_stack = tail(prs.stack) # always pop the top node
    if type == Fn
        if input == ObX 
            input = sampleobs!(prs, root.input)
        end
        distr = distr_fn(prs.grammar, sym, input)
        setdistr!(prs, root.output, distr)
        samplevalue!(prs, root.output)
    elseif type == Cn
        distr = distr_cn(prs.grammar, sym)
        setdistr!(prs, root.output, distr)
        samplevalue!(prs, root.output)
    elseif type == Id
        input == ObX && sampleobs!(prs, root.input)
        propagate!(prs, root.input, root.output)
    elseif type == Ob # this will include the case sym == ObX
        finish, weight_update = parse_obs!(prs, root, obs)
    else # composite nodes
        # create children node
        left_child, right_child, dynam_child = parse_create_children!(prs, root, type)
        # sample left child immediately
        setdistr!(prs, left_child.sym, distr_left(prs.grammar, sym))
        left_val = samplevalue!(prs, left_child.sym)
        # if left is observation, process it immediately
        if left_val == ObX
            finish, weight_update = parse_obs!(prs, left_child, obs)
            left_val = getvalue(prs, left_child.sym)
            left_child_processed = true
        end
        # setup right child distribution, but don't sample it yet
        if weight_update > 0
            setdistr!(prs, right_child.sym, distr_right(prs.grammar, sym, left_val))
        end
        # update stack
        if !isnothing(dynam_child) new_stack = cons(dynam_child, new_stack) end
        new_stack = cons(right_child, new_stack)
        new_stack = cons(left_child, new_stack)
    end
    # create a new parser with the new stack and the old mutated value
    new_prs = Parser(new_stack, prs.value, prs.distr, prs.grammar, prs.max_iter, prs)
    # pop stack top is left child is already process
    if left_child_processed
        prs = new_prs # keep this in parse history for tree printing
        new_prs = Parser(tail(prs.stack), prs.value, prs.distr, prs.grammar, prs.max_iter, prs)
    end
    # println("Observation: $obs")
    # println("Processing node: $(prs.grammar.label[sym])")
    # println("Weight update: $weight_update")
    # printstatus(prs)
    return new_prs, finish, weight_update
end

"part of parse step that creates children node"
function parse_create_children!(prs::Parser, root, type)
    left_sym = adddistr!(prs)
    right_sym = adddistr!(prs)
    if type in (Sl, Pl)
        left_out = root.output
        right_out = adddistr!(prs)
        dynam_child = nothing
    elseif type in (Sr, Pr)
        left_out = adddistr!(prs)
        right_out = root.output
        dynam_child = nothing
    elseif type in (Slr, Plr)
        left_out = adddistr!(prs)
        right_out = adddistr!(prs)
        dynam_child = Node(left_out, right_out, root.output)
    else # Srl or Prl
        left_out = adddistr!(prs)
        right_out = adddistr!(prs)
        dynam_child = Node(right_out, left_out, root.output)
    end
    left_in = root.input
    right_in = type in (Sl, Sr, Slr, Srl) ? left_out : root.input
    left_child = Node(left_sym, left_in, left_out)
    right_child = Node(right_sym, right_in, right_out)
    return left_child, right_child, dynam_child
end

"get a truncated distribution of observations"
function distr_obs(prs::Parser, idx)
    distr = prs.distr[idx]
    range_Ob = range(prs.grammar, Ob)
    obs_weight = view(distr.weight, range_Ob)
    return Distribution(obs_weight, distr.bias)
end

"get the total probability of generating an observation"
function obsprobability(prs::Parser, idx)
    return distr_obs(prs, idx).totalsum / prs.distr[idx].totalsum
end

"sample an ObX to a value.
Use a truncated distribution, then add back the offset"
function sampleobs!(prs::Parser, idx)
    obs_distr = distr_obs(prs, idx)
    prs.value[idx] = sample(obs_distr) + first(range(prs.grammar, Ob)) - 1
    return prs.value[idx]
end

"record the distribution for future sampling"
function setdistr!(prs::Parser, idx, distr)
    prs.distr[idx] = distr
end

"expand the size of prs.distr and prs.value, 
record the distribution, an return the new index"
function adddistr!(prs::Parser, distr)
    push!(prs.distr, distr)
    push!(prs.value, 0)
    return length(prs.value)
end
function adddistr!(prs::Parser)
    push!(prs.distr, nothing)
    push!(prs.value, 0)
    return length(prs.value)
end

"sample a value from its distribution.
Optionally replace observation with ObX"
function samplevalue!(prs::Parser, idx, useObX=true)
    val = sample(prs.distr[idx])
    if useObX && getnodetype(prs.grammar, val) == Ob
        val = ObX
    end
    prs.value[idx] = val
    return val
end

"propagate distribtuion and value"
function propagate!(prs::Parser, source, target)
    prs.value[target] = prs.value[source]
    prs.distr[target] = prs.distr[source]
end

"force setting a value, and return its probability"
function setvalue!(prs::Parser, idx, val)
    prs.value[idx] = val
    return getprobability(prs.distr[idx], val)
end

"Create a new Parser with a root node 1, input val 1, and output val 0(unknown)"
function initialize(prs::Parser)
    distr = onehotdistr(Float, size(prs.grammar), 1)
    return Parser(
        newstack(Node(1, 2, 3)),
        [1, 1, 0],
        [distr, distr, nothing],
        prs.grammar,
        prs.max_iter,
        nothing
    )
end

function getobservation(gm::DFGrammar, str::AbstractString)
    return [gm.labelindex[string(char)] for char in str]
end
function getobservation(gm::DFGrammar, char::Char)
    return gm.labelindex[string(char)]
end
function getobservation(prs::Parser, data)
    return getobservation(prs.grammar, data)
end

#=======
Parse Tree
=======#

"parse tree"
struct Tree
    node::Node # a tree node
    left::Union{Tree, Nothing} # left child
    right::Union{Tree, Nothing} # right child
    dynam::Union{Tree, Nothing} # dynamic child
end

"Recursively overwrite ValId with actual values"
fill_value(t::Tree, v) = Tree(
    Node(v[t.node.sym], v[t.node.input], v[t.node.output]),
    fill_value(t.left, v), fill_value(t.right, v), fill_value(t.dynam, v))
fill_value(t::Nothing, v) = nothing

"Get tree from parser history"
function gettree(prs::Parser)
    prs0 = prs
    stack = []
    while prs !== nothing
        stack_new = []
        pstack = collect(prs.stack)
        while length(pstack) > 0 && length(stack) > 0 && pstack[end] === stack[end].node
            pop!(pstack)
            s = pop!(stack)
            push!(stack_new, s)
        end
        for node in reverse(pstack)
            stack = vcat(stack, repeat([nothing], 3 - length(stack)))
            push!(stack_new, Tree(node, stack...))
        end
        stack = reverse(stack_new)
        prs = prs.prev
    end
    # fill actual values instead of indices to p.values
    t = fill_value(stack[1], prs0.value)
    return t
end

"Print the tree"
function printtree(t::Tree, g::DFGrammar, prefix::String="", lastchild::Bool=true)
    sym, input, output = t.node.sym, t.node.input, t.node.output
    print(prefix)
    print(lastchild ? " └─" : " ├─")
    prefix_new = prefix * (lastchild ? "   " : " │ ")
    label = x -> x == 0 ? '?' : x == ObX ? '\$' : g.label[x]
    type = x -> x == 0 ? '?' : getnodetype(g, x)
    printstyled(' ', label(sym), bold=true, color=(
                type(sym) == Ob ? :red : :black))
    print(' ', label(input), ' ', label(output), ' ')
    printstyled('(', type(sym), ')', color=:green)
    println()
    if t.dynam isa Tree
        printtree(t.left, g, prefix_new, false)
        printtree(t.right, g, prefix_new, false)
        printtree(t.dynam, g, prefix_new, true)
    elseif t.right isa Tree
        printtree(t.left, g, prefix_new, false)
        printtree(t.right, g, prefix_new, true)
    elseif t.left isa Tree
        printtree(t.left, g, prefix_new, true)
    end
end

#=======
Particle filter
=======#

"Particle"
struct Particle{T<:ParticleItem}
    item::T
    weight::Float
    ancestor::Union{Particle{T}, Nothing}
end

"Initialize a Particle with 1.0 weight and no ancestor"
Particle(item) = Particle(item, Float(1), nothing)

"Get a particle's weight"
getweight(ptl::Particle) = ptl.weight

"Simulate a particle and return a new particle.
The old particle becomes the ancestor."
function simulate(ptl::Particle, obs::Observation)
    new_item, weight_update = simulate(ptl.item, obs)
    # new_weight = ptl.weight * weight_update
    new_weight = weight_update # the old weight is gone due to resampling
    return Particle(new_item, new_weight, ptl)
end

"Print the status of a parser"
function printstatus(prs::Parser)
    printtree(gettree(prs), prs.grammar)
end

"Print the status of a particle"
function printstatus(ptl::Particle)
    println("Weight: ", ptl.weight)
    printstatus(ptl.item)
end

"Particle system"
mutable struct ParticleSystem{T}
    particles::Vector{Particle{T}}
    loglikelihood::Float
end

"Initialize a new particle system from item"
ParticleSystem(n::Int, item) = 
    ParticleSystem([Particle(initialize(item)) for _=1:n], Float(0))

"Inherite loglikelihood from an old particle system, clean up the particles"
ParticleSystem(ps::ParticleSystem{T}) where T = 
    ParticleSystem(Particle{T}[], ps.loglikelihood)

"Get particle weights and return as a vector"
getweights(ps::ParticleSystem) = getweight.(ps.particles)

"Get a particle by index.
Return nothing if index=0"
getparticle(ps::ParticleSystem, idx) = idx > 0 ? ps.particles[idx] : nothing

"push a particle to a particle system"
Base.push!(ps::ParticleSystem, ptl) = push!(ps.particles, ptl)

"Effective sample size of a particle system"
function effective_sample_size(ps::ParticleSystem)
    weights = getweights(ps)
    return sum(weights) ^ 2 / sum(weights .^ 2)
end

"Print the status of a particle system"
function printstatus(ps::ParticleSystem, temperature)
    println("ESS: ", effective_sample_size(ps))
    printstatus(sample(ps, temperature))
end
function printstatus(_::Nothing)
    println("Nothing to print")
end

"Particle Filter"
struct ParticleFilter
    num_particles::Int
    log::Dict
end
ParticleFilter(n) = ParticleFilter(n, Dict())

"Sample one particle from the particle system.
Use temperature to control the sampling.
Return nothing if all particles have zero weight."
function sample(ps::ParticleSystem, temperature=1)
    weights = getweights(ps) .^ (1 / Float(temperature))
    distr = Distribution(weights)
    idx = sample(distr)
    return getparticle(ps, idx)
end

"Sample N particle from the particle system with replacement.
Return nothing if all particles have zero weight."
function sample_n(ps::ParticleSystem, n::Int)
    ret = [sample(ps) for _ in 1:n]
    return nothing in ret ? nothing : ret
end

"Run the particle filter with a seed item.
Return a particle system is succeed. Otherwise return nothing"
function simulate(pf::ParticleFilter, item::ParticleItem, data::Data)
    # use item as seed to generate a particle system
    ps = ParticleSystem(pf.num_particles, item)
    pf.log["ESS"] = Float[]
    # go one more step after the last observation
    for step in 1:length(data) + 1
        new_ps = ParticleSystem(ps)
        # draw new particles independently
        for _ in 1:pf.num_particles
            particle = sample(ps)
            if isnothing(particle)
                return nothing
            end
            # use dummy obs = 0 for the last step
            obs = step <= length(data) ? getobservation(item, data[step]) : 0
            new_particle = simulate(particle, obs)
            push!(new_ps, new_particle)
        end
        update_loglikelihood(new_ps)
        ps = new_ps
        push!(pf.log["ESS"], effective_sample_size(ps))
        # printstatus(ps)
    end
    return ps
end

"Conditional Particle Filter"
struct ConditionalParticleFilter
    num_particles::Int # total number of particles
    num_conditions::Int # number of conditional particles
    log::Dict
end
ConditionalParticleFilter(np, nc) = ConditionalParticleFilter(np, nc, Dict())

function get_trajectory(ptl::Particle)
    trajectory = Particle[]
    while !isnothing(ptl)
        push!(trajectory, ptl)
        ptl = ptl.ancestor
    end
    return reverse(trajectory)
end

"Run conditional particle filter with an input particle for conditioning.
A special case of conditional particle filter with ancestor sampling.
Return nothing is the particle have a mismatch trajectory length.
Otherwise return a particle system with its last particle being the input one." 
function simulate(pf::ConditionalParticleFilter, ptls::Vector{Particle{T}}, data::Data) where T
    pf_with_as = ConditionalParticleFilterAS(
        pf.num_particles, pf.num_conditions, 0, pf.log)
    simulate(pf_with_as, ptls, data)
end

"Conditional Particle Filter with ancestor sampling"
struct ConditionalParticleFilterAS
    num_particles::Int # total number of particles
    num_conditions::Int # number of conditional particles
    eta::Float # probability of performing ancetor sampling
    log::Dict
end
ConditionalParticleFilterAS(np, nc, eta) = ConditionalParticleFilterAS(np, nc, eta, Dict())

"Run conditional particle filter with ancestor sampling (AS), 
using an input particle for conditioning.
In each step, resample the ancestor of the input particle with probability Eta
This is done by creating a new chain of particles, without mutating the input one.
Return nothing is the input particle have a mismatch trajectory length.
Otherwise return a particle system with its last particle being the input one." 
function simulate(pf::ConditionalParticleFilterAS, ptls::Vector{Particle{T}}, data::Data) where T
    # get the whole trajectorys of ptls and check length
    length(ptls) == pf.num_conditions || return nothing
    ptl_trajectorys = get_trajectory.(ptls)
    if any([length(traj) != length(data) + 2 for traj in ptl_trajectorys])
        return nothing
    end
    # initialize a particle system and append the conditioning particle
    item = ptls[1].item
    ps = ParticleSystem(pf.num_particles - pf.num_conditions, item)
    push!(ps, first.(ptl_trajectorys)...)
    pf.log["ESS"] = Float[]
    # go one more step after the last observation
    for step in 1:length(data) + 1
        new_ps = ParticleSystem(ps)
        # draw new particles independently, but excluding the last few
        for _ in 1:pf.num_particles - pf.num_conditions
            ptl = sample(ps) # this will never fail
            # use dummy obs = 0 for the last step
            obs = step <= length(data) ? getobservation(item, data[step]) : 0
            new_ptl = simulate(ptl, obs)
            push!(new_ps, new_ptl)
        end
        # add the conditioning particle, but with resampled ancestor
        for traj in ptl_trajectorys
            new_ptl = traj[step + 1]
            if rand() < pf.eta
                new_ptl = ancestorsample(new_ptl, ps) # use the old ps
            end
            push!(new_ps, new_ptl)
        end
        update_loglikelihood(new_ps)
        ps = new_ps
        push!(pf.log["ESS"], effective_sample_size(ps))
        # printstatus(ps)
    end
    return ps
end

function ancestorsample(ptl::Particle, ps::ParticleSystem)
    return ptl
end


"Particle EM"
struct ParticleEM
    ptlfilter::ParticleFilter
    condptlfilter::ConditionalParticleFilter
    dataset::Dataset
    epochs::Int
    gamma::Float
    memory::Vector{Any}
    log::Dict
end
"Initialize particle EM with empty memory"
function ParticleEM(ptlfilter, condptlfilter, dataset, epochs, gamma)
    return ParticleEM(ptlfilter, condptlfilter, dataset, epochs, gamma,
        repeat(Any[nothing], length(dataset)), Dict())
end

"Particle EM with ancestor sampling"
struct ParticleEMAS
    ptlfilter::ParticleFilter
    condptlfilterAS::ConditionalParticleFilterAS
end

"Run particle EM on a dataset.
For each data, run a standard particle filter until it succeeds, memorize some samples.
The following runs will use conditional particle filter on these samples, and update them."
function simulate(pe::ParticleEM, item::ParticleItem)
    pe.log["NLL"] = Float[]
    for epoch in 1:pe.epochs
        for (data_id, data) in enumerate(pe.dataset)
            if isnothing(pe.memory[data_id])
                ps = simulate(pe.ptlfilter, item, data)
            else
                ps = simulate(pe.condptlfilter, pe.memory[data_id], data)
            end
            if !isnothing(ps)
                memory_update!(pe, data_id, ps)
                model_update!(pe, item, ps)
                nll(ps) < Inf && push!(pe.log["NLL"], nll(ps))
            end
        end
    end
end

"negative log likelihood of a particle system"
nll(ps::ParticleSystem) = -ps.loglikelihood

"Update the log likelihood of a particle system based on the new observations.
Become -Inf if all particle fails."
function update_loglikelihood(ps::ParticleSystem)
    ps.loglikelihood += log(sum(getweights(ps)) / length(getweights(ps)))
end

"Update the memory at data_id based on a particle system"
function memory_update!(pe::ParticleEM, data_id::Int, ps::ParticleSystem)
    pe.memory[data_id] = sample_n(ps, pe.condptlfilter.num_conditions)
end

"Update model parameters using a particle system"
function model_update!(pe::ParticleEM, item::Parser, ps::ParticleSystem)
    gm = item.grammar
    # grammar weight decay
    grammar_decay!(gm, pe.gamma)
    ps_weight_total = sum(getweights(ps))
    for ptl in ps.particles
        # get normalized particle weight
        ptl_weight = ptl.weight / ps_weight_total
        # grammar weight update
        if ptl_weight > 0
            tree = gettree(ptl.item)
            # printtree(tree, gm)
            grammar_update!(gm, tree, ptl_weight)
        end
    end
end

"Decay grammar weights by w"
function grammar_decay!(gm::DFGrammar, w::Float)
    gm.weight3d .*= 1 - w
    gm.weight2d .*= 1 - w
    gm.weight1d .*= 1 - w
end

"Update grammar from a parse tree.
Traverse the tree to collect patterns.
Update the pattern weights by w."
function grammar_update!(gm::DFGrammar, t::Tree, w::Float)
    sym, input, output = t.node.sym, t.node.input, t.node.output
    type = getnodetype(gm, sym)
    if type == Fn
        gm.weight3d[output, input, sym] += w
        gm.weight2d[input, sym] += w
        gm.weight1d[sym] += w
    elseif type == Cn
        gm.weight2d[output, sym] += w
        gm.weight1d[sym] += w
    elseif type == Id || type == Ob
        # do nothing
    else
        gm.weight3d[t.right.node.sym, t.left.node.sym, sym] += w
        gm.weight2d[t.left.node.sym, sym] += w
        gm.weight1d[sym] += w
    end
    # recursive update
    t.left isa Tree && grammar_update!(gm, t.left, w)
    t.right isa Tree && grammar_update!(gm, t.right, w)
    t.dynam isa Tree && grammar_update!(gm, t.dynam, w)
end

"smoothen a vector with exponential decay"
function smooth(xs, rate=0.99)
    length(xs) == 0 && return xs
    y = xs[1]
    ys = [y]
    for x in xs
        y = rate * y + (1 - rate) * x
        push!(ys, y)
    end
    return ys
end

"print the results"
function summarize(pe::ParticleEM, prs::Parser, smooth_rate)
    gm = prs.grammar
    for data in pe.dataset[1:5]
        ps = simulate(ParticleFilter(100), prs, data)
        printstatus(ps, 0.1)
    end
    p1 = plot(smooth(pe.log["NLL"], smooth_rate))
    p2 = histogram(filter(x->x>gm.alpha, reshape(gm.weight3d, :)), bins=10)
    display(plot(p1, p2, layout=(2, 1)))
end
