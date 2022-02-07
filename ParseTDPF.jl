#=
Top-down parsing with particle filter
WARNING: TDStates in different particles may interfere.
=#
include("Utility.jl")
include("GrammarEx.jl")

"State of TD stack.
 Store index to Particle.values, instead of actual values"
struct TDState
    sym::Int
    input::Int
    output::Int
end

"parse tree"
struct Tree
    state::TDState # a tree node
    prob::Float32 # tree probability
    left::Union{Tree, Nothing} # left child
    right::Union{Tree, Nothing} # right child
    dynam::Union{Tree, Nothing} # dynamic child
end

"Particle state"
struct Particle
    w::Float32 # particle weight
    stack::LinkedList{TDState} # top-down parse stack
    values::Vector{Int} # values in TDState
    ances::Union{Particle, Nothing} # ancestor of the particle
end
Base.copy(p::Particle) = Particle(p.w, p.stack, copy(p.values), p.ances)

"Progress a particle forward to the next observation"
function forward(ptl::Particle, obs::Int, g::GrammarEx)
    while !isa(ptl.stack, Nil)
        # pop the top symbol
        state = head(ptl.stack)
        stack_new = tail(ptl.stack)
        # retrieve Ref contents
        sym, input = ptl.values[state.sym], ptl.values[state.input]
        type = g.type[sym]
        if type in Cm
            # sample left child
            left, _ = sample(g.h.p_left[:, sym])
            if type == U
                # create a new child, pass input and output Ref's
                push!(ptl.values, left)
                state_child = TDState(length(ptl.values), state.input, state.output)
                stack_new = cons(state_child, stack_new)
            else
                # sample right child
                right, _ = sample(g.h.p_right[:, left, sym])
                # create 1 or 2 output placeholders for value sharing
                # push the dynamic sub-tree to stack is necessary
                if type in (Ll, Pl)
                    output_left = state.output
                    push!(ptl.values, 0) # 0 as a placeholder
                    output_right = length(ptl.values)
                elseif type in (Lr, Pr)
                    push!(ptl.values, 0)
                    output_left = length(ptl.values)
                    output_right = state.output
                elseif type in (Llr, Plr)
                    push!(ptl.values, 0)
                    output_left = length(ptl.values)
                    push!(ptl.values, 0)
                    output_right = length(ptl.values)
                    state_dynam = TDState(output_left, output_right, state.output)
                    stack_new = cons(state_dynam, stack_new)
                else # type in (Lrl, Prl)
                    push!(ptl.values, 0)
                    output_left = length(ptl.values)
                    push!(ptl.values, 0)
                    output_right = length(ptl.values)
                    state_dynam = TDState(output_right, output_left, state.output)
                    stack_new = cons(state_dynam, stack_new)
                end
                # create left and right sub-trees, and push to stack
                if type in (Ll, Lr, Llr, Lrl)
                    push!(ptl.values, left)
                    state_left = TDState(length(ptl.values), state.input, output_left)
                    push!(ptl.values, right)
                    state_right = TDState(length(ptl.values), output_left, output_right)
                else # (Pl, Pr, Plr, Prl)
                    push!(ptl.values, left)
                    state_left = TDState(length(ptl.values), state.input, output_left)
                    push!(ptl.values, right)
                    state_right = TDState(length(ptl.values), state.input, output_right)
                end
                stack_new = cons(state_left, cons(state_right, stack_new))
            end
        elseif type == Fn
            # write to shared output
            ptl.values[state.output], _ = sample(g.h.p_fn[:, input, offset(sym, g)])
        elseif type == Cn
            # write to shared output
            ptl.values[state.output], _ = sample(g.h.p_cn[:, offset(sym, g)])
        elseif type == Tr
            # write to shared output, update weight, pop stack
            ptl.values[state.output] = ptl.values[state.input]
            w_new = ptl.w * (sym == obs ? 1. : 0.)
            # exit the loop
            return Particle(w_new, stack_new, ptl.values, ptl)
        else # type == Id
            # write to shared output, pop stack
            ptl.values[state.output] = ptl.values[state.input]
        end
        # if not producing an observation, update particle and continue
        ptl = Particle(ptl.w, stack_new, ptl.values, ptl)
        label = x -> ptl.values[x] == 0 ? '?' : g.label[ptl.values[x]]
        for s in ptl.stack
            print('(', label(s.sym), ' ', label(s.input), ' ', label(s.output), ") ")
        end
        println()
    end
    println("Parse finished")
    return ptl
end

"Effective sample size (ESS) of a particle system"
function effective_size(particles::Vector{Particle})
    weights = [p.w for p in particles]
    ess = sum(weights) ^ 2 / sum(weights .^ 2)
    return isnan(ess) ? 0 : floor(Int, ess)
end

"Resample the particle system to the target number"
function resample(particles::Vector{Particle}, target_num::Int)
    weights = [p.w for p in particles]
    samples = first.(sample_n(normalize(weights), target_num, :systematic))
    println("Resampling: ancestor indices are $samples")
    particles_new = Vector{Particle}()
    for i in samples
        push!(particles_new, copy(particles[i]))
    end
    return particles_new
end

"Print a particle"
function print_particle(ptl::Particle, g::GrammarEx)
    println("Particle weight: $(ptl.w)")
    label = x -> ptl.values[x] == 0 ? '?' : g.label[ptl.values[x]]
    for s in ptl.stack
        print('(', label(s.sym), ' ', label(s.input), ' ', label(s.output), ") ")
    end
    println()
end

"Parse a string and return a particle system"
function parse_tdpf(str::String, g::GrammarEx)
    # compile function and rule probabilities
    p_fn = normalize(g.w_fn, dims=1)
    p_cn = normalize(g.w_cn, dims=1)
    p_left = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1, dims=1)
    p_right = normalize(g.w_cm2, dims=1)
    g.h = (; p_fn, p_cn, p_left, p_right)
    # initialize particles
    n_particle = 4
    state_init = TDState(1, 2, 3)
    particle_init = Particle(1., list(state_init), [1, 1, 0], nothing)
    particles = Vector{Particle}()
    for i in 1:n_particle
        push!(particles, copy(particle_init))
    end
    # start parsing
    for char in str
        obs = g.index[string(char)]
        # progress particles and assign weight according to obs
        for (i, p) in enumerate(particles)
            println("Simulating particle $i")
            particles[i] = forward(p, obs, g)
        end
        # resample is effective sample size too small
        ess = effective_size(particles)
        if ess == 0
            println("All particles fail")
            return Vector{Particle}()
        end
        if ess <= n_particle / 2
            # for binary weights, this step introduces zero variance
            particles = resample(particles, ess * Int(floor(n_particle / ess)))
        end
    end
    # finalize the simulation after the last observation
    for (i, p) in enumerate(particles)
        println("Simulating particle $i")
        particles[i] = forward(p, 0, g)
    end
    return particles
end

"Print a parse tree"
function print_tree(t::Tree, g::GrammarEx, v::Vector{Int}, prefix::String="", lastchild::Bool=true)
    sym = v[t.state.sym]
    input = v[t.state.input]
    output = v[t.state.output]
    print(prefix)
    print(lastchild ? " └─" : " ├─")
    prefix_new = prefix * (lastchild ? "   " : " │ ")
    label = x -> x == 0 ? '?' : g.label[x]
    printstyled(' ', label(sym), bold=true, color=(
        g.type[sym] == Tr ? :red : :black))
    print(' ', label(input), ' ', label(output), ' ')
    printstyled('(', g.type[sym], ')', color=:green)
    println()
    if t.dynam !== nothing
        print_tree(t.left, g, v, prefix_new, false)
        print_tree(t.right, g, v, prefix_new, false)
        print_tree(t.dynam, g, v, prefix_new, true)
    elseif t.right !== nothing
        print_tree(t.left, g, v, prefix_new, false)
        print_tree(t.right, g, v, prefix_new, true)
    elseif t.left !== nothing
        print_tree(t.left, g, v, prefix_new, true)
    end
end

"Sample a parse tree from a particle system"
function sample_tdpf(particles::Vector{Particle})
    if length(particles) == 0 
        println("No particles to sample")
        return
    end
    weights = [p.w for p in particles]
    idx, _ = sample(normalize(weights)) 
    p = particles[idx]
    v = p.values
    stack = []
    while p !== nothing
        stack_new = []
        pstack = collect(p.stack)
        # println("before:", pstack, stack)
        while length(pstack) > 0 && length(stack) > 0 && pstack[end] === stack[end].state
            pop!(pstack)
            s = pop!(stack)
            push!(stack_new, s)
        end
        if length(pstack) > 0
            state = pstack[end]
            stack = vcat(stack, repeat([nothing], 3 - length(stack)))
            push!(stack_new, Tree(state, 1., stack...))
        end
        stack = reverse(stack_new)
        # println("after:", pstack, stack)
        p = p.ances
    end
    print_tree(stack[1], g, v)
    return stack[1]
end

function grammar_update!(g::GrammarEx, tree::Tree)
end

# testing
g = test_grammar3()
# data = "1+0=1"
data = "3=xxx"
ps = parse_tdpf(data, g)
tr = sample_tdpf(ps)
grammar_update!(g, tr)