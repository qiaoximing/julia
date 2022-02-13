#=
Top-down parsing with stochastic search
Combines the idea of Huffman coding and beam search
=#
include("Utility.jl")
include("GrammarEx.jl")
include("ParseUtil.jl")

"Item state"
struct Item
    w::Float32 # weight for sorting
    loc::Int # location of next scan in input string
    stack::LinkedList{TDState} # top-down parse stack
    values::Vector{Int} # values in TDState
    iter::Int # number of step before an observation
    ances::Union{Item, Nothing} # ancestor of the particle
end

"Sort Items by weight"
Base.:isless(x::Item,y::Item) = isless(x.w, y.w)

"Expand an item by one step" 
function forward(item::Item, str::String, g::GrammarEx)
    items_new = Vector{Item}()
    item.stack isa Nil && return items_new
    state = head(item.stack)
    iter = item.iter
    sym, input = item.values[state.sym], item.values[state.input]
    type = g.type[sym]
    if type in Cm
        for (left, prob_left) in enumerate(g.h.p_left[:, sym])
            prob_left > 0 || continue
            if type == U
                w_new = item.w + log(prob_left)
                values_new = copy(item.values)
                push!(values_new, left)
                state_child = TDState(length(values_new), state.input, state.output)
                stack_new = cons(state_child, tail(item.stack))
                item_new = Item(w_new, item.loc, stack_new, values_new, iter + 1, item)
                push!(items_new, item_new)
            else
                for (right, prob_right) in enumerate(g.h.p_right[:, left, sym])
                    prob_right > 0 || continue
                    w_new = item.w + log(prob_left) + log(prob_right)
                    values_new = copy(item.values)
                    stack_new = tail(item.stack)
                    if type in (Ll, Pl)
                        output_left = state.output
                        push!(values_new, 0) # 0 as a placeholder
                        output_right = length(values_new)
                    elseif type in (Lr, Pr)
                        push!(values_new, 0)
                        output_left = length(values_new)
                        output_right = state.output
                    elseif type in (Llr, Plr)
                        push!(values_new, 0)
                        output_left = length(values_new)
                        push!(values_new, 0)
                        output_right = length(values_new)
                        state_dynam = TDState(output_left, output_right, state.output)
                        stack_new = cons(state_dynam, stack_new)
                    else # type in (Lrl, Prl)
                        push!(values_new, 0)
                        output_left = length(values_new)
                        push!(values_new, 0)
                        output_right = length(values_new)
                        state_dynam = TDState(output_right, output_left, state.output)
                        stack_new = cons(state_dynam, stack_new)
                    end
                    # create left and right sub-trees, and push to stack
                    if type in (Ll, Lr, Llr, Lrl)
                        push!(values_new, left)
                        state_left = TDState(length(values_new), state.input, output_left)
                        push!(values_new, right)
                        state_right = TDState(length(values_new), output_left, output_right)
                    else # (Pl, Pr, Plr, Prl)
                        push!(values_new, left)
                        state_left = TDState(length(values_new), state.input, output_left)
                        push!(values_new, right)
                        state_right = TDState(length(values_new), state.input, output_right)
                    end
                    stack_new = cons(state_left, cons(state_right, stack_new))
                    item_new = Item(w_new, item.loc, stack_new, values_new, iter + 1, item)
                    push!(items_new, item_new)
                end
            end
        end
    elseif type == Fn
        for (output, prob_output) in enumerate(g.h.p_fn[:, input, offset(sym, g)])
            prob_output > 0 || continue
            w_new = item.w + log(prob_output)
            values_new = copy(item.values)
            values_new[state.output] = output
            item_new = Item(w_new, item.loc, tail(item.stack), values_new, iter + 1, item)
            push!(items_new, item_new)
        end
    elseif type == Cn
        for (output, prob_output) in enumerate(g.h.p_cn[:, offset(sym, g)])
            prob_output > 0 || continue
            w_new = item.w + log(prob_output)
            values_new = copy(item.values)
            values_new[state.output] = output
            item_new = Item(w_new, item.loc, tail(item.stack), values_new, iter + 1, item)
            push!(items_new, item_new)
        end
    elseif type == Tr
        if item.loc > length(str)
            # println("input outbound")
        elseif sym != g.index[string(str[item.loc])]
            # println("input mismatch")
        else
            values_new = copy(item.values)
            values_new[state.output] = input
            item_new = Item(item.w, item.loc + 1, tail(item.stack), values_new, 1, item)
            push!(items_new, item_new)
        end
    else # type == Id
        values_new = copy(item.values)
        values_new[state.output] = input
        item_new = Item(item.w, item.loc, tail(item.stack), values_new, iter + 1, item)
        push!(items_new, item_new)
    end
    return items_new
end

"Merge new items to heap.
 Randomly remove lowest-score items and merge their weights (Huffman)"
function merge!(heap::BinaryMinMaxHeap, items::Vector, heap_size::Int)
    length(heap) > heap_size && warning("Heap size too big before merge")
    for item in items
        push!(heap, item)
        if length(heap) > heap_size
            item1 = popmin!(heap)
            item2 = popmin!(heap)
            w_max = max(item1.w, item2.w)
            prob1 = exp(item1.w - w_max) / (exp(item1.w - w_max) + exp(item2.w - w_max))
            itemx = rand() < prob1 ? item1 : item2
            w_new = log(exp(item1.w - w_max) + exp(item2.w - w_max)) + w_max
            item_new = Item(w_new, itemx.loc, itemx.stack, itemx.values, itemx.iter, itemx.ances)
            push!(heap, item_new)
        end
    end
end

"Check is a parse is finished"
finish(item::Item, str) = item.loc > length(str) && item.stack isa Nil

function get_tree(p::Item)
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
        for state in reverse(pstack)
            stack = vcat(stack, repeat([nothing], 3 - length(stack)))
            push!(stack_new, Tree(state, 1., stack...))
        end
        stack = reverse(stack_new)
        # println("after:", pstack, stack)
        p = p.ances
    end
    # fill actual values instead of indices to p.values
    t = fill_value(stack[1], v)
    # print_tree(t, g)
    return t
end

function print_item(item::Item, g::GrammarEx)
    tag(x) = string(hash(x), base=16)[1:4]
    println("Weight $(item.w) hash $(tag(item)) -> $(tag(item.ances))")
    label = x -> item.values[x] == 0 ? '?' : g.label[item.values[x]]
    for s in item.stack
        print('(', label(s.sym), ' ', label(s.input), ' ', label(s.output), ") ")
    end
    println()
    print_tree(get_tree(item), g)
end

function parse_tdss(str::String, g::GrammarEx, heap_size::Int=4, max_iter::Int=100, training::Bool=false)
    # compile function and rule probabilities
    if training
        p_fn = normalize(g.w_fn .+ g.alpha, dims=1)
        p_cn = normalize(g.w_cn .+ g.alpha, dims=1)
        p_left = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1 .+ g.alpha, dims=1)
        p_right = normalize(g.w_cm2 .+ g.alpha, dims=1)
    else
        p_fn = normalize(g.w_fn, dims=1)
        p_cn = normalize(g.w_cn, dims=1)
        p_left = normalize(sum(g.w_cm2, dims=1)[1, :, :] .+ g.w_cm1, dims=1)
        p_right = normalize(g.w_cm2, dims=1)
    end
    g.h = (; p_fn, p_cn, p_left, p_right)
    # initialize a minmax heap sorted by item.w
    state_init = TDState(1, 2, 3)
    item_init = Item(0., 1, list(state_init), [1, 1, 0], 1, nothing)
    heap = BinaryMinMaxHeap([item_init])
    result = Vector{Item}()
    # start parsing
    while length(heap) > 0
        # progress the particle with the highest score
        item = popmax!(heap)
        # print_item(item, g)
        if finish(item, str)
            push!(result, item)
            heap_size -= 1
        elseif item.iter <= max_iter
            items_new = forward(item, str, g)
            # merge to heap
            merge!(heap, items_new, heap_size)
        else
            # println("exceeding max iter")
        end
    end
    ws = [item.w for item in result]
    logprob = length(ws) > 0 ? log(sum(exp.(ws .- maximum(ws)))) + maximum(ws) : NaN
    return result, logprob
end

# testing
# g = test_grammar1()
# data = "xx"
# g = test_grammar2()
# data = "1+0=1"
# g = test_grammar3()
# data = "3=xxx"
# res, logprob = parse_tdss(data, g, 10, 30, false)
# println(logprob)
# print_tree(get_tree(res[1]), g)
