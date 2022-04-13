"State of TD stack.
 Store index to values, instead of actual values"
struct TDState
    sym::Int
    input::Int
    output::Int
end

struct BUState
    sym::Int
    in::Int
    out::Int
    left::Union{BUState, Nothing}
    right::Union{BUState, Nothing}
    dynam::Union{BUState, Nothing}
end
BUState(sym, in, out) = BUState(sym, in, out, nothing, nothing, nothing)

"parse tree"
struct Tree
    state::TDState # a tree node
    prob::Float32 # tree probability
    left::Union{Tree, Nothing} # left child
    right::Union{Tree, Nothing} # right child
    dynam::Union{Tree, Nothing} # dynamic child
end

"Helper function for print_tree"
fill_value(t::Nothing, v) = nothing
fill_value(t::Tree, v) = Tree(
    TDState(v[t.state.sym], v[t.state.input], v[t.state.output]), t.prob, 
    fill_value(t.left, v), fill_value(t.right, v), fill_value(t.dynam, v))
fill_value(s::BUState, ptl) = BUState(
    getval(ptl, s.sym), getval(ptl, s.in), getval(ptl, s.out),
    fill_value(s.left, ptl), fill_value(s.right, ptl), fill_value(s.dynam, ptl))

"Print a parse tree"
function print_tree(t::Tree, g::GrammarEx, prefix::String="", lastchild::Bool=true)
    sym, input, output = t.state.sym, t.state.input, t.state.output
    print(prefix)
    print(lastchild ? " └─" : " ├─")
    prefix_new = prefix * (lastchild ? "   " : " │ ")
    label = x -> x == 0 ? '?' : g.label[x]
    type = x -> x == 0 ? '?' : g.type[x]
    printstyled(' ', label(sym), bold=true, color=(
                type(sym) == Tr ? :red : :black))
    print(' ', label(input), ' ', label(output), ' ')
    printstyled('(', type(sym), ')', color=:green)
    println()
    if t.dynam isa Tree
        print_tree(t.left, g, prefix_new, false)
        print_tree(t.right, g, prefix_new, false)
        print_tree(t.dynam, g, prefix_new, true)
    elseif t.right isa Tree
        print_tree(t.left, g, prefix_new, false)
        print_tree(t.right, g, prefix_new, true)
    elseif t.left isa Tree
        print_tree(t.left, g, prefix_new, true)
    end
end

"Print a parse tree"
function print_tree(s::BUState, g::GrammarEx, prefix::String="", lastchild::Bool=true)
    sym, input, output = s.sym, s.in, s.out
    print(prefix)
    print(lastchild ? " └─" : " ├─")
    prefix_new = prefix * (lastchild ? "   " : " │ ")
    label = x -> x == 0 ? '?' : g.label[x]
    type = x -> x == 0 ? '?' : g.type[x]
    printstyled(' ', label(sym), bold=true, color=(
                type(sym) == Tr ? :red : :black))
    print(' ', label(input), ' ', label(output), ' ')
    printstyled('(', type(sym), ')', color=:green)
    println()
    if s.dynam isa BUState
        print_tree(s.left, g, prefix_new, false)
        print_tree(s.right, g, prefix_new, false)
        print_tree(s.dynam, g, prefix_new, true)
    elseif s.right isa BUState
        print_tree(s.left, g, prefix_new, false)
        print_tree(s.right, g, prefix_new, true)
    elseif s.left isa BUState
        print_tree(s.left, g, prefix_new, true)
    end
end

function print_trees(s_list::Vector{BUState}, g)
    for s in s_list
        print_tree(s, g)
    end
end