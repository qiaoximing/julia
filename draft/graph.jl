@enum Etype S L R D B F K I
@enum Vtype H P0 P1

mutable struct Edge
    p::Int64 # parent node / function symbol
    l::Int64 # left child / input value
    r::Int64 # right child / output value
    t::Etype # edge type
    w::Float64 # weight
end

mutable struct Vertex
    rules::Vector{Edge} # production rule edges
    rules_l::Vector{Edge} # rules as left child
    rules_r::Vector{Edge} # rules as right child
    cache::Vector{Edge} # i/o value edges
    cache_l::Vector{Edge} # cache as input
    cache_r::Vector{Edge} # cache as output
    nr::Int64 # number of rules
    nc::Int64 # number of cache entries
    t::Vtype # vertex type
end
Vertex(t::Vtype) = Vertex([], 0, t)

mutable struct Graph
    vertices::Vector{Vertex}
    nv::Int64 # number of vertices
    root::Int64 # root state
    init::Int64 # init state
    prints::Vector{Int64} # printing states
end
Graph() = Graph([], 0)

function set_root!(g::Graph, root)
    g.root = root
end

function set_init!(g::Graph, init)
    g.init = init
end

function set_prints!(g::Graph, prints)
    g.prints = prints
end

function add_vertex!(g::Graph, t::Vtype=H)
    push!(g.vertices, Vertex(t))
    g.nv += 1
    return g.nv
end

function add_edge!(g::Graph, v, l, r, t, w)
    edge = Edge(v, l, r, t, w)
    if t == F
        # share the edge in 3 lists
        push!(g.vertices[v].cache, edge)
        push!(g.vertices[l].cache_l, edge)
        push!(g.vertices[r].cache_r, edge)
        g.vertices[v].nc += 1
        return g.vertices[v].nc
    else
        push!(g.vertices[v].rules, edge)
        push!(g.vertices[l].rules_l, edge)
        push!(g.vertices[r].rules_r, edge)
        g.vertices[v].nr += 1
        return g.vertices[v].nr
    end
end