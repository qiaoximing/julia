using GLMakie
using GraphMakie
using Graphs, MetaGraphs, NetworkLayout

push!(LOAD_PATH, pwd())
using Utility, DenseGrammar, DetParse, Dataset
function graph_init(gram)
    n, ns = gram.n, gram.ns
    ur, br, c = gram.ur, gram.br, gram.c
    nlabels = [String([i]) for i in gram.label]
    g = MetaDiGraph(SimpleDiGraph(), 0.001)
    for i in 1:n
        add_vertex!(g)
        set_prop!(g, nv(g), :weight, c.d[i] / c.d0)
        set_prop!(g, nv(g), :color, :black)
    end
    # unary rules
    for i in 1:ns
        for j in 1:n
            add_edge!(g, i, j)
            set_prop!(g, i, j, :weight, ur[j, i] / c.tw[i] * c.d[i] / c.d0)
            set_prop!(g, i, j, :color, :black)
        end
    end
    # binary rules
    for j in 1:n
        for k in 1:n
            add_vertex!(g)
            v, w = nv(g), c.db[k, j] / c.d0
            set_prop!(g, v, :weight, w)
            set_prop!(g, v, :color, :red)
            push!(nlabels, gram.label[j] * gram.label[k])
            add_edge!(g, v, j)
            set_prop!(g, v, j, :weight, w)
            set_prop!(g, v, j, :color, :red)
            add_edge!(g, v, k)
            set_prop!(g, v, k, :weight, w)
            set_prop!(g, v, k, :color, :red)
        end
    end
    for i in 1:ns
        for j in 1:n
            for k in 1:n
                v, w = j * n + k, br[k, j, i] / c.tw[i] * c.d[i] / c.d0
                add_edge!(g, i, v)
                set_prop!(g, i, v, :weight, w)
                set_prop!(g, i, v, :color, :black)
            end
        end
    end
    node_color = [get_prop(g, node, :color) for node in 1:nv(g)]
    node_size = 10 .* sqrt.([get_prop(g, node, :weight) for node in 1:nv(g)])
    nlabels_textsize = 5 .* node_size
    edge_color = [get_prop(g, edge, :color) for edge in edges(g)]
    edge_width = 3 .* [get_prop(g, edge, :weight)  for edge in edges(g)]
    arrow_size = 5 .* edge_width
    layout = Spring(Ptype=Float32)
    f, ax, p = graphplot(g; nlabels, nlabels_textsize, layout, node_color, node_size, edge_color, edge_width, arrow_show=true, arrow_size)
    hidedecorations!(ax); hidespines!(ax)
    ax.aspect = DataAspect()
    display(f)
    return g, f, ax, p
end

function pos_update(pos, w)
    K = 1.0
    new_pos = similar(pos)
    for i in 1:length(pos)
        force = zeros(Float32, 2)
        for j in 1:length(pos)
            i == j && continue
            d = sqrt(sum((pos[j] .- pos[i]).^2))
            f = w[j, i] * d / K - K^2 / d^2
            force .+= f .* (pos[j] .- pos[i]) * 0.05
        end
        if i == 1 
            force .= 0. 
        else
            force .= sign.(force) .* max.(0., abs.(force) .- 0.01)
        end
        new_pos[i] = Point2f(pos[i] .+ force)
    end
    return new_pos
end

function graph_update!(g, f, ax, p, gram)
    n, ns = gram.n, gram.ns
    ur, br, c = gram.ur, gram.br, gram.c
    for i in 1:n
        set_prop!(g, i, :weight, c.d[i] / c.d0)
    end
    for i in 1:ns
        for j in 1:n
            set_prop!(g, i, j, :weight, ur[j, i] / c.tw[i] * c.d[i] / c.d0)
        end
    end
    for j in 1:n
        for k in 1:n
            v, w = j * n + k, c.db[k, j] / c.d0
            set_prop!(g, v, :weight, w)
            set_prop!(g, v, j, :weight, w)
            set_prop!(g, v, k, :weight, w)
        end
    end
    for i in 1:ns
        for j in 1:n
            for k in 1:n
                v, w = j * n + k, br[k, j, i] / c.tw[i] * c.d[i] / c.d0
                set_prop!(g, i, v, :weight, w)
            end
        end
    end
    p[:node_size][] = 10 .* sqrt.([get_prop(g, node, :weight) for node in 1:nv(g)])
    p[:nlabels_textsize][] = 5 .* p[:node_size][]
    p[:edge_width][] = 10 .* [get_prop(g, edge, :weight)  for edge in edges(g)]
    p[:arrow_size][] = 5 .* p[:edge_width][]
    p[:node_pos][] = pos_update(p[:node_pos][], weights(g))
    autolimits!(ax)
    print("")
end

ds = generate_dataset(test_grammar(), 1000)
g = init_grammar()
parse_init!(g)
graph, f, ax, p = graph_init(g)
for epoch in 1:5
    println("Epoch $epoch")
    for (i, d) in enumerate(ds)
        result, count = parse_str(d, g)
        if length(result) == 0 continue end
        # estimate posteiror
        prob = [r.s.t.p for r in result]
        post = normalize(prob)
        # grammar update
        for (r, v) in zip(result, post)
            grammar_update!(r.s.t, g, v)
        end
        # weight decay and cache update
        if i % 5 == 0
            decay!(0.01, g)
            parse_update!(g)
            graph_update!(graph, f, ax, p, g)
        end
    end
end