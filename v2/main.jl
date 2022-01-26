push!(LOAD_PATH, pwd())
using Random
using Utility, DenseGrammar, DetParse, Dataset

function print_tree(t::Tree, g::Grammar, isroot=true)
    if isroot print(t.p, ": ") end
    print("(", g.label[t.s])
    if t.l !== nothing print_tree(t.l, g, false) end
    if t.r !== nothing print_tree(t.r, g, false) end
    print(")")
    if isroot println() end
end

function print_stack(s::PStack, g::Grammar, isroot=true)
    if s.prev !== nothing print_stack(s.prev, g, false) end
    print(g.label[s.s])
    # if isroot println() end
end

function print_item(i::Item, g::Grammar, isroot=true)
    if isroot
        println("Prob: $(i.q)")
        print_tree(i.s.t, g)
    end
    if i.prev !== nothing 
        print_item(i.prev, g, false) 
        print("->")
    end
    print_stack(i.s, g)
    if isroot println() end
end

function test_parse()
    g = test_grammar()
    ds = generate_dataset(g, 1)
    parse_init!(g)
    for d in ds
        println("Data: ", d)
        result, count = parse_str(d, g)
        println("Parse steps/data length: $count/$(length(d))")
        if length(result) == 0
            println("Parse fails")
        end
        for (i, item) in enumerate(result)
            println("Item $i:")
            print_item(item, g)
        end
    end
end

# g0 = test_grammar()
# parse_init!(g0)
# ds = generate_dataset(g0, 1000)
ds = test_dataset(100, 30)
# ds0 = [repeat(["big"],1000);repeat(["you"],1000);ds]
g = init_grammar()
parse_init!(g)
for epoch in 1:10
    counts = Vector{Float64}()
    for (i, d) in enumerate(ds)
        # println("Parsing $i: $d")
        # get the parse results
        result, count = parse_str(d, g)
        # result, chart = parse_cyk(d, g)
        # count = length(chart)
        # push!(counts, count / length(d))
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
            decay!(0.03, g)
            parse_update!(g)
        end
    end
    println("Max weight: ", maximum(g.br))
    # counts = counts[1:10div(length(counts),10)]
    # counts_avg = sum(reshape(counts, (:,10)), dims=1)[1,:] ./ (length(counts)/10)
    # println("Average count: ", counts_avg)
    for d in generate_dataset(g, 30)
        println(d)
    end
end
t1 = d -> print_item(parse_str(d,g)[1][1],g)
t2 = d -> print_item(parse_cyk(d,g)[1][1],g)