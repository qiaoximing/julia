
function MCMC_learning()
    g0 = test_grammar()
    parse_init!(g0)
    ds = generate_dataset(g0, 1000)
    # ds = test_dataset(1000)
    g = init_grammar()
    parse_init!(g)
    prev_parse = []
    for epoch in 1:10
        println("epoch:$epoch")
        for (i, d) in enumerate(ds)
            # parse_init!(g)
            # result, count = parse_str(d, g)
            result, chart = parse_cyk(d, g0)
            if length(result) == 0 continue end
            # estimate posteiror
            prob = [r.s.t.p for r in result]
            idx, _ = sample(normalize(prob))
            t = result[idx].s.t
            # grammar update
            grammar_update!(t, g, 1.)
            if epoch > 1 
                # println("i:$i, d:$d")
                # print_tree(t, g)
                # print_tree(prev_parse[i], g)
                grammar_update!(prev_parse[i], g, -1.) 
                prev_parse[i] = t
            else
                push!(prev_parse, t)
            end
        end
        for d in generate_dataset(g, 30)
            println(d)
        end
    end
end