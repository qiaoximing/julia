#=
[Failed!]
Dynamic programming chart parser for the extended grammar
Analogous to CYK parser for CFG

Breaking issue: 
Recursive grammar with >1 null children will break the induction 
on string length. Need to repeat the program until fix point.
=#
include("Utility.jl")
include("GrammarEx.jl")

"""
Compile the chart of null symbols.
- Indexed by `[output val, input val, symbol]`
- Satisfies `sum([:, input val, symbol]) <= 1`. 
  Equal when a symbol is always null
"""
function get_chart_null(g::GrammarEx)
    n = size(g)
    chart = zeros(Float32, n, n, n)
    # Fn symbols
    chart[:, :, range(Fn, g)] .= g.h.p_fn
    # Cn symbols
    # - this will broadcast to all inputs
    chart[:, :, range(Cn, g)] .= g.h.p_cn 
    # Id symbol
    for sym in range(Id, g)
        for i in 1:n
            chart[i, i, sym] = 1.
        end
    end
    # Tr symbols 
    # chart[:, :, range(Tr, g)] = 0 since Tr's are always observed
    # Cm symbols
    for sym in range(U, g)
        type = g.type[sym]
        # check type, and loop over all rules
        if type == U
            for child in 1:n
                chart[:, :, sym] .+= g.h.p_cm1[child, sym] * chart[:, :, child]
            end
        elseif type == Lr
            for left in 1:n
                for right in 1:n
                    # the matmul sums over all outputs of left,
                    # which are equal to inputs of right
                    chart[:, :, sym] .+= (
                        g.h.p_cm2[right, left, sym] * 
                        chart[:, :, right] * chart[:, :, left])
                end
            end
        elseif type == Plr
            for left in 1:n
                for right in 1:n
                    # output of left will be a dynamic tree,
                    # which use outputs of right as its inputs
                    for dynam in 1:n
                        chart[:, :, sym] .+= (
                            g.h.p_cm2[right, left, sym] *
                            chart[:, :, dynam] * chart[:, :, right] .* 
                            chart[dynam:dynam, :, left])
                    end
                end
            end
        end
    end
    return chart
end

"""
Compile the main parse chart
- Depends on the null chart `g.h.nchart` (from `get_chart_null`)
- Indexed by `[output val, input val, symbol, end loc + 1, start loc]`
"""
function get_chart_main(str::String, g::GrammarEx)
    n = size(g)
    len = length(str)
    chart = zeros(Float32, n, n, n, len + 1, len)
    nchart = g.h.nchart
    # Terminals
    for loc in 1:len
        sym = g.index[string(str[loc])]
        for i in 1:n
            chart[i, i, sym, loc + 1, loc] = 1.
        end
    end
    # Composites
    for len0 in 1:n # substring length
        for loc_s in 1:(len - len0 + 1) # start loc
            loc_e = loc_s + len0 # end loc + 1
            for sym in range(U, g)
                type = g.type[sym]
                # check type and loop over all rules
                if type == U
                    for child in 1:n
                        chart[:, :, sym, loc_e, loc_s] .+= (
                            g.h.p_cm1[child, sym] * 
                            chart[:, :, child, loc_e, loc_s])
                    end
                elseif type == Lr
                    for loc_p in loc_s:loc_e # split loc
                        for left in 1:n # left child
                            for right in 1:n # right child
                                mat_left = (loc_p == loc_s ? nchart[:, :, left] :
                                    chart[:, :, left, loc_p, loc_s])
                                mat_right = (loc_e == loc_p ? nchart[:, :, right] :
                                    chart[:, :, right, loc_e, loc_p])
                                chart[:, :, sym, loc_e, loc_s] .+= (
                                    g.h.p_cm2[right, left, sym] * 
                                    mat_right * mat_left)
                            end
                        end
                    end
                elseif type == Plr
                    for loc_p1 in loc_s:loc_e # split loc 1
                        for loc_p2 in loc_p1:loc_e # split loc 2
                            for left in 1:n # left child
                                for right in 1:n # right child
                                    for dynam in 1:n # dynamic child
                                        mat_left = (loc_p1 == loc_s ? 
                                            nchart[dynam:dynam, :, left] :
                                            chart[dynam:dynam, :, left, loc_p1, loc_s])
                                        mat_right = (loc_p2 == loc_p1 ? 
                                            nchart[:, :, right] :
                                            chart[:, :, right, loc_p2, loc_p1])
                                        mat_dynam = (loc_e == loc_p2 ?
                                            nchart[:, :, dynam] :
                                            chart[:, :, dynam, loc_e, loc_p2])
                                        chart[:, :, sym, loc_e, loc_s] .+= (
                                            g.h.p_cm2[right, left, sym] *
                                            mat_dynam * mat_right .* 
                                            mat_left)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return chart
end

function parse_dp(str::String, g::GrammarEx)
    # compile function and rule probabilities
    # suppress NaN's in impossible rules
    p_fn = normalize0(g.w_fn, dims=1)
    p_cn = normalize0(g.w_cn, dims=1)
    w_sym = sum(g.w_cm1, dims=1)[:] .+ sum(g.w_cm2, dims=(1,2))[:]
    p_cm1 = div0.(g.w_cm1, reshape(w_sym, (1, g.n_cm)))
    p_cm2 = div0.(g.w_cm2, reshape(w_sym, (1, 1, g.n_cm)))
    g.h = (; p_fn, p_cn, p_cm1, p_cm2)
    # compile null chart (input independent)
    nchart = get_chart_null(g)
    g.h = (; nchart, g.h...)
    # start to process inputs
    chart = get_chart_main(str, g)
    # return the marginal probability of string
    # fix symbol = 1, input val = 1, sum over all output vals
    println(findall(x->x>0, chart))
    return sum(chart[:, 1, 1, length(str) + 1, 1])
end

# testing
g = test_grammar()
println(parse_dp("xx",g))