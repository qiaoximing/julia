module DetParse

using DataStructures
using Utility, DenseGrammar
export Tree, PStack, Item
export parse_init!, parse_update!, grammar_update!, parse_str

function parse_init!(g::Grammar)
    if !haskey(g.c, :c2i) # map char to int
        c2i = Dict(reverse.(enumerate(g.label)))
        g.c = (g.c..., c2i=c2i) 
    end
    if !haskey(g.c, :lr) # left symbol of rules
        lr = sum(g.br, dims=1)[1, :, :]
        g.c = (g.c..., lr=lr) 
    end
    if !haskey(g.c, :rr) # right symbol of rules
        rr = sum(g.br, dims=2)[:, 1, :]
        g.c = (g.c..., rr=rr) 
    end
    if !haskey(g.c, :tw) # total weight of rules
        tw = (sum(g.c.lr, dims=1) + sum(g.ur, dims=1))
        g.c = (g.c..., tw=tw) 
    end
    if !haskey(g.c, :d) # expected frequency of symbols
        pl = g.c.lr ./ g.c.tw
        pr = g.c.rr ./ g.c.tw
        pb = reshape(g.br, (g.n^2, g.ns)) ./ g.c.tw
        pu = g.ur ./ g.c.tw
        p = (pl + pr + pu)[1:g.ns, :]
        b = [1.; zeros(g.ns - 1)]
        d = (I - p) \ b
        dl = pl * d
        dr = pr * d
        db = reshape(pb * d, (g.n, g.n))
        du = pu * d
        dw = (dl + dr + du)[g.ns+1:end]
        d0 = sum(dw)
        g.c = (g.c..., d=[d;dw], dl=dl, dr=dr, db=db, du=du, d0=d0)
    end
end

function parse_update!(g::Grammar)
    g.c = (;)
    parse_init!(g)
end

struct Tree # parse tree
    s::Int # symbol
    p::Float64 # tree probability
    l::Union{Tree, Nothing} # left child
    r::Union{Tree, Nothing} # right child
end

struct PStack # parse stack
    s::Int # symbol
    t::Tree # parse tree
    prev::Union{PStack, Nothing} # previous stack state
end

struct Item # search item
    s::PStack # parse stack
    l::Int # input location
    q::Float64 # path probability
    prev::Union{Item, Nothing} # previous item
end

function grammar_update!(t::Tree, g::Grammar, v::Float64)
    if t.l === nothing && t.r === nothing # terminal
        return
    elseif t.r === nothing # unary rule
        g.ur[t.l.s, t.s] += v
        grammar_update!(t.l, g, v)
    else # binary rule
        g.br[t.r.s, t.l.s, t.s] += v
        grammar_update!(t.l, g, v)
        grammar_update!(t.r, g, v)
    end
end

function bounded_insert!(d::SortedMultiDict, k, v, n)
    if length(d) < n
        insert!(d, k, v)
        return true
    elseif k > first(d)[1]
        insert!(d, k, v)
        delete!((d, startof(d)))
        return true
    else
        return false
    end
end

function parse_str(str::String, g::Grammar)
    if minimum(g.c.d) < 0
        warning("bad grammar")
        return [], 0
    end
    input = [g.c.c2i[chr] for chr in str]
    d = SortedMultiDict(1.0 => Item(
        PStack(input[1], Tree(input[1], 1., nothing, nothing), nothing),
        1, 1., nothing))
    fin = []
    max_parse = 10 
    max_item = 10
    max_step = 500
    step = 1
    insertn!(d, k, v) = bounded_insert!(d, k, v, max_item)
    while length(d) > 0 && length(fin) < max_parse 
        step += 1
        if step > max_step 
            # warning("Parse timeout at $step")
            break 
        end
        z = last(d)[2] # item with highest score
        delete!((d, lastindex(d)))
        x = z.s.s # symbol at stack top
        # finish
        if x == 1 && z.l == length(input) && z.s.prev === nothing
            if !(z.s.t.p > 0)
                warning("Parse succeed with zero prob")
            else
                push!(fin, z)
            end
        end
        # shift
        q = g.c.dl[x] / g.c.d[x] # probability to shift
        if q > 0 && z.l < length(input)
            w = input[z.l + 1]
            q *= g.c.d[w] / g.c.d0 # probability to choose w
            insertn!(d, z.q * q, Item(
                PStack(w, Tree(w, 1., nothing, nothing), z.s), 
                z.l + 1, z.q * q, z))
        end
        # Unary reduce
        q = g.c.du[x] / g.c.d[x] # probability to unary reduce
        if q > 0
            qs = g.ur[x, :] ./ g.c.tw[1, :] .* g.c.d[1:g.ns] ./ g.c.du[x]
            perm = partialsortperm(qs, 1:min(length(qs), max_item), rev=true)
            for i in perm
                if isnan(qs[i]) continue end
                if qs[i] == 0 break end
                p = g.ur[x, i] / g.c.tw[i] * z.s.t.p
                if !insertn!(d, z.q * q * qs[i], Item(
                    PStack(i, Tree(i, p, z.s.t, nothing), z.s.prev), 
                    z.l, z.q * q * qs[i], z))
                    break
                end
            end
        end
        # Binary reduce
        q = g.c.dr[x] / g.c.d[x] # probability to binary reduce
        if q > 0 && z.s.prev !== nothing
            y = z.s.prev.s # next stack symbol, reduce lhs -> (y, x)
            q *= g.c.d0 * g.c.db[x, y] / g.c.dl[y] / g.c.dr[x] # merge probability
            qs = g.br[x, y, :] ./ g.c.tw[1, :] .* g.c.d[1:g.ns] ./ g.c.db[x, y] 
            perm = partialsortperm(qs, 1:min(length(qs), max_item), rev=true)
            for i in perm
                if isnan(qs[i]) continue end
                if qs[i] == 0 break end
                p = g.br[x, y, i] / g.c.tw[i] * z.s.t.p * z.s.prev.t.p
                # Note: q*qs[i] might > 1, breaking the optimality of A*
                if !insertn!(d, z.q * q * qs[i], Item(
                    PStack(i, Tree(i, p, z.s.prev.t, z.s.t), z.s.prev.prev), 
                    z.l, z.q * q * qs[i], z))
                    break
                end
            end
        end
    end
    return fin, step
end

include("DenseCYK.jl")

end