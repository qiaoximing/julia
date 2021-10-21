using Base: Tuple

function S(left, right)
    # return x -> right(x)(left(x))
    return ('S', left, right)
end

function L(left, right)
    # return x -> right(left(x))
    return ('L', left, right)
end

function R(left, right)
    # return x -> right(x)(left)
    return ('R', left, right)
end

function D(left, right)
    # return x -> right(left)
    return ('D', left, right)
end

function B(left, right)
    # return x -> (left(x); right(x))
    return ('B', left, right)
end

# node: a list of function transitions
# and a list of grammar rules
struct Node
    t::Char # name
    f::Array{Tuple{Int64, Int64, Int64}} # in, out, count
    g::Array{Tuple{Int64, Int64, Int64}} # left, right, count
end
Node() = Node('-', [], [])
Node(s) = Node(s, [], [])

# kg: knowledge graph as a list of nodes
kg = Node[]

# two print states
push!(kg, Node('0'))
push!(kg, Node('1'))
p0, p1 = 1, 2

# z: return null state
push!(kg, Node('z'))
z = length(kg)

# h: const function
push!(kg, Node('h',[(z,p0,1),(z,p1,1)],[]))
h = length(kg)

# f: state transition

# g: emission function

# p: print
push!(kg, Node('p',[],[]))
p = length(kg)

counter = D(z, L(h, p))
# bmm = D(D(z, h), B(L(f, p), L(g, p))) 
# vmm = D(D(D(z, L(h, p)), L(f, p)), L(f, p))
# hmm = D(D(D(D(z, h), 
#             B(L(g, p), f)), 
#           B(L(g, p), f)), 
#         B(L(g, p), f))

data = [0]
pointer = 1
tree_prob = 1

function sample(tree, act)
    if typeof(tree) <: Tuple
        t, l, r = tree
        if t == 'D'
            act = sample(l, [])
            act = sample(r, act)
        elseif t == 'L'
            act = sample(l, act)
            act = sample(r, act)
        elseif t == 'B'
            sample(l, copy(act))
            act = sample(r, act)
        else
            error()
        end
    else
        t = kg[tree].t
        if t == 'z' # null state
            act = zeros(length(kg))
            act[z] = 1
        elseif t == 'p' # print
            prob = act[p0] / (act[p0] + act[p1])
            if rand() < prob 
                print('0')
            else
                print('1')
            end
            if data[pointer] == 0
                tree_prob *= prob
            else
                tree_prob *= 1 - prob
            end
            pointer += 1
        else # functions
            act1 = zeros(length(act))
            for (i, j, c) in kg[tree].f
                act1[j] += act[i] * c
            end
            act = act1 ./ sum(act1)
        end
    end
    return act
end

function test(tree)
    prob = 0.7
    for i in 1:10
        if rand() < prob 
            data = [0]
        else
            data = [1]
        end
        pointer = 1
        tree_prob = 1
        sample(tree, [])
        kg
    end
    return
end

test(counter)
