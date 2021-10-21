function getfirst(f::Function, x)
    i = findfirst(f, x)
    if i === nothing 
        return nothing 
    else
        return f[i]
    end
end

function deletefirst!(f::Function, x)
    i = findfirst(f, x)
    if i === nothing 
        println("ERROR: deletefirst fails")
    else
        deleteat!(x, i)
    end
end

function deleteall!(f::Function, x)
    idxs = findall(f, x)
    for i in idxs
        deleteat!(x, i)
    end
end

function warning(msg::AbstractString)
    printstyled("WARNING: ", bold=true, color=:yellow)
    println(msg)
end

function Base.show(io::IO,x::AbstractArray)
    print('[')
    for i in x[1:end-1]
        print(i, ", ")
    end
    print(x[end], ']')
end

Base.show(io::IO,x::Group) = print(x.factors)

Base.show(io::IO,x::Distr) = print(x.targ, ":", x.prob)

Base.show(io::IO,x::State) = print(x.trees, '\n', x.dgroups)

function Base.show(io::IO,x::Tree)
    print("Tree:", x.node, "->")
    print('e')
    if x.edge isa Etype
        print(x.edge)
    else
        print('?')
    end
    print('l')
    if x.left isa Tree
        print(x.left.node)
    elseif x.left isa Nothing
        print('*')
    else
        print('?')
    end
    print('r')
    if x.right isa Tree
        print(x.right.node)
    elseif x.right isa Nothing
        print('*')
    else
        print('?')
    end
    print('i')
    if x.input isa Int
        print(x.input)
    else
        print('?')
    end
    print('o')
    if x.output isa Int
        print(x.output)
    else
        print('?')
    end
end