function getfirst(f::Function, x)
    i = findfirst(f, x)
    if i === nothing 
        return nothing 
    else
        return x[i]
    end
end

function deletefirst!(f::Function, x)
    i = findfirst(f, x)
    if i === nothing 
        warning("deletefirst fails")
    else
        deleteat!(x, i)
    end
end

function deleteall!(f::Function, x)
    # idxs = findall(f, x)
    # for i in idxs
    #     deleteat!(x, i)
    # end
    filter!(i->f(i)==false, x)
end

function powerset(x::Vector{T}) where T
    result = Vector{T}[[]]
    for elem in x, j in eachindex(result)
        push!(result, [result[j] ; elem])
    end
    result
end

function warning(msg::String)
    printstyled("WARNING: ", bold=true, color=:yellow)
    println(msg)
end

# print str1 and highlight the difference comparing to str2
function println_diff(str1::String, str2::String)
    i, j, l = 1, 0, min(length.((str1, str2))...)
    while i < l && str1[i] == str2[i]
        i += 1
    end
    while j < l && str1[end-j] == str2[end-j]
        j += 1
    end
    print(str1[1:i-1])
    printstyled(str1[i:end-j], color=:red)
    print(str1[end-j+1:end])
    println()
end

function Base.show(io::IO, x::AbstractArray)
    print(io, '[')
    if !isempty(x)
        for i in x[1:end-1]
            print(io, i, ", ")
        end
        print(io, x[end], ']')
    else
        print(io, ']')
    end
end

Base.show(io::IO, x::Group) = print(io, x.prod.node.node, "->", 
                                    # x.prod.targ, ": ", x.factors)
                                    x.prod.targ)

Base.show(io::IO, x::Distr) = print(io, x.prob)

Base.show(io::IO, x::State) = print(io, x.trees)

function Base.show(io::IO, x::Tree)
    print(io, "T", x.node, "->")
    if x.edge isa Etype
        print(io, x.edge)
    else
        print(io, '?')
    end
    if x.left isa Tree
        print(io, '(', x.left, ')')
    elseif x.left isa Nothing
        print(io, "()")
    else
        print(io, "(?)")
    end
    if x.right isa Tree
        print(io, '(', x.right, ')')
    elseif x.right isa Nothing
        print(io, "()")
    else
        print(io, "(?)")
    end
    print(io, 'i')
    if x.input isa Int
        print(io, x.input)
    else
        print(io, '?')
    end
    print(io, 'o')
    if x.output isa Int
        print(io, x.output)
    else
        print(io, '?')
    end
end

function Base.show(io::IO, x::Option)
    print(io, '[')
    for i in 1:length(x.dgroups)-1
        print(io, x.dgroups[i].prod.node.node, ',')
    end
    print(io, x.dgroups[end].prod.node.node, "]-")
    print(x.prod.targ, '-')
    # print(x.prod, '-')
    @printf("%.2f", x.score)
    # println()
end

function Base.show(io::IO, x::Net)
    array = x.data[(1,1,1,1,1,1)]
    print(io, array.data)
end