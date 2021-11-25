function warning(msg::String)
    printstyled("WARNING: ", bold=true, color=:yellow)
    println(msg)
end

function relative_error(target, estimate)
    return maximum(abs.(target - estimate)) / maximum(abs.(target))
end