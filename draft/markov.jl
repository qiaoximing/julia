using Plots

function getdata()
    text = collect(String(read("input.txt")))
    alphabet = unique(text)
    return text, alphabet
end

function char2int(text, alphabet)
    c2i = Dict(alphabet .=> 1:length(alphabet))
    # i2c = Dict(values(c2i) .=> keys(c2i))
    i2c = alphabet
    if !(reduce(&, map(i -> i == i2c[c2i[i]], alphabet)) &&
        reduce(&, map(i -> i == c2i[i2c[i]], 1:length(alphabet))))
        @warn "Not 1-to-1 mapping"
    end
    ints = Array{Int8}(undef, length(text))
    for i in 1:length(text)
        ints[i] = c2i[text[i]]
    end
    return ints, i2c
end

function count0(data, label)
    cnt = ones(Int64, length(label))
    for i in data
        cnt[i] += 1
    end
    return Dict(label .=> cnt)
end

function count1(data, label)
    cnt = ones(Int64, length(label), length(label))
    prev = data[end]
    for i in data
        cnt[i, prev] += 1
        prev = i
    end
    return Dict((label[i], label[j])=>cnt[j, i] 
        for j in 1:length(label) for i in 1:length(label))
end

text, alphabet = getdata()
data, label = char2int(text, alphabet)
stat0 = sort(collect(count0(data, label)), by=x->x[2], rev=true)
println(stat0[1:20])
stat1 = sort(collect(count1(data, label)), by=x->x[2], rev=true)
println(stat1[1:20])
pred("Hello, ", )