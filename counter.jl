using Random

mutable struct Memory
    # size of total, hidden, input, output
    shape::Tuple{Int64, Int64, Int64, Int64} 
    Vnxt::Int64 # next free node
    # nodes: type, weight
    V::Array{Tuple{Int64, Float64}} 
    # directed 2-edges: target node => type, weight
    E::Array{Dict{Int64, Tuple{Int64, Float64}}} 
    # directed 3-edges: target1, target2 => type, weight
    Hi::Array{Dict{Tuple{Int64, Int64}, Tuple{Int64, Float64}}} 
    # directed 3-edges: target1 => target2, weight
    Hd::Array{Dict{Int64, Tuple{Int64, Float64}}} 
end

mutable struct Model
    iptr::Int64 # instruction pointer
    dptr::Int64 # data pointer
    erg::Int64 # energy cost (number of nodes creation)
    Vsta::Array{Bool} # activation state of nodes
    iptr::Array{Int64} # path of instruction pointer
    dptr::Array{Int64} # path of data pointer
    mem::Memory
end

function memory_init(hidden::Int64, input::Int64, output::Int64)
    total = hidden + input + output
    shape = (total, hidden, input, output)
    # init nodes with zero weight
    V = fill((0, 0.), total) # 0: empty nodes
    for i = hidden+1:hidden+input-1
        V[i] = (1, 0.) # 1: input nodes
    end
    for i = hidden+input+1:total-1
        V[i] = (1, 0.) # 2: output nodes
    end
    V[1] = (3, 0.) # 3: root node (initial iptr)
    V[2] = (4, 0.) # 4: terminate node (halt when iptr == 2)
    V[3] = (5, 0.) # 3: zero node (initial dptr)
    Vnxt = 4
    # init 2-edges, first dimension is Array, second dimension is Dict
    E = fill(Dict(), total)
    # connect inputs
    for i = hidden+1:hidden+input-1
        E[i][i+1] = (1, 1.) # typical edge with weight=1
        E[i+1][i] = (1, 1.) # typical edge with weight=1
    end
    # connect outputs
    for i = hidden+input+1:total-1
        E[i][i+1] = (1, 1.) # typical edge with weight=1
        E[i+1][i] = (1, 1.) # typical edge with weight=1
    end
    # connect root
    for i = hidden+1:hidden+input-1
        E[1][i] = (1, 1.)
    end
    for i = hidden+input+1:total-1
        E[1][i] = (1, 1.)
    end
    E[1][2] = (1, 1.) # root to terminate
    # init 3-edges, first dimension is Array, second dimension is Dict
    H = fill(Dict(), total)
    return Memory(shape, Vnxt, V, E, H)
end

function model_init(memory::Memory)
    iptr = 1 # root node
    dptr = 3 # zero node
    erg = 0 # zero energy
    Vsta = fill(false, memory.shape[1]) # node state = 0
    Vnxt = fill(0, memory.shape[1]) # 0 => not entered
    ipth = [iptr] # instr path
    dpth = [dptr] # data path
    return Model(iptr, dptr, ipth, dpth, erg, Vsta, memory)
end

function categorical_rand(prob::Array{Float64})
    psum = cumsum(prob)
    r = rand(psum[end])
    idx = findfirst(==(0), r .> psum)
    return idx
end

function model_step!(model::Model; train::Bool)
    # if first enter:
    if model.Vnxt[model.iptr] == 0
        # get all neighbors
        H = model.mem.H[model.iptr] # (t1, t2) => (type, weight)
        Hwgt = getfield.(values(H), 2)
        # select next function
        if train
            # sample the next function by probability
            softmax(x) = (t = exp.(x); t ./ sum(t))
            Hprob = softmax(Hwgt)
            # explore new functions with ϵ-greedy
            # (implicitly regularize long programs)
            eps = 0
            if rand(1) < eps
                # perform a new random walk
                # if meet a data node, try branch condition

                # if meet a func node, try composition

                model.erg += 1
            else
                # choose random path
                Hidx = categorical_rand(Hprob)
                t1, t2 = keys(H)[Hidx]
            end
        else
            # follow the max-probability path
            Hidx = argmax(Hprob)
            t1, t2 = keys(H)[Hidx]
        end
        # condition on type of H
        Htyp = H[(t1, t2)][1]
        if Htyp == 2
            # composition
            # store right node for re-enter
            model.Vnxt[model.iptr] = t2
            # go to left node
            model.iptr = t1
        elseif Htyp == 3
            # branching
            ()
            model.iptr = H.left
        else
            # atomic
            ()
            model.iptr = H.top
        end
    else
    # previously entered:
        model.iptr = model.Vnxt[model.iptr]
    end
    # if new data node
    model.erg += 1
end

function model_execute!(model::Model, data::Array{Bool}; train::Bool)
    # hook inputs
    total, hidden, input, output = model.mem.shape 
    model.Vsta[hidden+1:hidden+input] = data
    # model update
    max_steps = 10
    for step = 1:max_steps
        model_step!(model, train=train)
        # terminate or energy cost too high
        if model.iptr == 2 || model.erg > 3
            break
        end
    end
    # return outputs
    return model.Vsta[hidden+input+1:total]
end

function memory_update!(memory::Memory, path::Array{Int64}, reward::Float64)
    # need the trace of iptr and dptr
    return true
end

function memory_decay!(memory::Memory, wd::Float64)
    return true
end

function data_gen(level=3::Int, review_ratio=0.5::Real)
    #=
        level: difficulty of data (start from level 1)
    =#
    trainsize, testsize = 100, 100
    totalsize = trainsize + testsize
    maxlevel = 10
    dataset = fill((Bool[], Bool[]), totalsize)
    for i = 1:totalsize
        if level == 1 || i > review_ratio * totalsize # learn new stuff
            lv = level
        else # review old stuff
            lv = rand(1:level-1)
        end
        # random permutation
        data = vcat(fill(true, lv), fill(false, maxlevel - lv))
        shuffle!(data)
        # one hot label
        label = fill(false, maxlevel)
        label[lv] = true
        dataset[i] = (data, label)
    end
    shuffle!(dataset)
    return dataset[1:trainsize], dataset[trainsize+1:totalsize]
end

function train_level(memory, level::Int, epochs::Int)
    #=
        level: difficulty of data (start from level 1)
        epochs: number of training epochs
    =#
    trainset, testset = data_gen(level, 0.5)
    for epoch = 1:epochs
        print("Epoch ", epoch, ": ")
        # train loop
        correct = 0
        for (data, label) in trainset
            model = model_init(memory)
            output = model_execute!(model, data, train=true)
            if output == label
                reward = 1 - tanh(model.step / 10)
                memory_update!(memory, model.ipth, reward)
                memory_update!(memory, model.dpth, reward)
            end
            memory_decay!(memory, 1e-4)
        end
        print("Train: $(correct / length(trainset)); ")
        # test loop
        correct = 0
        for (data, label) in testset
            model = model_init(memory)
            output = model_execute!(model, data, train=false)
            correct += output == label
        end
        println("Test: $(correct / length(testset))")
    end
end

function test_level(memory, level::Int)
    _, testset = data_gen(level, 0)
    correct = 0
    for (data, label) in testset
        model = model_init(memory)
        output = model_execute!(model, data, train=false)
        correct += output == label
    end
    println("Test level $(level): $(correct / length(testset))")
end

function main_loop()
    memory = memory_init(10, 10, 100)
    for level = 1:10
        println("-- Level $level --")
        train_level(memory, level, 10)
        for lv = 1:level
            test_level(memory, lv)
        end
    end
end

main_loop()