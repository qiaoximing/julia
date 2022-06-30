include("main.jl")
using Random
using Profile

const LineBreak = "="^60

# pattern ABA
rules1 = """
S Sr -> c A 1
A Sr -> B C 1
B Plr -> I I 1
C Plr -> z I 1
c Cn -> x 1 | y 1
I Id
"""

# pattern ABA
rules1x = """
S Sr -> A B 1
A Slr -> c I 1
B Slr -> I z 1
c Cn -> x 1 | y 1
I Id
"""

# pattern ABA', where A' = f(A)
rules1y = """
S Sr -> A B 1
A Slr -> c I 1
B Slr -> f z 1
f Fn -> x y 1 | y x 1
c Cn -> x 1 | y 1
I Id
"""

# pattern ABA'BA''BA'''...
rules1z = """
S Sr -> A B 1
A Slr -> c I 1
B Sr -> C B 3 | C z 1
C Slr -> f z 1
f Fn -> x y 1 | y x 1
c Cn -> x 1 | y 1
I Id
"""

# Boolean logic
rules2 = """
S Sr -> E F 1
F Sr -> = P 1
E Plr -> E1 X 1
E1 Prl -> X cop 1
X Sr -> cx P 1
P Plr -> I I 1
And Sr -> and * 1
Or Sr -> or + 1
and Fn -> 0 c0 1 | 1 I 1
or Fn -> 0 I 1 | 1 c1 1
c0 Cn -> 0 1
c1 Cn -> 1 1
cx Cn -> 0 1 | 1 1
cop Cn -> And 1 | Or 1
I Id
"""

# recursive counting
rules3 = """
S Sr -> E F 1
E Sr -> cx P 1
F Sr -> = Rec 1
Rec Plr -> G H 1
G Sr -> isone fRec 1
H Sr -> x prev 1
P Plr -> I I 1
isone Fn -> 1 true 1 | 2 false 1 | 3 false 1 | 4 false 1
prev Fn -> 1 0 1 | 2 1 1 | 3 2 1 | 4 3 1
fRec Fn -> true I 1 | false Rec 1
cx Cn -> 1 1 | 2 1 | 3 1 | 4 1
I Id
true Id
false Id
"""

"Test the parser on simple grammars"
function parser_test()
    # Random.seed!(1)
    gm = init_grammar(rules1)
    data = generate_sentence(gm)
    println("Data: $data")
    prs = Parser(gm, 10)
    for i in 1:10
        println("trying particle filter $i")
        pf = ParticleFilter(2)
        ps = simulate(pf, prs, data)
        if !isnothing(ps)
            println("particle filter succeeds, now starting conditional particle filter")
            cpf = ConditionalParticleFilter(2, 1, false) 
            ps = simulate(cpf, sample_n(ps, cpf.num_conditions), data)
            break
        end
    end
    println(LineBreak)
end
# parser_test()

"Test grammar learning limited to CFG"
function word_learning(learner::Symbol, alpha)
    gm = init_grammar((4,0,0,0,0,0,0,0,0,0,1), "yobapedi", alpha)
    dict = ["yo", "ba", "pe", "di"]
    dataset = [reduce(*, [rand(dict) for j in 1:3]) for i in 1:100]
    curriculum = Curriculum([dataset], [100], [0])
    prs = Parser(gm, 10)
    num_particles, num_epochs = 10, curriculum.totalepochs
    pf = ParticleFilter(num_particles)
    cpf = ConditionalParticleFilter(num_particles, 1, false)
    println("Start learning")
    if learner == :EM
        schedule = Schedule(1e-3, 1e-3, num_epochs)
        pl = ParticleEM(pf, cpf, curriculum, num_epochs, schedule)
    elseif learner == :Gibbs
        pl = ParticleGibbs(pf, cpf, curriculum, num_epochs, 1)
    end
    simulate(pl, prs)
    summarize(pl, prs, 100, 0.999)
    println(LineBreak)
end
# @time word_learning(:EM, 1e-3)
# @time word_learning(:Gibbs, 1e-1)

"Test grammar learning of DFG"
function repetition_learning(learner::Symbol, alpha)
    alphabet = "0123456789"
    gm = init_grammar((0,3,0,0,3,0,0,0,2,2,1), alphabet, alpha)
    addrule!(gm, parserule("S6 -> C1 I1 50"))
    function getdata(alphabet)
        word1 = string(rand(alphabet))
        word2 = string(rand(alphabet))
        return word1 * word2 * word1
    end
    dataset1 = [getdata(alphabet) for _ in 1:100]
    curriculum = Curriculum([dataset1], [100], [0])
    prs = Parser(gm, 10)
    num_ptl, num_epochs = 100, curriculum.totalepochs
    pf = ParticleFilter(num_ptl)
    cpf = ConditionalParticleFilter(num_ptl, 1, true)
    println("Starting particle learner")
    if learner == :EM
        schedule = Schedule(1e-3, 1e-3, num_epochs)
        pl = ParticleEM(pf, cpf, curriculum, num_epochs, schedule)
    elseif learner == :Gibbs
        pl = ParticleGibbs(pf, cpf, curriculum, num_epochs, 1)
    end
    simulate(pl, prs)
    summarize(pl, prs, 100, 0.9)
    println(LineBreak)
    return pl, prs
end
for i in 1:10
    path = "exp1_$i.jld"
    if !isfile(path)
        pl, prs = repetition_learning(:Gibbs, 1e-1)
        JLD.save(path, "NLL", pl.log["NLL"])
    end
end

function relation_learning(learner::Symbol, alpha)
    alphabet = "0123456789"
    gm = init_grammar((0,3,0,0,3,0,0,0,2,2,1), alphabet, alpha)
    addrule!(gm, parserule("S6 -> C1 I1 100"))
    # for i in 0:9
    #     addrule!(gm, parserule("F1 -> $i $((i+1)%10) 100"))
    # end
    function getdata1(alphabet)
        word1 = string(rand(alphabet))
        word2 = string(rand(alphabet))
        return word1 * word2 * word1
    end
    function getdata2(alphabet)
        x = Int(rand(alphabet)) - Int('0')
        word1 = string(x)
        word2 = string(rand(alphabet))
        word3 = string((x+2)%length(alphabet))
        # return word1 * word2 * word3 # experiment 2
        return word1 * word2 * word1 * word3 # experiment 3
    end
    num_data = 100
    dataset1 = [getdata1(alphabet) for i in 1:num_data]
    dataset2 = [getdata2(alphabet) for i in 1:num_data]
    # good curriculum
    curriculum = Curriculum([dataset1, dataset2], [50,50], [50])
    # bad curriculum
    # curriculum = Curriculum([dataset2], [100], [0])
    prs = Parser(gm, 10)
    num_ptl, num_epochs = 400, curriculum.totalepochs
    pf = ParticleFilter(num_ptl)
    cpf = ConditionalParticleFilter(num_ptl, 1, true)
    println("Starting particle learner")
    if learner == :EM
        schedule = Schedule(1e-3, 1e-3, num_epochs)
        pl = ParticleEM(pf, cpf, curriculum, num_epochs, schedule)
    elseif learner == :Gibbs
        pl = ParticleGibbs(pf, cpf, curriculum, num_epochs, 1)
    end
    simulate(pl, prs)
    summarize(pl, prs, 100, 0.9)
    println(LineBreak)
    return pl, prs
end
# pl, prs = relation_learning(:Gibbs, 1e-1)
for i in 1:10
    path = "exp3_$i.jld"
    if !isfile(path)
        pl, prs = relation_learning(:Gibbs, 1e-1)
        JLD.save(path, "NLL", pl.log["NLL"])
    end
end

function composition_init(learner::Symbol, alpha)
    alphabet = "0123456789"
    gm = init_grammar((0,5,0,0,5,0,0,0,2,2,1), alphabet * "a", alpha)
    addrule!(gm, parserule("S6 -> C1 I1 100"))
    for i in 0:9
        addrule!(gm, parserule("F1 -> $i $((i+2)%10) 100"))
    end
    function getdata1(alphabet)
        word1 = string(rand(alphabet))
        word2 = string(rand(alphabet))
        return word1 * word2 * word1
    end
    function getdata2(alphabet)
        x = Int(rand(alphabet)) - Int('0')
        word1 = string(x)
        word2 = string(rand(alphabet))
        word3 = string((x+1)%length(alphabet))
        return word1 * word2 * word1 * word3
    end
    num_data = 100
    dataset1 = [getdata1(alphabet) for i in 1:num_data]
    dataset2 = [getdata2(alphabet) for i in 1:num_data]
    # good curriculum
    curriculum = Curriculum([dataset1, dataset2], [50,100], [50])
    # bad curriculum
    # curriculum = Curriculum([dataset2], [100], [0])
    prs = Parser(gm, 10)
    num_ptl, num_epochs = 400, curriculum.totalepochs
    pf = ParticleFilter(num_ptl)
    cpf = ConditionalParticleFilter(num_ptl, 1, true)
    println("Starting particle learner")
    if learner == :EM
        schedule = Schedule(1e-3, 1e-3, num_epochs)
        pl = ParticleEM(pf, cpf, curriculum, num_epochs, schedule)
    elseif learner == :Gibbs
        pl = ParticleGibbs(pf, cpf, curriculum, num_epochs, 1)
    end
    # simulate(pl, prs)
    # summarize(pl, prs, 100, 0.9)
    println(LineBreak)
    return pl, prs
end
# pl, prs = composition_init(:Gibbs, 1e-1)
# save(prs.grammar, "save.jld")

function composition_learning(learner::Symbol, alpha)
    alphabet = "0123456789"
    gm = init_grammar((0,3,0,0,3,0,0,0,2,2,1), alphabet * "a", alpha)
    # load!(gm, "save.jld")
    addrule!(gm, parserule("S6 -> C1 I1 100"))
    for i in 0:9 # experiment 6
        addrule!(gm, parserule("F1 -> $i $((i+1)%10) 100"))
    end
    # addrule!(gm, parserule("S3 -> F1 F1 100")) # experiment 7
    function getdata1(alphabet)
        word1 = string(rand(alphabet))
        word2 = string(rand(alphabet))
        return word1 * word2 * word1
    end
    function getdata2(alphabet)
        x = Int(rand(alphabet)) - Int('0')
        word1 = string(x)
        word2 = string(rand(alphabet))
        word3 = string((x+1)%length(alphabet))
        return word1 * word2 * word3
    end
    function getdata3(alphabet)
        x = Int(rand(alphabet)) - Int('0')
        word1 = string(x)
        word2 = string(rand(alphabet))
        word3 = string((x+2)%length(alphabet))
        return word1 * word2 * word3
    end
    function getdata4(alphabet)
        x = Int(rand(alphabet)) - Int('0')
        word1 = string(x)
        word2 = string(rand(alphabet))
        word3 = string((x+1)%length(alphabet))
        return word1 * word2 * word1 * word3
    end
    function getdata5(alphabet)
        x = Int(rand(alphabet)) - Int('0')
        word1 = string(x)
        word2 = string(rand(alphabet))
        word3 = string((x+1)%length(alphabet))
        word4 = string((x+3)%length(alphabet))
        return word1 * word2 * word1 * word3 * word4
    end
    num_data = 100
    dataset1 = [getdata1(alphabet) for i in 1:num_data]
    dataset2 = [getdata2(alphabet) for i in 1:num_data]
    dataset3 = [getdata3(alphabet) for i in 1:num_data]
    dataset4 = [getdata4(alphabet) for i in 1:num_data]
    dataset5 = [getdata5(alphabet) for i in 1:num_data]
    # curriculum = Curriculum([dataset1, dataset2, dataset3], [50,50,50], [50,50]) # experiment 4
    # curriculum = Curriculum([dataset1, dataset4, dataset5], [50,50,50], [50,50]) # experiment 5
    curriculum = Curriculum([dataset1, dataset3], [50,50], [50]) # experiment 6
    prs = Parser(gm, 10)
    num_ptl, num_epochs = 400, curriculum.totalepochs
    pf = ParticleFilter(num_ptl)
    cpf = ConditionalParticleFilter(num_ptl, 1, true)
    println("Starting particle learner")
    if learner == :EM
        schedule = Schedule(1e-3, 1e-3, num_epochs)
        pl = ParticleEM(pf, cpf, curriculum, num_epochs, schedule)
    elseif learner == :Gibbs
        pl = ParticleGibbs(pf, cpf, curriculum, num_epochs, 1)
    end
    simulate(pl, prs)
    summarize(pl, prs, 100, 0.9)
    println(LineBreak)
    return pl, prs
end
# pl, prs = composition_learning(:Gibbs, 1e-1)
for i in 1:10
    path = "exp6_$i.jld"
    if !isfile(path)
        pl, prs = composition_learning(:Gibbs, 1e-1)
        JLD.save(path, "NLL", pl.log["NLL"])
    end
end