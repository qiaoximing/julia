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
    gm = init_grammar(rules1z)
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
function word_learning(learner::Symbol)
    gm = init_grammar((4,0,0,0,0,0,0,0,0,0,1), "xyz", 1e-3)
    dict = ["xx", "yy", "z"]
    dataset = [reduce(*, [rand(dict) for j in 1:2]) for i in 1:100]
    prs = Parser(gm, 10)
    pf = ParticleFilter(10)
    cpf = ConditionalParticleFilter(10, 1, false)
    # cpf = ConditionalParticleFilterAS(10, 1, 1, false)
    println("Start learning")
    if learner == :EM
        schedule = Schedule(1e-3, 1e-3, 100)
        pl = ParticleEM(pf, cpf, dataset, 100, schedule)
    elseif learner == :Gibbs
        pl = ParticleGibbs(pf, cpf, dataset, 100, 1e-2)
    end
    simulate(pl, prs)
    summarize(pl, prs, 0.999)
    println(LineBreak)
end
# @time word_learning(:EM)
# @time word_learning(:Gibbs)

"Test grammar learning of DFG"
function repetition_learning(learner::Symbol)
    alphabet = "rstuvwxyz"
    gm = init_grammar((3,0,0,0,3,0,0,0,0,1,1), alphabet, 1e-3, 0)
    addrule!(gm, parserule("S5 -> C1 I1 1"))
    # addrule!(gm, parserule("S6 -> I1 I1 1"))
    function getdata(alphabet)
        word1 = string(rand(alphabet))
        word2 = string(rand(alphabet))
        word3 = string(rand(alphabet))
        return word1 * word2 * word1# * word3 * word1
    end
    dataset = [getdata(alphabet) for i in 1:100]
    prs = Parser(gm, 10)
    num_ptl, num_epochs = 100, 100
    pf = ParticleFilter(num_ptl)
    cpf = ConditionalParticleFilter(num_ptl, 1, true)
    println("Starting particle learner")
    if learner == :EM
        schedule = Schedule(1e-3, 1e-3, num_epochs)
        pl = ParticleEM(pf, cpf, dataset, num_epochs, schedule)
    elseif learner == :Gibbs
        pl = ParticleGibbs(pf, cpf, dataset, num_epochs, 1/length(dataset))
    end
    simulate(pl, prs)
    summarize(pl, prs, 0.999)
    println(LineBreak)
end
# @time repetition_learning(:Gibbs)

function relation_learning(learner::Symbol)
    alphabet = "123456"
    gm = init_grammar((3,0,0,0,3,0,0,0,3,1,1), alphabet, 1e-3, 0)
    addrule!(gm, parserule("S5 -> C1 I1 1"))
    # addrule!(gm, parserule("S6 -> F1 I1 1"))
    function getdata1(alphabet)
        word1 = string(rand(alphabet))
        word2 = string(rand(alphabet))
        word3 = word1
        return word1 * word2 * word3
    end
    function getdata2(alphabet)
        x = Int(rand(alphabet)) - Int('0')
        word1 = string(x)
        word2 = string(rand(alphabet))
        word3 = string(x == length(alphabet) ? 1 : x + 1)
        return word1 * word2 * word3
    end
    num_data = 100
    dataset1 = [getdata1(alphabet) for i in 1:num_data]
    dataset2 = [getdata2(alphabet) for i in 1:num_data]
    curriculum = Curriculum([dataset1, dataset2], [50,50], [100])
    prs = Parser(gm, 10)
    num_ptl, num_epochs = 400, curriculum.totalepochs
    pf = ParticleFilter(num_ptl)
    cpf = ConditionalParticleFilter(num_ptl, 1, true)
    println("Starting particle learner")
    if learner == :EM
        schedule = Schedule(1e-3, 1e-3, num_epochs)
        pl = ParticleEM(pf, cpf, curriculum, num_epochs, schedule)
    elseif learner == :Gibbs
        pl = ParticleGibbs(pf, cpf, curriculum, num_epochs, 1/num_data)
    end
    simulate(pl, prs)
    summarize(pl, prs, 0.999)
    println(LineBreak)
end
relation_learning(:Gibbs)