include("main.jl")
using Random

const LineBreak = "="^60

rules1 = """
S Sr -> c A 1
A Sr -> B C 1
B Plr -> I I 1
C Plr -> z I 1
c Cn -> x 1 | y 1
I Id
"""

rules1x = """
S Sr -> A B 1
A Slr -> c I 1
B Plr -> z I 1
c Cn -> x 1 | y 1
I Id
"""

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
    gm = init_grammar(rules3)
    data = generate_sentence(gm)
    println("Data: $data")
    prs = Parser(gm, 10)
    for i in 1:10
        println("trying particle filter $i")
        pf = ParticleFilter(2)
        ps = simulate(pf, prs, data)
        if !isnothing(ps)
            println("particle filter succeeds, now starting conditional particle filter")
            cpf = ConditionalParticleFilter(2, 1) 
            ps = simulate(cpf, sample_n(ps, cpf.num_conditions), data)
            break
        end
    end
    println(LineBreak)
end
# parser_test()

"Test grammar learning limited to CFG"
function cfg_learning_test()
    gm = init_grammar((4,0,0,0,0,0,0,0,0,0,1), "xyz", 0.1)
    dict = ["xx", "yy", "z"]
    dataset = [reduce(*, [rand(dict) for j in 1:2]) for i in 1:100]
    prs = Parser(gm, 10)
    pf = ParticleFilter(10)
    cpf = ConditionalParticleFilter(10, 1)
    pe = ParticleEM(pf, cpf, dataset, 100, 0.001)
    simulate(pe, prs)
    summarize(pe)
    println(LineBreak)
end
cfg_learning_test()