include("main.jl")

rules1 = """
S Sr -> c A 1
A Sr -> B C 1
B Plr -> I I 1
C Plr -> z I 1
c Cn -> x 10 | y 1
"""

gm = init_grammar(rules1)
data = generate_sentence(gm)
println("Data: $data")
obs = getobservation(gm, data)
prs = Parser(gm, 10)
pf = ParticleFilter(1, obs)
ps = simulate(pf, prs)
println("Finish")