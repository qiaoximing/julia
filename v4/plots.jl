include("main.jl")

for j in 1:7
    plot([smooth(step_average(JLD.load("exp$(j)_$i.jld")["NLL"],100),0.9)
            for i in 1:10])
    savefig("exp$j.pdf")
end