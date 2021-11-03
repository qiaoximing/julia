using LightGraphs
using Plots

@gif for i in 1:10
    plot(sin.(1:0.1:i))
end