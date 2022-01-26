# using GLMakie
# fig = Figure()
# ax = Axis(fig[1,1])
# limits!(ax, 0, 1, 0, 1)
# xs = rand(10)
# ys = rand(10)
# scatter!(xs,ys)
# scatter!(ys,xs)
# arrows!(xs,ys,ys-xs,xs-ys)
# display(fig)
using GLMakie
using GraphMakie
using Graphs
g = wheel_graph(10)
f, ax, p = graphplot(g)
hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()
display(f)

# run the following script
#= 
using PackageCompiler
create_sysimage(["GraphMakie"], sysimage_path="sysimage.so", precompile_execution_file="sysimage.jl")
=#
