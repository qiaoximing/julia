using Plots

f(x, p, s) = exp(-(x - p)^2/(2s)) / sqrt(2pi*s)
g(x) = x^2
u, su = 2, 1
vp, sp = 3, 1
T, dt = 5, 0.01

p(v, u) = f(v, vp, sp) * f(u, g(v), su)

P = map(x->p(x, u), dt:dt:T)
P ./= sum(P)
# plot(P) |> display

ϕ = zeros(Int(T/dt))
ϕ[1] = vp
g′(x) = x
for i in 1:length(ϕ)-1
    ϵp = vp - ϕ[i]
    ϵu = u - g(ϕ[i])
    grad = ϵp / sp + g′(ϕ[i]) * ϵu / su
    ϕ[i+1] = ϕ[i] + dt * grad
end
plot(ϕ, ylims=(0,4)) |> display
