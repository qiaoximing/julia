using Distributions, Random

gaussian(μ, σ, x) = exp(-(x - μ)^2 / (2σ^2))
N = 50
μs = [-3.5, 0.0]
x = [rand(Normal(μs[1], 1.), N); rand(Normal(μs[2], 1.), N)]

K = 2
μ = rand(Float64, K)
σ = repeat([1.], K)
z = zeros(Int64, length(x))
for iter in 1:10
	x_sum = zeros(Float64, K)
	x_cnt = zeros(Int64, K)
	for i in 1:length(x)
		γ = zeros(Float64, K)
		for j in 1:K
			γ[j] = gaussian(μ[j], σ[j], x[i])
		end
		γ /= sum(γ)
		z[i] = rand(Categorical(γ))
		x_sum[z[i]] += x[i]
		x_cnt[z[i]] += 1
	end
	μ = x_sum ./ x_cnt
    println(μ)
end