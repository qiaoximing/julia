using OpenAIGym
using Reinforce

env = GymEnv(:CartPole, :v0)

mutable struct MyPolicy <: AbstractPolicy 
    p1::Array{Float64}
end
function explore!(p::MyPolicy, s)
    # create a new node; threshold depends on state
end
function select!(p::MyPolicy)
    # select one random policy and store the selection
end
function learn!(p::MyPolicy, r)
    # update policy based on reward
end
function Reinforce.reset!(p::MyPolicy)
    num_in = 4
    num_out = 2
    explore!(p, zeros(num_in))
end
function Reinforce.action(p::MyPolicy, r, s′, A′)
    return rand(A′)
end
policy = MyPolicy()

for i ∈ 1:2
  T = 0
  reset!(policy)
  R = run_episode(env, policy) do (s, a, r, s′)
    # render(env)
    # println(a, ' ', r)
    T += 1
  end
  learn!(policy, R)
  @info("Episode $i finished after $T steps. Total reward: $R")
end
close(env)