struct RandomPolicy{RNG} <: Policy
    pomdp::AlzheimerPOMDPProblem
    rng::RNG
end

POMDPs.action(p::RandomPolicy, b) = rand(p.rng, actions(p.pomdp))
