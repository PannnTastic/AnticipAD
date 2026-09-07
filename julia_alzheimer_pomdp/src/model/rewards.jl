# Dimensionless decision utilities, not measured treatment benefits or costs.
function default_rewards()
    rewards = Dict(
        :CN => Dict(:A_Wait => 30, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => -50, :A_Treat_Dementia => -500),
        :MCI => Dict(:A_Wait => -10, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => 150, :A_Treat_Dementia => -100),
        :Dementia => Dict(:A_Wait => -1000, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => -50, :A_Treat_Dementia => 300)
    )
    return rewards
end

POMDPs.reward(p::AlzheimerPOMDPProblem, s::AlzheimerState, a::AlzheimerAction) = p.rewards[s.name][a.name]
