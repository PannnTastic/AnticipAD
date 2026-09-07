struct ExpertRollout <: Policy
    pomdp::AlzheimerPOMDPProblem
end

function POMDPs.action(p::ExpertRollout, s::AlzheimerState)
    if s.name == :Dementia
        return AlzheimerAction(:A_Treat_Dementia)
    elseif s.name == :MCI
        return AlzheimerAction(:A_Treat_MCI)
    else
        return AlzheimerAction(:A_Wait)
    end
end
