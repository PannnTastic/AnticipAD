POMDPs.states(p::AlzheimerPOMDPProblem) = STATES
POMDPs.actions(p::AlzheimerPOMDPProblem) = ACTIONS
POMDPs.observations(p::AlzheimerPOMDPProblem) = OBSERVATIONS

POMDPs.stateindex(p::AlzheimerPOMDPProblem, s::AlzheimerState) = findfirst(x -> x == s, STATES)
POMDPs.actionindex(p::AlzheimerPOMDPProblem, a::AlzheimerAction) = findfirst(x -> x == a, ACTIONS)
POMDPs.obsindex(p::AlzheimerPOMDPProblem, o::AlzheimerObs) = findfirst(x -> x == o, OBSERVATIONS)
POMDPs.discount(p::AlzheimerPOMDPProblem) = p.discount
POMDPs.isterminal(p::AlzheimerPOMDPProblem, s::AlzheimerState) = false
