mutable struct AlzheimerPOMDPProblem <: POMDP{AlzheimerState, AlzheimerAction, AlzheimerObs}
    transitions::Dict{Symbol, Dict{Symbol, Float64}}
    mmse_probs::Dict{Symbol, Dict{Symbol, Float64}}
    cdr_probs::Dict{Symbol, Dict{Symbol, Float64}}
    apoe4_probs::Dict{Symbol, Dict{Symbol, Float64}}
    rewards::Dict{Symbol, Dict{Symbol, Float64}}
    discount::Float64
end
function AlzheimerPOMDPProblem(;discount=0.95)
    transitions = default_transitions()
    mmse, cdr, apoe4 = default_observation_probabilities()
    rewards = default_rewards()
    _validate_observation_probs(mmse, "MMSE")
    _validate_observation_probs(cdr, "CDR")
    _validate_observation_probs(apoe4, "APOE4")
    _validate_transition_probs(transitions)
    return AlzheimerPOMDPProblem(transitions, mmse, cdr, apoe4, rewards, discount)
end
