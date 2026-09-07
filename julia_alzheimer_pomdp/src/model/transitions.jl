# Consecutive ADNI visit frequencies with no calendar-time rescaling.
function default_transitions()
    transitions = Dict(
        :CN => Dict(:CN => 0.929, :MCI => 0.068, :Dementia => 0.003),
        :MCI => Dict(:CN => 0.000, :MCI => 0.892, :Dementia => 0.108),
        :Dementia => Dict(:CN => 0.000, :MCI => 0.000, :Dementia => 1.000)
    )
    return transitions
end

function _validate_transition_probs(transitions::Dict)
    for (from_state, to_dict) in transitions
        total = sum(values(to_dict))
        if abs(total - 1.0) > 1e-6
            error("Transition probabilities from $from_state sum to $total (expected 1.0)")
        end
    end
end

function POMDPs.transition(p::AlzheimerPOMDPProblem, s::AlzheimerState, a::AlzheimerAction)
    probs = Float64[p.transitions[s.name][sp.name] for sp in STATES]
    return SparseCat(STATES, probs)
end
