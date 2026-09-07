# MMSE/CDR are hand-specified; APOE4 frequencies are normalized below.
function default_observation_probabilities()
    mmse = Dict(
        :CN => Dict(:MMSE_Normal => 0.85, :MMSE_Sedang => 0.13, :MMSE_Rendah => 0.02),
        :MCI => Dict(:MMSE_Normal => 0.55, :MMSE_Sedang => 0.35, :MMSE_Rendah => 0.10),
        :Dementia => Dict(:MMSE_Normal => 0.20, :MMSE_Sedang => 0.45, :MMSE_Rendah => 0.35)
    )

    cdr = Dict(
        :CN => Dict(:CDR_0 => 0.95, :CDR_0_5 => 0.04, :CDR_1_plus => 0.01),
        :MCI => Dict(:CDR_0 => 0.25, :CDR_0_5 => 0.70, :CDR_1_plus => 0.05),
        :Dementia => Dict(:CDR_0 => 0.05, :CDR_0_5 => 0.30, :CDR_1_plus => 0.65)
    )

    apoe4_raw = Dict(
        :CN => Dict(:APOE4_Negatif => 0.6073, :APOE4_Hetero => 0.2472, :APOE4_Homo => 0.0244),
        :MCI => Dict(:APOE4_Negatif => 0.5071, :APOE4_Hetero => 0.3225, :APOE4_Homo => 0.0842),
        :Dementia => Dict(:APOE4_Negatif => 0.3198, :APOE4_Hetero => 0.4486, :APOE4_Homo => 0.1759)
    )
    # Normalize APOE4 probabilities so each state's observations sum to 1.0
    apoe4 = Dict{Symbol, Dict{Symbol, Float64}}()
    for (state, probs) in apoe4_raw
        total = sum(values(probs))
        apoe4[state] = Dict(k => v / total for (k, v) in probs)
    end
    return mmse, cdr, apoe4
end

function _validate_observation_probs(probs::Dict, name::String)
    for (state, pdict) in probs
        total = sum(values(pdict))
        if abs(total - 1.0) > 1e-6
            error("$name observation probabilities for $state sum to $total (expected 1.0)")
        end
    end
end

function POMDPs.observation(p::AlzheimerPOMDPProblem, a::AlzheimerAction, sp::AlzheimerState)
    if a.name == :A_Test_MMSE
        obs = [AlzheimerObs(k) for k in keys(p.mmse_probs[sp.name])]
        probs = collect(values(p.mmse_probs[sp.name]))
        return SparseCat(obs, probs)
    elseif a.name == :A_Test_CDR
        obs = [AlzheimerObs(k) for k in keys(p.cdr_probs[sp.name])]
        probs = collect(values(p.cdr_probs[sp.name]))
        return SparseCat(obs, probs)
    elseif a.name == :A_Test_APOE4
        obs = [AlzheimerObs(k) for k in keys(p.apoe4_probs[sp.name])]
        probs = collect(values(p.apoe4_probs[sp.name]))
        return SparseCat(obs, probs)
    else
        return SparseCat([AlzheimerObs(:None)], [1.0])
    end
end
