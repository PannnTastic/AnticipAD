mutable struct ExpertPolicy <: Policy
    pomdp::AlzheimerPOMDPProblem
    tested::Set{Symbol}
    p_dem_thres::Float64
    p_cn_thres::Float64
    p_mci_thres::Float64
end

ExpertPolicy(pomdp::AlzheimerPOMDPProblem) = ExpertPolicy(pomdp, Set{Symbol}(), 0.85, 0.80, 0.80)

function POMDPs.action(p::ExpertPolicy, b::DiscreteBelief)
    p_cn = pdf(b, AlzheimerState(:CN))
    p_mci = pdf(b, AlzheimerState(:MCI))
    p_dem = pdf(b, AlzheimerState(:Dementia))

    if p_dem >= p.p_dem_thres
        return AlzheimerAction(:A_Treat_Dementia)
    elseif p_cn >= p.p_cn_thres
        return AlzheimerAction(:A_Wait)
    elseif p_mci >= p.p_mci_thres
        return AlzheimerAction(:A_Treat_MCI)
    end

    if :A_Test_MMSE ∉ p.tested
        push!(p.tested, :A_Test_MMSE)
        return AlzheimerAction(:A_Test_MMSE)
    elseif :A_Test_CDR ∉ p.tested
        push!(p.tested, :A_Test_CDR)
        return AlzheimerAction(:A_Test_CDR)
    elseif :A_Test_APOE4 ∉ p.tested
        push!(p.tested, :A_Test_APOE4)
        return AlzheimerAction(:A_Test_APOE4)
    end

    if p_dem > p_mci && p_dem > p_cn
        return AlzheimerAction(:A_Treat_Dementia)
    elseif p_mci > p_cn
        return AlzheimerAction(:A_Treat_MCI)
    else
        return AlzheimerAction(:A_Wait)
    end
end

function reset_policy!(p::ExpertPolicy)
    empty!(p.tested)
    return nothing
end
