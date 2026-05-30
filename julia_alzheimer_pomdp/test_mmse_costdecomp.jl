#!/usr/bin/env julia
# Decomposition: is the MMSE-takeover driven by COST or by DISCRIMINABILITY?
# Cell 3 of the 2x2: raw observed-ADNI MMSE likelihoods, but MMSE cost RAISED
# to -20 (equal to CDR). If CDR returns to dominance -> the flip was cost-driven.
# If MMSE still dominates -> it was discriminability-driven.

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, SARSOP, Random, Printf

function raw_mmse()
    d = Dict(
        :CN       => Dict(:MMSE_Normal=>0.9969, :MMSE_Sedang=>0.0031, :MMSE_Rendah=>0.0000),
        :MCI      => Dict(:MMSE_Normal=>0.9488, :MMSE_Sedang=>0.0485, :MMSE_Rendah=>0.0027),
        :Dementia => Dict(:MMSE_Normal=>0.3757, :MMSE_Sedang=>0.4617, :MMSE_Rendah=>0.1627),
    )
    for (s,p) in d; t=sum(values(p)); for k in keys(p); p[k]=p[k]/t; end; end
    return d
end

const ACTS = [:A_Wait,:A_Test_MMSE,:A_Test_CDR,:A_Test_APOE4,:A_Treat_MCI,:A_Treat_Dementia]
function regions(pomdp,policy;res=0.02)
    c=Dict(a=>0 for a in ACTS); tot=0
    for p_cn in 0.0:res:1.0, p_mci in 0.0:res:(1.0-p_cn)
        p_dem=1.0-p_cn-p_mci; p_dem< -1e-8 && continue; p_dem=max(0.0,p_dem)
        t=p_cn+p_mci+p_dem; b=DiscreteBelief(pomdp,[p_cn/t,p_mci/t,p_dem/t])
        c[action(policy,b).name]+=1; tot+=1
    end
    c,tot
end

function main()
    println("CELL 3: raw MMSE values + MMSE cost raised to -20 (= CDR cost)")
    p = AlzheimerPOMDPProblem(discount=0.95)
    p.mmse_probs = raw_mmse()
    # raise MMSE cost to -20 in every state's reward
    for s in [:CN,:MCI,:Dementia]; p.rewards[s][:A_Test_MMSE] = -20.0; end
    solver = SARSOPSolver(precision=1e-2, timeout=120.0,
                          pomdp_filename="model_cell3.pomdpx",
                          policy_filename="policy_cell3.out")
    pol = solve(solver, p)
    c,tot = regions(p,pol)
    for a in ACTS; @printf("  %-16s %5.1f%%\n", String(a), 100*c[a]/tot); end
    cdr=100*c[:A_Test_CDR]/tot; mmse=100*c[:A_Test_MMSE]/tot
    @printf("\nCDR=%.1f%%  MMSE=%.1f%%\n", cdr, mmse)
    println(cdr>mmse ? "=> CDR returns to dominance: flip was COST-driven (MMSE wins because it is cheap)." :
                       "=> MMSE still dominates even at equal cost: flip was DISCRIMINABILITY-driven.")
end
main()
