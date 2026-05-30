#!/usr/bin/env julia
# Blocker A: does CDR still dominate the SARSOP policy when MMSE uses the
# RAW observed-only ADNI likelihoods (not the smoothed/noised model)?
# Builds a POMDP variant with observed MMSE, re-solves SARSOP to a SEPARATE
# policy file (does NOT touch the frozen policy.out), and re-extracts the
# belief-simplex decision regions for comparison with the 79.2% CDR baseline.

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, SARSOP, Random, Printf

# Observed-only ADNI MMSE likelihoods (normalized), keys match model convention
# MMSE_Normal / MMSE_Sedang(=Moderate) / MMSE_Rendah(=Low)
function raw_mmse()
    d = Dict(
        :CN       => Dict(:MMSE_Normal=>0.9969, :MMSE_Sedang=>0.0031, :MMSE_Rendah=>0.0000),
        :MCI      => Dict(:MMSE_Normal=>0.9488, :MMSE_Sedang=>0.0485, :MMSE_Rendah=>0.0027),
        :Dementia => Dict(:MMSE_Normal=>0.3757, :MMSE_Sedang=>0.4617, :MMSE_Rendah=>0.1627),
    )
    for (s,p) in d
        t = sum(values(p)); for k in keys(p); p[k] = p[k]/t; end
    end
    return d
end

const ACTION_NAMES = [:A_Wait, :A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4, :A_Treat_MCI, :A_Treat_Dementia]

function extract_regions(pomdp, policy; resolution=0.02)
    counts = Dict(a=>0 for a in ACTION_NAMES); total = 0
    for p_cn in 0.0:resolution:1.0, p_mci in 0.0:resolution:(1.0-p_cn)
        p_dem = 1.0 - p_cn - p_mci
        p_dem < -1e-8 && continue
        p_dem = max(0.0, p_dem)
        t = p_cn + p_mci + p_dem
        b = DiscreteBelief(pomdp, [p_cn/t, p_mci/t, p_dem/t])
        a = action(policy, b)
        counts[a.name] += 1; total += 1
    end
    return counts, total
end

function main()
    println("="^70)
    println("BLOCKER A: CDR dominance under RAW observed-only ADNI MMSE")
    println("="^70)

    # 1. baseline (smoothed MMSE) — re-extract from frozen policy.out for apples-to-apples
    base = AlzheimerPOMDPProblem(discount=0.95)
    base_policy = SARSOP.load_policy(base, "policy.out")
    bc, bt = extract_regions(base, base_policy)
    println("\n[Baseline smoothed-MMSE, from frozen policy.out]")
    for a in ACTION_NAMES
        @printf("  %-16s %5.1f%%\n", String(a), 100*bc[a]/bt)
    end

    # 2. raw-MMSE variant — re-solve SARSOP to a SEPARATE file
    raw = AlzheimerPOMDPProblem(discount=0.95)
    raw.mmse_probs = raw_mmse()
    println("\nRe-solving SARSOP on raw-MMSE model (separate policy file)...")
    solver = SARSOPSolver(precision=1e-2, timeout=120.0,
                          pomdp_filename="model_rawmmse.pomdpx",
                          policy_filename="policy_rawmmse.out")
    t0 = time()
    raw_policy = solve(solver, raw)
    @printf("  solved in %.1fs\n", time()-t0)
    rc, rt = extract_regions(raw, raw_policy)
    println("\n[Raw-MMSE variant]")
    for a in ACTION_NAMES
        @printf("  %-16s %5.1f%%\n", String(a), 100*rc[a]/rt)
    end

    cdr_base = 100*bc[:A_Test_CDR]/bt
    cdr_raw  = 100*rc[:A_Test_CDR]/rt
    mmse_raw = 100*rc[:A_Test_MMSE]/rt
    println("\n=== VERDICT ===")
    @printf("CDR coverage: baseline %.1f%%  ->  raw-MMSE %.1f%%\n", cdr_base, cdr_raw)
    @printf("MMSE coverage under raw model: %.1f%% (baseline 0.0%%)\n", mmse_raw)
    println(cdr_raw >= 50 ? "CDR STILL DOMINANT under raw MMSE." :
                            "CDR dominance WEAKENED — MMSE now competes.")
end
main()
