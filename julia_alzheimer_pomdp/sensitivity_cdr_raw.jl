#!/usr/bin/env julia
# CDR raw-likelihoods sensitivity
# Compares belief-simplex extraction with raw ADNI CDR MCI row (0.075/0.904/0.021)
# vs hand-specified (0.25/0.70/0.05)

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP
using POMDPs, POMDPTools, Random, Statistics, SARSOP, Printf

const N = 10_000
const SEED = 42
const MAX_STEPS = 20
const SIMPLEX_N = 20

function extract_regions(policy, pomdp; n=SIMPLEX_N)
    updater = DiscreteUpdater(pomdp)
    counts = Dict{Symbol, Int}()
    total = 0
    for i in 0:n, j in 0:(n-i)
        k = n - i - j
        p_cn  = i / n
        p_mci = j / n
        p_dem = k / n
        b = initialize_belief(updater, SparseCat(AlzheimerPOMDP.STATES, [p_cn, p_mci, p_dem]))
        try
            a = action(policy, b)
            counts[a.name] = get(counts, a.name, 0) + 1
            total += 1
        catch
        end
    end
    return Dict(k => round(100.0 * v / total, digits=1) for (k, v) in counts)
end

function run_and_summarize(pomdp, policy, updater, b0, true_states; rng)
    results = [run_episode(pomdp, policy, updater, b0, s;
                           max_steps=MAX_STEPS, rng=rng) for s in true_states]
    rew = mean(r.total_reward for r in results)
    acc = mean(r.correct for r in results) * 100
    return (reward=round(rew,digits=1), accuracy=round(acc,digits=1))
end

function main()
    # --- Raw CDR model (MCI row only changed) ---
    pomdp_raw = AlzheimerPOMDPProblem()
    pomdp_raw.cdr_probs[:MCI] = Dict(
        :CDR_0      => 0.075,
        :CDR_0_5    => 0.904,
        :CDR_1_plus => 0.021
    )

    updater = DiscreteUpdater(pomdp_raw)
    b0      = initialize_belief(updater, initialstate(pomdp_raw))
    rng_st  = MersenneTwister(SEED)
    true_states = [rand(rng_st, initialstate(pomdp_raw)) for _ in 1:N]

    println("=" ^ 60)
    println("CDR RAW-LIKELIHOODS SENSITIVITY")
    println("Raw CDR MCI row: CDR=0 → 0.075 | CDR=0.5 → 0.904 | CDR≥1 → 0.021")
    println("Baseline MCI row: 0.250 / 0.700 / 0.050")
    println("=" ^ 60)

    println("\nTraining SARSOP on raw-CDR model (timeout 600s)...")
    sarsop_raw = try
        SARSOP.solve(SARSOPSolver(precision=1e-2, timeout=600.0), pomdp_raw)
    catch e
        println("SARSOP training failed: $e"); return
    end
    println("SARSOP trained.")

    println("\n--- Belief-simplex decision regions (raw CDR) ---")
    regions = extract_regions(sarsop_raw, pomdp_raw)
    for (a, pct) in sort(collect(regions), by=x -> -x[2])
        println("  $a: $pct%")
    end

    println("\n--- Policy performance (raw CDR model, N=$N) ---")
    for (name, pol) in [
        ("SARSOP-raw-CDR", sarsop_raw),
        ("MyopicPOMDP",    AlzheimerPOMDP.MyopicPOMDPPlanner(pomdp_raw, updater)),
        ("Expert",         AlzheimerPOMDP.ExpertPolicy(pomdp_raw)),
    ]
        r = run_and_summarize(pomdp_raw, pol, updater, b0, true_states;
                              rng=MersenneTwister(SEED))
        @printf("  %-20s  reward=%6.1f  acc=%5.1f%%\n", name, r.reward, r.accuracy)
    end

    println("\nCDR raw sensitivity analysis complete.")
end

main()
