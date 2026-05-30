#!/usr/bin/env julia
# Multi-seed variance benchmark for Table VII
# Runs 10 seeds × N=10,000 patients; reports mean ± std across seeds.
# SARSOP policy loaded from policy.out (no retraining).

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, Random, Statistics, Printf
using SARSOP

const N_PER_SEED = 10_000
const MAX_STEPS  = 20
const SEEDS = [42, 123, 456, 789, 1000, 2024, 314, 271, 100, 999]

struct SeedResult
    reward   :: Float64
    accuracy :: Float64
    cn_acc   :: Float64
    mci_acc  :: Float64
    dem_acc  :: Float64
    test_cost:: Float64
    steps    :: Float64
end

function eval_policy(pomdp, policy, updater, b0, n, seed)
    rng = MersenneTwister(seed)
    true_states = [rand(rng, initialstate(pomdp)) for _ in 1:n]
    results = [run_episode(pomdp, policy, updater, b0, s;
                           max_steps=MAX_STEPS, rng=rng) for s in true_states]
    SeedResult(
        mean(r.total_reward for r in results),
        mean(r.correct      for r in results) * 100,
        mean(r.correct for r in results if r.true_state == :CN)        * 100,
        mean(r.correct for r in results if r.true_state == :MCI)       * 100,
        mean(r.correct for r in results if r.true_state == :Dementia)  * 100,
        mean(r.test_cost    for r in results),
        mean(r.steps        for r in results),
    )
end

function stats(vals)
    m = mean(vals); s = std(vals)
    (mean=round(m,digits=1), std=round(s,digits=2))
end

function print_row(name, seed_results)
    rw  = stats([r.reward    for r in seed_results])
    acc = stats([r.accuracy  for r in seed_results])
    cn  = stats([r.cn_acc    for r in seed_results])
    mc  = stats([r.mci_acc   for r in seed_results])
    dm  = stats([r.dem_acc   for r in seed_results])
    tc  = stats([r.test_cost for r in seed_results])
    st  = stats([r.steps     for r in seed_results])
    @printf("%-14s | %+6.1f±%-5.2f | %5.1f±%-4.2f | %5.1f±%-4.2f | %5.1f±%-4.2f | %5.1f±%-4.2f | %+6.1f±%-5.2f | %4.2f±%-4.2f\n",
            name,
            rw.mean, rw.std,
            acc.mean, acc.std,
            cn.mean, cn.std,
            mc.mean, mc.std,
            dm.mean, dm.std,
            tc.mean, tc.std,
            st.mean, st.std)
end

function main()
    pomdp   = AlzheimerPOMDPProblem(discount=0.95)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0      = initialize_belief(updater, initialstate(pomdp))

    println("Loading SARSOP policy from policy.out ...")
    sarsop = SARSOP.load_policy(pomdp, "policy.out")

    println("="^100)
    println("MULTI-SEED BENCHMARK  n_seeds=$(length(SEEDS))  N_per_seed=$N_PER_SEED  max_steps=$MAX_STEPS")
    println("="^100)
    @printf("%-14s | %-12s | %-10s | %-10s | %-10s | %-10s | %-12s | %-9s\n",
            "Solver", "Reward", "Acc", "CN", "MCI", "Dem", "TestCost", "Steps")
    println("-"^100)

    policies = [
        ("SARSOP",      sarsop),
        ("MyopicPOMDP", nothing),
        ("Expert",      nothing),
        ("Random",      nothing),
    ]

    for (name, _) in policies
        seed_results = SeedResult[]
        for seed in SEEDS
            pol = if name == "SARSOP"
                sarsop
            elseif name == "MyopicPOMDP"
                MyopicPOMDPPlanner(pomdp, updater)
            elseif name == "Expert"
                ExpertPolicy(pomdp)
            else
                AlzheimerPOMDP.RandomPolicy(pomdp, MersenneTwister(seed))
            end
            r = eval_policy(pomdp, pol, updater, b0, N_PER_SEED, seed)
            push!(seed_results, r)
        end
        print_row(name, seed_results)
    end

    println("="^100)
    println("\nDone. Use mean±std values to update Table VII (tab:benchmark) in main.tex.")
end

main()
