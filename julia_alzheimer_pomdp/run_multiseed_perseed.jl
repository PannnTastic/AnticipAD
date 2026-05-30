#!/usr/bin/env julia
# Per-seed dump for paired significance testing (Gap #1).
# Loads frozen SARSOP policy.out (no retrain), evaluates all policies on the
# SAME 10 seeds, and writes per-seed metrics to JSON for paired Wilcoxon / bootstrap.

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, Random, Statistics, JSON
using SARSOP

const N_PER_SEED = 10_000
const MAX_STEPS  = 20
const SEEDS = [42, 123, 456, 789, 1000, 2024, 314, 271, 100, 999]

function eval_policy(pomdp, policy, updater, b0, n, seed)
    rng = MersenneTwister(seed)
    true_states = [rand(rng, initialstate(pomdp)) for _ in 1:n]
    results = [run_episode(pomdp, policy, updater, b0, s; max_steps=MAX_STEPS, rng=rng) for s in true_states]
    Dict(
        "reward"    => mean(r.total_reward for r in results),
        "accuracy"  => mean(r.correct for r in results) * 100,
        "cn_acc"    => mean(r.correct for r in results if r.true_state == :CN) * 100,
        "mci_acc"   => mean(r.correct for r in results if r.true_state == :MCI) * 100,
        "dem_acc"   => mean(r.correct for r in results if r.true_state == :Dementia) * 100,
        "test_cost" => mean(r.test_cost for r in results),
        "steps"     => mean(r.steps for r in results),
    )
end

function main()
    pomdp   = AlzheimerPOMDPProblem(discount=0.95)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0      = initialize_belief(updater, initialstate(pomdp))
    println("Loading frozen SARSOP policy.out (no retrain)...")
    sarsop = SARSOP.load_policy(pomdp, "policy.out")

    out = Dict{String,Any}()
    for name in ["SARSOP","MyopicPOMDP","Expert","Random"]
        out[name] = Dict{String,Vector{Float64}}()
        for seed in SEEDS
            pol = name == "SARSOP" ? sarsop :
                  name == "MyopicPOMDP" ? MyopicPOMDPPlanner(pomdp, updater) :
                  name == "Expert" ? ExpertPolicy(pomdp) :
                  AlzheimerPOMDP.RandomPolicy(pomdp, MersenneTwister(seed))
            m = eval_policy(pomdp, pol, updater, b0, N_PER_SEED, seed)
            for (k,v) in m
                push!(get!(out[name], k, Float64[]), v)
            end
        end
        println("done: $name")
    end
    open("per_seed_results.json","w") do f; JSON.print(f, out) end
    println("Saved per_seed_results.json")
end
main()
