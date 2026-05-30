#!/usr/bin/env julia
# SOLVER: Random Policy (3 states, max_steps=20, N=100k)

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, JSON, Random, Statistics, Dates

const N_SAMPLES = 100000
const SEED = 42
const MAX_STEPS = 20

function main()
    println("=" ^ 70); println("SOLVER: Random Policy (3 states | max_steps=$MAX_STEPS)"); println("=" ^ 70)
    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    rng = MersenneTwister(SEED)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    true_states = [rand(rng, initialstate(pomdp)) for _ in 1:N_SAMPLES]
    policy = POMDPTools.RandomPolicy(pomdp; rng=rng)
    
    println("Evaluating on $N_SAMPLES patients (max_steps=$MAX_STEPS)...")
    t0 = time()
    results = [run_episode(pomdp, policy, updater, b0, s; max_steps=MAX_STEPS, rng=rng) for s in true_states]
    println("Completed in $(round(time()-t0,digits=2))s")
    
    save_results("results_random_$(N_SAMPLES).json", results)
    print_summary(results)
end

function save_results(filename, data)
    out = Dict("solver"=>"Random", "n_samples"=>N_SAMPLES, "max_steps"=>MAX_STEPS, "seed"=>SEED,
        "avg_reward"=>round(mean([d.total_reward for d in data]),digits=2),
        "reward_std"=>round(std([d.total_reward for d in data]),digits=2),
        "accuracy"=>round(mean([d.correct for d in data])*100,digits=1),
        "cn_acc"=>round(mean([d.correct for d in data if d.true_state==:CN])*100,digits=1),
        "mci_acc"=>round(mean([d.correct for d in data if d.true_state==:MCI])*100,digits=1),
        "dem_acc"=>round(mean([d.correct for d in data if d.true_state==:Dementia])*100,digits=1),
        "test_cost"=>round(mean([d.test_cost for d in data]),digits=2),
        "steps"=>round(mean([d.steps for d in data]),digits=2))
    open(filename,"w") do f; JSON.print(f,out) end; println("Saved: $filename")
end

function print_summary(data)
    println("\n--- Summary ---")
    println("Avg Reward: ", round(mean([d.total_reward for d in data]),digits=2))
    println("Accuracy:   ", round(mean([d.correct for d in data])*100,digits=1),"%")
    println("CN Acc:     ", round(mean([d.correct for d in data if d.true_state==:CN])*100,digits=1),"%")
    println("MCI Acc:    ", round(mean([d.correct for d in data if d.true_state==:MCI])*100,digits=1),"%")
    println("Dem Acc:    ", round(mean([d.correct for d in data if d.true_state==:Dementia])*100,digits=1),"%")
    println("Test Cost:  ", round(mean([d.test_cost for d in data]),digits=2))
    println("Steps:      ", round(mean([d.steps for d in data]),digits=2))
end

main()
