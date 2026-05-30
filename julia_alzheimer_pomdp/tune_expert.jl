#!/usr/bin/env julia
# Gap #3: threshold-tuned Expert baseline.
# Tunes Expert (p_cn, p_mci, p_dem) thresholds on a TRAIN seed by grid search
# (max reward), then evaluates the best config on the held-out 9 seeds.
# Answers: does a fair, tuned Expert close the gap to belief-driven policies?

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, Random, Statistics, Printf

const N = 10_000
const MAX_STEPS = 20
const TRAIN_SEED = 42
const TEST_SEEDS = [123, 456, 789, 1000, 2024, 314, 271, 100, 999]

function eval_expert(pomdp, updater, b0, pcn, pmci, pdem, n, seed)
    rng = MersenneTwister(seed)
    truth = [rand(rng, initialstate(pomdp)) for _ in 1:n]
    pol = ExpertPolicy(pomdp, Set{Symbol}(), pdem, pcn, pmci)
    res = [run_episode(pomdp, pol, updater, b0, s; max_steps=MAX_STEPS, rng=rng) for s in truth]
    (reward=mean(r.total_reward for r in res), acc=mean(r.correct for r in res)*100)
end

function main()
    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))

    grid = 0.50:0.05:0.90
    best = (reward=-Inf, pcn=0.8, pmci=0.8, pdem=0.85)
    for pcn in grid, pmci in grid, pdem in grid
        r = eval_expert(pomdp, updater, b0, pcn, pmci, pdem, N, TRAIN_SEED)
        if r.reward > best.reward
            best = (reward=r.reward, pcn=pcn, pmci=pmci, pdem=pdem)
        end
    end
    @printf("BEST on train seed %d: pcn=%.2f pmci=%.2f pdem=%.2f -> reward=%.1f\n",
            TRAIN_SEED, best.pcn, best.pmci, best.pdem, best.reward)

    # default Expert (0.80/0.80/0.85) vs tuned, on held-out seeds
    def_r=Float64[]; def_a=Float64[]; tun_r=Float64[]; tun_a=Float64[]
    for s in TEST_SEEDS
        d = eval_expert(pomdp, updater, b0, 0.80, 0.80, 0.85, N, s)
        t = eval_expert(pomdp, updater, b0, best.pcn, best.pmci, best.pdem, N, s)
        push!(def_r,d.reward); push!(def_a,d.acc); push!(tun_r,t.reward); push!(tun_a,t.acc)
    end
    @printf("\nHeld-out (%d seeds):\n", length(TEST_SEEDS))
    @printf("Expert default : reward %.1f±%.1f  acc %.1f±%.1f\n", mean(def_r),std(def_r),mean(def_a),std(def_a))
    @printf("Expert tuned   : reward %.1f±%.1f  acc %.1f±%.1f\n", mean(tun_r),std(tun_r),mean(tun_a),std(tun_a))
    @printf("(compare: MyopicPOMDP reward 73.0/acc 78.1 ; SARSOP 68.7/78.3)\n")
end
main()
