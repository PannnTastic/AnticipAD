#!/usr/bin/env julia
# Generate per-episode discounted reward distributions for all evaluated policies.

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP
using POMDPs, POMDPTools, SARSOP, Random, Statistics, StatsBase, JSON, Plots

const N_SAMPLES = 100000
const SEED = 42
const MAX_STEPS = 20
const POLICY_ORDER = ["SARSOP", "MyopicPOMDP", "Expert", "Random"]
const POLICY_COLORS = Dict(
    "SARSOP" => "#e74c3c",
    "MyopicPOMDP" => "#2ecc71",
    "Expert" => "#3498db",
    "Random" => "#95a5a6"
)
const STATE_ORDER = [:CN, :MCI, :Dementia]
const STATE_COLORS = Dict(
    :CN => "#2ecc71",
    :MCI => "#f39c12",
    :Dementia => "#e74c3c"
)
const STATE_LABELS = Dict(
    :CN => "CN",
    :MCI => "MCI",
    :Dementia => "Dementia"
)
const COMMON_BINS = collect(-1050.0:25.0:325.0)

function percentile(sorted_vals::Vector{Float64}, p::Float64)
    n = length(sorted_vals)
    idx = 1 + (n - 1) * p
    lo = floor(Int, idx)
    hi = ceil(Int, idx)
    lo == hi && return sorted_vals[lo]
    w = idx - lo
    return (1 - w) * sorted_vals[lo] + w * sorted_vals[hi]
end

function summarize_rewards(rewards::Vector{Float64})
    sorted_rewards = sort(rewards)
    return Dict(
        "mean" => round(mean(rewards), digits=3),
        "std" => round(std(rewards), digits=3),
        "variance" => round(var(rewards), digits=3),
        "min" => round(minimum(rewards), digits=3),
        "p05" => round(percentile(sorted_rewards, 0.05), digits=3),
        "q1" => round(percentile(sorted_rewards, 0.25), digits=3),
        "median" => round(median(rewards), digits=3),
        "q3" => round(percentile(sorted_rewards, 0.75), digits=3),
        "p95" => round(percentile(sorted_rewards, 0.95), digits=3),
        "max" => round(maximum(rewards), digits=3)
    )
end

function make_policy(name::String, pomdp, updater)
    if name == "SARSOP"
        return SARSOP.load_policy(pomdp, "policy.out")
    elseif name == "MyopicPOMDP"
        return MyopicPOMDPPlanner(pomdp, updater)
    elseif name == "Expert"
        return ExpertPolicy(pomdp)
    elseif name == "Random"
        return AlzheimerPOMDP.RandomPolicy(pomdp, MersenneTwister(SEED + 999))
    else
        error("Unknown policy: $name")
    end
end

function evaluate_policy(name::String, pomdp, updater, b0, true_states)
    policy = make_policy(name, pomdp, updater)
    rng = MersenneTwister(SEED)
    rewards = Float64[]
    correct = Bool[]
    steps = Int[]
    true_state_names = Symbol[]

    for (i, s) in enumerate(true_states)
        if i % 10000 == 0
            println("  $name progress: $i / $(length(true_states))")
        end
        r = run_episode(pomdp, policy, updater, b0, s; max_steps=MAX_STEPS, rng=rng)
        push!(rewards, r.total_reward)
        push!(correct, r.correct)
        push!(steps, r.steps)
        push!(true_state_names, r.true_state)
    end

    return (
        rewards = rewards,
        correct = correct,
        steps = steps,
        true_states = true_state_names
    )
end

function plot_reward_distributions(reward_data, summaries)
    plots = []
    for name in POLICY_ORDER
        rewards = reward_data[name]
        s = summaries[name]
        title_text = "$(name)\nmean=$(s["mean"]), sd=$(s["std"])"
        p = histogram(
            rewards,
            bins=80,
            normalize=:probability,
            color=POLICY_COLORS[name],
            alpha=0.82,
            linecolor=:white,
            linewidth=0.2,
            label=false,
            title=title_text,
            xlabel="Discounted reward",
            ylabel="Proportion",
            size=(650, 420),
            dpi=300,
            grid=true,
            gridalpha=0.25
        )
        vline!([s["mean"]], color=:black, linewidth=2, linestyle=:dash, label=false)
        push!(plots, p)
    end

    fig = plot(
        plots...,
        layout=(2, 2),
        size=(1300, 850),
        dpi=300,
        plot_title="Distribution of Discounted Reward Across 100,000 Simulated Patients",
        margin=5Plots.mm
    )

    mkpath("figures_paper")
    savefig(fig, "figures_paper/v2_14_discounted_reward_distribution.png")
    savefig(fig, "figures_paper/v2_14_discounted_reward_distribution.pdf")
    println("Saved: figures_paper/v2_14_discounted_reward_distribution.png")
end

function plot_reward_distributions_by_state(reward_data, state_data)
    plots = []
    bin_width = COMMON_BINS[2] - COMMON_BINS[1]
    ymax = 0.0

    for policy in POLICY_ORDER
        rewards = reward_data[policy]
        states_for_policy = state_data[policy]
        for st in STATE_ORDER
            vals = rewards[findall(==(st), states_for_policy)]
            counts = fit(Histogram, vals, COMMON_BINS).weights
            density_max = maximum(counts ./ (length(vals) * bin_width))
            ymax = max(ymax, density_max)
        end
    end

    ylims_common = (0.0, ymax * 1.12)

    for policy in POLICY_ORDER
        rewards = reward_data[policy]
        states_for_policy = state_data[policy]
        p = plot(
            title=policy,
            xlabel="Discounted reward",
            ylabel="Normalized density",
            xlims=(first(COMMON_BINS), last(COMMON_BINS)),
            ylims=ylims_common,
            size=(650, 420),
            dpi=300,
            grid=false,
            titlefontsize=18,
            guidefontsize=16,
            tickfontsize=14,
            legendfontsize=14,
            legend=:topleft
        )

        for st in STATE_ORDER
            vals = rewards[findall(==(st), states_for_policy)]
            histogram!(
                p,
                vals,
                bins=COMMON_BINS,
                normalize=:pdf,
                alpha=0.42,
                color=STATE_COLORS[st],
                linecolor=:white,
                linewidth=0.2,
                label=STATE_LABELS[st]
            )
        end

        push!(plots, p)
    end

    fig = plot(
        plots...,
        layout=(2, 2),
        size=(1300, 850),
        dpi=300,
        plot_title="Discounted Reward Distribution by True Clinical State and Policy",
        plot_titlefontsize=20,
        margin=5Plots.mm
    )

    mkpath("figures_paper")
    savefig(fig, "figures_paper/v2_15_discounted_reward_by_state.png")
    savefig(fig, "figures_paper/v2_15_discounted_reward_by_state.pdf")
    println("Saved: figures_paper/v2_15_discounted_reward_by_state.png")
end

function summarize_by_state(rewards::Vector{Float64}, states_for_policy::Vector{Symbol})
    out = Dict{String, Any}()
    for st in STATE_ORDER
        vals = rewards[findall(==(st), states_for_policy)]
        out[string(st)] = summarize_rewards(vals)
    end
    return out
end

function main()
    println("="^72)
    println("DISCOUNTED REWARD DISTRIBUTION")
    println("N=$N_SAMPLES, max_steps=$MAX_STEPS, seed=$SEED")
    println("="^72)

    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    updater = DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    rng_states = MersenneTwister(SEED)
    true_states = [rand(rng_states, initialstate(pomdp)) for _ in 1:N_SAMPLES]

    reward_data = Dict{String, Vector{Float64}}()
    state_data = Dict{String, Vector{Symbol}}()
    summaries = Dict{String, Any}()
    by_state_summaries = Dict{String, Any}()

    for name in POLICY_ORDER
        println("\nEvaluating $name...")
        t0 = time()
        result = evaluate_policy(name, pomdp, updater, b0, true_states)
        elapsed = round(time() - t0, digits=2)
        reward_data[name] = result.rewards
        state_data[name] = result.true_states
        summaries[name] = summarize_rewards(result.rewards)
        by_state_summaries[name] = summarize_by_state(result.rewards, result.true_states)
        summaries[name]["accuracy"] = round(mean(result.correct) * 100, digits=2)
        summaries[name]["avg_steps"] = round(mean(result.steps), digits=3)
        summaries[name]["eval_time_sec"] = elapsed
        println("  completed in $(elapsed)s")
        println("  mean=$(summaries[name]["mean"]), sd=$(summaries[name]["std"]), var=$(summaries[name]["variance"])")
    end

    out = Dict(
        "n_samples" => N_SAMPLES,
        "seed" => SEED,
        "max_steps" => MAX_STEPS,
        "summaries" => summaries,
        "by_state_summaries" => by_state_summaries,
        "rewards" => reward_data
    )

    open("reward_distribution_100000.json", "w") do f
        JSON.print(f, out, 2)
    end
    println("\nSaved: reward_distribution_100000.json")

    plot_reward_distributions(reward_data, summaries)
    plot_reward_distributions_by_state(reward_data, state_data)
end

main()
