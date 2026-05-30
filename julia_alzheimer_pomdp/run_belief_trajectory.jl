#!/usr/bin/env julia
# Belief Trajectory Figure — anticipation in action
#
# For patients whose TRUE initial state is MCI, track how the belief
# distribution evolves step-by-step until a terminal decision is reached.
# Shows that SARSOP reaches confident MCI recognition faster than Expert,
# demonstrating early anticipation of cognitive decline.

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, Random, Statistics, Plots
using SARSOP
import .AlzheimerPOMDP: reset_policy!, isterminal_action

const SEED      = 42
const MAX_STEPS = 20
const N_PATIENTS = 3000   # per true-state group

# Run episode and return per-step belief snapshots (before each action)
function run_episode_tracked(pomdp, policy, updater, b0, true_state0;
                              max_steps=MAX_STEPS, rng=Random.GLOBAL_RNG)
    reset_policy!(policy)
    s  = true_state0
    b  = b0
    belief_history = Vector{Float64}[]   # p_MCI at each step
    action_history = Symbol[]

    for step in 1:max_steps
        push!(belief_history, [pdf(b, AlzheimerState(:CN)),
                               pdf(b, AlzheimerState(:MCI)),
                               pdf(b, AlzheimerState(:Dementia))])
        a  = action(policy, b)
        push!(action_history, a.name)

        sp = rand(rng, transition(pomdp, s, a))
        o  = rand(rng, observation(pomdp, a, sp))

        if !isterminal_action(a)
            b = update(updater, b, a, o)
        end
        s = sp
        isterminal_action(a) && break
    end
    return (belief_history=belief_history, action_history=action_history,
            n_steps=length(belief_history))
end

# Compute average belief trajectory up to a fixed horizon T,
# padding with the terminal belief for episodes that finished early.
function mean_belief_trajectory(all_histories, component, T)
    traj = zeros(T)
    for hist in all_histories
        for t in 1:T
            idx = min(t, length(hist))
            traj[t] += hist[idx][component]
        end
    end
    traj ./= length(all_histories)
    return traj
end

function main()
    pomdp   = AlzheimerPOMDPProblem(discount=0.95)
    rng     = MersenneTwister(SEED)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0      = initialize_belief(updater, initialstate(pomdp))

    sarsop = SARSOP.load_policy(pomdp, "policy.out")
    myopic = MyopicPOMDPPlanner(pomdp, updater)
    expert = ExpertPolicy(pomdp)

    POLICIES = [
        ("SARSOP",       sarsop,  "#3498db"),
        ("MyopicPOMDP",  myopic,  "#2ecc71"),
        ("Expert",       expert,  "#f39c12"),
    ]

    T = 8   # plot horizon (most episodes finish within 4-5 steps)

    fig = plot(size=(2000, 750), dpi=300, layout=(1, 3),
               left_margin=10Plots.mm, bottom_margin=22Plots.mm,
               right_margin=5Plots.mm, top_margin=8Plots.mm)

    true_states_info = [
        (AlzheimerState(:CN),       :CN,       "(a) True state: CN",        1),
        (AlzheimerState(:MCI),      :MCI,      "(b) True state: MCI",       2),
        (AlzheimerState(:Dementia), :Dementia, "(c) True state: Dementia",  3),
    ]

    # Target belief label shown on each panel's y-axis title
    target_ylabels = ["p(CN | history)", "p(MCI | history)", "p(Dem | history)"]

    println("="^65)
    println("BELIEF TRAJECTORY ANALYSIS  N=$N_PATIENTS per group")
    println("="^65)

    for (pi, (true_s, true_clinical, panel_title, target_comp)) in
            enumerate(true_states_info)

        # Collect belief histories for each policy
        policy_histories = Dict{String, Vector{Vector{Vector{Float64}}}}()
        for (pname, pol, _) in POLICIES
            policy_histories[pname] = Vector{Vector{Float64}}[]
        end

        for i in 1:N_PATIENTS
            for (pname, pol, _) in POLICIES
                ep = run_episode_tracked(pomdp, pol, updater, b0, true_s;
                                         max_steps=MAX_STEPS,
                                         rng=MersenneTwister(SEED + pi*1000 + i))
                push!(policy_histories[pname], ep.belief_history)
            end
        end

        # Print convergence stats
        println("\n--- True state: $true_clinical ---")
        for (pname, pol, _) in POLICIES
            hists = policy_histories[pname]
            avg_steps = mean(length(h) for h in hists)
            term_target = mean(h[end][target_comp] for h in hists)
            println("  $pname: avg steps=$(round(avg_steps,digits=2)), " *
                    "avg p(target) at terminal=$(round(term_target,digits=3))")
        end

        # Decision threshold for this true state
        threshold = (true_clinical == :Dementia) ? 0.52 :
                    (true_clinical == :MCI)       ? 0.50 : 0.82

        # Plot target-belief trajectory for each policy — one clean line each
        linestyles = [:solid, :dash, :dashdot]
        for (j, (pname, pol, pcolor)) in enumerate(POLICIES)
            traj = mean_belief_trajectory(policy_histories[pname], target_comp, T)
        plot!(subplot=pi, 1:T, traj,
              color=pcolor, lw=3.0, ls=linestyles[j], alpha=0.95,
              label=(pi == 3 ? pname : ""),
              legend=(pi == 3 ? :topright : false))
        end

        # Decision threshold reference line
        hline!(subplot=pi, [threshold], color=:gray40, lw=1.5, ls=:dot, alpha=0.7,
               label=(pi == 3 ? "Threshold" : ""))

        plot!(subplot=pi,
              xlabel="Visit number",
              ylabel=target_ylabels[target_comp],
              title=panel_title,
              titlefontsize=24, guidefontsize=22, tickfontsize=20,
              legendfontsize=16,
              ylims=(0, 1.05), xticks=1:T, yticks=0:0.2:1.0,
              grid=true, gridalpha=0.2,
              framestyle=:box)
    end

    mkpath("figures_paper")
    savefig("figures_paper/v2_belief_trajectory.png")
    savefig("figures_paper/v2_belief_trajectory.pdf")
    println("\nSaved: figures_paper/v2_belief_trajectory.png")
end

main()
