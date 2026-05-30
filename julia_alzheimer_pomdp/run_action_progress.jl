#!/usr/bin/env julia
# Generate Action Progress Figure
# Run: cd julia_alzheimer_pomdp && julia --project=. run_action_progress.jl

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, JSON, Random, Plots
using SARSOP

const N_SAMPLES = 1000
const SEED = 42
const MAX_STEPS = 20
const SOLVER_ORDER = ["SARSOP", "MyopicPOMDP", "Expert", "Random"]

function main()
    println("=" ^ 70)
    println("ACTION PROGRESS FIGURE GENERATION")
    println("N=$N_SAMPLES per solver | Max Steps=$MAX_STEPS")
    println("=" ^ 70)

    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    rng = MersenneTwister(SEED)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    true_states = [rand(rng, initialstate(pomdp)) for _ in 1:N_SAMPLES]

    println("Loading policies...")
    policies = Dict(
        "SARSOP"      => SARSOP.load_policy(pomdp, "policy.out"),
        "MyopicPOMDP" => MyopicPOMDPPlanner(pomdp, updater),
        "Expert"      => ExpertPolicy(pomdp),
        "Random"      => POMDPTools.RandomPolicy(pomdp; rng=rng)
    )
    println("All policies ready.")

    all_action_data = Dict{String, Vector{Vector{Symbol}}}()

    for name in SOLVER_ORDER
        policy = policies[name]
        println("Running $name on $N_SAMPLES patients...")
        action_histories = Vector{Vector{Symbol}}()
        for s in true_states
            r = run_episode(pomdp, policy, updater, b0, s; max_steps=MAX_STEPS, rng=rng)
            push!(action_histories, r.action_history)
        end
        all_action_data[name] = action_histories
        println("  Done: $(length(action_histories)) episodes")
    end

    println("\nGenerating action progress figure...")
    generate_action_progress_figure(all_action_data)
    println("Done!")
end

function generate_action_progress_figure(all_data)
    action_names  = [:A_Wait, :A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4, :A_Treat_MCI, :A_Treat_Dementia]
    action_labels = ["Wait", "MMSE", "CDR", "APOE4", "Tr.MCI", "Tr.Dem"]
    action_colors = ["#2ecc71", "#f1c40f", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"]

    max_steps = 20
    fig = plot(size=(1400, 900), dpi=300, layout=(2, 2),
               left_margin=5Plots.mm, bottom_margin=5Plots.mm)

    for (pi, pol_name) in enumerate(SOLVER_ORDER)
        histories = all_data[pol_name]
        n = length(histories)

        step_counts = [Dict(a => 0 for a in action_names) for _ in 1:max_steps]
        for hist in histories
            for (step_idx, act) in enumerate(hist)
                step_idx <= max_steps && (step_counts[step_idx][act] += 1)
            end
        end

        step_pcts = [Dict(a => 100 * step_counts[s][a] / n for a in action_names) for s in 1:max_steps]
        x_vals = collect(1:max_steps)
        bottoms = zeros(length(x_vals))

        for (aname, alabel, acolor) in zip(action_names, action_labels, action_colors)
            y_vals = [step_pcts[s][aname] for s in x_vals]
            bar!(x_vals, y_vals, bottom=bottoms, bar_width=0.85, color=acolor, alpha=0.88,
                 linecolor=:white, linewidth=0.3, label=(pi == 1 ? alabel : ""), subplot=pi)
            bottoms .+= y_vals
        end

        plot!(subplot=pi, xlabel="Visit (t)", ylabel="% Patients",
              title=pol_name, titlefontsize=13,
              guidefontsize=10, tickfontsize=9,
              ylims=(0, 100),
              legend=(pi == 1 ? :topright : false),
              legendfontsize=9,
              xticks=1:2:max_steps,
              grid=true, gridalpha=0.25,
              framestyle=:box)
    end

    plot!(plot_title="Action Distribution Over 20 Visits — Mixed Population (N=$N_SAMPLES per solver)",
          plot_titlefontsize=12)

    mkpath("figures_paper")
    savefig("figures_paper/v2_06_action_progress.png")
    savefig("figures_paper/v2_06_action_progress.pdf")
    println("Saved: figures_paper/v2_06_action_progress.png")
end

main()
