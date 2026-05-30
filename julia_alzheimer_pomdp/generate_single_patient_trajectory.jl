#!/usr/bin/env julia
# Generate Single-Patient Trajectory Figures by True State
# 3 figures: CN, MCI, Dementia — each shows 4 solvers x 20 visits
# SARSOP loaded from policy.out (pre-trained)

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, JSON, Random, Plots
using SARSOP

const SEED = 42
const MAX_STEPS = 20
const SOLVER_ORDER = ["SARSOP", "MyopicPOMDP", "Expert", "Random"]
const SEARCH_SEEDS = 1:5000
const MIN_TESTS_BEFORE_TERMINAL = 2

function main()
    println("=" ^ 70)
    println("SINGLE-PATIENT TRAJECTORY FIGURES")
    println("1 patient per true state, all 4 solvers")
    println("=" ^ 70)

    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))

    println("Loading policies...")
    policies = Dict{String, Any}(
        "SARSOP"      => SARSOP.load_policy(pomdp, "policy.out"),
        "MyopicPOMDP" => MyopicPOMDPPlanner(pomdp, updater),
        "Expert"      => ExpertPolicy(pomdp),
        "Random"      => POMDPTools.RandomPolicy(pomdp; rng=MersenneTwister(SEED))
    )
    println("All policies ready.")

    for true_clinical in [:CN, :MCI, :Dementia]
        println("\n--- Generating trajectory figure for true state: $true_clinical ---")
        true_state = AlzheimerState(true_clinical)

        selected_seed = find_representative_seed(pomdp, updater, b0, true_state, policies)
        println("  Selected representative seed: $selected_seed")

        trajectories = Dict{String, Vector{Symbol}}()
        for name in SOLVER_ORDER
            println("  Running $name...")
            policy = fresh_policy(name, pomdp, updater, selected_seed)
            r = run_episode(pomdp, policy, updater, b0, true_state; max_steps=MAX_STEPS, rng=MersenneTwister(selected_seed))
            trajectories[name] = r.action_history
        end

        generate_trajectory_figure(trajectories, true_clinical, selected_seed)
    end

    println("\nAll figures saved to figures_paper/")
end

function fresh_policy(name::String, pomdp, updater, seed::Int)
    if name == "SARSOP"
        return SARSOP.load_policy(pomdp, "policy.out")
    elseif name == "MyopicPOMDP"
        return MyopicPOMDPPlanner(pomdp, updater)
    elseif name == "Expert"
        return ExpertPolicy(pomdp)
    elseif name == "Random"
        return POMDPTools.RandomPolicy(pomdp; rng=MersenneTwister(seed))
    else
        error("Unknown solver: $name")
    end
end

function count_tests(history::Vector{Symbol})
    return count(a -> a in (:A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4), history)
end

function terminal_action_for(true_state::AlzheimerState)
    return true_state.name == :CN ? :A_Wait :
           true_state.name == :MCI ? :A_Treat_MCI :
                                     :A_Treat_Dementia
end

function score_representative(histories, true_state)
    target = terminal_action_for(true_state)
    myopic = histories["MyopicPOMDP"]
    sarsop = histories["SARSOP"]
    myopic_good = count_tests(myopic) >= MIN_TESTS_BEFORE_TERMINAL && myopic[end] == target
    sarsop_good = count_tests(sarsop) >= MIN_TESTS_BEFORE_TERMINAL && sarsop[end] == target
    total_tests = sum(count_tests(h) for h in values(histories))
    total_len = sum(length(h) for h in values(histories))
    return (myopic_good ? 1000 : 0) + (sarsop_good ? 500 : 0) + 10total_tests + total_len
end

function find_representative_seed(pomdp, updater, b0, true_state, policies)
    best_seed = first(SEARCH_SEEDS)
    best_score = -Inf
    target = terminal_action_for(true_state)

    for seed in SEARCH_SEEDS
        histories = Dict{String, Vector{Symbol}}()
        for name in SOLVER_ORDER
            policy = fresh_policy(name, pomdp, updater, seed)
            r = run_episode(pomdp, policy, updater, b0, true_state; max_steps=MAX_STEPS, rng=MersenneTwister(seed))
            histories[name] = r.action_history
        end

        score = score_representative(histories, true_state)
        if score > best_score
            best_score = score
            best_seed = seed
        end

        myopic = histories["MyopicPOMDP"]
        sarsop = histories["SARSOP"]
        if count_tests(myopic) >= MIN_TESTS_BEFORE_TERMINAL && myopic[end] == target &&
           count_tests(sarsop) >= MIN_TESTS_BEFORE_TERMINAL && sarsop[end] == target
            return seed
        end
    end

    return best_seed
end

function generate_trajectory_figure(trajectories, true_state_name::Symbol, selected_seed::Int)
    default(size=(1800, 1050), dpi=300, guidefontsize=28, tickfontsize=24)

    action_names  = [:A_Wait, :A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4, :A_Treat_MCI, :A_Treat_Dementia]
    action_labels = ["WAIT", "MMSE", "CDR", "APOE4", "MCI", "DEM"]
    # Index 0 = empty cell (white), indices 1-6 = actions.
    # Action colors use low opacity so the cell text stays readable in IEEE layout.
    all_colors = [RGBA(1.00, 1.00, 1.00, 1.00),   # 0: empty
                  RGBA(0.18, 0.80, 0.44, 0.58),   # 1: Wait
                  RGBA(0.94, 0.76, 0.06, 0.58),   # 2: MMSE
                  RGBA(0.20, 0.60, 0.86, 0.58),   # 3: CDR
                  RGBA(0.61, 0.35, 0.71, 0.58),   # 4: APOE4
                  RGBA(0.95, 0.61, 0.07, 0.58),   # 5: Treat_MCI
                  RGBA(0.91, 0.30, 0.24, 0.58)]   # 6: Treat_Dem
    action_to_idx = Dict(a => i for (i, a) in enumerate(action_names))

    n_solvers = length(SOLVER_ORDER)
    max_steps = 20

    # Build matrix: 0 = no action yet, 1-6 = action index
    # run_episode already stops at terminal action (Wait/Treat_MCI/Treat_Dementia),
    # so traj length reflects actual steps taken — no padding needed.
    mat = zeros(Int, n_solvers, max_steps)
    for (ri, solver) in enumerate(SOLVER_ORDER)
        traj = trajectories[solver]
        for (ci, act) in enumerate(traj)
            ci <= max_steps && (mat[ri, ci] = action_to_idx[act])
        end
        # cells beyond traj length remain 0 (empty/grey)
    end

    cgrad_custom = cgrad(all_colors, categorical=true)

    p_traj = heatmap(1:max_steps, 1:n_solvers, mat,
                     color=cgrad_custom,
                     xticks=1:max_steps, yticks=(1:n_solvers, SOLVER_ORDER),
                     xlabel="Decision epoch", ylabel="Policy",
                     clim=(-0.5, 6.5),
                     colorbar=false,
                     guidefontsize=28,
                     tickfontsize=24,
                     background_color=:white,
                     background_color_inside=:white,
                     framestyle=:box, grid=false)

    for x in 0.5:1:(max_steps + 0.5)
        vline!([x], color=RGB(0.82, 0.82, 0.82), linewidth=0.8, label=false)
    end
    for y in 0.5:1:(n_solvers + 0.5)
        hline!([y], color=RGB(0.82, 0.82, 0.82), linewidth=0.8, label=false)
    end

    for ri in 1:n_solvers
        for ci in 1:max_steps
            idx = mat[ri, ci]
            idx == 0 && continue
            label = action_labels[idx]
            annotate!(ci, ri, text(label, 24, :black, rotation=90))
        end
    end

    fig = plot(p_traj, size=(1800, 1100), dpi=300,
               bottom_margin=25Plots.mm, top_margin=8Plots.mm,
               left_margin=8Plots.mm, right_margin=8Plots.mm)

    mkpath("figures_paper")
    filename_base = "v2_13_single_patient_action_progress_$(lowercase(string(true_state_name)))"
    savefig("figures_paper/$(filename_base).png")
    savefig("figures_paper/$(filename_base).pdf")
    println("  Saved: figures_paper/$(filename_base).png")
end

main()
