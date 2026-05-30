#!/usr/bin/env julia
# Generate Single-Patient Trajectory for the LONGEST episode per true state
# This demonstrates "test several times until confident, then terminal action"

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, JSON, Random, Plots
using SARSOP

const SEED = 42
const MAX_STEPS = 20
const SOLVER_ORDER = ["SARSOP", "MyopicPOMDP", "Expert", "Random"]

function main()
    println("=" ^ 70)
    println("LONGEST TRAJECTORY FIGURES")
    println("Finding patients that require the most tests before terminal action")
    println("=" ^ 70)
    
    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    rng = MersenneTwister(SEED)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    
    policies = Dict{String, Any}(
        "SARSOP"      => SARSOP.load_policy(pomdp, "policy.out"),
        "MyopicPOMDP" => MyopicPOMDPPlanner(pomdp, updater),
        "Expert"      => ExpertPolicy(pomdp),
        "Random"      => POMDPTools.RandomPolicy(pomdp; rng=rng)
    )
    
    for true_clinical in [:CN, :MCI, :Dementia]
        println("\n--- Finding longest trajectory for true state: $true_clinical ---")
        true_state = AlzheimerState(true_clinical)
        
        # Search for patient with longest episode among first 500
        best_results = Dict{String, NamedTuple}()
        for name in SOLVER_ORDER
            best_len = 0
            best_traj = nothing
            search_rng = MersenneTwister(SEED)
            for patient_id in 1:500
                # Fresh policy state for each patient
                pol = name == "SARSOP" ? SARSOP.load_policy(pomdp, "policy.out") :
                      name == "MyopicPOMDP" ? MyopicPOMDPPlanner(pomdp, updater) :
                      name == "Expert" ? ExpertPolicy(pomdp) :
                      POMDPTools.RandomPolicy(pomdp; rng=search_rng)
                
                r = run_episode(pomdp, pol, updater, b0, true_state; max_steps=MAX_STEPS, rng=search_rng)
                if length(r.action_history) > best_len
                    best_len = length(r.action_history)
                    best_traj = r
                end
            end
            best_results[name] = best_traj
            println("  $name: longest episode = $best_len visits")
        end
        
        generate_trajectory_figure(best_results, true_clinical)
    end
    
    println("\nAll figures saved to figures_paper/")
end

function generate_trajectory_figure(results, true_state_name::Symbol)
    action_names  = [:A_Wait, :A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4, :A_Treat_MCI, :A_Treat_Dementia]
    action_labels = ["Wait", "MMSE", "CDR", "APOE4", "Tr.MCI", "Tr.Dem"]
    all_colors = [RGB(0.88, 0.88, 0.88),
                  RGB(0.18, 0.80, 0.44),
                  RGB(0.94, 0.76, 0.06),
                  RGB(0.20, 0.60, 0.86),
                  RGB(0.61, 0.35, 0.71),
                  RGB(0.95, 0.61, 0.07),
                  RGB(0.91, 0.30, 0.24)]
    action_to_idx = Dict(a => i for (i, a) in enumerate(action_names))
    
    n_solvers = length(SOLVER_ORDER)
    max_steps = 20
    
    mat = zeros(Int, n_solvers, max_steps)
    for (ri, solver) in enumerate(SOLVER_ORDER)
        traj = results[solver].action_history
        for (ci, act) in enumerate(traj)
            ci <= max_steps && (mat[ri, ci] = action_to_idx[act])
        end
    end
    
    cgrad_custom = cgrad(all_colors, categorical=true)
    
    state_title = true_state_name == :CN ? "CN (Cognitively Normal)" : 
                  (true_state_name == :MCI ? "MCI (Mild Cognitive Impairment)" : "Dementia")
    correct_action = true_state_name == :CN ? "Wait" : (true_state_name == :MCI ? "Treat MCI" : "Treat Dementia")
    
    fig = heatmap(1:max_steps, 1:n_solvers, mat,
                  color=cgrad_custom,
                  xticks=1:max_steps, yticks=(1:n_solvers, SOLVER_ORDER),
                  xlabel="Visit Number (t)", ylabel="Solver",
                  clim=(-0.5, 6.5),
                  colorbar_ticks=(1:6, action_labels),
                  title="Longest Episode Trajectory — True State: $state_title | Correct: $correct_action",
                  size=(1400, 420), dpi=300,
                  framestyle=:box, grid=false)
    
    for x in 0.5:1:(max_steps+0.5)
        vline!([x], color=:white, linewidth=1.5, label=false)
    end
    for y in 0.5:1:(n_solvers+0.5)
        hline!([y], color=:white, linewidth=1.5, label=false)
    end
    
    mkpath("figures_paper")
    filename_base = "v2_11_longest_trajectory_$(lowercase(string(true_state_name)))"
    savefig("figures_paper/$(filename_base).png")
    savefig("figures_paper/$(filename_base).pdf")
    println("  Saved: figures_paper/$(filename_base).png")
end

main()
