#!/usr/bin/env julia
# Combine all solver JSON results and generate comparison table + figures
# Run AFTER all solver_*.jl scripts complete
# Usage: cd julia_alzheimer_pomdp && julia --project=. combine_results.jl

using Pkg; Pkg.activate(".")
using JSON, Statistics, Plots

const N_SAMPLES = 100000
const SOLVERS = ["Random", "Expert", "MyopicPOMDP", "SARSOP"]

function main()
    println("=" ^ 70)
    println("COMBINING RESULTS & GENERATING FIGURES")
    println("=" ^ 70)
    
    all_results = Dict{String, Dict}()
    for solver in SOLVERS
        file_key = solver == "MyopicPOMDP" ? "myopic" : lowercase(solver)
        filename = "results_$(file_key)_$(N_SAMPLES).json"
        if isfile(filename)
            data = JSON.parsefile(filename)
            all_results[solver] = data
            println("Loaded: $filename")
        else
            println("WARNING: $filename not found! Run solver_$(lowercase(solver)).jl first.")
        end
    end
    
    if isempty(all_results)
        println("No results found. Please run solver scripts first.")
        return
    end
    
    # Print combined table
    println("\n" * "=" ^ 90)
    println("COMBINED RESULTS TABLE (N=$N_SAMPLES)")
    println("=" ^ 90)
    println("\nSolver        | Reward  | Acc    | CN     | MCI    | Dem    | Cost   | Steps  | Train(s)")
    println("-" ^ 90)
    
    for solver in SOLVERS
        if haskey(all_results, solver)
            d = all_results[solver]
            train_t = get(d, "train_time_sec", 0.0)
            line = "$(rpad(solver,13)) | $(lpad(get(d,"avg_reward",0.0),7)) | $(lpad(get(d,"accuracy",0.0),5))% | $(lpad(get(d,"cn_acc",0.0),5))% | $(lpad(get(d,"mci_acc",0.0),5))% | $(lpad(get(d,"dem_acc",0.0),5))% | $(lpad(get(d,"test_cost",0.0),6)) | $(lpad(get(d,"steps",0.0),6)) | $(lpad(train_t,8))"
            println(line)
        end
    end
    println("-" ^ 90)
    
    # Save combined JSON
    open("combined_results_$(N_SAMPLES).json", "w") do f
        JSON.print(f, all_results)
    end
    println("\nSaved: combined_results_$(N_SAMPLES).json")
    
    # Generate figures
    println("\nGenerating comparison figures...")
    generate_figures(all_results)
    println("All figures saved to figures_paper/")
end

function generate_figures(results)
    mkpath("figures_paper")
    solvers = [s for s in SOLVERS if haskey(results, s)]
    cmap = Dict("Random"=>"#95a5a6", "Expert"=>"#3498db", "MyopicPOMDP"=>"#2ecc71", "SARSOP"=>"#e74c3c")
    colors = [get(cmap, s, :gray) for s in solvers]
    
    # Figure 1: Reward comparison
    fig = plot(size=(1000,600), dpi=300, legend=false, xlabel="Solver", ylabel="Average Reward",
               title="Average Reward Comparison (N=$N_SAMPLES, Max Steps=20)")
    rewards = [get(results[s], "avg_reward", 0.0) for s in solvers]
    bar!(solvers, rewards, color=colors, alpha=0.85, linecolor=:black, linewidth=1.2)
    hline!([0], color=:black, linewidth=1, linestyle=:dash, label="")
    for (i, r) in enumerate(rewards)
        annotate!(i, r >= 0 ? r + 20 : r - 50, text("$(round(r,digits=1))", 10, :center))
    end
    savefig("figures_paper/v2_01_reward_comparison.png")
    savefig("figures_paper/v2_01_reward_comparison.pdf")
    println("Saved: v2_01_reward_comparison.png")
    
    # Figure 2: Accuracy comparison
    fig = plot(size=(1000,600), dpi=300, legend=false, xlabel="Solver", ylabel="Accuracy (%)",
               title="Overall Diagnostic Accuracy (N=$N_SAMPLES)", ylims=(0,100))
    accs = [get(results[s], "accuracy", 0.0) for s in solvers]
    bar!(solvers, accs, color=colors, alpha=0.85, linecolor=:black, linewidth=1.2)
    for (i, a) in enumerate(accs)
        annotate!(i, a + 2, text("$(round(a,digits=1))%", 10, :center))
    end
    savefig("figures_paper/v2_02_accuracy_comparison.png")
    savefig("figures_paper/v2_02_accuracy_comparison.pdf")
    println("Saved: v2_02_accuracy_comparison.png")
    
    # Figure 3: Per-state accuracy
    fig = plot(size=(1100,600), dpi=300)
    state_names = ["CN", "MCI", "Dementia"]
    state_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    x = 1:length(solvers)
    width = 0.25
    for (si, (sn, sc)) in enumerate(zip(state_names, state_colors))
        vals = [get(results[s], "$(lowercase(sn))_acc", 0.0) for s in solvers]
        bar!(x .+ (si-2)*width, vals, bar_width=width, color=sc, alpha=0.85, linecolor=:black, label=sn)
        for (i, v) in enumerate(vals)
            annotate!(x[i] + (si-2)*width, v + 2, text("$(round(Int, v))", 8, :center))
        end
    end
    xticks!(x, solvers); xlabel!("Solver"); ylabel!("Accuracy (%)")
    title!("Per-State Diagnostic Accuracy (N=$N_SAMPLES)")
    ylims!(0, 110)
    savefig("figures_paper/v2_03_perstate_accuracy.png")
    savefig("figures_paper/v2_03_perstate_accuracy.pdf")
    println("Saved: v2_03_perstate_accuracy.png")
    
    # Figure 4: Test cost
    fig = plot(size=(1000,600), dpi=300, legend=false, xlabel="Solver", ylabel="Average Test Cost",
               title="Average Testing Cost per Solver (N=$N_SAMPLES)")
    costs = [get(results[s], "test_cost", 0.0) for s in solvers]
    bar!(solvers, costs, color=colors, alpha=0.85, linecolor=:black, linewidth=1.2)
    for (i, c) in enumerate(costs)
        annotate!(i, c - 3, text("$(round(c,digits=1))", 10, c < -30 ? :white : :black, :center))
    end
    savefig("figures_paper/v2_04_testing_cost.png")
    savefig("figures_paper/v2_04_testing_cost.pdf")
    println("Saved: v2_04_testing_cost.png")
    
    # Figure 5: Training time comparison
    fig = plot(size=(1000,600), dpi=300, legend=false, xlabel="Solver", ylabel="Training Time (seconds, log scale)",
               title="Offline Solver Training Time Comparison", yscale=:log10)
    train_times = [max(0.01, get(results[s], "train_time_sec", 0.0)) for s in solvers]
    bar!(solvers, train_times, color=colors, alpha=0.85, linecolor=:black, linewidth=1.2)
    for (i, t) in enumerate(train_times)
        if t > 0.1
            annotate!(i, t * 1.5, text("$(round(t,digits=1))s", 9, :center))
        end
    end
    savefig("figures_paper/v2_05_training_time.png")
    savefig("figures_paper/v2_05_training_time.pdf")
    println("Saved: v2_05_training_time.png")
end

main()
