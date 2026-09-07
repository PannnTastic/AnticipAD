function plot_initial_belief()
    fig = plot(size=(800, 600), dpi=300)
    states = ["CN\n(Cognitively\nNormal)", "MCI\n(Mild Cognitive\nImpairment)", "Dementia\n(Alzheimer)"]
    probs = [0.385189, 0.417041, 0.197770]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    bar!(states, probs, color=colors, alpha=0.8, linecolor=:black, linewidth=1.5,
         label="", ylabel="Probability", title="Initial Belief Distribution (b₀)\nADNI Retained Visit Records (N=12,464)")

    for (i, pr) in enumerate(probs)
        annotate!(i, pr + 0.02, text("$(round(pr*100,digits=1))%\n($(round(Int,pr*12464)) visit records)", 10, :center))
    end
    ylims!(0, 0.5)

    savefig("figures_paper/01_initial_belief.png")
    savefig("figures_paper/01_initial_belief.pdf")
    println("Saved: 01_initial_belief.png")
end

function plot_transition_matrix()
    mat = [0.929 0.068 0.003;
           0.000 0.892 0.108;
           0.000 0.000 1.000]
    states = ["CN", "MCI", "Dementia"]

    fig = heatmap(states, states, mat, color=:Blues, clim=(0,1),
                  xlabel="Next State (s')", ylabel="Current State (s)",
                  title="Transition Probability Matrix T(s'|s)\nConsecutive ADNI Visits with Structural Constraints",
                  size=(900, 700), dpi=300)

    for i in 1:3, j in 1:3
        color = mat[i,j] > 0.5 ? :white : :black
        annotate!(j, i, text("$(round(mat[i,j],digits=3))", 14, color, :center))
    end

    savefig("figures_paper/02_transition_matrix.png")
    savefig("figures_paper/02_transition_matrix.pdf")
    println("Saved: 02_transition_matrix.png")
end

function plot_reward_function()
    rewards = [30 -5 -20 -50 -50 -500;
               -10 -5 -20 -50 150 -100;
               -1000 -5 -20 -50 -50 300]
    actions = ["Wait", "Test\nMMSE", "Test\nCDR", "Test\nAPOE4", "Treat\nMCI", "Treat\nDementia"]
    states = ["CN\n(Healthy)", "MCI", "Dementia\n(Severe)"]

    fig = heatmap(actions, states, rewards, color=:RdYlGn, clim=(-1000, 300),
                  xlabel="Action (a)", ylabel="State (s)",
                  title="Reward Function R(s, a)\nGreen=Positive, Red=Negative (Malpractice Penalty: -500)",
                  size=(1000, 700), dpi=300)

    for i in 1:3, j in 1:6
        val = rewards[i,j]
        c = (val < -100 || val > 100) ? :white : :black
        annotate!(j, i, text("$(val)", 11, c, :center))
    end

    savefig("figures_paper/04_reward_function.png")
    savefig("figures_paper/04_reward_function.pdf")
    println("Saved: 04_reward_function.png")
end

function plot_solver_comparisons(results::Dict{String, Vector{NamedTuple}})
    policies = sort(collect(keys(results)))
    n = length(policies)

    cmap = Dict(
        "Random" => "#95a5a6",
        "Expert" => "#3498db",
        "MyopicPOMDP" => "#2ecc71",
        "POMCP" => "#e74c3c",
        "DESPOT" => "#e67e22",
        "PBVI" => "#9b59b6",
        "MCTS" => "#f1c40f"
    )
    colors = [get(cmap, p, :gray) for p in policies]

    # 1. Reward comparison
    fig = plot(size=(1000, 600), dpi=300, legend=false)
    rewards = [mean([d.total_reward for d in results[p]]) for p in policies]
    stds = [std([d.total_reward for d in results[p]]) for p in policies]

    bar!(policies, rewards, yerror=stds, color=colors, alpha=0.85, linecolor=:black, linewidth=1.2,
         ylabel="Average Reward (utility points)", title="Average Simulated Reward Across Policies")
    hline!([0], color=:black, linewidth=1, label="")
    for (i, r) in enumerate(rewards)
        annotate!(i, r >= 0 ? r + 30 : r - 70, text("$(round(r,digits=1))", 10, :center))
    end
    savefig("figures_paper/07_solver_reward_comparison.png")
    savefig("figures_paper/07_solver_reward_comparison.pdf")
    println("Saved: 07_solver_reward_comparison.png")

    # 2. Accuracy comparison
    fig = plot(size=(1000, 600), dpi=300, legend=false)
    accs = [mean([d.correct for d in results[p]]) * 100 for p in policies]
    bar!(policies, accs, color=colors, alpha=0.85, linecolor=:black, linewidth=1.2,
         ylabel="Intake-Stage Agreement (%)", title="Overall Intake-Stage Agreement", ylims=(0,100))
    for (i, a) in enumerate(accs)
        annotate!(i, a + 2, text("$(round(a,digits=1))%", 10, :center))
    end
    savefig("figures_paper/08_solver_accuracy_comparison.png")
    savefig("figures_paper/08_solver_accuracy_comparison.pdf")
    println("Saved: 08_solver_accuracy_comparison.png")

    # 3. Per-state accuracy (grouped bar)
    fig = plot(size=(1100, 600), dpi=300)
    state_names = ["CN", "MCI", "Dementia"]
    state_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    x = 1:length(policies)
    width = 0.25

    for (si, (sn, sc)) in enumerate(zip(state_names, state_colors))
        vals = [begin
            d = [r.correct for r in results[p] if r.true_state == Symbol(sn)]
            isempty(d) ? 0.0 : mean(d) * 100
        end for p in policies]
        bar!(x .+ (si-2)*width, vals, bar_width=width, color=sc, alpha=0.85, linecolor=:black, label=sn)
        for (i, v) in enumerate(vals)
            annotate!(x[i] + (si-2)*width, v + 2, text("$(round(Int, v))", 8, :center))
        end
    end

    xticks!(x, policies)
    xlabel!("Policy")
    ylabel!("Intake-Stage Agreement (%)")
    title!("Per-State Intake-Stage Agreement")
    ylims!(0, 110)
    savefig("figures_paper/09_solver_per_state_accuracy.png")
    savefig("figures_paper/09_solver_per_state_accuracy.pdf")
    println("Saved: 09_solver_per_state_accuracy.png")

    # 4. Testing cost
    fig = plot(size=(1000, 600), dpi=300, legend=false)
    costs = [mean([d.test_cost for d in results[p]]) for p in policies]
    std_costs = [std([d.test_cost for d in results[p]]) for p in policies]
    bar!(policies, costs, yerror=std_costs, color=colors, alpha=0.85, linecolor=:black, linewidth=1.2,
         ylabel="Average Testing Cost", title="Average Testing Cost per Policy")
    for (i, c) in enumerate(costs)
        annotate!(i, c - 3, text("$(round(c,digits=1))", 10, c < -30 ? :white : :black, :center))
    end
    savefig("figures_paper/10_solver_testing_cost.png")
    savefig("figures_paper/10_solver_testing_cost.pdf")
    println("Saved: 10_solver_testing_cost.png")
end

function generate_all_figures(results::Dict{String, Vector{NamedTuple}})
    println("\n" * "=" ^ 60)
    println("GENERATING FIGURES")
    println("=" ^ 60)
    mkpath("figures_paper")
    plot_initial_belief()
    plot_transition_matrix()
    plot_reward_function()
    plot_solver_comparisons(results)
    println("All figures saved to figures_paper/")
end
