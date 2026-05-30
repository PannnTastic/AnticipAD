#!/usr/bin/env julia
# generate_paper_figures_v2.jl
# Generates all publication-quality comparison figures for the paper.
# Run: cd julia_alzheimer_pomdp && julia --project=. generate_paper_figures_v2.jl
#
# Outputs (figures_paper/ → copy to paper_v2/figures/):
#   fig1_reward.png / .pdf
#   fig2_accuracy.png / .pdf
#   fig3_perstate.png / .pdf
#   fig4_testcost.png / .pdf
#   fig5_sens_reward.png / .pdf
#   fig6_sens_acc.png / .pdf
#   fig10_regions.png / .pdf

using Pkg; Pkg.activate(".")
using JSON, Statistics, Plots
const mm = Plots.mm

const N_SAMPLES = 100000
# Worst-to-best order for left→right visual progression in bar charts
const SOLVERS = ["Random", "Expert", "MyopicPOMDP", "SARSOP"]
const COLORS = Dict(
    "Random"      => "#95a5a6",
    "Expert"      => "#3498db",
    "MyopicPOMDP" => "#2ecc71",
    "SARSOP"      => "#e74c3c",
)
const FS_TITLE  = 20   # title fontsize
const FS_AXIS   = 17   # axis label fontsize
const FS_TICK   = 15   # tick label fontsize
const FS_ANNOT  = 16   # bar annotation fontsize

# ─────────────────────────────────────────────────────────────────────────────
function load_results()
    results = Dict{String,Dict}()
    for s in SOLVERS
        key  = s == "MyopicPOMDP" ? "myopic" : lowercase(s)
        file = "results_$(key)_$(N_SAMPLES).json"
        if isfile(file)
            results[s] = JSON.parsefile(file)
            println("  Loaded $file")
        else
            println("  WARNING: $file not found")
        end
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
function fig1_reward(results, solvers, colors)
    rewards = [get(results[s], "avg_reward", 0.0) for s in solvers]
    min_r   = minimum(rewards)
    max_r   = maximum(rewards)

    fig = bar(solvers, rewards,
              color = colors, alpha = 0.88,
              linecolor = :black, linewidth = 1.2,
              legend = false,
              ylims  = (min_r * 1.08, max_r * 1.22),   # headroom above tallest bar
              size = (1100, 650), dpi = 300,
              top_margin = 6mm, left_margin = 10mm, bottom_margin = 8mm,
              xlabel = "Solver",
              ylabel = "Average Discounted Reward",
              title  = "Average Reward Comparison  (N = $N_SAMPLES)",
              titlefontsize  = FS_TITLE,
              guidefontsize  = FS_AXIS,
              tickfontsize   = FS_TICK,
              framestyle = :box,
              grid = true, gridalpha = 0.25)

    hline!([0], color = :black, linewidth = 1.2, linestyle = :dash, label = "")

    # Labels: positive bars → above bar top; negative bar → above zero line
    pad = max_r * 0.04
    for (i, r) in enumerate(rewards)
        y_pos = r >= 0 ? r + pad : pad
        annotate!(i, y_pos,
                  text("$(round(r, digits=1))", FS_ANNOT + 1, :center, :black))
    end

    savefig("figures_paper/fig1_reward.png")
    savefig("figures_paper/fig1_reward.pdf")
    println("  Saved fig1_reward")
    return fig
end

# ─────────────────────────────────────────────────────────────────────────────
function fig2_accuracy(results, solvers, colors)
    accs = [get(results[s], "accuracy", 0.0) for s in solvers]

    fig = bar(solvers, accs,
              color = colors, alpha = 0.88,
              linecolor = :black, linewidth = 1.2,
              legend = false,
              ylims = (0, 108),
              size = (1100, 650), dpi = 300,
              left_margin = 10mm, bottom_margin = 8mm,
              xlabel = "Solver",
              ylabel = "Diagnostic Accuracy (%)",
              title  = "Overall Diagnostic Accuracy  (N = $N_SAMPLES)",
              titlefontsize = FS_TITLE,
              guidefontsize = FS_AXIS,
              tickfontsize  = FS_TICK,
              framestyle = :box,
              grid = true, gridalpha = 0.25)

    for (i, a) in enumerate(accs)
        annotate!(i, a + 2.5,
                  text("$(round(a, digits=1))%", FS_ANNOT, :center, :black))
    end

    savefig("figures_paper/fig2_accuracy.png")
    savefig("figures_paper/fig2_accuracy.pdf")
    println("  Saved fig2_accuracy")
    return fig
end

# ─────────────────────────────────────────────────────────────────────────────
function fig3_perstate(results, solvers)
    # FIX: correct JSON keys are cn_acc, mci_acc, dem_acc (NOT dementia_acc)
    state_labels = ["CN", "MCI", "Dementia"]
    state_keys   = ["cn_acc", "mci_acc", "dem_acc"]
    state_colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    n  = length(solvers)
    x  = 1:n
    w  = 0.24  # bar width for 3 groups

    fig = plot(size = (1200, 650), dpi = 300,
               left_margin = 10mm, bottom_margin = 8mm,
               legend = :topright, legendfontsize = FS_TICK,
               xlabel = "Solver", ylabel = "Accuracy (%)",
               title  = "Per-State Diagnostic Accuracy  (N = $N_SAMPLES)",
               titlefontsize = FS_TITLE,
               guidefontsize = FS_AXIS,
               tickfontsize  = FS_TICK,
               ylims = (0, 108),
               framestyle = :box,
               grid = true, gridalpha = 0.25)

    for (si, (slabel, skey, scol)) in enumerate(zip(state_labels, state_keys, state_colors))
        vals = [get(results[s], skey, 0.0) for s in solvers]
        offsets = x .+ (si - 2) * w
        bar!(offsets, vals,
             bar_width = w, color = scol, alpha = 0.88,
             linecolor = :black, linewidth = 0.8,
             label = slabel)
        for (i, v) in enumerate(vals)
            annotate!(offsets[i], v + 2.2,
                      text("$(round(Int, v))", 9, :center, :black))
        end
    end

    xticks!(collect(x), solvers)

    savefig("figures_paper/fig3_perstate.png")
    savefig("figures_paper/fig3_perstate.pdf")
    println("  Saved fig3_perstate")
    return fig
end

# ─────────────────────────────────────────────────────────────────────────────
function fig4_testcost(results, solvers, colors)
    costs = [get(results[s], "test_cost", 0.0) for s in solvers]
    y_min = minimum(costs) * 1.22   # extra room below deepest bar for labels

    fig = bar(solvers, costs,
              color = colors, alpha = 0.88,
              linecolor = :black, linewidth = 1.2,
              legend = false,
              ylims = (y_min, 3),
              size = (1100, 650), dpi = 300,
              top_margin = 6mm, left_margin = 10mm, bottom_margin = 8mm,
              xlabel = "Solver",
              ylabel = "Average Diagnostic Test Cost",
              title  = "Average Test Cost per Patient  (N = $N_SAMPLES)",
              titlefontsize = FS_TITLE,
              guidefontsize = FS_AXIS,
              tickfontsize  = FS_TICK,
              framestyle = :box,
              grid = true, gridalpha = 0.25)

    for (i, c) in enumerate(costs)
        # Place label just below the bar bottom, black text on white background
        annotate!(i, c * 1.10,
                  text("$(round(c, digits=1))", FS_ANNOT, :center, :black))
    end

    savefig("figures_paper/fig4_testcost.png")
    savefig("figures_paper/fig4_testcost.pdf")
    println("  Saved fig4_testcost")
    return fig
end

# ─────────────────────────────────────────────────────────────────────────────
function fig5_fig6_sensitivity()
    sens_file = "sensitivity_analysis_results.json"
    if !isfile(sens_file)
        println("  SKIP sensitivity: $sens_file not found")
        return
    end

    results = JSON.parsefile(sens_file)

    # Full variant names (for tooltip / table) → short display names
    variants_full = [
        "Baseline", "LowDemWait(-500)", "HighDemWait(-1500)",
        "LowMCITreat(+100)", "HighMCITreat(+200)",
        "LowMalpractice(-300)", "HighMalpractice(-700)",
        "ExpensiveTests", "EqualTestCosts(-20)"
    ]
    variants_short = [
        "Baseline", "LowDemW\n(-500)", "HighDemW\n(-1500)",
        "LowMCI\n(+100)", "HighMCI\n(+200)",
        "LowMalpr\n(-300)", "HighMalpr\n(-700)",
        "ExpTests", "EqCosts\n(-20)"
    ]

    policies = ["SARSOP", "MyopicPOMDP", "Expert"]
    pol_colors  = Dict("SARSOP"=>"#e74c3c", "MyopicPOMDP"=>"#2ecc71", "Expert"=>"#3498db")
    pol_markers = Dict("SARSOP"=>:circle, "MyopicPOMDP"=>:square, "Expert"=>:diamond)

    get_val(vname, pol, key) =
        haskey(results, vname) && haskey(results[vname], pol) ?
            get(results[vname][pol], key, NaN) : NaN

    nv = length(variants_full)

    # ── Fig 5: Reward sensitivity ─────────────────────────────────────────────
    fig = plot(size = (1300, 680), dpi = 300,
               legend = :topright, legendfontsize = FS_TICK,
               xticks = (1:nv, variants_short),
               xrotation = 0,
               bottom_margin = 16mm, left_margin = 10mm,
               xlabel = "Reward Variant",
               ylabel = "Average Discounted Reward",
               title  = "SARSOP Reward Sensitivity Across Reward Variants",
               titlefontsize = FS_TITLE,
               guidefontsize = FS_AXIS,
               tickfontsize  = 10,
               framestyle = :box,
               grid = true, gridalpha = 0.25)

    for pol in policies
        vals = [get_val(v, pol, "avg_reward") for v in variants_full]
        plot!(1:nv, vals,
              color = pol_colors[pol], marker = pol_markers[pol],
              linewidth = 2.5, markersize = 7, label = pol)
    end
    hline!([0], color = :black, linewidth = 1, linestyle = :dash, label = "")

    savefig("figures_paper/fig5_sens_reward.png")
    savefig("figures_paper/fig5_sens_reward.pdf")
    println("  Saved fig5_sens_reward")

    # ── Fig 6: Accuracy sensitivity ───────────────────────────────────────────
    fig = plot(size = (1300, 680), dpi = 300,
               legend = :topright, legendfontsize = FS_TICK,
               xticks = (1:nv, variants_short),
               xrotation = 0,
               ylims = (60, 85),
               bottom_margin = 16mm, left_margin = 10mm,
               xlabel = "Reward Variant",
               ylabel = "Overall Accuracy (%)",
               title  = "SARSOP Accuracy Sensitivity Across Reward Variants",
               titlefontsize = FS_TITLE,
               guidefontsize = FS_AXIS,
               tickfontsize  = 10,
               framestyle = :box,
               grid = true, gridalpha = 0.25)

    for pol in policies
        vals = [get_val(v, pol, "accuracy") for v in variants_full]
        plot!(1:nv, vals,
              color = pol_colors[pol], marker = pol_markers[pol],
              linewidth = 2.5, markersize = 7, label = pol)
    end

    savefig("figures_paper/fig6_sens_acc.png")
    savefig("figures_paper/fig6_sens_acc.pdf")
    println("  Saved fig6_sens_acc")
end

# ─────────────────────────────────────────────────────────────────────────────
function fig10_sarsop_regions()
    grid_file = "sarsop_policy_grid.json"
    if !isfile(grid_file)
        println("  SKIP sarsop regions: $grid_file not found")
        return
    end

    data   = JSON.parsefile(grid_file)
    points = data["grid_points"]

    action_labels = ["Wait", "Test MMSE", "Test CDR", "Test APOE4", "Treat MCI", "Treat Dementia"]
    action_colors = Dict(
        1 => "#2ecc71",
        2 => "#f1c40f",
        3 => "#3498db",
        4 => "#9b59b6",
        5 => "#f39c12",
        6 => "#e74c3c",
    )

    fig = plot(size = (900, 860), dpi = 300,
               legend = :outertopright, legendfontsize = 15,
               xlabel = "p(CN)", ylabel = "p(MCI)",
               title  = "SARSOP Decision Regions on Belief Simplex",
               titlefontsize = 20,
               guidefontsize = 17,
               tickfontsize  = 15,
               xlims = (-0.02, 1.02), ylims = (-0.02, 1.02),
               framestyle = :box,
               aspect_ratio = :equal)

    for aidx in 1:6
        pts = filter(p -> p["action_idx"] == aidx, points)
        isempty(pts) && continue
        scatter!([p["p_cn"] for p in pts],
                 [p["p_mci"] for p in pts],
                 color = action_colors[aidx],
                 label = action_labels[aidx],
                 markersize = 5, markerstrokewidth = 0, alpha = 0.8)
    end

    savefig("figures_paper/fig10_regions.png")
    savefig("figures_paper/fig10_regions.pdf")
    println("  Saved fig10_regions")
end

# ─────────────────────────────────────────────────────────────────────────────
# Table 9 visualization: SARSOP sensitivity across 9 reward variants
function fig_sensitivity_table9()
    variants = [
        "Baseline", "LowDemW\n(-500)", "HighDemW\n(-1500)",
        "LowMCI\n(+100)", "HighMCI\n(+200)",
        "ExpTests\n(2×)", "EqCosts",
        "LowMalpr\n(-300)", "HighMalpr\n(-700)"
    ]
    rewards    = [68.1, 68.5, 67.8, 55.2, 78.4, 42.1, 68.0, 68.9, 62.3]
    accuracies = [78.1, 78.2, 78.0, 77.9, 78.3, 78.1, 78.1, 78.2, 78.0]
    test_costs = [29.7, 29.5, 30.1, 35.8, 28.9, 58.2, 29.8, 29.6, 30.2]  # abs values

    nv = length(variants)
    xs = 1:nv
    bar_col = "#3498db"
    base_col = "#e74c3c"  # highlight Baseline (index 1)
    cols = [i == 1 ? base_col : bar_col for i in xs]

    fig = plot(layout = (3, 1), size = (1300, 900), dpi = 300,
               left_margin = 14mm, bottom_margin = 6mm, top_margin = 2mm)

    # Panel 1: Average Reward
    bar!(xs, rewards, color = cols, alpha = 0.88,
         linecolor = :black, linewidth = 0.8, legend = false,
         subplot = 1,
         ylims = (0, maximum(rewards) * 1.22),
         ylabel = "Avg. Discounted Reward",
         title  = "SARSOP Sensitivity Across Reward Variants",
         titlefontsize = FS_TITLE,
         guidefontsize = FS_AXIS, tickfontsize = FS_TICK,
         xticks = (xs, fill("", nv)),
         framestyle = :box, grid = true, gridalpha = 0.25)
    for (i, v) in enumerate(rewards)
        annotate!(i, v + maximum(rewards)*0.035,
                  text("$(round(v, digits=1))", 9, :center, :black); subplot=1)
    end

    # Panel 2: Accuracy (narrow ylim to show variation)
    bar!(xs, accuracies, color = cols, alpha = 0.88,
         linecolor = :black, linewidth = 0.8, legend = false,
         subplot = 2,
         ylims = (76.5, 79.5),
         ylabel = "Accuracy (%)",
         guidefontsize = FS_AXIS, tickfontsize = FS_TICK,
         xticks = (xs, fill("", nv)),
         framestyle = :box, grid = true, gridalpha = 0.25)
    for (i, v) in enumerate(accuracies)
        annotate!(i, v + 0.12,
                  text("$(round(v, digits=1))%", 9, :center, :black); subplot=2)
    end

    # Panel 3: Test Cost (absolute value; larger = more testing burden)
    bar!(xs, test_costs, color = cols, alpha = 0.88,
         linecolor = :black, linewidth = 0.8, legend = false,
         subplot = 3,
         ylims = (0, maximum(test_costs) * 1.28),
         ylabel = "Avg. Test Cost (abs.)",
         guidefontsize = FS_AXIS, tickfontsize = FS_TICK,
         xticks = (xs, variants), xrotation = 0,
         framestyle = :box, grid = true, gridalpha = 0.25)
    for (i, v) in enumerate(test_costs)
        annotate!(i, v + maximum(test_costs)*0.04,
                  text("$(round(v, digits=1))", 9, :center, :black); subplot=3)
    end

    savefig("figures_paper/fig_sensitivity_sarsop.png")
    savefig("figures_paper/fig_sensitivity_sarsop.pdf")
    println("  Saved fig_sensitivity_sarsop")
end

# ─────────────────────────────────────────────────────────────────────────────
function main()
    println("=" ^ 65)
    println("GENERATING PAPER FIGURES v2")
    println("=" ^ 65)

    mkpath("figures_paper")

    println("\nLoading benchmark results...")
    results = load_results()
    isempty(results) && (println("No results — aborting."); return)

    solvers = [s for s in SOLVERS if haskey(results, s)]
    colors  = [COLORS[s] for s in solvers]

    println("\nGenerating figures...")
    fig1_reward(results, solvers, colors)
    fig2_accuracy(results, solvers, colors)
    fig3_perstate(results, solvers)
    fig4_testcost(results, solvers, colors)
    fig5_fig6_sensitivity()
    fig_sensitivity_table9()
    fig10_sarsop_regions()

    println("\nDone. Now copy figures_paper/fig*.png → paper_v2/figures/")
end

main()
