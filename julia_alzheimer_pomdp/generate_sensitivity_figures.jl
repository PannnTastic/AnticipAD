# ============================================================
# Generate Sensitivity Analysis Figures
# ============================================================

using Plots, JSON, Statistics

results = JSON.parsefile("sensitivity_analysis_results.json")

variants = ["Baseline", "LowDemWait(-500)", "HighDemWait(-1500)", 
            "LowMCITreat(+100)", "HighMCITreat(+200)",
            "LowMalpractice(-300)", "HighMalpractice(-700)",
            "ExpensiveTests", "EqualTestCosts(-20)"]

policies = ["SARSOP", "MyopicPOMDP", "Expert"]
colors = Dict("SARSOP"=>"#e74c3c", "MyopicPOMDP"=>"#2ecc71", "Expert"=>"#3498db")
markers = Dict("SARSOP"=>:circle, "MyopicPOMDP"=>:square, "Expert"=>:diamond)

function get_metric(variant, policy, metric)
    haskey(results, variant) && haskey(results[variant], policy) && results[variant][policy] !== nothing ? results[variant][policy][metric] : NaN
end

mkpath("figures_paper")

# --- Figure 1: Reward across variants ---
fig = plot(size=(1100,600), dpi=300, legend=:outertopright,
           xlabel="Reward Variant", ylabel="Average Reward",
           title="Sensitivity of Average Reward to Reward Function Variations",
           xticks=(1:length(variants), variants), xrotation=30)
for pol in policies
    vals = [get_metric(v, pol, "avg_reward") for v in variants]
    plot!(1:length(variants), vals, color=colors[pol], marker=markers[pol], 
          linewidth=2.5, markersize=7, label=pol)
end
hline!([0], color=:black, linewidth=1, linestyle=:dash, label="")
savefig("figures_paper/12_sensitivity_reward.png")
savefig("figures_paper/12_sensitivity_reward.pdf")
println("Saved: 12_sensitivity_reward.png")

# --- Figure 2: Accuracy across variants ---
fig = plot(size=(1100,600), dpi=300, legend=:outertopright,
           xlabel="Reward Variant", ylabel="Overall Accuracy (%)",
           title="Sensitivity of Diagnostic Accuracy to Reward Function Variations",
           xticks=(1:length(variants), variants), xrotation=30, ylims=(60,85))
for pol in policies
    vals = [get_metric(v, pol, "accuracy") for v in variants]
    plot!(1:length(variants), vals, color=colors[pol], marker=markers[pol],
          linewidth=2.5, markersize=7, label=pol)
end
savefig("figures_paper/13_sensitivity_accuracy.png")
savefig("figures_paper/13_sensitivity_accuracy.pdf")
println("Saved: 13_sensitivity_accuracy.png")

# --- Figure 3: Test Cost across variants ---
fig = plot(size=(1100,600), dpi=300, legend=:outertopright,
           xlabel="Reward Variant", ylabel="Average Test Cost",
           title="Sensitivity of Testing Cost to Reward Function Variations",
           xticks=(1:length(variants), variants), xrotation=30)
for pol in policies
    vals = [get_metric(v, pol, "test_cost") for v in variants]
    plot!(1:length(variants), vals, color=colors[pol], marker=markers[pol],
          linewidth=2.5, markersize=7, label=pol)
end
savefig("figures_paper/14_sensitivity_testcost.png")
savefig("figures_paper/14_sensitivity_testcost.pdf")
println("Saved: 14_sensitivity_testcost.png")

# --- Figure 4: MCI Accuracy across variants ---
fig = plot(size=(1100,600), dpi=300, legend=:outertopright,
           xlabel="Reward Variant", ylabel="MCI Accuracy (%)",
           title="Sensitivity of MCI Detection to Reward Function Variations",
           xticks=(1:length(variants), variants), xrotation=30, ylims=(55,90))
for pol in policies
    vals = [get_metric(v, pol, "mci_acc") for v in variants]
    plot!(1:length(variants), vals, color=colors[pol], marker=markers[pol],
          linewidth=2.5, markersize=7, label=pol)
end
savefig("figures_paper/15_sensitivity_mci_acc.png")
savefig("figures_paper/15_sensitivity_mci_acc.pdf")
println("Saved: 15_sensitivity_mci_acc.png")

println("All sensitivity figures saved to figures_paper/")
