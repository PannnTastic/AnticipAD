# ============================================================
# SARSOP Policy Extraction for Alzheimer POMDP
# Extracts interpretable decision rules and flowchart from
# the SARSOP alpha-vector policy.
# ============================================================

using POMDPs, POMDPTools, SARSOP, Random

include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP
import .AlzheimerPOMDP: isterminal_action, AlzheimerState, AlzheimerAction

# Mapping
const STATE_NAMES = [:CN, :MCI, :Dementia]
const ACTION_NAMES = [:A_Wait, :A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4, :A_Treat_MCI, :A_Treat_Dementia]
const ACTION_LABELS = ["Wait", "Test MMSE", "Test CDR", "Test APOE4", "Treat MCI", "Treat Dementia"]

# Load POMDP and SARSOP policy
pomdp = AlzheimerPOMDPProblem()
policy = SARSOP.load_policy(pomdp, "policy.out")
updater = DiscreteUpdater(pomdp)
b0 = initialize_belief(updater, initialstate(pomdp))

println("="^70)
println("SARSOP POLICY EXTRACTION")
println("="^70)
println("Number of alpha vectors: $(length(policy.alphas))")
println()

# ============================================================
# 1. Evaluate on fine grid over belief simplex
# ============================================================
println("Building belief grid and evaluating policy...")

# For 3 states, belief simplex is 2D: [p_CN, p_MCI, 1-p_CN-p_MCI]
# We sample p_CN and p_MCI with fine resolution
resolution = 0.02  # 0.02 step = ~2600 valid points
grid_results = []

for p_cn in 0.0:resolution:1.0
    for p_mci in 0.0:resolution:(1.0 - p_cn)
        p_dem = 1.0 - p_cn - p_mci
        # small numerical errors might make p_dem slightly negative
        p_dem < -1e-8 && continue
        p_dem = max(0.0, p_dem)
        
        # Normalize to ensure exact sum=1
        total = p_cn + p_mci + p_dem
        b = DiscreteBelief(pomdp, [p_cn/total, p_mci/total, p_dem/total])
        
        a = action(policy, b)
        val = value(policy, b)
        
        push!(grid_results, (
            p_cn = p_cn/total,
            p_mci = p_mci/total,
            p_dem = p_dem/total,
            action = a.name,
            action_idx = actionindex(pomdp, a),
            value = val
        ))
    end
end

println("Grid points evaluated: $(length(grid_results))")
println()

# ============================================================
# 2. Decision Region Summary
# ============================================================
println("="^70)
println("DECISION REGION SUMMARY")
println("="^70)

for (aidx, aname) in enumerate(ACTION_NAMES)
    alabel = ACTION_LABELS[aidx]
    points = filter(r -> r.action_idx == aidx, grid_results)
    n = length(points)
    pct = 100 * n / length(grid_results)
    println("$(lpad(alabel, 14)): $(lpad(n, 5)) points ($(round(pct, digits=1))%)")
end
println()

# ============================================================
# 3. Extract Dominant Terminal Decision Boundaries
# ============================================================
println("="^70)
println("TERMINAL ACTION THRESHOLDS (extracted from grid)")
println("="^70)

# For terminal actions, find regions where they dominate
term_actions = [:A_Wait, :A_Treat_MCI, :A_Treat_Dementia]
term_labels = ["Wait (CN)", "Treat MCI", "Treat Dementia"]
term_indices = [1, 5, 6]  # action indices

for (tidx, tname, tlabel) in zip(term_indices, term_actions, term_labels)
    pts = filter(r -> r.action_idx == tidx, grid_results)
    if isempty(pts)
        println("$tlabel: NO dominant region found")
        continue
    end
    
    # Find approximate convex hull / min-max of each coordinate
    min_cn = minimum(p.p_cn for p in pts)
    max_cn = maximum(p.p_cn for p in pts)
    min_mci = minimum(p.p_mci for p in pts)
    max_mci = maximum(p.p_mci for p in pts)
    min_dem = minimum(p.p_dem for p in pts)
    max_dem = maximum(p.p_dem for p in pts)
    
    println("$tlabel region:")
    println("  p_CN    ∈ [$(round(min_cn,digits=2)), $(round(max_cn,digits=2))]")
    println("  p_MCI   ∈ [$(round(min_mci,digits=2)), $(round(max_mci,digits=2))]")
    println("  p_Dem   ∈ [$(round(min_dem,digits=2)), $(round(max_dem,digits=2))]")
    
    # Find tightest approximate rule: find points where only ONE belief component is high
    # This gives us interpretable threshold-like rules
    high_conf = filter(p -> 
        (tname == :A_Wait && p.p_cn >= 0.7) ||
        (tname == :A_Treat_MCI && p.p_mci >= 0.6) ||
        (tname == :A_Treat_Dementia && p.p_dem >= 0.5)
    , pts)
    
    if !isempty(high_conf)
        println("  High-confidence examples:")
        for p in high_conf[1:min(3, length(high_conf))]
            println("    b=[$(round(p.p_cn,digits=2)), $(round(p.p_mci,digits=2)), $(round(p.p_dem,digits=2))] → $tlabel")
        end
    end
    println()
end

# ============================================================
# 4. Test Sequences: What tests does SARSOP prefer?
# ============================================================
println("="^70)
println("TEST ACTION REGIONS")
println("="^70)

test_actions = [:A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4]
test_labels = ["Test MMSE", "Test CDR", "Test APOE4"]
test_indices = [2, 3, 4]

for (tidx, tname, tlabel) in zip(test_indices, test_actions, test_labels)
    pts = filter(r -> r.action_idx == tidx, grid_results)
    if isempty(pts)
        println("$tlabel: NO region")
        continue
    end
    
    n = length(pts)
    avg_cn = mean(p.p_cn for p in pts)
    avg_mci = mean(p.p_mci for p in pts)
    avg_dem = mean(p.p_dem for p in pts)
    
    println("$tlabel: $(n) points ($(round(100*n/length(grid_results),digits=1))%)")
    println("  Avg belief in region: CN=$(round(avg_cn,digits=2)), MCI=$(round(avg_mci,digits=2)), Dem=$(round(avg_dem,digits=2))")
    
    # What would be chosen if belief shifts after this test?
    println()
end

# ============================================================
# 5. Extract Clinical Flowchart Rules (Approximate)
# ============================================================
println("="^70)
println("APPROXIMATE CLINICAL FLOWCHART (from grid analysis)")
println("="^70)

# Greedy extraction: find linear separations
function find_approximate_rule(grid, target_action_idx; step=0.05)
    # Try threshold rules on each belief component
    best_acc = 0.0
    best_rule = "none"
    
    # Rule type 1: single threshold
    for thresh in 0.1:step:0.95
        # p_dem >= thresh -> action?
        pred = [p.p_dem >= thresh ? target_action_idx : -1 for p in grid]
        actual = [p.action_idx for p in grid]
        acc = mean(pred[i] == actual[i] || pred[i] == -1 for i in 1:length(grid))
        tp = sum(pred[i] == actual[i] == target_action_idx for i in 1:length(grid))
        fn = sum(pred[i] == -1 && actual[i] == target_action_idx for i in 1:length(grid))
        precision = tp / max(tp + sum(pred[i] == target_action_idx && actual[i] != target_action_idx for i in 1:length(grid)), 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        if f1 > best_acc
            best_acc = f1
            best_rule = "p_Dem >= $thresh → $(ACTION_LABELS[target_action_idx]) (F1=$(round(f1,digits=2)))"
        end
        
        # p_mci >= thresh
        pred = [p.p_mci >= thresh ? target_action_idx : -1 for p in grid]
        tp = sum(pred[i] == actual[i] == target_action_idx for i in 1:length(grid))
        fn = sum(pred[i] == -1 && actual[i] == target_action_idx for i in 1:length(grid))
        fp = sum(pred[i] == target_action_idx && actual[i] != target_action_idx for i in 1:length(grid))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        if f1 > best_acc
            best_acc = f1
            best_rule = "p_MCI >= $thresh → $(ACTION_LABELS[target_action_idx]) (F1=$(round(f1,digits=2)))"
        end
        
        # p_cn >= thresh
        pred = [p.p_cn >= thresh ? target_action_idx : -1 for p in grid]
        tp = sum(pred[i] == actual[i] == target_action_idx for i in 1:length(grid))
        fn = sum(pred[i] == -1 && actual[i] == target_action_idx for i in 1:length(grid))
        fp = sum(pred[i] == target_action_idx && actual[i] != target_action_idx for i in 1:length(grid))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        if f1 > best_acc
            best_acc = f1
            best_rule = "p_CN >= $thresh → $(ACTION_LABELS[target_action_idx]) (F1=$(round(f1,digits=2)))"
        end
    end
    
    return best_rule, best_acc
end

println("\nBest approximate single-threshold rules:")
for (aidx, alabel) in zip([1,5,6,2,3,4], ACTION_LABELS)
    rule, f1 = find_approximate_rule(grid_results, aidx)
    println("  $(lpad(alabel,12)): $rule")
end

# ============================================================
# 6. Simulate typical trajectories to show test sequences
# ============================================================
println()
println("="^70)
println("TYPICAL SARSOP DECISION TRAJECTORIES")
println("="^70)

function simulate_trajectory(pomdp, policy, updater, b0, true_state_name; max_steps=5)
    s = AlzheimerState(true_state_name)
    b = b0
    traj = []
    
    for step in 1:max_steps
        a = action(policy, b)
        push!(traj, (step=step, belief=[round(pdf(b,AlzheimerState(:CN)),digits=2), round(pdf(b,AlzheimerState(:MCI)),digits=2), round(pdf(b,AlzheimerState(:Dementia)),digits=2)], action=a.name))
        
        if isterminal_action(a)
            break
        end
        
        sp = rand(Random.GLOBAL_RNG, transition(pomdp, s, a))
        o = rand(Random.GLOBAL_RNG, observation(pomdp, a, sp))
        b = update(updater, b, a, o)
        s = sp
    end
    return traj
end

rng_fixed = Random.MersenneTwister(999)
for sname in [:CN, :MCI, :Dementia]
    println("\nTrue State = $sname (simulated observations)")
    traj = simulate_trajectory(pomdp, policy, updater, b0, sname)
    for t in traj
        println("  Step $(t.step): b=$(t.belief) → $(t.action)")
    end
end

# ============================================================
# 7. Save grid results for visualization
# ============================================================
println()
println("="^70)
println("SAVING RESULTS")
println("="^70)

using JSON

json_data = Dict(
    "grid_points" => [Dict("p_cn"=>r.p_cn, "p_mci"=>r.p_mci, "p_dem"=>r.p_dem, 
                           "action"=>string(r.action), "action_idx"=>r.action_idx, "value"=>r.value) for r in grid_results],
    "action_names" => ACTION_NAMES,
    "action_labels" => ACTION_LABELS
)

open("sarsop_policy_grid.json", "w") do f
    JSON.print(f, json_data)
end
println("Saved: sarsop_policy_grid.json")

# ============================================================
# 8. Generate decision region plot (if Plots available)
# ============================================================
try
    using Plots
    println("\nGenerating decision region plot...")
    
    # Simplex plot: p_cn vs p_mci, color by action
    action_colors = Dict(
        1 => "#2ecc71",   # Wait - green
        2 => "#f1c40f",   # Test MMSE - yellow
        3 => "#3498db",   # Test CDR - blue
        4 => "#9b59b6",   # Test APOE4 - purple
        5 => "#f39c12",   # Treat MCI - orange
        6 => "#e74c3c"    # Treat Dementia - red
    )
    
    fig = plot(size=(900,800), dpi=300, legend=:outertopright,
               xlabel="p_CN", ylabel="p_MCI",
               title="SARSOP Decision Regions over Belief Simplex")
    
    for aidx in 1:6
        pts = filter(r -> r.action_idx == aidx, grid_results)
        if !isempty(pts)
            scatter!([p.p_cn for p in pts], [p.p_mci for p in pts],
                     color=action_colors[aidx], label=ACTION_LABELS[aidx],
                     markersize=2.5, markerstrokewidth=0, alpha=0.7)
        end
    end
    
    # Draw simplex boundary
    plot!([0,1,0,0], [0,0,1,0], color=:black, linewidth=2, label="")
    
    savefig("figures_paper/11_sarsop_decision_regions.png")
    savefig("figures_paper/11_sarsop_decision_regions.pdf")
    println("Saved: figures_paper/11_sarsop_decision_regions.png")
catch e
    println("Plot generation skipped (Plots may need GR backend): $e")
end

println()
println("="^70)
println("EXTRACTION COMPLETE")
println("="^70)
