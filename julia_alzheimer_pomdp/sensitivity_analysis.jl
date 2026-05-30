# ============================================================
# Sensitivity Analysis: Reward Function Variations
# Tests robustness of SARSOP and MyopicPOMDP policies under
# different clinical reward assumptions.
# ============================================================

using POMDPs, POMDPTools, SARSOP, Random, Statistics, JSON

include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP

const N_SAMPLES = 1000
const SEED = 42
const MAX_STEPS = 5

# ============================================================
# Reward variants
# ============================================================
function make_baseline_rewards()
    return Dict(
        :CN => Dict(:A_Wait => 30, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => -50, :A_Treat_Dementia => -500),
        :MCI => Dict(:A_Wait => -10, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => 150, :A_Treat_Dementia => -100),
        :Dementia => Dict(:A_Wait => -1000, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => -50, :A_Treat_Dementia => 300)
    )
end

function copy_rewards(base::Dict)
    return Dict(s => Dict(a => v for (a,v) in av) for (s,av) in base)
end

reward_variants = Dict{String, Dict{Symbol, Dict{Symbol, Float64}}}()

# 1. Baseline
reward_variants["Baseline"] = make_baseline_rewards()

# 2. Lower Dementia Wait Penalty (less severe consequence of waiting)
r = copy_rewards(make_baseline_rewards())
r[:Dementia][:A_Wait] = -500
reward_variants["LowDemWait(-500)"] = r

# 3. Higher Dementia Wait Penalty (more severe)
r = copy_rewards(make_baseline_rewards())
r[:Dementia][:A_Wait] = -1500
reward_variants["HighDemWait(-1500)"] = r

# 4. Lower MCI Treatment Reward
r = copy_rewards(make_baseline_rewards())
r[:MCI][:A_Treat_MCI] = 100
reward_variants["LowMCITreat(+100)"] = r

# 5. Higher MCI Treatment Reward
r = copy_rewards(make_baseline_rewards())
r[:MCI][:A_Treat_MCI] = 200
reward_variants["HighMCITreat(+200)"] = r

# 6. Lower Malpractice Penalty (Treat Dementia on CN)
r = copy_rewards(make_baseline_rewards())
r[:CN][:A_Treat_Dementia] = -300
reward_variants["LowMalpractice(-300)"] = r

# 7. Higher Malpractice Penalty
r = copy_rewards(make_baseline_rewards())
r[:CN][:A_Treat_Dementia] = -700
reward_variants["HighMalpractice(-700)"] = r

# 8. More Expensive Tests (double cost)
r = copy_rewards(make_baseline_rewards())
for s in [:CN, :MCI, :Dementia]
    r[s][:A_Test_MMSE] = -10
    r[s][:A_Test_CDR] = -40
    r[s][:A_Test_APOE4] = -100
end
reward_variants["ExpensiveTests"] = r

# 9. Equal Test Costs (all tests cost -20)
r = copy_rewards(make_baseline_rewards())
for s in [:CN, :MCI, :Dementia]
    r[s][:A_Test_MMSE] = -20
    r[s][:A_Test_CDR] = -20
    r[s][:A_Test_APOE4] = -20
end
reward_variants["EqualTestCosts(-20)"] = r

println("="^70)
println("SENSITIVITY ANALYSIS: $(length(reward_variants)) Reward Variants")
println("N=$(N_SAMPLES) patients per variant, seed=$(SEED)")
println("="^70)

# ============================================================
# Evaluation function
# ============================================================
function evaluate_policy(pomdp, policy, updater, b0, true_states; rng)
    results = NamedTuple[]
    for s in true_states
        r = run_episode(pomdp, policy, updater, b0, s; max_steps=MAX_STEPS, rng=rng)
        push!(results, r)
    end
    return results
end

function summarize(data)
    rewards = [d.total_reward for d in data]
    acc = mean([d.correct for d in data]) * 100
    cost = mean([d.test_cost for d in data])
    steps = mean([d.steps for d in data])
    
    cn_data = [d.correct for d in data if d.true_state == :CN]
    mci_data = [d.correct for d in data if d.true_state == :MCI]
    dem_data = [d.correct for d in data if d.true_state == :Dementia]
    
    cn_acc = isempty(cn_data) ? 0.0 : mean(cn_data) * 100
    mci_acc = isempty(mci_data) ? 0.0 : mean(mci_data) * 100
    dem_acc = isempty(dem_data) ? 0.0 : mean(dem_data) * 100
    
    return (
        avg_reward = round(mean(rewards), digits=2),
        reward_std = round(std(rewards), digits=2),
        accuracy = round(acc, digits=1),
        test_cost = round(cost, digits=2),
        steps = round(steps, digits=2),
        cn_acc = round(cn_acc, digits=1),
        mci_acc = round(mci_acc, digits=1),
        dem_acc = round(dem_acc, digits=1)
    )
end

# ============================================================
# Run experiments
# ============================================================
rng = MersenneTwister(SEED)
updater_template = DiscreteUpdater(AlzheimerPOMDPProblem())
b0_template = initialize_belief(updater_template, initialstate(AlzheimerPOMDPProblem()))
true_states = [rand(rng, initialstate(AlzheimerPOMDPProblem())) for _ in 1:N_SAMPLES]

all_results = Dict{String, Dict{String, Any}}()

for (variant_name, rewards_dict) in reward_variants
    println("\n" * "-"^70)
    println("VARIANT: $variant_name")
    println("-"^70)
    
    # Create POMDP with this reward
    pomdp = AlzheimerPOMDPProblem()
    pomdp.rewards = rewards_dict
    
    updater = DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    
    variant_results = Dict{String, Any}()
    
    # 1. SARSOP
    println("  Training SARSOP...")
    try
        sarsop_solver = SARSOPSolver(precision=1e-2, timeout=60.0)
        sarsop_policy = solve(sarsop_solver, pomdp)
        sarsop_data = evaluate_policy(pomdp, sarsop_policy, updater, b0, true_states; rng=MersenneTwister(SEED))
        variant_results["SARSOP"] = summarize(sarsop_data)
        println("    SARSOP: reward=$(variant_results["SARSOP"].avg_reward), acc=$(variant_results["SARSOP"].accuracy)%")
    catch e
        println("    SARSOP FAILED: $e")
        variant_results["SARSOP"] = nothing
    end
    
    # 2. MyopicPOMDP
    println("  Running MyopicPOMDP...")
    myopic = MyopicPOMDPPlanner(pomdp, updater)
    myopic_data = evaluate_policy(pomdp, myopic, updater, b0, true_states; rng=MersenneTwister(SEED))
    variant_results["MyopicPOMDP"] = summarize(myopic_data)
    println("    Myopic: reward=$(variant_results["MyopicPOMDP"].avg_reward), acc=$(variant_results["MyopicPOMDP"].accuracy)%")
    
    # 3. Expert
    println("  Running Expert...")
    expert = ExpertPolicy(pomdp)
    expert_data = evaluate_policy(pomdp, expert, updater, b0, true_states; rng=MersenneTwister(SEED))
    variant_results["Expert"] = summarize(expert_data)
    println("    Expert: reward=$(variant_results["Expert"].avg_reward), acc=$(variant_results["Expert"].accuracy)%")
    
    all_results[variant_name] = variant_results
end

# ============================================================
# Summary table
# ============================================================
println("\n" * "="^70)
println("SENSITIVITY ANALYSIS SUMMARY")
println("="^70)

policies = ["SARSOP", "MyopicPOMDP", "Expert"]
println()
println("Variant               | Policy      | Reward | Acc   | Cost   | Steps | CN   | MCI  | Dem")
println("-"^90)

for (vname, vres) in all_results
    for pol in policies
        if vres[pol] !== nothing
            r = vres[pol]
            line = "$(rpad(vname, 20)) | $(rpad(pol, 11)) | $(lpad(r.avg_reward, 6)) | $(lpad(r.accuracy, 4))% | $(lpad(r.test_cost, 6)) | $(lpad(r.steps, 5)) | $(lpad(r.cn_acc, 4))% | $(lpad(r.mci_acc, 4))% | $(lpad(r.dem_acc, 4))%"
            println(line)
        end
    end
    println()
end

# ============================================================
# Save results
# ============================================================
save_data = Dict()
for (vname, vres) in all_results
    save_data[vname] = Dict()
    for (pol, res) in vres
        if res !== nothing
            save_data[vname][pol] = Dict(
                "avg_reward" => res.avg_reward,
                "reward_std" => res.reward_std,
                "accuracy" => res.accuracy,
                "test_cost" => res.test_cost,
                "steps" => res.steps,
                "cn_acc" => res.cn_acc,
                "mci_acc" => res.mci_acc,
                "dem_acc" => res.dem_acc
            )
        else
            save_data[vname][pol] = nothing
        end
    end
end

open("sensitivity_analysis_results.json", "w") do f
    JSON.print(f, save_data)
end
println("\nSaved: sensitivity_analysis_results.json")

println("\n" * "="^70)
println("SENSITIVITY ANALYSIS COMPLETE")
println("="^70)
