module AlzheimerPOMDP

using POMDPs
using POMDPTools
using BasicPOMCP
using ARDESPOT
using PointBasedValueIteration
using SARSOP
using MCTS
using Distributions
using Random
using Statistics
using Plots
using StatsBase
using LinearAlgebra

export AlzheimerState, AlzheimerAction, AlzheimerObs, AlzheimerPOMDPProblem
export RandomPolicy, ExpertPolicy, MyopicPOMDPPlanner
export make_pomcp_policy, make_despot_policy, make_pbvi_policy, make_sarsop_policy, make_mcts_policy
export run_episode, run_all_policies, summarize_results
export generate_all_figures, plot_solver_comparisons

# ============================================================================
# Types
# ============================================================================

struct AlzheimerState
    name::Symbol
end

struct AlzheimerAction
    name::Symbol
end

struct AlzheimerObs
    name::Symbol
end

Base.:(==)(a::AlzheimerState, b::AlzheimerState) = a.name == b.name
Base.hash(s::AlzheimerState, h::UInt) = hash(s.name, h)

Base.:(==)(a::AlzheimerAction, b::AlzheimerAction) = a.name == b.name
Base.hash(a::AlzheimerAction, h::UInt) = hash(a.name, h)

Base.:(==)(a::AlzheimerObs, b::AlzheimerObs) = a.name == b.name
Base.hash(o::AlzheimerObs, h::UInt) = hash(o.name, h)

# ============================================================================
# POMDP Model
# ============================================================================

mutable struct AlzheimerPOMDPProblem <: POMDP{AlzheimerState, AlzheimerAction, AlzheimerObs}
    transitions::Dict{Symbol, Dict{Symbol, Float64}}
    mmse_probs::Dict{Symbol, Dict{Symbol, Float64}}
    cdr_probs::Dict{Symbol, Dict{Symbol, Float64}}
    apoe4_probs::Dict{Symbol, Dict{Symbol, Float64}}
    rewards::Dict{Symbol, Dict{Symbol, Float64}}
    discount::Float64
end

function AlzheimerPOMDPProblem(;discount=0.95)
    transitions = Dict(
        :CN => Dict(:CN => 0.929, :MCI => 0.068, :Dementia => 0.003),
        :MCI => Dict(:CN => 0.000, :MCI => 0.892, :Dementia => 0.108),
        :Dementia => Dict(:CN => 0.000, :MCI => 0.000, :Dementia => 1.000)
    )
    
    mmse = Dict(
        :CN => Dict(:MMSE_Normal => 0.85, :MMSE_Sedang => 0.13, :MMSE_Rendah => 0.02),
        :MCI => Dict(:MMSE_Normal => 0.55, :MMSE_Sedang => 0.35, :MMSE_Rendah => 0.10),
        :Dementia => Dict(:MMSE_Normal => 0.20, :MMSE_Sedang => 0.45, :MMSE_Rendah => 0.35)
    )
    
    cdr = Dict(
        :CN => Dict(:CDR_0 => 0.95, :CDR_0_5 => 0.04, :CDR_1_plus => 0.01),
        :MCI => Dict(:CDR_0 => 0.25, :CDR_0_5 => 0.70, :CDR_1_plus => 0.05),
        :Dementia => Dict(:CDR_0 => 0.05, :CDR_0_5 => 0.30, :CDR_1_plus => 0.65)
    )
    
    apoe4_raw = Dict(
        :CN => Dict(:APOE4_Negatif => 0.6073, :APOE4_Hetero => 0.2472, :APOE4_Homo => 0.0244),
        :MCI => Dict(:APOE4_Negatif => 0.5071, :APOE4_Hetero => 0.3225, :APOE4_Homo => 0.0842),
        :Dementia => Dict(:APOE4_Negatif => 0.3198, :APOE4_Hetero => 0.4486, :APOE4_Homo => 0.1759)
    )
    # Normalize APOE4 probabilities so each state's observations sum to 1.0
    apoe4 = Dict{Symbol, Dict{Symbol, Float64}}()
    for (state, probs) in apoe4_raw
        total = sum(values(probs))
        apoe4[state] = Dict(k => v / total for (k, v) in probs)
    end
    
    rewards = Dict(
        :CN => Dict(:A_Wait => 30, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => -50, :A_Treat_Dementia => -500),
        :MCI => Dict(:A_Wait => -10, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => 150, :A_Treat_Dementia => -100),
        :Dementia => Dict(:A_Wait => -1000, :A_Test_MMSE => -5, :A_Test_CDR => -20, :A_Test_APOE4 => -50, :A_Treat_MCI => -50, :A_Treat_Dementia => 300)
    )
    
    # Validate all observation probabilities sum to 1.0
    _validate_observation_probs(mmse, "MMSE")
    _validate_observation_probs(cdr, "CDR")
    _validate_observation_probs(apoe4, "APOE4")
    
    # Validate transition probabilities sum to 1.0
    _validate_transition_probs(transitions)
    
    return AlzheimerPOMDPProblem(transitions, mmse, cdr, apoe4, rewards, discount)
end

function _validate_observation_probs(probs::Dict, name::String)
    for (state, pdict) in probs
        total = sum(values(pdict))
        if abs(total - 1.0) > 1e-6
            error("$name observation probabilities for $state sum to $total (expected 1.0)")
        end
    end
end

function _validate_transition_probs(transitions::Dict)
    for (from_state, to_dict) in transitions
        total = sum(values(to_dict))
        if abs(total - 1.0) > 1e-6
            error("Transition probabilities from $from_state sum to $total (expected 1.0)")
        end
    end
end

const STATES = [AlzheimerState(:CN), AlzheimerState(:MCI), AlzheimerState(:Dementia)]
const ACTIONS = [AlzheimerAction(:A_Wait), AlzheimerAction(:A_Test_MMSE), AlzheimerAction(:A_Test_CDR), 
                 AlzheimerAction(:A_Test_APOE4), AlzheimerAction(:A_Treat_MCI), AlzheimerAction(:A_Treat_Dementia)]
const OBSERVATIONS = [AlzheimerObs(:None), AlzheimerObs(:MMSE_Normal), AlzheimerObs(:MMSE_Sedang), AlzheimerObs(:MMSE_Rendah),
                      AlzheimerObs(:CDR_0), AlzheimerObs(:CDR_0_5), AlzheimerObs(:CDR_1_plus),
                      AlzheimerObs(:APOE4_Negatif), AlzheimerObs(:APOE4_Hetero), AlzheimerObs(:APOE4_Homo)]

POMDPs.states(p::AlzheimerPOMDPProblem) = STATES
POMDPs.actions(p::AlzheimerPOMDPProblem) = ACTIONS
POMDPs.observations(p::AlzheimerPOMDPProblem) = OBSERVATIONS

POMDPs.stateindex(p::AlzheimerPOMDPProblem, s::AlzheimerState) = findfirst(x -> x == s, STATES)
POMDPs.actionindex(p::AlzheimerPOMDPProblem, a::AlzheimerAction) = findfirst(x -> x == a, ACTIONS)
POMDPs.obsindex(p::AlzheimerPOMDPProblem, o::AlzheimerObs) = findfirst(x -> x == o, OBSERVATIONS)

function POMDPs.transition(p::AlzheimerPOMDPProblem, s::AlzheimerState, a::AlzheimerAction)
    probs = Float64[p.transitions[s.name][sp.name] for sp in STATES]
    return SparseCat(STATES, probs)
end

function POMDPs.observation(p::AlzheimerPOMDPProblem, a::AlzheimerAction, sp::AlzheimerState)
    if a.name == :A_Test_MMSE
        obs = [AlzheimerObs(k) for k in keys(p.mmse_probs[sp.name])]
        probs = collect(values(p.mmse_probs[sp.name]))
        return SparseCat(obs, probs)
    elseif a.name == :A_Test_CDR
        obs = [AlzheimerObs(k) for k in keys(p.cdr_probs[sp.name])]
        probs = collect(values(p.cdr_probs[sp.name]))
        return SparseCat(obs, probs)
    elseif a.name == :A_Test_APOE4
        obs = [AlzheimerObs(k) for k in keys(p.apoe4_probs[sp.name])]
        probs = collect(values(p.apoe4_probs[sp.name]))
        return SparseCat(obs, probs)
    else
        return SparseCat([AlzheimerObs(:None)], [1.0])
    end
end

POMDPs.reward(p::AlzheimerPOMDPProblem, s::AlzheimerState, a::AlzheimerAction) = p.rewards[s.name][a.name]
POMDPs.initialstate(p::AlzheimerPOMDPProblem) = SparseCat(STATES, [0.385189, 0.417041, 0.197770])
POMDPs.discount(p::AlzheimerPOMDPProblem) = p.discount
POMDPs.isterminal(p::AlzheimerPOMDPProblem, s::AlzheimerState) = false

# ============================================================================
# Helper Functions
# ============================================================================

isterminal_action(a::AlzheimerAction) = a.name in (:A_Wait, :A_Treat_MCI, :A_Treat_Dementia)
is_test(a::AlzheimerAction) = a.name in (:A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4)

function is_diagnosis_correct(s::AlzheimerState, a::AlzheimerAction)
    return (s.name == :CN && a.name == :A_Wait) ||
           (s.name == :MCI && a.name == :A_Treat_MCI) ||
           (s.name == :Dementia && a.name == :A_Treat_Dementia)
end

# ============================================================================
# Policies
# ============================================================================

struct RandomPolicy{RNG} <: Policy
    pomdp::AlzheimerPOMDPProblem
    rng::RNG
end

POMDPs.action(p::RandomPolicy, b) = rand(p.rng, actions(p.pomdp))

# ----------------------------------------------------------------------------
mutable struct ExpertPolicy <: Policy
    pomdp::AlzheimerPOMDPProblem
    tested::Set{Symbol}
    p_dem_thres::Float64
    p_cn_thres::Float64
    p_mci_thres::Float64
end

ExpertPolicy(pomdp::AlzheimerPOMDPProblem) = ExpertPolicy(pomdp, Set{Symbol}(), 0.85, 0.80, 0.80)

function POMDPs.action(p::ExpertPolicy, b::DiscreteBelief)
    p_cn = pdf(b, AlzheimerState(:CN))
    p_mci = pdf(b, AlzheimerState(:MCI))
    p_dem = pdf(b, AlzheimerState(:Dementia))
    
    if p_dem >= p.p_dem_thres
        return AlzheimerAction(:A_Treat_Dementia)
    elseif p_cn >= p.p_cn_thres
        return AlzheimerAction(:A_Wait)
    elseif p_mci >= p.p_mci_thres
        return AlzheimerAction(:A_Treat_MCI)
    end
    
    if :A_Test_MMSE ∉ p.tested
        push!(p.tested, :A_Test_MMSE)
        return AlzheimerAction(:A_Test_MMSE)
    elseif :A_Test_CDR ∉ p.tested
        push!(p.tested, :A_Test_CDR)
        return AlzheimerAction(:A_Test_CDR)
    elseif :A_Test_APOE4 ∉ p.tested
        push!(p.tested, :A_Test_APOE4)
        return AlzheimerAction(:A_Test_APOE4)
    end
    
    if p_dem > p_mci && p_dem > p_cn
        return AlzheimerAction(:A_Treat_Dementia)
    elseif p_mci > p_cn
        return AlzheimerAction(:A_Treat_MCI)
    else
        return AlzheimerAction(:A_Wait)
    end
end

# ----------------------------------------------------------------------------
struct ExpertRollout <: Policy
    pomdp::AlzheimerPOMDPProblem
end

function POMDPs.action(p::ExpertRollout, s::AlzheimerState)
    if s.name == :Dementia
        return AlzheimerAction(:A_Treat_Dementia)
    elseif s.name == :MCI
        return AlzheimerAction(:A_Treat_MCI)
    else
        return AlzheimerAction(:A_Wait)
    end
end

# ----------------------------------------------------------------------------
struct MyopicPOMDPPlanner <: Policy
    pomdp::AlzheimerPOMDPProblem
    updater::DiscreteUpdater
    gamma::Float64
end

MyopicPOMDPPlanner(pomdp::AlzheimerPOMDPProblem, updater::DiscreteUpdater) = MyopicPOMDPPlanner(pomdp, updater, discount(pomdp))

function reset_policy!(p::ExpertPolicy)
    empty!(p.tested)
    return nothing
end

reset_policy!(p) = nothing

function POMDPs.action(planner::MyopicPOMDPPlanner, b::DiscreteBelief)
    pomdp = planner.pomdp
    term_acts = [AlzheimerAction(:A_Wait), AlzheimerAction(:A_Treat_MCI), AlzheimerAction(:A_Treat_Dementia)]
    test_acts = [AlzheimerAction(:A_Test_MMSE), AlzheimerAction(:A_Test_CDR), AlzheimerAction(:A_Test_APOE4)]
    
    # Best terminal action now
    best_term_val = -Inf
    best_term_act = term_acts[1]
    for a in term_acts
        val = sum(pdf(b, s) * reward(pomdp, s, a) for s in states(pomdp))
        if val > best_term_val
            best_term_val = val
            best_term_act = a
        end
    end
    
    # Best test action with exact Bayesian one-step lookahead.
    best_test_val = -Inf
    best_test_act = test_acts[1]
    for a in test_acts
        immediate = sum(pdf(b, s) * reward(pomdp, s, a) for s in states(pomdp))
        
        future = 0.0
        for o in observations(pomdp)
            obs_prob = 0.0
            for s in states(pomdp)
                ps = pdf(b, s)
                ps <= 0.0 && continue
                for (sp, p_sp) in weighted_iterator(transition(pomdp, s, a))
                    p_obs = pdf(observation(pomdp, a, sp), o)
                    obs_prob += ps * p_sp * p_obs
                end
            end
            
            if obs_prob > 1e-10
                bp = update(planner.updater, b, a, o)
                val_post = maximum(
                    sum(pdf(bp, sp) * reward(pomdp, sp, a2) for sp in states(pomdp))
                    for a2 in term_acts
                )
                future += obs_prob * val_post
            end
        end
        
        total = immediate + planner.gamma * future
        if total > best_test_val
            best_test_val = total
            best_test_act = a
        end
    end
    
    return best_test_val > best_term_val ? best_test_act : best_term_act
end

# ============================================================================
# Solver Factories (mature Julia packages from POMDPs ecosystem)
# ============================================================================

function make_pomcp_policy(pomdp::AlzheimerPOMDPProblem; rng=Random.GLOBAL_RNG)
    solver = POMCPSolver(
        max_depth=8,
        c=50.0,
        tree_queries=200,
        estimate_value=FORollout(ExpertRollout(pomdp)),
        rng=rng
    )
    return solve(solver, pomdp)
end

function make_despot_policy(pomdp::AlzheimerPOMDPProblem; rng=Random.GLOBAL_RNG)
    # ARDESPOT.jl - DESPOT for real-time online planning
    # Tuned for small medical POMDPs: lower K and T_max to prevent over-testing
    solver = DESPOTSolver(
        K=10,
        D=6,
        lambda=0.8,
        T_max=0.005,
        rng=rng
    )
    return solve(solver, pomdp)
end

function make_pbvi_policy(pomdp::AlzheimerPOMDPProblem)
    # PBVI - offline point-based solver suitable for small state spaces (Navarro 2025)
    solver = PBVISolver(verbose=false, max_iterations=100)
    return solve(solver, pomdp)
end

function make_sarsop_policy(pomdp::AlzheimerPOMDPProblem)
    # SARSOP - offline solver with optimality bounds for small POMDPs
    solver = SARSOPSolver(precision=1e-2, timeout=60.0)
    return solve(solver, pomdp)
end

function make_mcts_policy(pomdp::AlzheimerPOMDPProblem; rng=Random.GLOBAL_RNG)
    # Convert POMDP to MDP for MCTS via QMDP heuristic
    # For small state spaces, use DPW solver on the underlying belief MDP
    solver = DPWSolver(
        n_iterations=500,
        depth=8,
        exploration_constant=50.0,
        rng=rng
    )
    return solve(solver, pomdp)
end

# ============================================================================
# Experiment Framework
# ============================================================================

function run_episode(pomdp::AlzheimerPOMDPProblem, policy, updater::DiscreteUpdater, b0::DiscreteBelief, 
                     true_state::AlzheimerState; max_steps=5, rng=Random.GLOBAL_RNG)
    reset_policy!(policy)
    
    s = true_state
    b = b0
    total_reward = 0.0
    action_history = AlzheimerAction[]
    steps = 0
    terminal_action = nothing
    
    for step in 1:max_steps
        a = action(policy, b)
        push!(action_history, a)
        steps = step
        
        r = reward(pomdp, s, a)
        total_reward += discount(pomdp)^(step - 1) * r
        
        sp = rand(rng, transition(pomdp, s, a))
        o = rand(rng, observation(pomdp, a, sp))

        if !isterminal_action(a)
            b = update(updater, b, a, o)
        end

        s = sp
        
        if isterminal_action(a)
            terminal_action = a
            break
        end
    end
    
    if terminal_action === nothing
        terminal_action = AlzheimerAction(:A_Wait)
        r = reward(pomdp, s, terminal_action)
        total_reward += discount(pomdp)^steps * r
        push!(action_history, terminal_action)
        steps += 1
    end
    
    correct = is_diagnosis_correct(true_state, terminal_action)
    test_cost = sum((reward(pomdp, AlzheimerState(:CN), a) for a in action_history if is_test(a)); init=0.0)
    
    return (
        total_reward = total_reward,
        correct = correct,
        test_cost = test_cost,
        steps = steps,
        terminal_action = terminal_action.name,
        true_state = true_state.name,
        action_history = [a.name for a in action_history]
    )
end

function run_all_policies(pomdp::AlzheimerPOMDPProblem, n_samples::Int=1000; seed::Int=42)
    rng = MersenneTwister(seed)
    updater = DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    
    true_states = [rand(rng, initialstate(pomdp)) for _ in 1:n_samples]
    
    policies = Dict{String, Any}()
    
    # Baselines
    policies["Random"] = RandomPolicy(pomdp, rng)
    policies["Expert"] = ExpertPolicy(pomdp)
    policies["MyopicPOMDP"] = MyopicPOMDPPlanner(pomdp, updater)
    
    # Online solvers
    println("Building POMCP solver...")
    policies["POMCP"] = make_pomcp_policy(pomdp; rng=rng)
    
    # DESPOT (ARDESPOT.jl) - online solver from Khatim et al. (2024)
    println("Building DESPOT solver...")
    try
        policies["DESPOT"] = make_despot_policy(pomdp; rng=rng)
    catch e
        println("  Warning: DESPOT solver failed: $e")
    end
    
    # Offline point-based solver - suitable for small state spaces (Navarro 2025)
    # PBVI skipped: Julia implementation does not converge on this problem (infinite loop in alpha improvement)
    println("Building SARSOP solver...")
    try
        policies["SARSOP"] = make_sarsop_policy(pomdp)
        println("  SARSOP built successfully")
    catch e
        println("  Warning: SARSOP solver failed: $e")
    end
    
    results = Dict{String, Vector{NamedTuple}}()
    for (name, policy) in policies
        println("Running $name on $n_samples patients...")
        t0 = time()
        policy_results = [
            run_episode(pomdp, policy, updater, b0, s; max_steps=5, rng=rng)
            for s in true_states
        ]
        elapsed = round(time() - t0, digits=2)
        println("  Completed in $(elapsed)s")
        results[name] = policy_results
    end
    
    return results
end

# ============================================================================
# Results Summary
# ============================================================================

function summarize_results(results::Dict{String, Vector{NamedTuple}})
    println("\n" * "=" ^ 80)
    println("MONTE CARLO RESULTS - JULIA ENVIRONMENT")
    println("=" ^ 80)
    
    println("\nPolicy          Avg Reward    Accuracy    Test Cost    Steps    CN     MCI    Dem")
    println("-" ^ 80)
    
    for name in sort(collect(keys(results)))
        data = results[name]
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
        
        line = lpad(name, 14) * " " *
               lpad(round(mean(rewards), digits=2), 10) * " " *
               lpad(round(acc, digits=1), 9) * "% " *
               lpad(round(cost, digits=2), 10) * " " *
               lpad(round(steps, digits=2), 8) * " " *
               lpad(round(cn_acc, digits=1), 6) * "% " *
               lpad(round(mci_acc, digits=1), 6) * "% " *
               lpad(round(dem_acc, digits=1), 6) * "%"
        println(line)
    end
    println("-" ^ 80)
end

# ============================================================================
# Figure Generation
# ============================================================================

function plot_initial_belief()
    fig = plot(size=(800, 600), dpi=300)
    states = ["CN\n(Cognitively\nNormal)", "MCI\n(Mild Cognitive\nImpairment)", "Dementia\n(Alzheimer)"]
    probs = [0.385189, 0.417041, 0.197770]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    
    bar!(states, probs, color=colors, alpha=0.8, linecolor=:black, linewidth=1.5,
         label="", ylabel="Probability", title="Initial Belief Distribution (b₀)\nExtracted from ADNI Dataset (N=15,836)")
    
    for (i, pr) in enumerate(probs)
        annotate!(i, pr + 0.02, text("$(round(pr*100,digits=1))%\n($(round(Int,pr*15836)) patients)", 10, :center))
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
                  title="Transition Probability Matrix T(s'|s)\n6-Month Progression from ADNI Longitudinal Data",
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
         ylabel="Average Reward", title="Average Reward Comparison Across Policies (N=1,000 Patients)")
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
         ylabel="Accuracy (%)", title="Overall Diagnostic Accuracy Comparison (N=1,000 Patients)", ylims=(0,100))
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
    ylabel!("Accuracy (%)")
    title!("Per-State Diagnostic Accuracy Comparison")
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

end # module
