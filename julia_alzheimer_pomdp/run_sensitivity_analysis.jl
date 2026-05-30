#!/usr/bin/env julia
# Sensitivity Analysis for Revision
# Tests: (1) MCI->CN reversion (0%, 5%, 10%)
#        (2) Dementia-Wait penalty (-1000, -500, -200)

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, Random, Statistics
using SARSOP

const N_SAMPLES = 10_000
const SEED = 42
const MAX_STEPS = 20

function run_solver(pomdp, policy, updater, b0, true_states; rng=MersenneTwister(SEED))
    results = []
    for s in true_states
        r = run_episode(pomdp, policy, updater, b0, s; max_steps=MAX_STEPS, rng=rng)
        push!(results, r)
    end
    return results
end

function summarize(results)
    rew  = mean(r.total_reward for r in results)
    acc  = mean(r.correct for r in results) * 100
    return (reward=round(rew, digits=1), accuracy=round(acc, digits=1))
end

function make_pomdp_with_transitions(transitions_dict; discount=0.95)
    # Build a custom POMDP with modified transitions using the existing model
    pomdp = AlzheimerPOMDPProblem(discount=discount)
    for (k, v) in transitions_dict
        pomdp.transitions[k] = v
    end
    # Validate
    for (from_state, to_dict) in pomdp.transitions
        total = sum(values(to_dict))
        if abs(total - 1.0) > 1e-6
            error("Transition from $from_state sums to $total")
        end
    end
    return pomdp
end

function make_pomdp_with_reward(state::Symbol, action::Symbol, value::Float64; discount=0.95)
    pomdp = AlzheimerPOMDPProblem(discount=discount)
    pomdp.rewards[state][action] = value
    return pomdp
end

function main()
    println("=" ^ 70)
    println("SENSITIVITY ANALYSIS")
    println("N=$N_SAMPLES | MaxSteps=$MAX_STEPS | Seed=$SEED")
    println("=" ^ 70)

    rng = MersenneTwister(SEED)
    base_pomdp = AlzheimerPOMDPProblem(discount=0.95)
    updater = POMDPTools.DiscreteUpdater(base_pomdp)
    b0 = initialize_belief(updater, initialstate(base_pomdp))
    true_states = [rand(rng, initialstate(base_pomdp)) for _ in 1:N_SAMPLES]

    # =========================================================================
    println("\n--- Part 1: Monotonicity Sensitivity (MCI->CN reversion) ---")
    # =========================================================================
    reversion_rates = [0.000, 0.050, 0.100]
    println("\nRev%  | MyopicPOMDP (Rew / Acc) | Expert (Rew / Acc)")
    println("-" ^ 55)

    for rev in reversion_rates
        new_mci_stay = 0.892 - rev  # Keep MCI->Dem=0.108 fixed
        transitions = Dict(
            :CN       => Dict(:CN => 0.929, :MCI => 0.068, :Dementia => 0.003),
            :MCI      => Dict(:CN => rev,   :MCI => new_mci_stay, :Dementia => 0.108),
            :Dementia => Dict(:CN => 0.000, :MCI => 0.000, :Dementia => 1.000)
        )
        pomdp = make_pomdp_with_transitions(transitions)
        upd   = POMDPTools.DiscreteUpdater(pomdp)
        b0_   = initialize_belief(upd, initialstate(pomdp))

        myopic  = MyopicPOMDPPlanner(pomdp, upd)
        expert  = ExpertPolicy(pomdp)

        rng2 = MersenneTwister(SEED)
        r_myopic = run_solver(pomdp, myopic, upd, b0_, true_states; rng=rng2)
        rng2 = MersenneTwister(SEED)
        r_expert = run_solver(pomdp, expert, upd, b0_, true_states; rng=rng2)

        sm = summarize(r_myopic)
        se = summarize(r_expert)
        pct = round(Int, rev * 100)
        println("$pct%   | $(sm.reward) / $(sm.accuracy)%       | $(se.reward) / $(se.accuracy)%")
    end

    # SARSOP with 5% reversion (requires retraining)
    println("\nTraining SARSOP for 5% MCI->CN model...")
    rev = 0.05
    new_mci_stay = 0.892 - rev
    transitions_rev = Dict(
        :CN       => Dict(:CN => 0.929, :MCI => 0.068, :Dementia => 0.003),
        :MCI      => Dict(:CN => rev,   :MCI => new_mci_stay, :Dementia => 0.108),
        :Dementia => Dict(:CN => 0.000, :MCI => 0.000, :Dementia => 1.000)
    )
    pomdp_rev = make_pomdp_with_transitions(transitions_rev)
    upd_rev   = POMDPTools.DiscreteUpdater(pomdp_rev)
    b0_rev    = initialize_belief(upd_rev, initialstate(pomdp_rev))

    try
        sarsop_rev = SARSOP.solve(SARSOPSolver(precision=1e-2, timeout=300.0), pomdp_rev)
        rng3 = MersenneTwister(SEED)
        r_sarsop_rev = run_solver(pomdp_rev, sarsop_rev, upd_rev, b0_rev, true_states; rng=rng3)
        ss = summarize(r_sarsop_rev)
        println("SARSOP (5% rev): reward=$(ss.reward), accuracy=$(ss.accuracy)%")
    catch e
        println("SARSOP retraining failed: $e")
    end

    # =========================================================================
    println("\n--- Part 2: Reward Sensitivity (Dementia-Wait penalty) ---")
    # =========================================================================
    penalties = [-1000.0, -500.0, -200.0]
    println("\nPenalty | SARSOP (Rew / Acc)     | MyopicPOMDP (Rew / Acc)")
    println("-" ^ 60)

    # Load baseline SARSOP policy (trained on baseline model)
    println("Loading baseline SARSOP policy...")
    base_sarsop = SARSOP.load_policy(base_pomdp, "policy.out")

    for penalty in penalties
        pomdp_p = make_pomdp_with_reward(:Dementia, :A_Wait, penalty)
        upd_p   = POMDPTools.DiscreteUpdater(pomdp_p)
        b0_p    = initialize_belief(upd_p, initialstate(pomdp_p))

        # For SARSOP: evaluate the baseline policy under modified rewards
        # (re-training for each penalty variant would be expensive)
        myopic_p = MyopicPOMDPPlanner(pomdp_p, upd_p)

        rng4 = MersenneTwister(SEED)
        r_sarsop_p = run_solver(pomdp_p, base_sarsop, upd_p, b0_p, true_states; rng=rng4)
        rng4 = MersenneTwister(SEED)
        r_myopic_p = run_solver(pomdp_p, myopic_p, upd_p, b0_p, true_states; rng=rng4)

        ss = summarize(r_sarsop_p)
        sm = summarize(r_myopic_p)
        println("$(Int(penalty))    | $(ss.reward) / $(ss.accuracy)% | $(sm.reward) / $(sm.accuracy)%")
    end

    # Double test costs variant
    println("\n--- Double test cost variant ---")
    pomdp_dt = AlzheimerPOMDPProblem(discount=0.95)
    for state in [:CN, :MCI, :Dementia]
        pomdp_dt.rewards[state][:A_Test_MMSE]  = -10.0
        pomdp_dt.rewards[state][:A_Test_CDR]   = -40.0
        pomdp_dt.rewards[state][:A_Test_APOE4] = -100.0
    end
    upd_dt = POMDPTools.DiscreteUpdater(pomdp_dt)
    b0_dt  = initialize_belief(upd_dt, initialstate(pomdp_dt))
    myopic_dt = MyopicPOMDPPlanner(pomdp_dt, upd_dt)
    expert_dt = ExpertPolicy(pomdp_dt)

    rng5 = MersenneTwister(SEED)
    r_sarsop_dt = run_solver(pomdp_dt, base_sarsop, upd_dt, b0_dt, true_states; rng=rng5)
    rng5 = MersenneTwister(SEED)
    r_myopic_dt = run_solver(pomdp_dt, myopic_dt, upd_dt, b0_dt, true_states; rng=rng5)
    rng5 = MersenneTwister(SEED)
    r_expert_dt = run_solver(pomdp_dt, expert_dt, upd_dt, b0_dt, true_states; rng=rng5)

    ss_dt = summarize(r_sarsop_dt)
    sm_dt = summarize(r_myopic_dt)
    se_dt = summarize(r_expert_dt)
    println("SARSOP (double costs):   reward=$(ss_dt.reward), accuracy=$(ss_dt.accuracy)%")
    println("Myopic (double costs):   reward=$(sm_dt.reward), accuracy=$(sm_dt.accuracy)%")
    println("Expert (double costs):   reward=$(se_dt.reward), accuracy=$(se_dt.accuracy)%")

    println("\nSensitivity analysis complete.")
end

main()
