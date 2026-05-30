#!/usr/bin/env julia

# ============================================================================
# Main Runner Script for Alzheimer POMDP Experiments (Julia)
# ============================================================================
# 
# This script runs a comprehensive Monte Carlo experiment comparing:
#   - Random (baseline)
#   - Expert (rule-based heuristic)
#   - MyopicPOMDP (one-step exact lookahead)
#   - POMCP (Monte Carlo Tree Search via BasicPOMCP.jl)
#   - QMDP (MDP-based heuristic)
#   - FIB (Fast Informed Bound heuristic)
#   - SARSOP (offline point-based optimal solver)
#
# Usage:
#   cd julia_alzheimer_pomdp
#   julia --project=. scripts/run_experiments.jl
#
# ============================================================================

using Pkg
Pkg.activate(dirname(@__DIR__))

# Install dependencies if missing
# Pkg.instantiate()  # Uncomment on first run

push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
using AlzheimerPOMDP
using POMDPs

function main()
    println("=" ^ 80)
    println("ALZHEIMER POMDP - JULIA ENVIRONMENT")
    println("Fair benchmark using mature POMDP solvers from the Julia ecosystem")
    println("=" ^ 80)
    
    # Create POMDP model
    pomdp = AlzheimerPOMDP.AlzheimerPOMDPProblem(discount=0.95)
    
    println("\nModel Specification:")
    println("  States: ", [s.name for s in states(pomdp)])
    println("  Actions: ", [a.name for a in actions(pomdp)])
    println("  Observations: ", [o.name for o in observations(pomdp)])
    println("  Discount: ", discount(pomdp))
    println("  Initial Belief: P(CN)=0.385, P(MCI)=0.417, P(Dem)=0.198")
    
    # Run experiments
    n_samples = 1000
    seed = 42
    println("\nRunning Monte Carlo experiment with N=$n_samples patients (seed=$seed)...")
    println()
    
    results = run_all_policies(pomdp, n_samples; seed=seed)
    
    # Print summary table
    summarize_results(results)
    
    # Generate figures
    println("\nGenerating comparison figures...")
    generate_all_figures(results)
    
    println("\n" * "=" ^ 80)
    println("DONE")
    println("=" ^ 80)
end

main()
