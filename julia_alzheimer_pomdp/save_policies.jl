#!/usr/bin/env julia
# Solve offline policies once and serialize to disk for reuse

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, Random, Serialization
using QMDP, FIB, SARSOP

function main()
    println("Solving policies (one-time)...")
    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    
    println("  Solving QMDP...")
    qmdp_policy = solve(QMDPSolver(), pomdp)
    serialize("policy_qmdp.jls", qmdp_policy)
    println("    Saved policy_qmdp.jls")
    
    println("  Solving FIB...")
    fib_policy = solve(FIBSolver(), pomdp)
    serialize("policy_fib.jls", fib_policy)
    println("    Saved policy_fib.jls")
    
    println("  Solving SARSOP (timeout=300s)...")
    sarsop_solver = SARSOPSolver(precision=1e-2, timeout=300.0)
    sarsop_policy = solve(sarsop_solver, pomdp)
    serialize("policy_sarsop.jls", sarsop_policy)
    println("    Saved policy_sarsop.jls")
    
    println("All policies saved.")
end

main()
