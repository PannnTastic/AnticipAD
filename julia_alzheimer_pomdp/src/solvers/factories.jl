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
