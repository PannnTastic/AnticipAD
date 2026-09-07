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
