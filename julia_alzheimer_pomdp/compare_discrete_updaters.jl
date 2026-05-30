#!/usr/bin/env julia
# Compare exact and conservative discrete belief updaters.
# This is a sensitivity experiment only; it does not modify the main model.

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP
using POMDPs, POMDPTools, Random, Statistics, JSON

const N_SAMPLES = 1000
const SEED = 42
const MAX_STEPS = 20
const STATE_NAMES = [:CN, :MCI, :Dementia]
const TEST_ACTIONS = [:A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4]
const TERMINAL_ACTIONS = [:A_Wait, :A_Treat_MCI, :A_Treat_Dementia]

struct TemperedDiscreteUpdater
    pomdp::AlzheimerPOMDPProblem
    alpha::Float64
end

function initial_belief(pomdp)
    return initialstate(pomdp)
end

function exact_observation_probability(pomdp, b, a::AlzheimerAction, o::AlzheimerObs)
    p = 0.0
    for s in states(pomdp)
        ps = pdf(b, s)
        ps <= 0.0 && continue
        for (sp, p_sp) in weighted_iterator(transition(pomdp, s, a))
            p += ps * p_sp * pdf(observation(pomdp, a, sp), o)
        end
    end
    return p
end

function POMDPs.update(updater::TemperedDiscreteUpdater, b, a::AlzheimerAction, o::AlzheimerObs)
    pomdp = updater.pomdp
    posterior = Float64[]

    for sp in states(pomdp)
        predicted = 0.0
        for s in states(pomdp)
            predicted += pdf(b, s) * pdf(transition(pomdp, s, a), sp)
        end

        likelihood = pdf(observation(pomdp, a, sp), o)
        push!(posterior, predicted * likelihood^updater.alpha)
    end

    total = sum(posterior)
    if total <= 1e-12
        return initial_belief(pomdp)
    end

    return SparseCat(states(pomdp), posterior ./ total)
end

mutable struct TemperedMyopicPolicy
    pomdp::AlzheimerPOMDPProblem
    updater::TemperedDiscreteUpdater
    gamma::Float64
    tested::Set{Symbol}
    min_tests_before_terminal::Int
end

function TemperedMyopicPolicy(pomdp, updater; min_tests_before_terminal=1)
    return TemperedMyopicPolicy(pomdp, updater, discount(pomdp), Set{Symbol}(), min_tests_before_terminal)
end

function reset!(p::TemperedMyopicPolicy)
    empty!(p.tested)
end

function best_terminal(p::TemperedMyopicPolicy, b)
    best_val = -Inf
    best_action = AlzheimerAction(:A_Wait)
    for name in TERMINAL_ACTIONS
        a = AlzheimerAction(name)
        v = sum(pdf(b, s) * reward(p.pomdp, s, a) for s in states(p.pomdp))
        if v > best_val
            best_val = v
            best_action = a
        end
    end
    return best_action, best_val
end

function POMDPs.action(p::TemperedMyopicPolicy, b)
    term_action, term_value = best_terminal(p, b)
    test_candidates = [AlzheimerAction(a) for a in TEST_ACTIONS if a ∉ p.tested]

    best_test_action = nothing
    best_test_value = -Inf

    for a in test_candidates
        immediate = sum(pdf(b, s) * reward(p.pomdp, s, a) for s in states(p.pomdp))
        future = 0.0

        for o in observations(p.pomdp)
            o.name == :None && continue
            po = exact_observation_probability(p.pomdp, b, a, o)
            if po > 1e-10
                bp = update(p.updater, b, a, o)
                _, post_val = best_terminal(p, bp)
                future += po * post_val
            end
        end

        q = immediate + p.gamma * future
        if q > best_test_value
            best_test_value = q
            best_test_action = a
        end
    end

    must_test = length(p.tested) < p.min_tests_before_terminal
    if best_test_action !== nothing && (must_test || best_test_value > term_value)
        push!(p.tested, best_test_action.name)
        return best_test_action
    end

    return term_action
end

function correct_terminal(true_state::AlzheimerState, action::AlzheimerAction)
    return (true_state.name == :CN && action.name == :A_Wait) ||
           (true_state.name == :MCI && action.name == :A_Treat_MCI) ||
           (true_state.name == :Dementia && action.name == :A_Treat_Dementia)
end

function run_episode_tempered(pomdp, policy::TemperedMyopicPolicy, true_state::AlzheimerState; rng)
    reset!(policy)
    b = initial_belief(pomdp)
    s = true_state
    total_reward = 0.0
    actions_taken = Symbol[]
    terminal = AlzheimerAction(:A_Wait)

    for step in 1:MAX_STEPS
        a = action(policy, b)
        push!(actions_taken, a.name)
        total_reward += discount(pomdp)^(step - 1) * reward(pomdp, s, a)

        sp = rand(rng, transition(pomdp, s, a))
        o = rand(rng, observation(pomdp, a, sp))

        if AlzheimerPOMDP.is_test(a)
            b = update(policy.updater, b, a, o)
        end

        s = sp
        if AlzheimerPOMDP.isterminal_action(a)
            terminal = a
            break
        end
    end

    test_count = count(a -> a in TEST_ACTIONS, actions_taken)
    return (
        total_reward = total_reward,
        correct = correct_terminal(true_state, terminal),
        terminal_action = terminal.name,
        true_state = true_state.name,
        steps = length(actions_taken),
        tests = test_count,
        action_history = actions_taken
    )
end

function posterior_demo(pomdp)
    b0 = initial_belief(pomdp)
    a = AlzheimerAction(:A_Test_CDR)
    o = AlzheimerObs(:CDR_1_plus)
    println("\nPosterior after one CDR_1_plus observation:")
    for alpha in [1.0, 0.75, 0.5]
        updater = TemperedDiscreteUpdater(pomdp, alpha)
        bp = update(updater, b0, a, o)
        probs = [round(pdf(bp, AlzheimerState(s)), digits=4) for s in STATE_NAMES]
        println("  alpha=$(alpha): [CN, MCI, Dementia] = $probs")
    end
end

function summarize(name, results)
    acc = mean([r.correct for r in results]) * 100
    reward_mean = mean([r.total_reward for r in results])
    steps_mean = mean([r.steps for r in results])
    tests_mean = mean([r.tests for r in results])
    println(rpad(name, 22), " acc=", round(acc, digits=1), "% reward=", round(reward_mean, digits=2),
            " steps=", round(steps_mean, digits=2), " tests=", round(tests_mean, digits=2))
    return Dict(
        "accuracy" => round(acc, digits=2),
        "avg_reward" => round(reward_mean, digits=2),
        "avg_steps" => round(steps_mean, digits=2),
        "avg_tests" => round(tests_mean, digits=2)
    )
end

function main()
    println("="^72)
    println("DISCRETE BELIEF UPDATER SENSITIVITY")
    println("N=$N_SAMPLES, seed=$SEED")
    println("="^72)

    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    rng_states = MersenneTwister(SEED)
    true_states = [rand(rng_states, initialstate(pomdp)) for _ in 1:N_SAMPLES]

    posterior_demo(pomdp)

    all_results = Dict{String, Any}()
    for alpha in [1.0, 0.75, 0.5]
        label = alpha == 1.0 ? "ExactDiscrete alpha=1" : "TemperedDiscrete alpha=$alpha"
        updater = TemperedDiscreteUpdater(pomdp, alpha)
        policy = TemperedMyopicPolicy(pomdp, updater)
        rng = MersenneTwister(SEED)
        results = [run_episode_tempered(pomdp, policy, s; rng=rng) for s in true_states]
        all_results[label] = summarize(label, results)
    end

    open("discrete_updater_sensitivity.json", "w") do f
        JSON.print(f, all_results, 2)
    end
    println("\nSaved: discrete_updater_sensitivity.json")
end

main()
