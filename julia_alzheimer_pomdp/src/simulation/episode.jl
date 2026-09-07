function is_diagnosis_correct(s::AlzheimerState, a::AlzheimerAction)
    return (s.name == :CN && a.name == :A_Wait) ||
           (s.name == :MCI && a.name == :A_Treat_MCI) ||
           (s.name == :Dementia && a.name == :A_Treat_Dementia)
end

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
