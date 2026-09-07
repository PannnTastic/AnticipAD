struct MyopicPOMDPPlanner <: Policy
    pomdp::AlzheimerPOMDPProblem
    updater::DiscreteUpdater
    gamma::Float64
end

MyopicPOMDPPlanner(pomdp::AlzheimerPOMDPProblem, updater::DiscreteUpdater) = MyopicPOMDPPlanner(pomdp, updater, discount(pomdp))

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
