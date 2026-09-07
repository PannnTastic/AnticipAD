using Test, SHA
include("snapshot.jl")

@testset "Model contracts" begin
    p = AlzheimerPOMDPProblem()
    @test length(states(p)) == 3
    @test length(actions(p)) == 6
    @test length(observations(p)) == 10
    for (i, s) in enumerate(states(p))
        @test stateindex(p, s) == i
        @test !isterminal(p, s)
        for a in actions(p)
            probabilities = [pdf(transition(p, s, a), sp) for sp in states(p)]
            @test sum(probabilities) ≈ 1.0
            @test all(x -> 0 <= x <= 1, probabilities)
            probabilities = [pdf(observation(p, a, s), o) for o in observations(p)]
            @test sum(probabilities) ≈ 1.0
            @test all(x -> 0 <= x <= 1, probabilities)
        end
    end
    @test sum(pdf(initialstate(p), s) for s in states(p)) ≈ 1.0
    @test discount(AlzheimerPOMDPProblem(discount=0.8)) == 0.8
    @test p.transitions[:Dementia][:Dementia] == 1.0
    @test p.transitions[:MCI][:CN] == 0.0
    @test p.rewards[:Dementia][:A_Wait] == -1000
    @test_throws ErrorException AlzheimerPOMDP._validate_transition_probs(
        Dict(:CN => Dict(:CN => 0.5)))
    @test_throws ErrorException AlzheimerPOMDP._validate_observation_probs(
        Dict(:CN => Dict(:None => 0.5)), "invalid")
    second = AlzheimerPOMDPProblem()
    p.rewards[:CN][:A_Wait] = -99.0
    p.cdr_probs[:MCI][:CDR_0] = 0.1
    @test second.rewards[:CN][:A_Wait] == 30.0
    @test second.cdr_probs[:MCI][:CDR_0] == 0.25
    for factory in [make_pomcp_policy, make_despot_policy, make_pbvi_policy,
            make_sarsop_policy, make_mcts_policy]
        @test hasmethod(factory, Tuple{AlzheimerPOMDPProblem})
    end
end

@testset "Exact pre-refactor behavior" begin
    fixture = JSON.parsefile(joinpath(@__DIR__, "fixtures", "pre_refactor.json"))
    @test bytes2hex(sha256(read(test_policy_path()))) ==
        fixture["policy_sha256"]
    actual = JSON.parse(JSON.json(behavior_snapshot()))
    expected = fixture["snapshot"]
    for section in ["model", "posteriors", "decisions", "episodes", "fallback"]
        @test actual[section] == expected[section]
    end
    @test length(actual["episodes"]) == 288
    @test length(actual["decisions"]) == 198
    for result in actual["fallback"]
        @test result["steps"] == 21
        @test result["terminal_action"] == "A_Wait"
        @test result["correct"] == (result["true_state"] == "CN")
    end
end
