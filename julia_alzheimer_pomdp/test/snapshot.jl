using POMDPs, POMDPTools, Random, SARSOP, JSON

include(joinpath(@__DIR__, "..", "src", "AlzheimerPOMDP.jl"))
using .AlzheimerPOMDP

struct AlwaysTestPolicy <: Policy end
POMDPs.action(::AlwaysTestPolicy, b) = AlzheimerAction(:A_Test_MMSE)

test_policy_path() = get(ENV, "ANTICIPAD_TEST_POLICY", joinpath(@__DIR__, "..", "policy.out"))

function behavior_snapshot()
    p = AlzheimerPOMDPProblem()
    updater = DiscreteUpdater(p)
    b0 = initialize_belief(updater, initialstate(p))
    frozen = SARSOP.load_policy(p, test_policy_path())
    make_policy(name, seed) = name == "SARSOP" ? frozen :
        name == "MyopicPOMDP" ? MyopicPOMDPPlanner(p, updater) :
        name == "Expert" ? ExpertPolicy(p) : AlzheimerPOMDP.RandomPolicy(p, MersenneTwister(seed))

    model = Dict(
        "states" => [string(s.name) for s in states(p)],
        "actions" => [string(a.name) for a in actions(p)],
        "observations" => [string(o.name) for o in observations(p)],
        "prior" => [pdf(initialstate(p), s) for s in states(p)],
        "discount" => discount(p),
        "transitions" => [pdf(transition(p, s, a), sp)
            for s in states(p) for a in actions(p) for sp in states(p)],
        "likelihoods" => [pdf(observation(p, a, s), o)
            for s in states(p) for a in actions(p) for o in observations(p)],
        "rewards" => [reward(p, s, a) for s in states(p) for a in actions(p)],
    )
    posteriors = []
    for a in actions(p), o in observations(p)
        AlzheimerPOMDP.is_test(a) || continue
        prob = sum(pdf(b0, s) * pdf(transition(p, s, a), sp) *
            pdf(observation(p, a, sp), o) for s in states(p), sp in states(p))
        prob > 0 || continue
        bp = update(updater, b0, a, o)
        push!(posteriors, (action=string(a.name), observation=string(o.name),
            belief=[pdf(bp, s) for s in states(p)]))
    end
    decisions = []
    for name in ["SARSOP", "MyopicPOMDP", "Expert"], i in 0:10, j in 0:(10-i)
        b = initialize_belief(updater,
            SparseCat(AlzheimerPOMDP.STATES, [i, j, 10-i-j] ./ 10))
        pol = make_policy(name, 42)
        push!(decisions, (policy=name, i=i, j=j, action=string(action(pol, b).name)))
    end
    episodes = []
    for name in ["SARSOP", "MyopicPOMDP", "Expert", "Random"],
            seed in [42, 123], horizon in [5, 20]
        pol = make_policy(name, seed)
        rng = MersenneTwister(seed)
        for s in states(p), repetition in 1:6
            result = run_episode(p, pol, updater, b0, s; max_steps=horizon, rng=rng)
            push!(episodes, (policy=name, seed=seed, horizon=horizon,
                repetition=repetition, result=result))
        end
    end
    fallback = [run_episode(p, AlwaysTestPolicy(), updater, b0, s;
        max_steps=20, rng=MersenneTwister(42)) for s in states(p)]
    return Dict("model" => model, "posteriors" => posteriors,
        "decisions" => decisions, "episodes" => episodes, "fallback" => fallback)
end
