module AlzheimerPOMDP

using POMDPs
using POMDPTools
using BasicPOMCP
using ARDESPOT
using PointBasedValueIteration
using SARSOP
using MCTS
using Distributions
using Random
using Statistics
using Plots
using StatsBase
using LinearAlgebra

export AlzheimerState, AlzheimerAction, AlzheimerObs, AlzheimerPOMDPProblem
export RandomPolicy, ExpertPolicy, MyopicPOMDPPlanner
export make_pomcp_policy, make_despot_policy, make_pbvi_policy, make_sarsop_policy, make_mcts_policy
export run_episode, run_all_policies, summarize_results
export generate_all_figures, plot_solver_comparisons

# Preserve the public module and script entry point while separating responsibilities.
include("model/states.jl")
include("model/actions.jl")
include("model/observations.jl")
include("model/problem.jl")
include("model/transitions.jl")
include("model/observation_model.jl")
include("model/rewards.jl")
include("model/interfaces.jl")
include("model/beliefs.jl")
include("policies/random.jl")
include("policies/expert.jl")
include("policies/rollout.jl")
include("policies/myopic.jl")
include("policies/reset.jl")
include("solvers/factories.jl")
include("simulation/episode.jl")
include("simulation/benchmark.jl")
include("simulation/summary.jl")
include("visualization/plots.jl")

end # module
