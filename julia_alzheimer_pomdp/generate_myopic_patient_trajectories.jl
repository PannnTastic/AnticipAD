#!/usr/bin/env julia
# Generate representative single-patient MyopicPOMDP trajectories.
# Each case starts with a diagnostic test, updates belief by exact Bayes,
# repeats testing while useful, and stops at a terminal action.

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP
using POMDPs, POMDPTools, Random, Plots

const MAX_STEPS = 20
const SEARCH_SEEDS = 1:5000

const STATE_ORDER = [:CN, :MCI, :Dementia]
const STATE_LABELS = Dict(:CN => "CN", :MCI => "MCI", :Dementia => "Dementia")
const ACTION_LABELS = Dict(
    :A_Wait => "Wait",
    :A_Test_MMSE => "MMSE",
    :A_Test_CDR => "CDR",
    :A_Test_APOE4 => "APOE4",
    :A_Treat_MCI => "Treat MCI",
    :A_Treat_Dementia => "Treat Dementia"
)
const ACTION_COLORS = Dict(
    :A_Wait => RGB(0.18, 0.80, 0.44),
    :A_Test_MMSE => RGB(0.94, 0.76, 0.06),
    :A_Test_CDR => RGB(0.20, 0.60, 0.86),
    :A_Test_APOE4 => RGB(0.61, 0.35, 0.71),
    :A_Treat_MCI => RGB(0.95, 0.61, 0.07),
    :A_Treat_Dementia => RGB(0.91, 0.30, 0.24)
)
const STATE_COLORS = Dict(:CN => RGB(0.18, 0.80, 0.44), :MCI => RGB(0.95, 0.61, 0.07), :Dementia => RGB(0.91, 0.30, 0.24))

correct_terminal(state::Symbol) = state == :CN ? :A_Wait : (state == :MCI ? :A_Treat_MCI : :A_Treat_Dementia)

function belief_vector(b)
    return [pdf(b, AlzheimerState(s)) for s in STATE_ORDER]
end

function run_trace(pomdp, updater, true_state::AlzheimerState, seed::Int)
    rng = MersenneTwister(seed)
    b0 = initialize_belief(updater, initialstate(pomdp))
    policy = MyopicPOMDPPlanner(pomdp, updater)
    AlzheimerPOMDP.reset_policy!(policy)

    s = true_state
    b = b0
    actions = Symbol[]
    observations = Symbol[]
    beliefs_before = Vector{Float64}[]
    beliefs_after = Vector{Float64}[]
    terminal = nothing

    for step in 1:MAX_STEPS
        push!(beliefs_before, belief_vector(b))
        a = action(policy, b)
        push!(actions, a.name)

        sp = rand(rng, transition(pomdp, s, a))
        o = rand(rng, observation(pomdp, a, sp))
        push!(observations, o.name)

        if !AlzheimerPOMDP.isterminal_action(a)
            b = update(updater, b, a, o)
        end
        push!(beliefs_after, belief_vector(b))

        s = sp
        if AlzheimerPOMDP.isterminal_action(a)
            terminal = a.name
            break
        end
    end

    return (
        seed = seed,
        actions = actions,
        observations = observations,
        beliefs_before = beliefs_before,
        beliefs_after = beliefs_after,
        terminal = terminal
    )
end

function find_representative_trace(pomdp, updater, true_state_name::Symbol)
    target = correct_terminal(true_state_name)
    true_state = AlzheimerState(true_state_name)
    fallback = nothing

    for seed in SEARCH_SEEDS
        tr = run_trace(pomdp, updater, true_state, seed)
        starts_with_test = !isempty(tr.actions) && tr.actions[1] in (:A_Test_MMSE, :A_Test_CDR, :A_Test_APOE4)
        correct = tr.terminal == target
        has_terminal = tr.terminal !== nothing
        if fallback === nothing && starts_with_test && has_terminal
            fallback = tr
        end
        if starts_with_test && correct && length(tr.actions) >= 2
            return tr
        end
    end

    fallback === nothing && error("No usable trajectory found for $true_state_name")
    return fallback
end

function action_strip_plot(actions, observations)
    n = length(actions)
    p = plot(xlims=(0.5, max(n, 3) + 0.5), ylims=(0, 1), yaxis=false, yticks=false,
             xlabel="Visit", legend=false, size=(900, 120), framestyle=:box)
    for (i, a) in enumerate(actions)
        plot!([i - 0.42, i + 0.42], [0.05, 0.05], fillrange=0.95,
              color=ACTION_COLORS[a], linecolor=:black, linewidth=0.5, label="")
        txt_color = a in (:A_Test_MMSE, :A_Test_APOE4, :A_Treat_MCI) ? :black : :white
        annotate!(i, 0.62, text(ACTION_LABELS[a], 8, :center, txt_color))
        obs = observations[i] == :None ? "Terminal" : string(observations[i])
        annotate!(i, 0.28, text(obs, 6, :center, txt_color))
    end
    return p
end

function legend_plot()
    items = [
        (:A_Wait, "Green: Wait / CN terminal"),
        (:A_Test_MMSE, "Yellow: MMSE test"),
        (:A_Test_CDR, "Blue: CDR test"),
        (:A_Test_APOE4, "Purple: APOE4 test"),
        (:A_Treat_MCI, "Orange: Treat MCI terminal"),
        (:A_Treat_Dementia, "Red: Treat Dementia terminal")
    ]
    p = plot(xlims=(0, 6), ylims=(0, 1), axis=false, border=:none, legend=false)
    for (i, (a, label)) in enumerate(items)
        x = i - 0.85
        plot!([x, x + 0.22], [0.45, 0.45], fillrange=0.75,
              color=ACTION_COLORS[a], linecolor=:black, linewidth=0.4, label="")
        annotate!(x + 0.32, 0.60, text(label, 7, :left, :black))
    end
    return p
end

function plot_trace(trace, true_state_name::Symbol)
    n = length(trace.actions)
    before = hcat(trace.beliefs_before...)'
    after = hcat(trace.beliefs_after...)'
    x_before = collect(1:n)
    x_after = x_before .+ 0.35

    state_title = true_state_name == :CN ? "CN (Cognitively Normal)" :
                  true_state_name == :MCI ? "MCI (Mild Cognitive Impairment)" : "Dementia"
    target = ACTION_LABELS[correct_terminal(true_state_name)]

    p_bel = plot(size=(900, 420), dpi=300,
                 title="MyopicPOMDP Belief Updating - True State: $state_title | Correct Terminal: $target | Seed=$(trace.seed)",
                 xlabel="Visit", ylabel="Belief Probability", ylims=(0, 1), xlims=(0.7, max(n, 3) + 0.75),
                 grid=true, gridalpha=0.25)
    for (idx, st) in enumerate(STATE_ORDER)
        plot!(x_before, before[:, idx], marker=:circle, linewidth=2.2,
              color=STATE_COLORS[st], label="P($(STATE_LABELS[st])) before action")
        scatter!(x_after, after[:, idx], marker=:diamond, markersize=4,
                 color=STATE_COLORS[st], label="P($(STATE_LABELS[st])) after update")
    end
    vline!(collect(1:n), color=:gray, linestyle=:dot, linewidth=0.8, label="")

    p_actions = action_strip_plot(trace.actions, trace.observations)
    p_legend = legend_plot()

    fig = plot(p_bel, p_actions, p_legend, layout=@layout([a{0.68h}; b{0.20h}; c{0.12h}]),
               size=(1100, 760), dpi=300, margin=5Plots.mm)

    mkpath("figures_paper")
    base = "v2_12_myopic_patient_trajectory_$(lowercase(string(true_state_name)))"
    savefig(fig, "figures_paper/$(base).png")
    savefig(fig, "figures_paper/$(base).pdf")
    println("Saved: figures_paper/$(base).png")
end

function main()
    println("=" ^ 72)
    println("MYOPIC SINGLE-PATIENT BELIEF + ACTION TRAJECTORIES")
    println("=" ^ 72)
    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    updater = DiscreteUpdater(pomdp)

    for true_state_name in STATE_ORDER
        println("Finding representative trajectory for $true_state_name...")
        tr = find_representative_trace(pomdp, updater, true_state_name)
        println("  seed=$(tr.seed), actions=$(tr.actions), terminal=$(tr.terminal)")
        plot_trace(tr, true_state_name)
    end

    println("All representative trajectory figures saved to figures_paper/")
end

main()
