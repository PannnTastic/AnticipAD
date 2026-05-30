#!/usr/bin/env julia
# Generate Belief + Action Trajectory Figures for Early Diagnosis Narrative

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, JSON, Random, Plots
using SARSOP

const SEED = 42
const MAX_STEPS = 20
const SOLVER_ORDER = ["SARSOP", "MyopicPOMDP", "Expert", "Random"]

function run_episode_with_belief(pomdp, policy, updater, b0, true_state; max_steps=20, rng=Random.GLOBAL_RNG)
    AlzheimerPOMDP.reset_policy!(policy)
    s = true_state
    b = b0
    action_history = Symbol[]
    obs_history = Symbol[]
    belief_history = Vector{Float64}[]
    for step in 1:max_steps
        a = action(policy, b)
        push!(action_history, a.name)
        push!(belief_history, [pdf(b, AlzheimerState(st)) for st in [:CN, :MCI, :Dementia]])
        sp = rand(rng, transition(pomdp, s, a))
        o = rand(rng, observation(pomdp, a, sp))
        push!(obs_history, o.name)
        if !AlzheimerPOMDP.isterminal_action(a)
            b = update(updater, b, a, o)
        end
        s = sp
        if AlzheimerPOMDP.isterminal_action(a)
            break
        end
    end
    return (actions=action_history, observations=obs_history, beliefs=belief_history)
end

function main()
    println("=" ^ 70)
    println("BELIEF + ACTION TRAJECTORY FIGURES")
    println("=" ^ 70)
    pomdp = AlzheimerPOMDPProblem(discount=0.95)
    rng = MersenneTwister(SEED)
    updater = POMDPTools.DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    policies = Dict{String, Any}(
        "SARSOP"      => SARSOP.load_policy(pomdp, "policy.out"),
        "MyopicPOMDP" => MyopicPOMDPPlanner(pomdp, updater),
        "Expert"      => ExpertPolicy(pomdp),
        "Random"      => POMDPTools.RandomPolicy(pomdp; rng=rng)
    )
    for true_clinical in [:CN, :MCI, :Dementia]
        println("\n--- Generating for true state: $true_clinical ---")
        true_state = AlzheimerState(true_clinical)
        results = Dict{String, NamedTuple}()
        for name in SOLVER_ORDER
            println("  Running $name...")
            r = run_episode_with_belief(pomdp, policies[name], updater, b0, true_state; max_steps=MAX_STEPS, rng=rng)
            results[name] = r
        end
        generate_combined_figure(results, true_clinical)
    end
    println("\nAll figures saved to figures_paper/")
end

function generate_combined_figure(results, true_state_name::Symbol)
    default(size=(1400, 2400), dpi=300, legendfontsize=16, guidefontsize=18,
            tickfontsize=15, titlefontsize=20, margin=8Plots.mm)

    action_colors = Dict(
        :A_Wait => RGB(0.18,0.80,0.44), :A_Test_MMSE => RGB(0.94,0.76,0.06),
        :A_Test_CDR => RGB(0.20,0.60,0.86), :A_Test_APOE4 => RGB(0.61,0.35,0.71),
        :A_Treat_MCI => RGB(0.95,0.61,0.07), :A_Treat_Dementia => RGB(0.91,0.30,0.24)
    )
    action_labels = Dict(:A_Wait=>"Wait", :A_Test_MMSE=>"MMSE", :A_Test_CDR=>"CDR",
                         :A_Test_APOE4=>"APOE4", :A_Treat_MCI=>"Tr.MCI", :A_Treat_Dementia=>"Tr.Dem")
    state_colors = [RGB(0.18,0.80,0.44), RGB(0.95,0.61,0.07), RGB(0.91,0.30,0.24)]
    state_names = ["CN", "MCI", "Dementia"]
    n_solvers = length(SOLVER_ORDER)
    max_steps = 20
    
    state_title = true_state_name == :CN ? "CN (Cognitively Normal)" :
                  (true_state_name == :MCI ? "MCI (Mild Cognitive Impairment)" : "Dementia")
    correct_action = true_state_name == :CN ? "Wait" : (true_state_name == :MCI ? "Treat MCI" : "Treat Dementia")
    
    all_plots = []
    for (ri, solver) in enumerate(SOLVER_ORDER)
        r = results[solver]
        n_steps = length(r.actions)
        t_range = 1:n_steps
        bel = hcat(r.beliefs...)'
        
        # Belief subplot
        p_bel = plot(t_range, bel[:,1], fillrange=0, fillalpha=0.3, color=state_colors[1],
                     label="", linewidth=2.5, legend=(ri==1 ? :outertopright : :none))
        plot!(t_range, bel[:,1] + bel[:,2], fillrange=bel[:,1], fillalpha=0.3,
              color=state_colors[2], label=(ri==1 ? "p(MCI)" : ""), linewidth=2.5)
        plot!(t_range, ones(n_steps), fillrange=bel[:,1] + bel[:,2], fillalpha=0.3,
              color=state_colors[3], label=(ri==1 ? "p(Dem)" : ""), linewidth=2.5)
        plot!(t_range, bel[:,1], color=state_colors[1], label=(ri==1 ? "p(CN)" : ""), linewidth=2.5)
        
        if true_state_name == :MCI
            hline!([0.50], color=:black, linestyle=:dash, linewidth=1.5, label="")
        elseif true_state_name == :Dementia
            hline!([0.52], color=:black, linestyle=:dash, linewidth=1.5, label="")
        elseif true_state_name == :CN
            hline!([0.82], color=:black, linestyle=:dash, linewidth=1.5, label="")
        end
        ylims!(0, 1)
        xlims!(0.5, max(n_steps, 5) + 0.5)
        ylabel!("Belief", fontsize=16)
        title!(solver, fontsize=18)
        if ri == n_solvers
            xlabel!("Visit (t)", fontsize=16)
        else
            plot!(xticks=:none)
        end
        
        # Action subplot
        p_act = plot(size=(900, 150), margin=4Plots.mm)
        for (ci, a) in enumerate(r.actions)
            plot!([ci-0.4, ci+0.4], [0, 0], fillrange=1, fillalpha=0.9,
                  color=action_colors[a], linecolor=:black, linewidth=0.8, label="")
            label_color = a in (:A_Test_MMSE, :A_Test_APOE4, :A_Treat_MCI) ? :black : :white
            annotate!(ci, 0.5, text(action_labels[a], 14, :center, label_color))
        end
        xlims!(0.5, max(n_steps, 5) + 0.5)
        ylims!(0, 1)
        yaxis!(false, ticks=:none)
        if ri == n_solvers
            xlabel!("Visit (t)", fontsize=9)
        else
            plot!(xticks=:none)
        end
        
        push!(all_plots, p_bel, p_act)
    end
    
    p_legend = plot(xlims=(0, 6), ylims=(0, 1), axis=false, border=:none, legend=false, size=(900, 240))
    legend_items = [
        (:A_Wait, "Green = Wait / healthy terminal"),
        (:A_Test_MMSE, "Yellow = MMSE test"),
        (:A_Test_CDR, "Blue = CDR test"),
        (:A_Test_APOE4, "Purple = APOE4 test"),
        (:A_Treat_MCI, "Orange = Treat MCI terminal"),
        (:A_Treat_Dementia, "Red = Treat Dementia terminal")
    ]
    for (i, (act, label)) in enumerate(legend_items)
        x0 = i - 0.85
        plot!([x0, x0 + 0.25], [0.45, 0.45], fillrange=0.75, color=action_colors[act],
              linecolor=:black, linewidth=0.6, label="")
        annotate!(x0 + 0.55, 0.60, text(label, 13, :left, :black))
    end
    push!(all_plots, p_legend)

    # Use @layout with explicit ratios
    l = @layout [a{0.6h}; b{0.4h}; c{0.6h}; d{0.4h}; e{0.6h}; f{0.4h}; g{0.6h}; h{0.4h}; i{0.28h}]
    fig = plot(all_plots..., layout=l, size=(1400, 320*n_solvers), dpi=300,
               plot_title="Belief Updating + Action Trajectory — True State: $state_title | Correct Terminal Action: $correct_action",
               margin=5Plots.mm)
    
    mkpath("figures_paper")
    filename_base = "v2_10_belief_trajectory_$(lowercase(string(true_state_name)))"
    savefig("figures_paper/$(filename_base).png")
    savefig("figures_paper/$(filename_base).pdf")
    println("  Saved: figures_paper/$(filename_base).png")
end

main()
