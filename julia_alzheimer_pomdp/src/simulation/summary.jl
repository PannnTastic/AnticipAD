function summarize_results(results::Dict{String, Vector{NamedTuple}})
    println("\n" * "=" ^ 80)
    println("MONTE CARLO RESULTS - JULIA ENVIRONMENT")
    println("=" ^ 80)

    println("\nPolicy          Avg Reward    Accuracy    Test Cost    Steps    CN     MCI    Dem")
    println("-" ^ 80)

    for name in sort(collect(keys(results)))
        data = results[name]
        rewards = [d.total_reward for d in data]
        acc = mean([d.correct for d in data]) * 100
        cost = mean([d.test_cost for d in data])
        steps = mean([d.steps for d in data])

        cn_data = [d.correct for d in data if d.true_state == :CN]
        mci_data = [d.correct for d in data if d.true_state == :MCI]
        dem_data = [d.correct for d in data if d.true_state == :Dementia]

        cn_acc = isempty(cn_data) ? 0.0 : mean(cn_data) * 100
        mci_acc = isempty(mci_data) ? 0.0 : mean(mci_data) * 100
        dem_acc = isempty(dem_data) ? 0.0 : mean(dem_data) * 100

        line = lpad(name, 14) * " " *
               lpad(round(mean(rewards), digits=2), 10) * " " *
               lpad(round(acc, digits=1), 9) * "% " *
               lpad(round(cost, digits=2), 10) * " " *
               lpad(round(steps, digits=2), 8) * " " *
               lpad(round(cn_acc, digits=1), 6) * "% " *
               lpad(round(mci_acc, digits=1), 6) * "% " *
               lpad(round(dem_acc, digits=1), 6) * "%"
        println(line)
    end
    println("-" ^ 80)
end
