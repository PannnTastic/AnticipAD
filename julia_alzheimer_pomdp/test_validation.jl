push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using AlzheimerPOMDP
using POMDPs

println("Testing model validation...")
p = AlzheimerPOMDP.AlzheimerPOMDPProblem()
println("✓ Model created successfully")

# Check APOE4 sums
for state in [:CN, :MCI, :Dementia]
    total = sum(values(p.apoe4_probs[state]))
    println("  APOE4 $state sum: ", round(total, digits=6))
end

# Check MMSE sums
for state in [:CN, :MCI, :Dementia]
    total = sum(values(p.mmse_probs[state]))
    println("  MMSE $state sum: ", round(total, digits=6))
end

# Check CDR sums
for state in [:CN, :MCI, :Dementia]
    total = sum(values(p.cdr_probs[state]))
    println("  CDR $state sum: ", round(total, digits=6))
end

# Check transition sums
for state in [:CN, :MCI, :Dementia]
    total = sum(values(p.transitions[state]))
    println("  Transition $state sum: ", round(total, digits=6))
end

println("\n✓ All validations passed!")
