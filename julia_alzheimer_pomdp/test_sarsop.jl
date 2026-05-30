using POMDPs, POMDPTools, SARSOP, AlzheimerPOMDP

pomdp = AlzheimerPOMDP.AlzheimerPOMDPProblem(discount=0.95)
println("Testing SARSOP...")
solver = SARSOPSolver(precision=1e-2, timeout=10.0)
policy = solve(solver, pomdp)
println("SARSOP solved successfully!")
