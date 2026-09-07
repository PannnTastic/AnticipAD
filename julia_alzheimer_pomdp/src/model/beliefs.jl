# Visit-level prior; Bayesian filtering uses POMDPTools.DiscreteUpdater unchanged.
POMDPs.initialstate(p::AlzheimerPOMDPProblem) = SparseCat(STATES, [0.385189, 0.417041, 0.197770])
