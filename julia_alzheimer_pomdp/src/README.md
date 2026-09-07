# Julia Source Layout

`AlzheimerPOMDP.jl` remains the public module and compatibility entry point.
Existing `include("src/AlzheimerPOMDP.jl")` calls and package imports continue
to work. Component files are included into that same module; do not include
them individually or introduce another copy of the model in experiment scripts.

```text
src/
  AlzheimerPOMDP.jl         imports, exports, ordered component includes
  model/
    states.jl              state type, identity, ordered state space
    actions.jl             action type, ordered actions, test/terminal predicates
    observations.jl        observation type and ordered observation space
    problem.jl             mutable model container and constructor
    transitions.jl         default T, validation, transition distribution
    observation_model.jl   MMSE/CDR/APOE4 defaults, validation, likelihoods
    rewards.jl             default utility table and reward interface
    beliefs.jl             visit-level initial belief
    interfaces.jl          POMDPs.jl spaces, indices, discount, terminal-state API
  policies/
    random.jl              uniform random action policy
    expert.jl              threshold policy and per-episode history reset
    myopic.jl              exact one-step Bayesian lookahead
    rollout.jl             fully observed rollout used by the POMCP factory
    reset.jl               default reset hook for stateless policies
  solvers/
    factories.jl           existing solver constructors and their defaults
  simulation/
    episode.jl             execution, stopping, fallback, intake agreement
    benchmark.jl           legacy multi-policy five-step driver
    summary.jl             aggregate console summaries
  visualization/
    plots.jl               plotting helpers, separate from model logic
```

Bayesian updating still uses `POMDPTools.DiscreteUpdater`, not a newly written
filter. All model values, public names, action/observation ordering, default
discount, random-number consumption, solver settings, and episode return
fields are preserved. Constructor calls allocate fresh parameter dictionaries,
so sensitivity scripts can continue modifying one model without changing another.

The legacy `run_episode` and `run_all_policies` defaults remain five steps.
The main benchmark explicitly passes 20. Evaluation stops at terminal actions,
while `isterminal(model, state)` remains false for the continuing solver model.
These distinctions are intentional compatibility constraints, not corrections
to the limitations discussed in the paper.

Top-level experiment scripts and stored result paths are retained to avoid
breaking existing commands, policy files, or artifact references. New source
logic belongs in the folders above; new regression tests belong in `test/`.

## Regression Tests

From `julia_alzheimer_pomdp/`:

```sh
julia --project=. test/runtests.jl
```

The fixture `test/fixtures/pre_refactor.json` was captured from the original
monolithic source before extraction, using the repository's frozen three-state
SARSOP policy. It records the source and policy SHA-256 hashes. Tests compare
model distributions, Bayesian posteriors, 198 belief-grid decisions, 288 seeded
episodes across four policies and two horizons, and forced 21-action fallback
episodes exactly. The fixture contains simulated records only, not ADNI patients.

Tests do not overwrite reference results, retrain solvers, or regenerate the
fixture. Additional online solver factories are checked for API presence, not
scientific validity or convergence. This suite is a behavior-preservation check,
not a new clinical validation or a rerun of the 100,000-episode benchmark.

If a local `policy.out` has been replaced by another experiment, use the exact
repository three-state policy and set `ANTICIPAD_TEST_POLICY` to its path. A hash
mismatch signals that the regression baseline no longer matches; do not silently
overwrite the reference fixture to make tests pass.
