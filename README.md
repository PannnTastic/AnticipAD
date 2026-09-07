# AnticipAD - POMDP Simulation for Alzheimer Staging

Reproduction code for *"A POMDP-Based Framework for Healthcare Decision-Making to Mitigate Cognitive Decline in Alzheimer's Disease."*

This is a **simulation benchmark**, not a validated clinical tool. Evaluation uses an ADNI-derived simulator with explicit observation and utility assumptions, not held-out patient outcomes. Management actions end an episode and do not modify progression, so the experiments do **not** establish mitigation of cognitive decline.

## Environment

- **Julia** 1.11.3.
- Packages are pinned in `julia_alzheimer_pomdp/Project.toml` and `Manifest.toml`: POMDPs.jl, POMDPTools, SARSOP.jl, BasicPOMCP, ARDESPOT, Distributions, StatsBase, and JSON.
- CPU execution; no GPU is required. Solver runtime depends on configuration and hardware.

```bash
cd julia_alzheimer_pomdp
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Data and Parameter Provenance

Source: ADNI via the [ADNIMERGE2 R package](https://atri-biostats.github.io/ADNIMERGE2), accessed under ADNI's Data Use Agreement. Patient-level data are **not redistributed** here. Obtain authorized [ADNI access](https://adni.loni.usc.edu/) and use the extraction scripts and preprocessing notebook in `preprocessing/` with your own data.

- The assembled table contains 15,836 visit records from 3,777 participants. Listwise exclusion of records missing diagnosis or MMSE leaves 12,464 records from 3,713 participants. No imputation is applied.
- The initial belief is `[0.385, 0.417, 0.198]` in CN, MCI, Dementia order, estimated from retained **visit-level** frequencies. It is not an independently estimated patient-intake distribution.
- The transition matrix, initial belief, and APOE4 likelihoods are ADNI-derived. The model additionally makes Dementia absorbing and suppresses MCI-to-CN reversion.
- **MMSE/CDR likelihoods and utilities are hand-specified**, not direct frequency estimates. Diagnostic labels partly incorporate these assessments; manually specified likelihoods do not establish independent validity or remove the need for external evaluation.
- The notebook sorts records by participant and date and counts consecutive diagnoses, without interval filtering or time rescaling. Transitions therefore describe a modeled visit step, **not calibrated six-month progression**. Irregular intervals, undated records, and duplicate-date records require further treatment for calendar-time analysis.

### Why the Horizon Is 20

The evaluation cap `H=20` was chosen from the maximum visit count per participant in the assembled dataset **before** listwise exclusion. The maximum is 19 after diagnosis/MMSE exclusion; the original cap is retained. One non-terminal test represents one modeled follow-up visit, not a same-day test bundle. Twenty epochs should not be converted into ten years.

If all 20 decisions are non-terminal tests, the evaluator appends a fallback `Wait`, so the recorded action count can reach 21. This cap is not SARSOP's planning horizon or a clinically optimized follow-up duration. Horizon sensitivity has not been established.

The read-only PowerShell audit in `reproducibility/audits/audit_visit_horizon.ps1` accepts an explicit `-Dataset` path. Its committed output contains aggregate counts and a dataset hash, not patient records.

## Core Model

`julia_alzheimer_pomdp/src/AlzheimerPOMDP.jl` remains the public entry point. Its implementation is now separated into `model/`, `policies/`, `solvers/`, `simulation/`, and `visualization/`. State, action, observation, transition, reward, and initial-belief definitions have dedicated files. Existing experiment imports and artifact paths remain compatible. See the [source layout and regression guide](julia_alzheimer_pomdp/src/README.md).

This refactor preserves model parameters, planner behavior, evaluation semantics, and archived seed results. Run `julia --project=. test/runtests.jl` from `julia_alzheimer_pomdp/` to compare against a pre-refactor simulation fixture. The check covers 198 belief-grid decisions, 288 seeded episodes, model distributions, Bayesian updates, and horizon fallback behavior. It is not a new benchmark experiment. Legacy plotting labels now distinguish retained visit records, visit-step transitions, and intake-stage agreement.

## Reproducing the Paper

Run the Julia commands below from `julia_alzheimer_pomdp/` with `julia --project=. <script>`. Solver scripts can overwrite `model.pomdpx` and `policy.out`; preserve frozen artifacts before re-solving.

| Element | Script | Output or evidence |
|---|---|---|
| Individual 100,000-episode benchmarks | `solver_sarsop.jl`, `solver_myopic.jl`, `solver_expert.jl`, `solver_random.jl`, then `combine_results.jl` | `results_*_100000.json`, `combined_results_100000.json` |
| Ten-seed benchmark used for revised Table II | `run_multiseed_perseed.jl` | `per_seed_results.json` |
| Mean, sample SD, and macro-accuracy audit | `../reproducibility/audits/audit_metrics.jl` | `camera_ready_metrics.json` beside the audit script |
| Paired Wilcoxon, bootstrap interval, Cliff's delta, Holm correction | `stats_paired.jl` | Console; `stats_paired.py` is an optional reference implementation |
| Threshold-tuned Expert baseline | `tune_expert.jl` | Console |
| SARSOP belief-region extraction | `extract_sarsop_policy.jl` | `sarsop_policy_grid.json` |
| Reward sensitivity cited in the revision | `sensitivity_analysis.jl` | `sensitivity_analysis_results.json`; 1,000 episodes, seed 42, five-decision cap |
| CDR MCI-row sensitivity cited in the revision | `sensitivity_cdr_raw.jl` | Archived `sensitivity_cdr_output.txt`; 10,000 episodes, seed 42, 20-decision cap |
| Additional sensitivity driver | `run_sensitivity_analysis.jl` | Separate protocol; do not conflate its output with the cited reward experiment |
| Additional raw-MMSE checks | `test_raw_mmse.jl`, `test_mmse_costdecomp.jl`, `rerun_rawmmse_benchmark.jl` | Console and archived logs |
| Model probability checks | `test_validation.jl` | Console |
| Figures | `generate_paper_figures_v2.jl`, `generate_*_trajectory.jl` | `figures_paper/` |

The ten benchmark seeds are `{42, 123, 456, 789, 1000, 2024, 314, 271, 100, 999}`, with 10,000 episodes per policy per seed and discount `gamma=0.95`. The frozen SARSOP policy is reused across seeds, not retrained per seed. Reproduction also requires matching model, policy, package versions, and random-number use.

## Audited Results and Interpretation

All accuracy values below measure **terminal commitment agreement with the intake stage**. They do not measure agreement with the evolving stage at commitment. An initially MCI episode that progresses to Dementia and receives dementia management is counted as incorrect against intake, even if that commitment matches the current state.

| Policy | Reward (utility points) | Episode-weighted accuracy (%) | Macro accuracy (%) | MCI agreement (%) | Dementia agreement (%) | Mean decisions |
|---|---:|---:|---:|---:|---:|---:|
| SARSOP | 68.7 +/- 1.5 | 78.3 +/- 0.4 | 76.6 +/- 0.5 | 81.2 +/- 0.6 | 68.2 +/- 1.3 | 2.50 |
| MyopicPOMDP | 73.0 +/- 1.2 | 78.1 +/- 0.3 | 79.8 +/- 0.3 | 71.4 +/- 0.6 | 87.3 +/- 0.7 | 2.87 |
| Expert | 44.7 +/- 1.9 | 71.1 +/- 0.4 | 72.1 +/- 0.5 | 61.3 +/- 0.5 | 75.1 +/- 1.0 | 3.34 |
| Random | -134.9 +/- 4.4 | 33.4 +/- 0.5 | 33.4 +/- 0.6 | 33.3 +/- 0.8 | 33.3 +/- 1.1 | 2.01 |

Values are means and sample standard deviations over ten seeds. Overall accuracy weights episodes; macro accuracy averages the three stage accuracies **within each seed** before aggregation. The two measures can rank policies differently. Reward uses the evolving pre-action state, whereas accuracy uses the intake stage. Testing expenditure sums undiscounted test costs; decision counts include the terminal action.

SARSOP has higher MCI agreement and lower testing expenditure among informed policies, while MyopicPOMDP has higher Dementia agreement and aggregate reward. SARSOP was planned on a continuing model, while evaluation terminates at commitment. Consequently, these rewards are not a matched-objective optimality comparison. Neither solver universally dominates, and sensitivity experiments can reverse rankings. The single-seed stress tests do not establish general robustness, clinical utility, or joint MMSE/CDR validity.

## Reproducibility Assets

The [reproducibility guide](reproducibility/README.md) documents numerical audits and standalone figure assets. `reproducibility/figures/` contains the figures and the illustrative trajectory figure source; `reproducibility/audits/` contains the audit code and aggregate outputs. Full manuscript PDFs, manuscript LaTeX, bibliography files, and conference templates are intentionally excluded from the current repository tree.

The documentation corrects accuracy definitions, explains the dataset-based horizon and visit-time abstraction, and clarifies model limitations. This update does not introduce new benchmark runs. Older experiment reports remain historical records and may contain superseded interpretations; use this README for current claims.

## Repository Status

- Preprocessing notebook and R extraction scripts are included; restricted patient data are not.
- `Project.toml` and `Manifest.toml` record the Julia environment.
- Numerical and visit-count audits and standalone figures are provided for reproduction.
- External validation, treatment-effect modeling, persistent genetic covariates, and decision-time evaluation remain future work.
- A repository-wide license has not been added; no new redistribution rights are granted by this revision.
