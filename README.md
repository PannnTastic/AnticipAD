# AnticipAD — POMDP Simulation for Alzheimer Staging

Reproduction code for *"A POMDP-Based Framework for Healthcare Decision-Making to Mitigate Cognitive Decline in Alzheimer’s Disease"*

This is a **simulation benchmark**, not a validated clinical tool. All accuracy figures are in-sample (ADNI-derived simulator); external validation (NACC/OASIS) is future work.

## Environment

- **Julia** 1.11.3
- **Packages** (pinned in `julia_alzheimer_pomdp/Project.toml` + `Manifest.toml`): POMDPs.jl, POMDPTools, SARSOP.jl, BasicPOMCP, ARDESPOT, Distributions, StatsBase, JSON
- CPU only; no GPU. Full benchmark (10 seeds × 4 policies × 10k episodes) runs in < 10 min on one core.

```bash
cd julia_alzheimer_pomdp
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Data

Source: ADNI via the **ADNIMERGE2** R data package (v0.1.1, ATRI Biostatistics, built 2025-12-17; https://atri-biostats.github.io/ADNIMERGE2), accessed under ADNI's Data Use Agreement (https://adni.loni.usc.edu). Per ADNI terms, processed data are **not redistributed** here — obtain your own ADNI access and regenerate the analytic table with the preprocessing notebook (see below).

- Retained analytic table: 15,836 visit records / 3,777 unique RIDs (no imputation; listwise deletion of rows missing DX or MMSE → 12,464 rows for the prior/transition estimates).
- Initial belief b₀ = [0.385, 0.417, 0.198] estimated on the 12,464 records (3,713 participants) with complete diagnosis + MMSE.
- **Hybrid provenance (important):** the transition matrix, APOE4 likelihoods, and b₀ are ADNI-derived (transitions reproduce from `preprocessing/Pra_pemrosesan_Dataset_POMDP.ipynb`; APOE4 matches the CSV to 3 decimals). The **MMSE and CDR observation likelihoods and the reward function are hand-specified clinical models**, not extracted by frequency counting — raw ADNI MMSE/CDR are near-deterministic and would induce degenerate over-testing (see `test_raw_mmse.jl`). Code comments in the Python prototype mark this design intent.

## Core model

`julia_alzheimer_pomdp/src/AlzheimerPOMDP.jl` — the 3-state POMDP (CN/MCI/Dementia), all policies (Random, Expert, MyopicPOMDP), solver factories, and the `run_episode` / belief-update framework. Every script `include()`s this file.

## Reproducing the paper

| Paper element | Script | Output |
|---|---|---|
| Main benchmark (Tables: per-policy) | `solver_sarsop.jl`, `solver_myopic.jl`, `solver_expert.jl`, `solver_random.jl` → `combine_results.jl` | `results_*_100000.json` |
| Multi-seed mean±std (Table VII) | `run_multiseed_benchmark.jl` | console |
| **Per-seed values for significance tests** | `run_multiseed_perseed.jl` | `per_seed_results.json` |
| **Paired Wilcoxon + bootstrap CI + Cliff's δ + Holm** | `stats_paired.jl` (run on `per_seed_results.json`; `stats_paired.py` is an equivalent Python reference) | console |
| **Threshold-tuned Expert baseline** | `tune_expert.jl` | console |
| SARSOP policy extraction (decision regions) | `extract_sarsop_policy.jl` | `sarsop_policy_grid.json` |
| **Obs-model sensitivity: raw-MMSE re-solve** | `test_raw_mmse.jl` | console + `policy_rawmmse.out` |
| **Cost-vs-discriminability decomposition (2×2)** | `test_mmse_costdecomp.jl` | console |
| **Full raw-MMSE re-run benchmark** | `rerun_rawmmse_benchmark.jl` | console |
| Reward / monotonicity sensitivity | `run_sensitivity_analysis.jl` | `sensitivity_analysis_results.json` |
| Model validation (probs sum to 1) | `test_validation.jl` | console |
| Figures | `generate_paper_figures_v2.jl`, `generate_*_trajectory.jl` | `figures_paper/` |

### Determinism
Seeds `{42, 123, 456, 789, 1000, 2024, 314, 271, 100, 999}`; γ=0.95; horizon 20. The simulator is deterministic given a seed (MersenneTwister streams). SARSOP is solved once and the frozen `policy.out` (638 α-vectors) is reused across all seeds — it is **not** retrained per seed.

## Key result and its scope

Under the adopted (smoothed-MMSE) observation model, MyopicPOMDP matches SARSOP's reward with no offline training. This parity is **conditional**: under raw observed-ADNI MMSE the myopic planner over-tests (~10 visits) and accuracy falls to ~40%, while SARSOP stays at ~78% (`rerun_rawmmse_benchmark.jl`). MyopicPOMDP is therefore competitive at realistic operating points; SARSOP is the robust choice.

## Repository status
- [x] Preprocessing notebook included (`preprocessing/Pra_pemrosesan_Dataset_POMDP.ipynb`) + R extraction scripts
- [x] `Project.toml` / `Manifest.toml` committed (exact Julia env)
- [x] `stats_paired.jl` (canonical, full-Julia) + `stats_paired.py` (reference) — both verified to reproduce the reported significance numbers
- [ ] LICENSE intentionally omitted for now (repo defaults to all-rights-reserved until added)
