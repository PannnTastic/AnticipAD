# Manual Run Guide: Time-State POMDP v2

**Model:** Alzheimer POMDP with explicit time counter t (t0 = first visit)  
**State Space:** 60 states (3 clinical × 20 time steps)  
**Patients:** 100,000  
**Max Steps:** 20 visits  
**Solvers:** 6 offline solvers (Random, Expert, MyopicPOMDP, QMDP, FIB, SARSOP)

---

## Prerequisites

```bash
cd julia_alzheimer_pomdp
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Packages required: `POMDPs`, `POMDPTools`, `SARSOP`, `QMDP`, `FIB`, `JSON`, `Plots`, `StatsBase`

---

## Step-by-Step Commands

Run each solver **one at a time** in separate terminal windows or sequentially:

### 1. Random Policy (Fastest, ~30 seconds)
```bash
cd julia_alzheimer_pomdp
julia --project=. solver_random.jl
```
Output: `results_random_100000.json`

### 2. Expert Policy (Fast, ~30 seconds)
```bash
cd julia_alzheimer_pomdp
julia --project=. solver_expert.jl
```
Output: `results_expert_100000.json`

### 3. MyopicPOMDP (~2-5 minutes)
```bash
cd julia_alzheimer_pomdp
julia --project=. solver_myopic.jl
```
Output: `results_myopic_100000.json`

### 4. QMDP (~1-2 minutes training + ~2 minutes eval)
```bash
cd julia_alzheimer_pomdp
julia --project=. solver_qmdp.jl
```
Output: `results_qmdp_100000.json`

### 5. FIB (~1-2 minutes training + ~2 minutes eval)
```bash
cd julia_alzheimer_pomdp
julia --project=. solver_fib.jl
```
Output: `results_fib_100000.json`

### 6. SARSOP (SLOWEST, 15-30 minutes training + ~2 minutes eval)
```bash
cd julia_alzheimer_pomdp
julia --project=. solver_sarsop.jl
```
**Tip:** Run this in background so it doesn't block your terminal:
```bash
cd julia_alzheimer_pomdp
nohup julia --project=. solver_sarsop.jl > sarsop_log.txt 2>&1 &
# Check progress: tail -f sarsop_log.txt
```
Output: `results_sarsop_100000.json`

---

## Step 7: Combine Results & Generate Figures

**Only run this AFTER all 6 solver scripts complete:**

```bash
cd julia_alzheimer_pomdp
julia --project=. combine_results.jl
```

This will:
- Load all `results_*.json` files
- Print a combined comparison table
- Generate 5 comparison figures in `figures_paper/`
- Save combined results to `combined_results_100000.json`

---

## Expected Runtime Summary

| Solver | Training | Evaluation | Total | Output File |
|--------|----------|------------|-------|-------------|
| Random | N/A | ~30s | ~30s | `results_random_100000.json` |
| Expert | N/A | ~30s | ~30s | `results_expert_100000.json` |
| MyopicPOMDP | N/A | ~2-5min | ~2-5min | `results_myopic_100000.json` |
| QMDP | ~1min | ~2min | ~3min | `results_qmdp_100000.json` |
| FIB | ~1min | ~2min | ~3min | `results_fib_100000.json` |
| SARSOP | 15-30min | ~2min | 17-32min | `results_sarsop_100000.json` |

**Total time if run sequentially:** ~25-45 minutes  
**Total time if run in parallel:** ~30 minutes (bottleneck = SARSOP)

---

## Quick Test (N=100, before full run)

To verify everything works before running 100,000 patients:

Edit any solver script and change `N_SAMPLES = 100000` to `N_SAMPLES = 100`, then run.

Or create a quick test:
```bash
cd julia_alzheimer_pomdp
julia --project=. -e '
include("src/AlzheimerPOMDP_v2.jl")
using .AlzheimerPOMDP
pomdp = AlzheimerPOMDPProblem(discount=0.95, max_steps=20)
println("States: ", length(states(pomdp)))
println("Model OK - ready for full run")
'
```

---

## Output Files

After all scripts complete, you will have:

```
julia_alzheimer_pomdp/
├── results_random_100000.json
├── results_expert_100000.json
├── results_myopic_100000.json
├── results_qmdp_100000.json
├── results_fib_100000.json
├── results_sarsop_100000.json
├── combined_results_100000.json
└── figures_paper/
    ├── v2_01_reward_comparison.png
    ├── v2_02_accuracy_comparison.png
    ├── v2_03_perstate_accuracy.png
    ├── v2_04_testing_cost.png
    └── v2_05_training_time.png
```

---

## Troubleshooting

### SARSOP takes too long
- Reduce `MAX_STEPS` from 20 to 10 in `solver_sarsop.jl` (state space = 30)
- Or increase timeout: change `timeout=1800.0` to `timeout=3600.0` (1 hour)

### Out of memory during evaluation
- Reduce `N_SAMPLES` from 100000 to 10000
- Or run evaluation in batches

### QMDP/FIB not installed
```bash
cd julia_alzheimer_pomdp
julia --project=. -e "using Pkg; Pkg.add(\"QMDP\"); Pkg.add(\"FIB\")"
```

---

## What Changed from v1?

| Aspect | v1 (Old) | v2 (New) |
|--------|----------|----------|
| States | 3 (CN, MCI, Dementia) | **60** (3 clinical × 20 time steps) |
| Time | Implicit (episode counter) | **Explicit state** (t = 0,1,...,19) |
| Max Steps | 5 | **20** |
| Patients | 1,000 | **100,000** |
| Solvers | 6 (incl. online) | **6 offline only** |
| Online Solvers | POMCP, DESPOT | **Removed** |
| Offline Solvers | SARSOP only | **SARSOP, QMDP, FIB** |
| Time-Dependent Reward | No | **Yes** (increasing wait penalty with t) |

---

*Run manually, one solver at a time. SARSOP is the bottleneck — run it first or in background.*
