# Experiment Setting: SARSOP Extraction & Sensitivity Analysis

**Date:** 29 April 2026  
**Location:** `D:\pomdp\pomdp_alzheimer\julia_alzheimer_pomdp`  
**Environment:** Julia 1.11.3 on Windows, `POMDPs.jl` ecosystem

---

## 1. Computing Environment

### 1.1 Platform
| Component | Specification |
|-----------|---------------|
| **Julia Version** | 1.11.3 |
| **Operating System** | Windows |
| **Project File** | `Project.toml` |
| **CPU Only** | No GPU required |

### 1.2 Julia Packages Used
| Package | Version | Purpose |
|---------|---------|---------|
| `POMDPs` | Latest (registry) | Core POMDP interface |
| `POMDPTools` | Latest | DiscreteUpdater, SparseCat, belief utilities |
| `SARSOP` | Latest | Offline point-based solver (`SARSOPSolver`, `load_policy`) |
| `BasicPOMCP` | Latest | Online POMCP solver (for comparison) |
| `ARDESPOT` | Latest | Online DESPOT solver (for comparison) |
| `Plots` | Latest | Figure generation (PNG + PDF) |
| `StatsBase` | Latest | Statistical summaries |
| `JSON` | Latest | Results serialization |
| `Random` | Stdlib | Reproducible random number generation |
| `Statistics` | Stdlib | Mean, std, etc. |

### 1.3 Activation Command
```bash
cd julia_alzheimer_pomdp
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

---

## 2. POMDP Model Specification

All experiments use the **identical** POMDP model defined in `src/AlzheimerPOMDP.jl`.

### 2.1 Tuple Definition
⟨S, A, O, T, Z, R, γ, b₀⟩

### 2.2 State Space S
| Index | State | Description |
|-------|-------|-------------|
| 1 | CN | Cognitively Normal |
| 2 | MCI | Mild Cognitive Impairment |
| 3 | Dementia | Alzheimer's Dementia |

**Property:** Dementia is an **absorbing state** (biologically plausible).

### 2.3 Action Space A
| Index | Action | Type | Cost/Reward |
|-------|--------|------|-------------|
| 1 | A_Wait | Terminal | State-dependent (+30, −10, −1000) |
| 2 | A_Test_MMSE | Test | −5 |
| 3 | A_Test_CDR | Test | −20 |
| 4 | A_Test_APOE4 | Test | −50 |
| 5 | A_Treat_MCI | Terminal | State-dependent (−50, +150, −50) |
| 6 | A_Treat_Dementia | Terminal | State-dependent (−500, −100, +300) |

### 2.4 Observation Space O
Observations are action-dependent (only the chosen test produces an observation; otherwise `None`).

**MMSE Observations:** `MMSE_Normal`, `MMSE_Sedang`, `MMSE_Rendah`

**CDR Observations:** `CDR_0`, `CDR_0_5`, `CDR_1_plus`

**APOE4 Observations:** `APOE4_Negatif`, `APOE4_Hetero`, `APOE4_Homo`

### 2.5 Transition Model T(s' | s, a)
Action-independent (disease progresses naturally regardless of intervention in this model).

| From \ To | CN | MCI | Dementia |
|-----------|-----|-----|----------|
| **CN** | 0.929 | 0.068 | 0.003 |
| **MCI** | **0.000** | **0.892** | 0.108 |
| **Dementia** | **0.000** | **0.000** | **1.000** |

### 2.6 Observation Model Z(o | s', a)

#### MMSE Probabilities
| State | Normal | Sedang | Rendah |
|-------|--------|--------|--------|
| CN | 0.85 | 0.13 | 0.02 |
| MCI | 0.55 | 0.35 | 0.10 |
| Dementia | 0.20 | 0.45 | 0.35 |

#### CDR Probabilities
| State | CDR_0 | CDR_0.5 | CDR_1_plus |
|-------|-------|---------|------------|
| CN | 0.95 | 0.04 | 0.01 |
| MCI | 0.25 | 0.70 | 0.05 |
| Dementia | 0.05 | 0.30 | 0.65 |

#### APOE4 Probabilities (Normalized from ADNI)
| State | Negatif | Hetero | Homo |
|-------|---------|--------|------|
| CN | 0.6910 | 0.2813 | 0.0277 |
| MCI | 0.5549 | 0.3529 | 0.0922 |
| Dementia | 0.3387 | 0.4751 | 0.1862 |

### 2.7 Reward Function R(s, a)

| State \ Action | Wait | Test MMSE | Test CDR | Test APOE4 | Treat MCI | Treat Dementia |
|----------------|------|-----------|----------|------------|-----------|----------------|
| **CN** | +30 | −5 | −20 | −50 | −50 | **−500** |
| **MCI** | −10 | −5 | −20 | −50 | **+150** | −100 |
| **Dementia** | **−1000** | −5 | −20 | −50 | −50 | **+300** |

### 2.8 Discount Factor
γ = 0.95 (reflecting 6-month clinical time preference)

### 2.9 Initial Belief b₀
```
b₀(CN)       = 0.385189
b₀(MCI)      = 0.417041
b₀(Dementia) = 0.197770
```

---

## 3. Experiment 1: SARSOP Policy Extraction

**Script:** `extract_sarsop_policy.jl`

### 3.1 Objective
Extract interpretable clinical decision rules from the SARSOP α-vector policy by evaluating it exhaustively over the belief simplex.

### 3.2 SARSOP Solver Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Precision** | 1e-2 | Optimality gap tolerance (1%) |
| **Timeout** | 60.0 seconds | Maximum offline training time |
| **Output** | `policy.out` | 638 α-vectors |
| **Model File** | `model.pomdpx` | POMDP description in POMDPX format |

**Solver instantiation:**
```julia
solver = SARSOPSolver(precision=1e-2, timeout=60.0)
policy = solve(solver, pomdp)
```

**Policy loading (for extraction):**
```julia
policy = SARSOP.load_policy(pomdp, "policy.out")
```

### 3.3 Belief Grid Evaluation
| Parameter | Value |
|-----------|-------|
| **Grid type** | Uniform over 2D belief simplex |
| **Resolution** | 0.02 step in p_CN and p_MCI |
| **Valid points** | 1,316 (excluding p_CN + p_MCI > 1.0) |
| **Belief representation** | `DiscreteBelief` (exact histogram) |

**Grid construction:**
```julia
for p_cn in 0.0:0.02:1.0
    for p_mci in 0.0:0.02:(1.0 - p_cn)
        p_dem = 1.0 - p_cn - p_mci
        # Normalize and evaluate
    end
end
```

### 3.4 Decision Rule Extraction
For each action, we extracted:
1. **Region coverage:** % of belief simplex where the action dominates
2. **Convex hull approximation:** min/max of each belief coordinate in the region
3. **Single-threshold rules:** Best approximate linear rule (F1 score) of the form `p_X ≥ θ → action`

### 3.5 Trajectory Simulation
| Parameter | Value |
|-----------|-------|
| **True states tested** | CN, MCI, Dementia |
| **Initial belief** | b₀ = [0.385, 0.417, 0.198] |
| **Max steps** | 5 |
| **Belief updater** | `DiscreteUpdater` (exact Bayesian) |
| **Random seed** | Not fixed per-trajectory (uses `Random.GLOBAL_RNG`) |

**Trajectory format:**
```
Step 1: b=[0.39, 0.42, 0.20] → A_Test_CDR
Step 2: b=[0.75, 0.22, 0.03] → A_Test_CDR
Step 3: b=[0.91, 0.08, 0.00] → A_Wait
```

### 3.6 Visualization
| Figure | Description |
|--------|-------------|
| `11_sarsop_decision_regions.png` | Belief simplex colored by SARSOP action (scatter plot, 900×800 px, DPI=300) |

**Color mapping:**
- Wait = Green (#2ecc71)
- Test MMSE = Yellow (#f1c40f)
- Test CDR = Blue (#3498db)
- Test APOE4 = Purple (#9b59b6)
- Treat MCI = Orange (#f39c12)
- Treat Dementia = Red (#e74c3c)

### 3.7 Output Artifacts
| File | Description |
|------|-------------|
| `policy.out` | SARSOP α-vector policy (638 vectors) |
| `model.pomdpx` | POMDP model in POMDPX XML format |
| `sarsop_policy_grid.json` | Grid evaluation results (1,316 points × action + value) |
| `figures_paper/11_sarsop_decision_regions.png/pdf` | Decision region visualization |

---

## 4. Experiment 2: Sensitivity Analysis

**Script:** `sensitivity_analysis.jl`

### 4.1 Objective
Test the robustness of SARSOP, MyopicPOMDP, and Expert policies to clinically plausible variations in the reward function.

### 4.2 Monte Carlo Simulation Settings
| Parameter | Value | Justification |
|-----------|-------|---------------|
| **N (patients)** | 1,000 | Statistically stable, computationally feasible |
| **Max steps per episode** | 5 | Sufficient for sequential testing |
| **Random seed** | 42 | Full reproducibility |
| **Shared random seed** | Yes | All solvers face identical true states |
| **Initial belief** | b₀ = [0.385, 0.417, 0.198] | ADNI population distribution |
| **True state sampling** | From b₀ | Each virtual patient sampled independently |
| **Episode termination** | Terminal action or max steps reached | Same for all solvers |

### 4.3 Belief Update
| Parameter | Value |
|-----------|-------|
| **Updater** | `DiscreteUpdater` (exact Bayesian) |
| **Update rule** | b'(s') = η · P(o | s', a) · Σ_s [P(s' | s, a) · b(s)] |
| **Complexity** | O(|S|²) = 9 operations per update |

### 4.4 Reward Variants

All variants modify the **baseline** reward function. Only changed values are shown.

#### Variant 1: Baseline
No changes. Reference model as specified in Section 2.7.

#### Variant 2: LowDemWait(-500)
| Change | Baseline | New Value |
|--------|----------|-----------|
| R(Dementia, Wait) | −1000 | **−500** |

**Clinical interpretation:** Less severe consequence of delaying dementia treatment (e.g., resource-limited setting where immediate treatment is difficult).

#### Variant 3: HighDemWait(-1500)
| Change | Baseline | New Value |
|--------|----------|-----------|
| R(Dementia, Wait) | −1000 | **−1500** |

**Clinical interpretation:** More severe penalty for delayed treatment (e.g., acute setting where every month of delay causes irreversible decline).

#### Variant 4: LowMCITreat(+100)
| Change | Baseline | New Value |
|--------|----------|-----------|
| R(MCI, Treat_MCI) | +150 | **+100** |

**Clinical interpretation:** Lower benefit from MCI intervention (e.g., less effective cognitive training program).

#### Variant 5: HighMCITreat(+200)
| Change | Baseline | New Value |
|--------|----------|-----------|
| R(MCI, Treat_MCI) | +150 | **+200** |

**Clinical interpretation:** Higher benefit from MCI intervention (e.g., new drug shows strong efficacy in slowing progression).

#### Variant 6: LowMalpractice(-300)
| Change | Baseline | New Value |
|--------|----------|-----------|
| R(CN, Treat_Dementia) | −500 | **−300** |

**Clinical interpretation:** Lower malpractice penalty (e.g., different legal jurisdiction or insurance structure).

#### Variant 7: HighMalpractice(-700)
| Change | Baseline | New Value |
|--------|----------|-----------|
| R(CN, Treat_Dementia) | −500 | **−700** |

**Clinical interpretation:** Higher malpractice penalty (e.g., litigious environment).

#### Variant 8: ExpensiveTests
| Change | Baseline | New Value |
|--------|----------|-----------|
| R(·, Test_MMSE) | −5 | **−10** |
| R(·, Test_CDR) | −20 | **−40** |
| R(·, Test_APOE4) | −50 | **−100** |

**Clinical interpretation:** All test costs doubled (e.g., private healthcare system, expensive biomarker assays).

#### Variant 9: EqualTestCosts(-20)
| Change | Baseline | New Value |
|--------|----------|-----------|
| R(·, Test_MMSE) | −5 | **−20** |
| R(·, Test_CDR) | −20 | **−20** |
| R(·, Test_APOE4) | −50 | **−20** |

**Clinical interpretation:** Flat reimbursement structure where all diagnostic tests cost the same.

### 4.5 Solvers Tested per Variant

| Solver | Type | Parameters |
|--------|------|------------|
| **SARSOP** | Offline point-based | `precision=1e-2, timeout=60.0` |
| **MyopicPOMDP** | Online exact 1-step | `γ=0.95` (no tuning) |
| **Expert** | Rule-based heuristic | `p_dem_thres=0.85, p_cn_thres=0.80, p_mci_thres=0.80` |

**Note:** POMCP and DESPOT were excluded from sensitivity analysis due to their poor baseline performance (negative rewards, clinically unsafe).

### 4.6 Fair Comparison Protocol
1. **Identical POMDP model** across all variants (only reward changes)
2. **Identical true state sequence** for all solvers within each variant (same RNG seed)
3. **Identical initial belief** b₀ for all episodes
4. **Identical episode termination logic**
5. **Identical evaluation metrics** computed uniformly

### 4.7 Evaluation Metrics

For each solver × variant combination, the following metrics were computed:

| Metric | Formula | Clinical Interpretation |
|--------|---------|------------------------|
| **Average Reward** | Σ_t γ^t · r_t / N | Overall cost-effectiveness |
| **Overall Accuracy** | % correct terminal actions | Diagnostic accuracy |
| **CN Accuracy** | % CN patients correctly Waited | False positive rate |
| **MCI Accuracy** | % MCI patients correctly treated | Early detection rate |
| **Dementia Accuracy** | % Dementia patients correctly treated | Critical detection rate |
| **Average Test Cost** | Mean cost of all test actions | Economic burden |
| **Average Steps** | Mean episode length | Time to diagnosis |
| **Reward Std** | Standard deviation of rewards | Policy consistency |

### 4.8 Visualization

Four sensitivity figures were generated (all 1100×600 px, DPI=300):

| Figure | Y-axis | Description |
|--------|--------|-------------|
| `12_sensitivity_reward.png` | Average Reward | How reward changes with clinical valuations |
| `13_sensitivity_accuracy.png` | Overall Accuracy (%) | Robustness of diagnostic accuracy |
| `14_sensitivity_testcost.png` | Average Test Cost | Economic burden variation |
| `15_sensitivity_mci_acc.png` | MCI Accuracy (%) | Robustness of early detection |

**Plot style:** Line plot with markers per policy (SARSOP=red circle, Myopic=green square, Expert=blue diamond).

### 4.9 Output Artifacts
| File | Description |
|------|-------------|
| `sensitivity_analysis_results.json` | Full numerical results for 9 variants × 3 policies × 8 metrics |
| `figures_paper/12_sensitivity_reward.png/pdf` | Reward sensitivity curve |
| `figures_paper/13_sensitivity_accuracy.png/pdf` | Accuracy sensitivity curve |
| `figures_paper/14_sensitivity_testcost.png/pdf` | Test cost sensitivity curve |
| `figures_paper/15_sensitivity_mci_acc.png/pdf` | MCI accuracy sensitivity curve |

---

## 5. Reproducibility Instructions

### 5.1 SARSOP Extraction
```bash
cd julia_alzheimer_pomdp
julia --project=. extract_sarsop_policy.jl
```
**Expected runtime:** ~2 minutes (mostly grid evaluation + plot generation)
**Expected output:** `sarsop_policy_grid.json` + `figures_paper/11_sarsop_decision_regions.png`

### 5.2 Sensitivity Analysis
```bash
cd julia_alzheimer_pomdp
julia --project=. sensitivity_analysis.jl
```
**Expected runtime:** ~10–15 minutes (9 variants × 60s SARSOP training + evaluation)
**Expected output:** `sensitivity_analysis_results.json` + 4 sensitivity figures

### 5.3 Prerequisites
- Julia 1.11.3 installed
- All packages instantiated (`Pkg.instantiate()`)
- SARSOP binary available (installed automatically via `SARSOP.jl` BinaryBuilder)
- `policy.out` file present (generated by previous SARSOP run, or re-generated during sensitivity)

---

## 6. Validation Checks Performed

| Check | Status | Method |
|-------|--------|--------|
| Transition probabilities sum to 1.0 | ✅ | `_validate_transition_probs()` at model construction |
| Observation probabilities sum to 1.0 | ✅ | `_validate_observation_probs()` for MMSE, CDR, APOE4 |
| Initial belief sums to 1.0 | ✅ | `SparseCat` normalization |
| Belief updates normalize to 1.0 | ✅ | `DiscreteUpdater` guarantees normalization |
| Grid beliefs sum to 1.0 | ✅ | Explicit normalization in grid loop |
| SARSOP precision gap < 1% | ✅ | Solver reports `Precision` at timeout |
| Shared RNG seed consistency | ✅ | Same `MersenneTwister(42)` for all solvers |

---

*Document generated: 29 April 2026*
