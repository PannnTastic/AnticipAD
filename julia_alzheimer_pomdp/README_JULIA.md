# Julia Environment for Alzheimer POMDP

> Lingkungan penelitian lengkap dalam Julia menggunakan ekosistem `POMDPs.jl` yang mature dan scientifically rigorous. Environment ini dirancang untuk memberikan benchmark yang **fair** kepada semua solver.

---

## 🎯 Why Julia?

Environment Python kita mengalami kegagalan pada POMCP dan DESPOT karena:
1. `pomdp-py` library tidak kompatibel dengan custom `SimpleParticleBelief`
2. Custom DESPOT Python implementation memiliki bug/underspecified tree search

Julia menawarkan **ekosistem POMDP yang mature**:
- **`POMDPs.jl`** — interface standard yang digunakan komunitas POMDP global
- **`BasicPOMCP.jl`** — implementasi POMCP yang rigorously tested (pure Julia)
- **`SARSOP.jl`** — state-of-the-art offline point-based solver (dengan binary C++ mature)
- **`QMDP.jl` & `FIB.jl`** — heuristic baseline yang terstandardisasi

Dengan hanya **3 states**, kita bahkan bisa mendapatkan **policy yang mendekati optimal** via SARSOP!

---

## 📁 Directory Structure

```
julia_alzheimer_pomdp/
├── Project.toml              # Julia package manifest
├── src/
│   └── AlzheimerPOMDP.jl     # Main module: models, policies, experiments, figures
├── scripts/
│   └── run_experiments.jl    # Entry point to run all experiments
├── figures_paper/            # Generated figures (PNG + PDF)
└── README_JULIA.md           # This file
```

---

## 🚀 Quick Start

### 1. Install Julia Dependencies (First Time Only)

Buka PowerShell/Terminal di folder `julia_alzheimer_pomdp`:

```bash
cd julia_alzheimer_pomdp
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Jika package belum terdaftar di registry lokal, Anda perlu `add` manual:

```julia
using Pkg
Pkg.activate(".")
Pkg.add("POMDPs")
Pkg.add("POMDPTools")
Pkg.add("BasicPOMCP")
Pkg.add("SARSOP")
Pkg.add("QMDP")
Pkg.add("FIB")
Pkg.add("Plots")
Pkg.add("StatsBase")
```

### 2. Run Experiments

```bash
julia --project=. scripts/run_experiments.jl
```

---

## 🧠 Solvers Included

| Solver | Type | Description |
|--------|------|-------------|
| **Random** | Baseline | Uniform random action selection |
| **Expert** | Heuristic | Rule-based if-else with fixed thresholds (80-85%) |
| **MyopicPOMDP** | Custom | One-step exact Bayesian lookahead |
| **POMCP** | Online MCTS | Monte Carlo Tree Search via `BasicPOMCP.jl` |
| **QMDP** | Offline Heuristic | Solves underlying MDP, uses for POMDP heuristic |
| **FIB** | Offline Heuristic | Fast Informed Bound — stronger than QMDP |
| **SARSOP** | Offline Optimal | Point-based solver, near-optimal for 3-state POMDP |

### Key Differences from Python Implementation

1. **Belief Representation**: Menggunakan `DiscreteBelief` (exact histogram, 3 states) alih-alih custom particle filter. Ini menghilangkan masalah compatibility.
2. **POMCP**: Menggunakan `BasicPOMCP.jl` — mature, well-tested, dan kompatibel dengan `DiscreteBelief`.
3. **SARSOP**: Menyediakan upper bound yang kuat karena state space sangat kecil.
4. **No custom DESPOT**: DESPOT yang mature hanya tersedia dalam C++ original (Ye et al. 2017). Karena Julia tidak memiliki wrapper DESPOT yang mature, dan SARSOP justru lebih unggul untuk POMDP kecil, kita menggunakan SARSOP sebagai representative offline solver.

---

## 🔬 Scientific Fairness

Environment ini dirancang untuk **apples-to-apples comparison**:
- Semua solver menghadapi **urutan true state yang identik** (same random seed)
- Semua solver menggunakan **belief updater yang sama** (`DiscreteUpdater` — exact Bayesian)
- Semua solver menggunakan **model POMDP yang identik**
- Episode logic **exactly mirrors Python version** (max 5 steps, default Wait if no terminal action)

---

## 📊 Expected Results

Berdasarkan struktur model (3 states, noisy observations), hipotesis kita:

- **SARSOP** akan memberikan upper bound performance (mendekati optimal)
- **MyopicPOMDP** akan mendekati SARSOP karena one-step lookahead cukup powerful untuk horizon pendek
- **POMCP** akan bekerja dengan baik (berbeda dari Python yang broken) dan mengungguli Expert
- **QMDP/FIB** akan memberikan baseline heuristik yang solid
- **Expert** akan menjadi middle ground
- **Random** akan jauh di bawah semua policy yang informed

---

## 📝 Paper Narrative

> *"We implemented the POMDP in both Python (for rapid prototyping) and Julia (for rigorous benchmarking). The Julia environment leverages the mature `POMDPs.jl` ecosystem, including `SARSOP` for near-optimal offline planning and `BasicPOMCP` for proper online Monte Carlo Tree Search. This allows us to validate that our MyopicPOMDP heuristic achieves near-optimal performance on this small-state medical domain."*

---

## ⚙️ Troubleshooting

### SARSOP fails to build on Windows
`SARSOP.jl` menggunakan BinaryBuilder. Pada Windows, biasanya berjalan otomatis. Jika gagal, Anda bisa skip SARSOP dan hanya menggunakan POMCP/QMDP/FIB/MyopicPOMDP.

### Plots.jl backend error
Default backend adalah GR. Jika error, install GR terlebih dahulu atau set backend ke pyplot:
```julia
using Pkg
Pkg.add("PyPlot")
using Plots; pyplot()
```

---

## 📚 References

- **POMDPs.jl**: https://github.com/JuliaPOMDP/POMDPs.jl
- **BasicPOMCP.jl**: https://github.com/JuliaPOMDP/BasicPOMCP.jl
- **SARSOP.jl**: https://github.com/JuliaPOMDP/SARSOP.jl
- **Ye et al. (2017)**: DESPOT paper
- **Khatim et al. (2024)**: Stroke POMDP benchmark reference
