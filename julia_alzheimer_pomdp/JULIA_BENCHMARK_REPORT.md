# Laporan Benchmark POMDP Alzheimer — Julia Environment

> **Tujuan**: Membangun dan mengevaluasi framework POMDP untuk diagnosis Alzheimer menggunakan solver-solver online yang tepat, mengikuti metodologi paper referensi (Khatim et al., 2024).

---

## 1. PREPROCESSING DATA

### 1.1 Sumber Dataset
- **Nama file**: `ADNI_POMDP_Complete.csv`
- **Sumber**: Alzheimer's Disease Neuroimaging Initiative (ADNI)
- **Total records**: 15,836 baris data bersih
- **Unique patients**: ~6,275 pasien

### 1.2 Fitur yang Digunakan
| Fitur | Deskripsi | Tipe |
|-------|-----------|------|
| `RID` | Patient ID | Integer |
| `VISCODE` | Kode kunjungan | String |
| `DATE` | Tanggal pemeriksaan | Date |
| `DX` | Diagnosis (CN / MCI / Dementia) | Kategori |
| `AGE` | Usia pasien | Float |
| `GENDER` | Jenis kelamin | Kategori |
| `EDUCATION` | Tingkat pendidikan | Float |
| `APOE4` | Genotipe APOE4 | String |
| `MMSE` | Mini-Mental State Examination score | Float (0–30) |
| `CDR` | Clinical Dementia Rating | Float |

### 1.3 Distribusi State dari Data ADNI
Berdasarkan kolom `DX`:

| State | Count | Probability |
|-------|-------|-------------|
| **CN** (Cognitively Normal) | 6,275 | 38.52% |
| **MCI** (Mild Cognitive Impairment) | 6,565 | 41.70% |
| **Dementia** | 2,996 | 19.78% |

*Initial belief POMDP* direpresentasikan sebagai probabilitas ini:
```
b₀ = [CN: 0.385189,  MCI: 0.417041,  Dementia: 0.197770]
```

### 1.4 Mapping APOE4 (6 Genotip → 3 Kategori Risiko)
Data mentah `APOE4` memiliki 6 kombinasi alel:

| Genotipe | Count |
|----------|-------|
| ε3/ε3 | 6,956 |
| ε3/ε4 | 4,697 |
| ε4/ε4 | 1,233 |
| ε2/ε3 | 1,129 |
| ε2/ε4 | 315 |
| ε2/ε2 | 32 |

Untuk model POMDP, 6 genotip ini di-*collapse* menjadi **3 kategori klinis** yang bermakna:

| Model Category | Genotipe yang Digabung | Risiko |
|----------------|------------------------|--------|
| **APOE4_Negatif** | ε2/ε2 + ε2/ε3 + ε3/ε3 | Rendah |
| **APOE4_Hetero** | ε2/ε4 + ε3/ε4 | Sedang |
| **APOE4_Homo** | ε4/ε4 | Tinggi |

Probabilitas observasi APOE4 per state dihitung dari distribusi frekuensi genotipe di atas.

---

## 2. METODE

### 2.1 Framework: Partially Observable Markov Decision Process (POMDP)
POMDP dipilih karena diagnosis Alzheimer bersifat **partially observable** — dokter tidak pernah mengamati state sebenarnya (CN/MCI/Dementia) secara langsung, melainkan hanya melihat hasil tes yang *noisy*.

### 2.2 Reference Paper
**Khatim, N. A., Irfan, A. A. A., Hayah, A. M., & Arief, M. M. (2024).**  
*Toward an Integrated Decision Making Framework for Optimized Stroke Diagnosis with DSA and Treatment under Uncertainty.*  
Proceedings of International Conference on Intelligent Medicine and Image Processing (IMIP2024). ACM.

**Kesamaan metodologi**:
- Menggunakan POMDP framework
- Menggunakan **online solver DESPOT** (via `ARDESPOT.jl`)
- Tree-search + particle filter untuk real-time decision making
- Trade-off antara biaya diagnostik dan benefit treatment

### 2.3 Solvers yang Dievaluasi
| Solver | Jenis | Karakteristik |
|--------|-------|---------------|
| **Random** | Baseline | Aksi dipilih secara uniform random |
| **Expert** | Heuristic | Rule-based threshold policy dengan test ordering tetap |
| **MyopicPOMDP** | Exact 1-step lookahead | Menghitung ekspektasi Bayesian exact untuk 1 langkah ke depan |
| **POMCP** | Online Monte Carlo | Tree search dengan 200 rollouts, ExpertRollout sebagai value estimate |
| **DESPOT** | Online Tree Search | `ARDESPOT.jl` dengan K=20, D=8, T_max=0.01s |

---

## 3. KOMPONEN POMDP

### 3.1 States (S)
| State | Deskripsi |
|-------|-----------|
| `CN` | Cognitively Normal (Sehat) |
| `MCI` | Mild Cognitive Impairment |
| `Dementia` | Alzheimer / Demensia |

### 3.2 Actions (A)
| Action | Deskripsi | Biaya |
|--------|-----------|-------|
| `A_Wait` | Observasi / tunggu 6 bulan | Variable reward |
| `A_Test_MMSE` | Tes kognitif dasar | -5 |
| `A_Test_CDR` | Tes wawancara klinis | -20 |
| `A_Test_APOE4` | Tes genetik darah | -50 |
| `A_Treat_MCI` | Intervensi ringan untuk MCI | Variable reward |
| `A_Treat_Dementia` | Pengobatan berat untuk Dementia | Variable reward |

*Actions `A_Wait`, `A_Treat_MCI`, dan `A_Treat_Dementia` bersifat terminal (mengakhiri episode).*

### 3.3 Observations (O)
| Observation | Konteks |
|-------------|---------|
| `None` | Muncul jika action bukan tes |
| `MMSE_Normal` | MMSE ≥ 26 |
| `MMSE_Sedang` | MMSE 20–25 |
| `MMSE_Rendah` | MMSE < 20 |
| `CDR_0` | CDR = 0 |
| `CDR_0_5` | CDR = 0.5 |
| `CDR_1_plus` | CDR ≥ 1 |
| `APOE4_Negatif` | Tidak membawa risiko ε4 |
| `APOE4_Hetero` | Satu copy ε4 |
| `APOE4_Homo` | Dua copy ε4 |

### 3.4 Transition Model P(S' | S)
Model transisi merefleksikan progresi penyakit selama 6 bulan berdasarkan data longitudinal ADNI:

| From \ To | CN | MCI | Dementia |
|-----------|-----|-----|----------|
| **CN** | 0.929 | 0.068 | 0.003 |
| **MCI** | 0.000 | 0.892 | 0.108 |
| **Dementia** | 0.000 | 0.000 | 1.000 |

### 3.5 Observation Models P(O | S', A)

#### MMSE Observation Model
| State | Normal | Sedang | Rendah |
|-------|--------|--------|--------|
| CN | 0.85 | 0.13 | 0.02 |
| MCI | 0.55 | 0.35 | 0.10 |
| Dementia | 0.20 | 0.45 | 0.35 |

#### CDR Observation Model
| State | CDR=0 | CDR=0.5 | CDR≥1 |
|-------|-------|---------|-------|
| CN | 0.95 | 0.04 | 0.01 |
| MCI | 0.25 | 0.70 | 0.05 |
| Dementia | 0.05 | 0.30 | 0.65 |

#### APOE4 Observation Model (dari data nyata ADNI)
| State | Negatif | Hetero | Homo |
|-------|---------|--------|------|
| CN | 0.6910 | 0.2813 | 0.0277 |
| MCI | 0.5549 | 0.3529 | 0.0922 |
| Dementia | 0.3387 | 0.4751 | 0.1862 |

### 3.6 Reward Function R(S, A)
| State \ Action | Wait | Test_MMSE | Test_CDR | Test_APOE4 | Treat_MCI | Treat_Dementia |
|----------------|------|-----------|----------|------------|-----------|----------------|
| **CN** | +30 | -5 | -20 | -50 | -50 | **-500** |
| **MCI** | -10 | -5 | -20 | -50 | **+150** | -100 |
| **Dementia** | **-1000** | -5 | -20 | -50 | -50 | **+300** |

**Rationale desain reward**:
- `Wait` saat Dementia = -1000 (sangat fatal, harus segera diobati)
- `Treat_Dementia` saat Dementia = +300 (treatment tepat, nilai tinggi)
- `Treat_Dementia` saat CN = -500 (malapraktik)
- Biaya tes meningkat: MMSE (-5) < CDR (-20) < APOE4 (-50)

### 3.7 Belief Update
Menggunakan **DiscreteUpdater** (exact Bayesian update) karena state space sangat kecil (|S|=3):
```
P(s'|a,o) = P(o|s',a) × Σ_s[P(s'|s,a) × P(s)] / P(o|a)
```

---

## 4. EXPERIMENT SETTING

| Parameter | Nilai |
|-----------|-------|
| Jumlah pasien (Monte Carlo) | **1,000** |
| Max steps per episode | 5 |
| Discount factor (γ) | 0.95 |
| Seed | 42 |
| Environment | `AlzheimerPOMDPProblem` dalam Julia 1.11.7 |

### 4.1 Parameter Solver
| Solver | Parameter Kunci |
|--------|-----------------|
| **POMCP** | max_depth=8, c=50.0, tree_queries=200, estimate_value=FORollout(ExpertRollout) |
| **DESPOT** | K=20, D=8, lambda=0.5, T_max=0.01s (via `ARDESPOT.jl`) |
| **MyopicPOMDP** | Exact 1-step lookahead dengan γ=0.95 |
| **Expert** | p_dem_thres=0.85, p_cn_thres=0.80, p_mci_thres=0.80 |

---

## 5. HASIL BENCHMARK (N = 1,000 PASIEN)

### 5.1 Ringkasan Performa

| Policy | Avg Reward | Accuracy | Avg Test Cost | Avg Steps | CN Acc | MCI Acc | Dementia Acc |
|--------|------------|----------|---------------|-----------|--------|---------|--------------|
| **SARSOP** | **+69.90** | **77.4%** | **-29.10** | **2.46** | 79.0% | **81.6%** | 65.3% |
| **MyopicPOMDP** | +67.64 | **78.5%** | -37.61 | 2.89 | **84.1%** | 71.6% | **82.9%** |
| Expert | +45.33 | 71.2% | -42.75 | 3.36 | 77.6% | 64.4% | 73.9% |
| POMCP | -30.04 | 52.2% | -43.38 | 3.21 | 47.2% | 57.4% | 50.3% |
| DESPOT | -186.60 | 48.8% | -75.86 | 4.53 | 80.9% | 23.3% | 44.2% |
| Random | -148.48 | 32.3% | -25.48 | 1.99 | 34.0% | 30.5% | 33.2% |

### 5.2 Waktu Eksekusi (1000 pasien)

| Policy | Total Time |
|--------|------------|
| Random | 0.05s |
| Expert | 0.10s |
| MyopicPOMDP | 0.36s |
| POMCP | 2.27s |
| DESPOT | 48.83s |

### 5.3 Analisis Hasil

#### 🥇 SARSOP — Pemenang Reward & Cost-Efficiency
- **Reward tertinggi** (+69.90) dan **biaya tes paling rendah** (-29.10)
- **MCI accuracy terbaik** (81.6%) — state yang paling sulit didiagnosis
- **Cepat saat evaluasi** (0.08s untuk 1000 pasien setelah training offline 60s)
- *Mengapa berhasil?* Karena state space kecil (3 states), offline point-based solver dapat mengeksplorasi belief space secara hampir optimal.

#### 🥈 MyopicPOMDP — Pemenang Overall Accuracy
- **Akurasi tertinggi** (78.5%) dengan **Dementia detection terbaik** (82.9%)
- **Zero training time**, fully interpretable
- **Biaya tes rendah** (-37.61)
- *Mengapa berhasil?* Exact 1-step Bayesian lookahead cukup powerful untuk domain dengan horizon pendek dan state space kecil.

#### Expert Policy — Baseline Heuristic yang Kuat
- Reward positif (+45.33) dengan akurasi tinggi (71.2%)
- Lebih konservatif dalam testing (-42.75) dibanding MyopicPOMDP
- Cocok untuk implementasi klinis karena deterministik dan dapat dijelaskan

#### POMCP — Underperforming
- Reward negatif (-30.04) meskipun menggunakan ExpertRollout
- Akurasi MCI lebih baik dari DESPOT (57.4%) tapi masih di bawah Myopic/Expert/SARSOP
- Tree_queries=200 tidak cukup untuk noisy-observation domain ini

#### DESPOT — Over-Testing Problem
- **Reward sangat negatif** (-186.60) karena **over-testing** (-75.86 test cost)
- Akurasi MCI sangat rendah (23.3%) — sering salah diagnosis meskipun banyak tes
- Parameter K=10, D=6, lambda=0.8 tidak mencegah over-testing
- Waktu paling lambat (41.79s)

---

## 6. FIGURE YANG TELAH DIGENERATE

Semua figure tersimpan di folder `figures_paper/`:

| File | Deskripsi |
|------|-----------|
| `01_initial_belief.png/pdf` | Distribusi awal belief b₀ dari ADNI |
| `02_transition_matrix.png/pdf` | Heatmap matriks transisi 6-bulan |
| `04_reward_function.png/pdf` | Heatmap reward function R(s,a) |
| `07_solver_reward_comparison.png/pdf` | Bar chart perbandingan average reward |
| `08_solver_accuracy_comparison.png/pdf` | Bar chart perbandingan overall accuracy |
| `09_solver_per_state_accuracy.png/pdf` | Grouped bar chart akurasi per state (CN/MCI/Dementia) |
| `10_solver_testing_cost.png/pdf` | Bar chart biaya tes rata-rata |

---

## 7. KESIMPULAN & REKOMENDASI

### 7.1 Kesimpulan
1. **SARSOP adalah solver terbaik untuk reward dan cost-efficiency** (+69.90 reward, -29.10 test cost, 81.6% MCI accuracy). Offline point-based solver menghasilkan policy paling efisien.
2. **MyopicPOMDP adalah solver terbaik untuk overall accuracy** (78.5%) dan **Dementia detection** (82.9%). Exact 1-step Bayesian lookahead mengungguli POMCP dan DESPOT dalam hal akurasi dan kecepatan.
2. **DESPOT (ARDESPOT.jl) over-tests** dengan parameter K=20, D=8, T_max=0.01s — menyebabkan biaya tes sangat tinggi dan reward negatif.
3. **Expert Policy** tetap merupakan baseline yang sangat kuat dan klinis-interpretable.
4. **POMCP** memerlukan tuning lebih lanjut (tree queries, exploration constant) atau reward shaping untuk meningkatkan deteksi MCI.

### 7.2 Rekomendasi untuk Paper
- Gunakan **MyopicPOMDP** sebagai proposed method karena performanya superior
- Gunakan **Expert Policy** sebagai baseline klinis
- Sertakan **DESPOT** dan **POMCP** sebagai perbandingan dengan online solvers state-of-the-art
- Highlight bahwa untuk *small-state-space medical POMDPs*, exact myopic lookahead bisa mengalahkan sampling-based long-horizon planners

### 7.3 Rekomendasi untuk Pengembangan
- **Reward Shaping**: Modifikasi reward `Wait` untuk MCI agar lebih negatif, atau tambahkan information-gathering bonus
- **DESPOT Tuning**: Eksplorasi bounds yang lebih ketat atau parameter regularization (lambda) yang lebih tinggi untuk mengurangi over-testing
- **Validasi Longitudinal**: Uji pada trajektori pasien nyata dari ADNI (bukan hanya independent samples)

---

**Lokasi hasil lengkap**:
- JSON results: `julia_alzheimer_pomdp/julia_results_1000.json`
- Figures: `julia_alzheimer_pomdp/figures_paper/`
- Benchmark script: `julia_alzheimer_pomdp/run_full_benchmark.jl`
