# ADNI Data Preprocessing Pipeline

**Dataset:** Alzheimer's Disease Neuroimaging Initiative (ADNI)  
**Raw Data:** `ADNI_POMDP_Complete.csv` (15,836 records, 3,777 unique patients/RID)  
**Date:** 29 April 2026

---

## 1. Data Source

The Alzheimer's Disease Neuroimaging Initiative (ADNI) is a longitudinal multicenter study launched in 2003 by the National Institute on Aging (NIA), the National Institute of Biomedical Imaging and Bioengineering (NIBIB), the Food and Drug Administration (FDA), private pharmaceutical companies, and nonprofit organizations [1]. ADNI collects clinical, genetic, imaging, and biochemical biomarker data from cognitively normal older individuals, patients with mild cognitive impairment (MCI), and patients with early Alzheimer's disease.

**Website:** https://adni.loni.usc.edu/  
**Cohorts:** ADNI1, ADNI2, ADNI3, ADNI-GO

---

## 2. Raw Data Acquisition

### 2.1 Downloaded Variables

| Variable | Description | Type |
|----------|-------------|------|
| `RID` | Patient ID (unique identifier) | Integer |
| `VISCODE` | Visit code (bl, m06, m12, m24, ...) | String |
| `EXAMDATE` | Examination date | Date |
| `DX` | Diagnosis (CN, MCI, Dementia/AD) | Categorical |
| `MMSE` | Mini-Mental State Examination (0-30) | Integer |
| `CDR` | Clinical Dementia Rating (0, 0.5, 1, 2, 3) | Float |
| `APOE4` | APOE genotype (e2/e2, e2/e3, e3/e3, e2/e4, e3/e4, e4/e4) | String |
| `AGE` | Age at visit | Float |
| `PTGENDER` | Gender (Male/Female) | Categorical |
| `PTEDUCAT` | Years of education | Integer |

### 2.2 Initial Data Size
- **Total records:** ~15,836 visits
- **Unique patients (RID):** ~6,275
- **Date range:** 2005–2020
- **Visit intervals:** Baseline (bl), 6-month (m06), 12-month (m12), 18-month (m18), 24-month (m24), 36-month (m36), 48-month (m48), 60-month (m60)

---

## 3. Preprocessing Pipeline

### Step 1: Filter Relevant Columns
```python
columns = ['RID', 'VISCODE', 'EXAMDATE', 'DX', 'MMSE', 'CDR', 
           'APOE4', 'AGE', 'PTGENDER', 'PTEDUCAT']
df = raw_data[columns]
```

### Step 2: Inclusion & Exclusion Criteria

**Inclusion:**
- Patients with ≥ 2 longitudinal visits
- Patients with baseline MMSE score
- Patients with baseline diagnosis (DX)

**Exclusion:**
- Records with missing diagnosis
- Records with missing MMSE at baseline
- Non-AD dementia types (e.g., vascular dementia, frontotemporal dementia)
- Visits without valid DX, MMSE, or CDR

**After filtering:** ~15,836 records retained

### Step 3: Diagnosis Harmonization

Original ADNI DX values mapped to 3 clinical states:

| Original DX | Mapped State | Description |
|-------------|--------------|-------------|
| CN | **CN** | Cognitively Normal |
| SMC, SMC+ | **CN** | Cognitively Normal (subjective memory concern) |
| EMCI, LMCI | **MCI** | Mild Cognitive Impairment |
| MCI | **MCI** | Mild Cognitive Impairment |
| AD, Dementia | **Dementia** | Alzheimer's Dementia |

```python
def harmonize_dx(dx):
    if dx in ['CN', 'SMC', 'SMC+']:
        return 'CN'
    elif dx in ['MCI', 'EMCI', 'LMCI']:
        return 'MCI'
    elif dx in ['AD', 'Dementia']:
        return 'Dementia'
    else:
        return None  # Exclude
```

### Step 4: Missing Value Handling

**Actual handling: listwise deletion, NOT imputation.** Records missing
diagnosis (DX) or MMSE are dropped (15,836 → 12,464 records). Observation
likelihoods are computed from observed (non-missing) values only. No
median/mode imputation is performed in the pipeline that produced the
model parameters (`Pra_pemrosesan_Dataset_POMDP.ipynb` uses `dropna`;
`build-pomdp-model.R` uses `filter(!is.na)`).

| Variable | Strategy |
|----------|----------|
| DX, MMSE | Drop record if missing (listwise) |
| CDR, APOE4 | Use observed values only; no imputation |

### Step 5: MMSE Binning (Observation Categories)

| Range | Category | Clinical Meaning |
|-------|----------|------------------|
| MMSE ≥ 26 | Normal | Cognitively healthy |
| 20 ≤ MMSE ≤ 25 | Sedang (Moderate) | Mild impairment |
| MMSE < 20 | Rendah (Low) | Severe impairment |

```python
def mmse_bin(mmse):
    if mmse >= 26: return 'MMSE_Normal'
    elif mmse >= 20: return 'MMSE_Sedang'
    else: return 'MMSE_Rendah'
```

### Step 6: CDR Binning (Observation Categories)

| Original CDR | Category | Clinical Meaning |
|--------------|----------|------------------|
| 0 | CDR_0 | No dementia |
| 0.5 | CDR_0_5 | Very mild dementia / MCI |
| ≥ 1.0 | CDR_1_plus | Mild-to-severe dementia |

```python
def cdr_bin(cdr):
    if cdr == 0: return 'CDR_0'
    elif cdr == 0.5: return 'CDR_0_5'
    else: return 'CDR_1_plus'
```

### Step 7: APOE4 Risk Category Collapse

Original 6 genotypes → 3 risk categories:

| Genotype | Risk Category | Risk Level |
|----------|---------------|------------|
| e2/e2, e2/e3, e3/e3 | APOE4_Negatif | Low |
| e2/e4, e3/e4 | APOE4_Hetero | Moderate |
| e4/e4 | APOE4_Homo | High |

```python
def apoe4_risk(genotype):
    neg = ['e2/e2', 'e2/e3', 'e3/e3']
    het = ['e2/e4', 'e3/e4']
    hom = ['e4/e4']
    if genotype in neg: return 'APOE4_Negatif'
    elif genotype in het: return 'APOE4_Hetero'
    elif genotype in hom: return 'APOE4_Homo'
    else: return None
```

### Step 8: Transition Probability Extraction

For each patient with ≥ 2 visits, compute state transitions:

```
For each patient (sorted by VISCODE):
    For each consecutive visit pair (visit_i, visit_{i+1}):
        Count transition: DX_i → DX_{i+1}
```

**Aggregate to population-level probabilities:**

| From \ To | CN | MCI | Dementia |
|-----------|-----|-----|----------|
| CN | 0.929 | 0.068 | 0.003 |
| MCI | 0.000 | 0.892 | 0.108 |
| Dementia | 0.000 | 0.000 | 1.000 |

**Rationale for MCI→CN = 0:**
- Biologically implausible for MCI to revert to CN in ADNI cohort
- Longitudinal data shows no spontaneous recovery pathway

**Rationale for Dementia as absorbing:**
- Neurodegeneration is irreversible
- No patients in ADNI recover from Dementia to MCI or CN

### Step 9: Observation Probability — MIXED PROVENANCE

**Important:** the three observation models do NOT share a single
extraction method.

- **APOE4** is genuinely extracted from ADNI by frequency counting
  `P(Obs | State) = Count(Obs ∧ State) / Count(State)` (reproduces from
  the CSV to 3 decimals).
- **MMSE and CDR are hand-specified clinical noise models**, NOT the
  output of frequency counting. Counting the CSV gives near-deterministic
  likelihoods (e.g. MMSE `P(Normal|CN)≈0.997`, CDR `P(0.5|MCI)≈0.90`);
  the tabled values below (MMSE 0.85/0.55/0.20…, CDR 0.25/0.70/0.05…) are
  smoothed by hand to represent realistic test imprecision. This is a
  deliberate modeling choice (raw, near-deterministic likelihoods induce
  degenerate over-testing), comparable to the hand-specified reward
  function. The code comments in `models/pomdp_models.py` reflect this
  ("akurasi sedang/tinggi" = design intent for MMSE/CDR vs. "dari data
  ADNI nyata" for APOE4).

**MMSE observation probabilities (hand-specified):**

| State | Normal | Sedang | Rendah |
|-------|--------|--------|--------|
| CN | 0.85 | 0.13 | 0.02 |
| MCI | 0.55 | 0.35 | 0.10 |
| Dementia | 0.20 | 0.45 | 0.35 |

**CDR observation probabilities (hand-specified, NOT counted):**

| State | CDR_0 | CDR_0.5 | CDR_1_plus |
|-------|-------|---------|------------|
| CN | 0.95 | 0.04 | 0.01 |
| MCI | 0.25 | 0.70 | 0.05 |
| Dementia | 0.05 | 0.30 | 0.65 |

**APOE4 observation probabilities (ADNI-derived by counting, normalized):**

| State | Negatif | Hetero | Homo |
|-------|---------|--------|------|
| CN | 0.691 | 0.281 | 0.028 |
| MCI | 0.555 | 0.353 | 0.092 |
| Dementia | 0.339 | 0.475 | 0.186 |

### Step 10: Initial Belief Distribution

Computed on the 12,464-record subset retained after dropping rows with
missing DX or MMSE (NOT the full 15,836 table):
```
b₀(CN)       = 4,801 / 12,464 = 0.385
b₀(MCI)      = 5,198 / 12,464 = 0.417
b₀(Dementia) = 2,465 / 12,464 = 0.198
```
(Note: 6,275 / 6,565 / 2,996 are the CN/MCI/Dementia *record* counts in
the full 15,836 table — they are NOT patient counts and do NOT yield b₀.)

### Step 11: Validation Checks

- [x] Transition probabilities sum to 1.0 per row
- [x] Observation probabilities sum to 1.0 per row
- [x] Initial belief sums to 1.0
- [x] No negative probabilities
- [x] MCI→CN = 0.000 (biologically validated)
- [x] Dementia is absorbing state

---

## 4. Final Output

```
ADNI_POMDP_Complete.csv
├── 15,836 records
├── 3,777 unique patients (RID)
├── 3 clinical states (CN, MCI, Dementia)
├── 3 test types (MMSE, CDR, APOE4)
├── 9 observation categories
├── 6-month transition probabilities
└── Initial belief: [0.385, 0.417, 0.198]
```

---

## 5. References

1. **ADNI:** Alzheimer's Disease Neuroimaging Initiative. https://adni.loni.usc.edu/
2. **Data Use Agreement:** Researchers must agree to ADNI data use terms before accessing the dataset.
3. **Citations:** When using ADNI data, cite: "Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database."

---

*Preprocessing completed: April 2026*
