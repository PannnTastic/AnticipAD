#!/usr/bin/env Rscript
# Script untuk membangun model POMDP dari data ADNIMERGE2
# Menghitung Transition Probabilities dan Observation Probabilities

setwd("D:/pomdp/ADNIMERGE2/ADNIMERGE2/ADNIMERGE2-full")
library(dplyr)

# Load data POMDP yang sudah diekstrak
load("pomdp-extracted/pomdp_data.rda")
message("📊 Data POMDP dimuat: ", nrow(pomdp_data), " records")

# ============================================================================
# 1. PREPROCESSING - Urutkan data longitudinal
# ============================================================================
message("\n=== 1. PREPROCESSING DATA ===")

# Buat urutan visit
visit_order <- c("sc", "bl", "m06", "m12", "m18", "m24", "m30", "m36", 
                 "m42", "m48", "m54", "m60", "m66", "m72", "m78", "m84",
                 "m90", "m96", "m102", "m108", "m114", "m120", "m126", 
                 "m132", "m138", "m144", "uns1", "f", "nv", "v03", "v06",
                 "v11", "v12", "v21", "v22", "v31", "v32", "v41", "v42",
                 "v51", "v52", "init", "y1", "y2", "y3", "y4", "y5", 
                 "y6", "y7", "y8", "y9", "y10", "y11", "y12", "y13", "y14")

pomdp_clean <- pomdp_data %>%
  filter(!is.na(DX)) %>%
  mutate(
    VISCODE_NUM = match(VISCODE, visit_order),
    STATE = factor(DX, levels = c("CN", "MCI", "Dementia")),
    STATE_NUM = as.numeric(STATE)  # 1=CN, 2=MCI, 3=Dementia
  ) %>%
  filter(!is.na(VISCODE_NUM)) %>%
  arrange(RID, VISCODE_NUM)

message("✅ Data setelah cleaning: ", nrow(pomdp_clean), " records")

# ============================================================================
# 2. TRANSITION PROBABILITY MATRIX (T)
# ============================================================================
message("\n=== 2. MENGHITUNG TRANSITION PROBABILITY MATRIX ===")

# Hitung transisi antar states
transitions <- pomdp_clean %>%
  group_by(RID) %>%
  mutate(
    NEXT_STATE = lead(STATE),
    TIME_DIFF = lead(VISCODE_NUM) - VISCODE_NUM
  ) %>%
  ungroup() %>%
  filter(!is.na(NEXT_STATE))

# Hitung matriks transisi
trans_matrix <- transitions %>%
  count(STATE, NEXT_STATE) %>%
  group_by(STATE) %>%
  mutate(PROB = n / sum(n)) %>%
  ungroup()

message("\n📊 Transition Counts:")
print(trans_matrix %>% select(STATE, NEXT_STATE, n) %>% tidyr::pivot_wider(names_from=NEXT_STATE, values_from=n, values_fill=0))

message("\n📊 Transition Probability Matrix (T):")
t_prob <- trans_matrix %>% 
  select(STATE, NEXT_STATE, PROB) %>%
  tidyr::pivot_wider(names_from=NEXT_STATE, values_from=PROB, values_fill=0)
print(t_prob)

# Simpan matriks transisi sebagai matrix
T_matrix <- matrix(0, nrow=3, ncol=3, dimnames=list(c("CN","MCI","Dementia"), c("CN","MCI","Dementia")))
for(i in 1:nrow(trans_matrix)) {
  from_state <- as.character(trans_matrix$STATE[i])
  to_state <- as.character(trans_matrix$NEXT_STATE[i])
  T_matrix[from_state, to_state] <- trans_matrix$PROB[i]
}

# Normalisasi agar setiap baris sum=1
T_matrix <- T_matrix / rowSums(T_matrix)
message("\n📊 Final Transition Matrix T (normalized):")
print(T_matrix)

# ============================================================================
# 3. OBSERVATION PROBABILITY (Z) - Discretisasi MMSE & CDR
# ============================================================================
message("\n=== 3. MENGHITUNG OBSERVATION PROBABILITY ===")

# Discretisasi MMSE menjadi kategori
pomdp_clean <- pomdp_clean %>%
  mutate(
    MMSE_CAT = case_when(
      is.na(MMSE_SCORE) ~ NA_character_,
      MMSE_SCORE >= 27 ~ "Normal",      # 27-30
      MMSE_SCORE >= 21 ~ "Mild",        # 21-26
      MMSE_SCORE >= 11 ~ "Moderate",    # 11-20
      TRUE ~ "Severe"                   # 0-10
    ),
    CDR_CAT = case_when(
      is.na(CDR_SCORE) ~ NA_character_,
      CDR_SCORE <= 0 ~ "Normal",        # 0
      CDR_SCORE <= 1 ~ "Mild",          # 0.5-1
      CDR_SCORE <= 2 ~ "Moderate",      # 1.5-2
      TRUE ~ "Severe"                   # 2.5-3
    )
  )

message("\n📊 Distribusi MMSE kategori per State:")
obs_mmse_table <- pomdp_clean %>%
  filter(!is.na(MMSE_CAT)) %>%
  count(STATE, MMSE_CAT) %>%
  group_by(STATE) %>%
  mutate(PROB = n / sum(n)) %>%
  ungroup()

print(obs_mmse_table %>% select(STATE, MMSE_CAT, PROB) %>% tidyr::pivot_wider(names_from=MMSE_CAT, values_from=PROB, values_fill=0))

message("\n📊 Distribusi CDR kategori per State:")
obs_cdr_table <- pomdp_clean %>%
  filter(!is.na(CDR_CAT)) %>%
  count(STATE, CDR_CAT) %>%
  group_by(STATE) %>%
  mutate(PROB = n / sum(n)) %>%
  ungroup()

print(obs_cdr_table %>% select(STATE, CDR_CAT, PROB) %>% tidyr::pivot_wider(names_from=CDR_CAT, values_from=PROB, values_fill=0))

# ============================================================================
# 4. SIMPAN MODEL POMDP
# ============================================================================
message("\n=== 4. MENYIMPAN MODEL POMDP ===")

pomdp_model <- list(
  states = c("CN", "MCI", "Dementia"),
  n_states = 3,
  n_observations = 4,  # Normal, Mild, Moderate, Severe
  transition_matrix = T_matrix,
  observation_mmse = obs_mmse_table,
  observation_cdr = obs_cdr_table,
  data = pomdp_clean,
  n_patients = length(unique(pomdp_clean$RID)),
  n_records = nrow(pomdp_clean)
)

save(pomdp_model, file = "pomdp-extracted/pomdp_model.rda")
write.csv(as.data.frame(T_matrix), "pomdp-extracted/transition_matrix.csv")

message("💾 Model POMDP disimpan di:")
message("   - pomdp-extracted/pomdp_model.rda")
message("   - pomdp-extracted/transition_matrix.csv")

# ============================================================================
# 5. SUMMARY
# ============================================================================
message("\n========================================")
message("📋 RINGKASAN MODEL POMDP")
message("========================================")
message("States: ", paste(pomdp_model$states, collapse=", "))
message("Jumlah Pasien: ", pomdp_model$n_patients)
message("Jumlah Records: ", pomdp_model$n_records)
message("\n📊 Transition Matrix T:")
print(round(T_matrix, 3))

message("\n✅ Model POMDP siap digunakan untuk DESPOT!")
message("\nContoh penggunaan:")
message("  # Load model")
message("  load('pomdp-extracted/pomdp_model.rda')")
message("  ")
message("  # Akses transition matrix")
message("  pomdp_model$transition_matrix")
message("  ")
message("  # Akses data")
message("  head(pomdp_model$data)")
