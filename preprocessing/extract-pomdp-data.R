#!/usr/bin/env Rscript
# Script untuk ekstrak data POMDP dari ADNIMERGE2
# Sesuai dengan nama kolom yang sebenarnya ada di data

setwd("D:/pomdp/ADNIMERGE2/ADNIMERGE2/ADNIMERGE2-full")
library(dplyr)

# Load data dari folder data/
message("Loading data...")
load("data/ADSL.rda")      # Demografi
load("data/DXSUM.rda")     # Diagnosis  
load("data/MMSE.rda")      # MMSE score
load("data/CDR.rda")       # CDR score

# Cek kolom yang tersedia
message("\n=== CEK KOLOM YANG TERSEDIA ===")
message("\nADSL cols: ", paste(names(ADSL), collapse=", "))
message("\nDXSUM cols: ", paste(names(DXSUM), collapse=", "))
message("\nMMSE cols: ", paste(names(MMSE), collapse=", "))
message("\nCDR cols: ", paste(names(CDR), collapse=", "))

# Ekstrak data untuk POMDP
message("\n\n=== MENGEKSTRAK DATA POMDP ===")

# 1. Ambil baseline demografi dari ADSL
# Catatan: ADSL pakai SUBJID, bukan RID
baseline_demo <- ADSL %>%
  select(SUBJID = SUBJID, 
         AGE = AGE,
         SEX = SEX,
         EDUCATION = EDUC,
         APOE = APOE) %>%
  distinct(SUBJID, .keep_all = TRUE)

message("✅ Baseline demografi: ", nrow(baseline_demo), " pasien")

# 2. Ambil states (DX/Diagnosis) dari DXSUM
# Catatan: DX pakai DIAGNOSIS, bukan DX
states <- DXSUM %>%
  select(RID = RID, 
         VISCODE = VISCODE,
         DATE = EXAMDATE,
         DX = DIAGNOSIS) %>%
  filter(!is.na(DX))

message("✅ States records: ", nrow(states), " diagnosis records")
message("   Unique DX values: ", paste(unique(states$DX), collapse=", "))

# 3. Ambil observations (MMSE)
obs_mmse <- MMSE %>%
  select(RID = RID,
         VISCODE = VISCODE,
         MMSE_SCORE = MMSCORE) %>%
  filter(!is.na(MMSE_SCORE))

message("✅ MMSE records: ", nrow(obs_mmse), " skor MMSE")

# 4. Ambil observations (CDR)
obs_cdr <- CDR %>%
  select(RID = RID,
         VISCODE = VISCODE,
         CDR_SCORE = CDGLOBAL) %>%
  filter(!is.na(CDR_SCORE))

message("✅ CDR records: ", nrow(obs_cdr), " skor CDR")

# Konversi SUBJID ke numeric agar match dengan RID
baseline_demo <- baseline_demo %>%
  mutate(RID = as.numeric(SUBJID)) %>%
  select(RID, AGE, SEX, EDUCATION, APOE)

# 5. Merge semua untuk membuat dataset POMDP lengkap
message("\n=== MERGING DATA POMDP ===")

pomdp_data <- states %>%
  left_join(baseline_demo, by = "RID") %>%
  left_join(obs_mmse, by = c("RID", "VISCODE")) %>%
  left_join(obs_cdr, by = c("RID", "VISCODE")) %>%
  arrange(RID, DATE)

message("✅ Dataset POMDP lengkap: ", nrow(pomdp_data), " records")
message("   Jumlah pasien unik: ", length(unique(pomdp_data$RID)))

# Simpan sebagai CSV dan RDA
output_dir <- "pomdp-extracted"
if (!dir.exists(output_dir)) dir.create(output_dir)

write.csv(pomdp_data, file.path(output_dir, "pomdp_data.csv"), row.names = FALSE)
save(pomdp_data, file = file.path(output_dir, "pomdp_data.rda"))

message("\n💾 Data disimpan di folder: ", output_dir)
message("   - pomdp_data.csv")
message("   - pomdp_data.rda")

message("\n📊 Preview data POMDP:")
print(head(pomdp_data, 10))

# Summary untuk POMDP
message("\n=== STATISTIK UNTUK POMDP ===")
message("\nDistribusi States (DX):")
print(table(pomdp_data$DX, useNA = "ifany"))

if ("MMSE_SCORE" %in% names(pomdp_data)) {
  message("\nStatistik MMSE:")
  print(summary(pomdp_data$MMSE_SCORE))
}

if ("CDR_SCORE" %in% names(pomdp_data)) {
  message("\nStatistik CDR:")
  print(summary(pomdp_data$CDR_SCORE))
}

message("\n📈 Contoh data longitudinal (pasien pertama):")
sample_patient <- pomdp_data %>%
  filter(RID == pomdp_data$RID[1]) %>%
  select(RID, VISCODE, DATE, DX, MMSE_SCORE, CDR_SCORE)
print(sample_patient)

message("\n✅ Data siap untuk dimodelkan dengan DESPOT!")
