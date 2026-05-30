#!/usr/bin/env Rscript
# Script untuk menggabungkan SEMUA data yang dibutuhkan Gemini ke 1 CSV
# Variabel: DX (State), MMSE/CDR (Observation), RID/VISCODE/DATE (Transition), 
#           AGE/SEX/EDUCATION/APOE (Demografi & Genetik)

setwd("D:/pomdp/ADNIMERGE2/ADNIMERGE2/ADNIMERGE2-full")
library(dplyr)

message("📦 MEMBUAT FILE CSV LENGKAP UNTUK POMDP/DESPOT\n")

# Load data
message("Loading data...")
load("data/ADSL.rda")      
load("data/DXSUM.rda")     
load("data/MMSE.rda")      
load("data/CDR.rda")       

# ============================================================================
# 1. EXTRACT & RENAME KOLOM SESUAI PERMINTAAN GEMINI
# ============================================================================

# ADSL - Demografi & Genetik
demografi <- ADSL %>%
  select(
    RID = SUBJID,           # ID Pasien
    AGE = AGE,              # Usia
    GENDER = SEX,           # Jenis Kelamin  
    EDUCATION = EDUC,       # Tingkat Pendidikan
    APOE4 = APOE            # Status Genetik APOE4
  ) %>%
  mutate(RID = as.numeric(RID)) %>%
  distinct(RID, .keep_all = TRUE)

message("✅ Demografi: ", nrow(demografi), " pasien")

# DXSUM - State (Diagnosis) + Transition info
states <- DXSUM %>%
  select(
    RID = RID,              # ID Pasien (untuk tracking)
    VISCODE = VISCODE,      # Kode Kunjungan (waktu)
    DATE = EXAMDATE,        # Tanggal Kunjungan
    DX = DIAGNOSIS          # Diagnosis (State: CN/MCI/Dementia)
  ) %>%
  filter(!is.na(DX)) %>%
  mutate(
    DATE = as.Date(DATE),
    # Simplify DX ke 3 kategori utama
    DX = case_when(
      DX %in% c("CN", "Normal") ~ "CN",
      DX %in% c("MCI") ~ "MCI", 
      DX %in% c("Dementia", "AD", "Alzheimer") ~ "Dementia",
      TRUE ~ DX
    )
  )

message("✅ States: ", nrow(states), " records")

# MMSE - Observation
mmse_obs <- MMSE %>%
  select(
    RID = RID,
    VISCODE = VISCODE,
    MMSE = MMSCORE          # Skor MMSE (Observation)
  ) %>%
  filter(!is.na(MMSE))

message("✅ MMSE: ", nrow(mmse_obs), " records")

# CDR - Observation  
cdr_obs <- CDR %>%
  select(
    RID = RID,
    VISCODE = VISCODE,
    CDR = CDGLOBAL           # Skor CDR (Observation)
  ) %>%
  filter(!is.na(CDR))

message("✅ CDR: ", nrow(cdr_obs), " records")

# ============================================================================
# 2. MERGE SEMUA KE DALAM 1 DATASET
# ============================================================================
message("\n=== MENGGABUNGKAN SEMUA DATA ===")

# Start dengan states (karena ini yang wajib ada)
final_data <- states %>%
  # Join demografi (1 pasien = 1 baris demografi)
  left_join(demografi, by = "RID") %>%
  # Join MMSE
  left_join(mmse_obs, by = c("RID", "VISCODE")) %>%
  # Join CDR
  left_join(cdr_obs, by = c("RID", "VISCODE")) %>%
  # Urutkan
  arrange(RID, DATE)

message("✅ Total records: ", nrow(final_data))
message("✅ Jumlah pasien unik: ", length(unique(final_data$RID)))

# ============================================================================
# 3. CEK KOMPLELENAS
# ============================================================================
message("\n=== CEK VARIABEL ===")
message("Kolom yang tersedia:")
print(names(final_data))

message("\n📊 STATISTIK PER KOLOM:")
message("\n1. STATE (DX) - Diagnosis:")
print(table(final_data$DX, useNA = "ifany"))

message("\n2. OBSERVATION - MMSE:")
message("   Non-missing: ", sum(!is.na(final_data$MMSE)), " records")
print(summary(final_data$MMSE))

message("\n3. OBSERVATION - CDR:")
message("   Non-missing: ", sum(!is.na(final_data$CDR)), " records")
print(summary(final_data$CDR))

message("\n4. TRANSITION TRACKING:")
message("   RID (ID Pasien): ", length(unique(final_data$RID)), " unik")
message("   VISCODE (Kunjungan): ", length(unique(final_data$VISCODE)), " unik")
message("   DATE (Tanggal): Range ", min(final_data$DATE, na.rm=TRUE), " sampai ", max(final_data$DATE, na.rm=TRUE))

message("\n5. DEMOGRAFI & GENETIK:")
message("   AGE: ", sum(!is.na(final_data$AGE)), " records")
message("   GENDER: ", sum(!is.na(final_data$GENDER)), " records")
print(table(final_data$GENDER, useNA = "ifany"))
message("   EDUCATION: ", sum(!is.na(final_data$EDUCATION)), " records")
message("   APOE4: ", sum(!is.na(final_data$APOE4)), " records")

# ============================================================================
# 4. SIMPAN KE CSV
# ============================================================================
message("\n=== MENYIMPAN KE CSV ===")

# Buat folder output jika belum ada
output_dir <- "pomdp-extracted"
if (!dir.exists(output_dir)) dir.create(output_dir)

# Simpan ke CSV
csv_file <- file.path(output_dir, "ADNI_POMDP_Complete.csv")
write.csv(final_data, csv_file, row.names = FALSE)

message("💾 File CSV tersimpan:")
message("   ", csv_file)
message("   Ukuran: ", round(file.size(csv_file)/1024, 2), " KB")

# ============================================================================
# 5. PREVIEW
# ============================================================================
message("\n=== PREVIEW 15 BARIS PERTAMA ===")
print(head(final_data, 15))

message("\n=== CONTOH 1 PASIEN (LONGITUDINAL) ===")
contoh_pasien <- final_data %>%
  filter(RID == final_data$RID[1]) %>%
  select(RID, VISCODE, DATE, DX, MMSE, CDR, AGE, GENDER, EDUCATION, APOE4)
print(contoh_pasien)

# ============================================================================
# 6. RINGKASAN
# ============================================================================
message("\n========================================")
message("✅ FILE CSV SIAP DIGUNAKAN!")
message("========================================")
message("File: ADNI_POMDP_Complete.csv")
message("Lokasi: pomdp-extracted/")
message("")
message("Isi file:")
message("• Total records: ", nrow(final_data))
message("• Total pasien: ", length(unique(final_data$RID)))
message("")
message("Kolom:")
message("• RID = ID Pasien (untuk tracking longitudinal)")
message("• VISCODE = Kode Kunjungan (bl, m06, m12, dst)")
message("• DATE = Tanggal Kunjungan")
message("• DX = Diagnosis/State (CN, MCI, Dementia)")
message("• MMSE = Skor MMSE (Observation)")
message("• CDR = Skor CDR (Observation)")
message("• AGE = Usia")
message("• GENDER = Jenis Kelamin")
message("• EDUCATION = Pendidikan")
message("• APOE4 = Status Genetik")
message("")
message("Langkah selanjutnya:")
message("Buka file CSV di Excel/R/Python untuk analisis POMDP/DESPOT")
