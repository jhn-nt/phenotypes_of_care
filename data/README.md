# Data

No data files are included in this repository. MIMIC-IV and eICU-CRD are credentialed databases whose data use agreements do not permit redistribution.

## Access

Both are available from PhysioNet. Access requires a PhysioNet account, completion of the CITI "Data or Specimens Only Research" training, and a signed data use agreement for each database.

- MIMIC-IV https://physionet.org/content/mimiciv/
- eICU Collaborative Research Database https://physionet.org/content/eicu-crd/

---

## MIMIC-IV

**Source.** Beth Israel Deaconess Medical Center, 2008–2022.

**Inclusion.** Adult ICU stays with at least 24 hours of invasive mechanical ventilation and a valid recorded weight. Source populations before eligibility filtering were 26,443 patient-stays (mouthcare pipeline) and 26,494 (turning pipeline); final analytic cohorts were 8,675 and 8,919.

**Severity.** SOFA total score, stratified into Low (0–6), Medium (7–11), High (≥12).

**Care phenotypes.** Mouthcare and patient repositioning (turning), operationalised as the mean interval in hours between consecutive documented care events. Within each severity stratum, patients are ranked by this interval and split into four equal-sized quartiles, Q1 (most frequent care) to Q4 (least).

### Expected file layout

`01_mimic_primary_analysis.ipynb` expects one folder per phenotype, each containing these pickled DataFrames, all keyed on `stay_id`:

```
<phenotype_folder>/
├── cohort.pkl          # stay_id, hospital_expire_flag, admission_age, los_icu, race
├── sofa.pkl            # SOFA total + six organ-system subscores
├── vitals.pkl          # heart rate, MBP, temperature, respiratory rate (min/max/mean)
├── lab.pkl             # hematocrit, WBC, creatinine, BUN, sodium, albumin,
│                       #   total bilirubin, glucose, bicarbonate (min/max)
├── gcs.pkl             # gcs_min
├── proxy.pkl           # stay_id, day, average_item_interval, item_volume, n_caregivers
└── comorbidities.pkl   # optional
```

`proxy.pkl` holds the care phenotype at patient-day level; `average_item_interval` is averaged per patient in the notebook to assign care-quality quartiles.

### Predictors (37)

Six SOFA organ-system subscores — respiration, cardiovascular, liver, coagulation, renal, CNS — used as a Logistic Organ Dysfunction Score proxy, plus 31 laboratory and vital-sign variables used as an APACHE III proxy:

- min, max and mean of heart rate, mean arterial pressure, temperature, respiratory rate (12)
- min and max of hematocrit, WBC, creatinine, BUN, sodium, albumin, total bilirubin, glucose, bicarbonate (18)
- minimum Glasgow Coma Scale (1)

Composite scoring totals are excluded to preserve granularity. **SOFA total, care frequency, and race are never model features** — SOFA total is used for severity stratification, care frequency for quartile assignment, and race for post-hoc evaluation only.

**Outcome.** `hospital_expire_flag`, ascertained from hospital discharge records.

**Race and ethnicity.** Consolidated into White, Black, Hispanic/Latino, and Asian. Patients recorded as unknown, declined, or in the heterogeneous "Other" category are dropped from race-stratified analyses only; they remain in model training and in the care-quartile evaluation.

---

## eICU-CRD

**Source.** 78 hospitals across all four U.S. census regions.

**Inclusion.** ICU stays with documented care events for the relevant phenotype. No mechanical ventilation or length-of-stay criterion is applied, so this is a broader population than the MIMIC-IV cohorts — glycemic monitoring and pain reassessment are not ventilation-specific in the way mouthcare and turning are.

**Severity.** APACHE IVa. The Clinical-Regime Decomposition sweep identifies the analysis window before the four-configuration analysis is run: APACHE IVa 50–125 for glucose (n=7,833, 78 hospitals) and 50–95 for pain (n=6,194, 70 hospitals).

### Expected file layout

`02_eicu_validation.ipynb` expects one combined CSV per phenotype:

```
<data_folder>/
├── glucose_combined.csv
└── pain_combined.csv
```

Each requires `patientunitstayid`, `hospitaldischargestatus`, `average_item_interval`, `apache_iva`, `ethnicity`, and the 20 `aps_*` columns.

### Predictors (20)

The APACHE IVa Acute Physiology Score components: `aps_heartrate`, `aps_meanbp`, `aps_creatinine`, `aps_glucose`, `aps_pao2`, `aps_pco2`, `aps_ph`, `aps_bun`, `aps_bilirubin`, `aps_albumin`, `aps_sodium`, `aps_hematocrit`, `aps_wbc`, `aps_temperature`, `aps_respiratoryrate`, `aps_urine`, `aps_eyes`, `aps_motor`, `aps_verbal`, `aps_fio2`.

These are day-one admission physiology by the definition of the APACHE IVa score, matching the first-24-hour design used in MIMIC-IV. Treatment-related columns (intubated, ventilated, dialysis, medications) are excluded so that no care-intensity variable enters the model as a predictor. Each database uses its own validated physiology set rather than a shared feature list, so the validation tests the transportability of the design rather than of one feature construction.

**Outcome.** `hospitaldischargestatus`.
