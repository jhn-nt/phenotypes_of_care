# Calibrated to Injustice and Accurately Unjust

Analysis code for *Calibrated to Injustice and Accurately Unjust: Clinical AI Has Been Predicting the Wrong World*.

The code implements the Four-Configuration Logical Asymmetry design, the Therapeutic-capability Reference Model (TRM), Clinical-Regime Decomposition, and the calibration analysis, in MIMIC-IV (primary) and eICU-CRD (external validation).

## Contents

| Path | Description |
|---|---|
| `notebooks/01_mimic_primary_analysis.ipynb` | MIMIC-IV analysis — mouthcare and turning phenotypes. Produces Figures 1–4 and Extended Data Table 2. |
| `notebooks/02_eicu_validation.ipynb` | eICU-CRD external validation — glycemic monitoring and pain reassessment phenotypes. Produces Extended Data Table 3. |
| `requirements.txt` | Pinned package versions. |
| `data/README.md` | Data access, cohort definitions, and expected file layout. |

The two notebooks are independent and can be run in either order.

## System requirements

- Python 3.9 (tested on 3.9.12)
- Tested on Ubuntu 20.04 and Windows 10
- No non-standard hardware. Approximately 8 GB RAM is sufficient.

## Installation

```bash
git clone https://github.com/jhn-nt/phenotypes_of_care.git
cd phenotypes_of_care
pip install -r requirements.txt
```

Typical install time on a standard desktop: 2–5 minutes.

## Data

Both analyses use credentialed public databases that cannot be redistributed here. Each requires a PhysioNet account, completion of the required training, and a signed data use agreement.

- MIMIC-IV — https://physionet.org/content/mimiciv/
- eICU Collaborative Research Database — https://physionet.org/content/eicu-crd/

Cohort definitions and the expected file layout are in [`data/README.md`](data/README.md).

## Running the analysis

**MIMIC-IV.** Set `MOUTHCARE_FOLDER` and `TURNING_FOLDER` in Cell 4 to your local extract folders, then run all cells.

**eICU-CRD.** Set the CSV folder path in Cell 2, then run all cells. The Clinical-Regime Decomposition sweep runs first and determines the APACHE IVa analysis windows used by the four-configuration analysis.

### Output

`01_mimic_primary_analysis.ipynb` writes to `figures/` and `mortality_results.xlsx`:

- Figure 1 — ΔAUROC across all four configurations, per phenotype and severity stratum
- Figure 2 — Clinical-Regime Decomposition across the continuous SOFA spectrum
- Figures 3–4 — TRM calibration curves and Expected Calibration Error
- Extended Data Table 2 — four-configuration results on both the race and care-quality axes

`02_eicu_validation.ipynb` produces the equivalent four-configuration results for both eICU phenotypes (Extended Data Table 3), together with the Clinical-Regime Decomposition sweep that defines the analysis windows.

### Runtime

On a standard desktop, end to end:

- `01_mimic_primary_analysis.ipynb` — approximately 30–60 minutes
- `02_eicu_validation.ipynb` — approximately 20–40 minutes

Runtime is dominated by the Clinical-Regime Decomposition sweep and the resampling: 2,000 bootstrap resamples for quartile confidence intervals, 1,000 for race, 1,000 permutations for the maximum-JT uniformity test, and 10,000 for the directional JT degradation test.

## Reproducibility

Results are deterministic given the same data, the pinned package versions, and the fixed random seed (42).

## Citation

Upariputtanggoon N, Giancotti R, Al Attrach R, AL-Louzi RM, Angelotti G, Cajas SA, Chaisutyakorn K, Ellen JG, Hernandez-Boussard T, Kleinlein R, Lorenzi LM, Nanyonjo J, Celi LA. *Calibrated to Injustice and Accurately Unjust: Clinical AI Has Been Predicting the Wrong World.*

## License

MIT — see [`LICENSE`](LICENSE).

## Contact

Nachanon Upariputtanggoon — nachanon.up@gmail.com
