# Residential energy efficiency inequalities in Greater Manchester

This repository contains the code and supporting material for my MSc Data Science dissertation. The analysis examines neighbourhood differences in domestic Energy Performance Certificate (EPC) ratings across Greater Manchester.

The EPC records were linked to 2021 Lower Layer Super Output Areas (LSOAs) and combined with Census 2021 tenure data and the English Indices of Deprivation 2025. The main outcome is the percentage of EPC-linked dwellings rated below C in each LSOA.

## Repository contents

- `src/` contains the Python scripts used for data construction, spatial analysis, modelling, SHAP analysis and sensitivity checks.
- `docs/` contains the technical appendix and data instructions.
- `verification_outputs/` contains the aggregate tables, model diagnostics and figures from the reported analysis.
- `run_pipeline.py` runs the scripts in their analytical order.
- `requirements.txt` records the Python packages used.

Raw source data are not included. The required files and their expected locations are listed in [`docs/data_access.md`](docs/data_access.md).

## Running the analysis

The submitted analysis used Python 3.12. From the repository directory, create an environment and install the required packages:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

After placing the source files in the documented locations, check that they can be found:

```bash
python check_setup.py --require-inputs
```

Run the full pipeline with:

```bash
python run_pipeline.py
```

Windows users can run the equivalent PowerShell wrapper:

```powershell
.\run_pipeline.ps1
```

The scripts write new results to the `outputs` folder. These can be compared with the files in `verification_outputs`. The complete run includes grouped cross-validation, spatial permutation tests, SHAP analysis and the sensitivity analyses reported in the dissertation.

## Reported checks

A successful run should produce an analytical sample of 540,310 EPC-linked dwellings in 1,702 Greater Manchester LSOAs. The main spatial result is a Global Moran's I of 0.417 (permutation p = .001). The held-out model results and SHAP summaries are saved in `verification_outputs/models/`.

The analysis is observational and conducted at LSOA level. EPC ratings describe certified, modelled building performance and should not be interpreted as household energy use or fuel poverty.
