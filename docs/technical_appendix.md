# Technical appendix

## Analytical sequence

The final analysis can be reproduced in the following order from the root of the `Materials` folder.

```powershell
python src/build_lsoa_dataset.py
python src/download_gm_lsoa_boundaries.py
python src/run_spatial_analysis.py --permutations 999 --seed 42
python src/run_explainable_models.py --n-iter 15 --inner-splits 4 --seed 42 --permutations 999
python src/render_explainability_figures.py
python src/run_sensitivity_analysis.py
python src/run_candidate_variable_audit.py
python src/build_methodology_workflow.py
```

## Data construction

`build_lsoa_dataset.py` restricts EPC records to Greater Manchester and to certificates lodged from 1 November 2015 to 31 October 2025. It retains the latest certificate per dwelling, normalises postcode strings, links postcodes to 2021 LSOAs through the May 2025 ONS Postcode Directory, constructs the EPC indicators, aggregates them to LSOA and joins Census 2021 tenure and IMD 2025 data.

Valid EPC ratings are A-G. Missing, `not recorded` and `unknown` property and built-form categories are excluded from the relevant denominators. Exact construction years outside 1700-2025 and non-positive floor areas are invalidated. Attribute-specific percentages use the valid record count for that attribute. No LSOA is removed from the main analysis solely because of EPC count or coverage.

## Spatial analysis

The spatial script constructs row-standardised first-order Queen contiguity weights for all 1,702 LSOAs. Global Moran's I and Local Moran's I use 999 permutations and seed 42. Local results are adjusted using the Benjamini-Hochberg false-discovery-rate procedure. The script also repeats the spatial analysis for LSOAs with EPC-to-Census household coverage between 0.25 and 1.25.

## Model evaluation and explanation

The main target is the LSOA Below-C rate. XGBoost and Random Forest are evaluated with Leave-One-LAD-Out validation across the ten Greater Manchester local-authority districts. XGBoost tuning is nested within the held-out evaluation: each outer fold uses a 15-combination random search with four-fold grouped inner validation. A mean-prediction Dummy regressor supplies the baseline.

The final explanatory XGBoost model is fitted to the complete sample after a separate 20-combination, five-fold grouped search. SHAP values describe how the fitted model distributes its predictions across the supplied features. They do not identify causal effects, and correlated variables can redistribute model contribution.

## Sensitivity analysis

The retained sensitivity tests cover:

- removal of social-rented share;
- removal of IMD score;
- substitution of non-mains-gas share for electric main fuel;
- restriction to EPC-to-Census coverage from 0.25 to 1.25;
- restriction to LSOAs with no more than 30% missing EPC age band;
- mean EPC score as the alternative outcome;
- an expanded 15-feature k-1 representation of the complete category groups.

## Verification

`verification_outputs/` contains the saved aggregate outputs from the dissertation run. The principal checks are listed in the root README. Exact output rows can also be compared through file hashes, although model serialisation metadata and figure bytes can vary across software builds.

