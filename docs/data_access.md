# Data access

Raw data are not included in this repository.

| Dataset | How to obtain it | Expected path |
|---|---|---|
| Domestic Energy Performance Certificates | The extract used in this dissertation was provided by the supervisor. The [official EPC service](https://get-energy-performance-data.communities.gov.uk/) may require an account and permission to access bulk data. | `EPC/epc_domestic_certificates_202607091646.csv` |
| ONS Postcode Directory, May 2025 | Open the [ONS download page](https://www.data.gov.uk/dataset/7c43745c-4c65-4447-b86a-d720def173c8/ons-postcode-directory-may-2025-for-the-uk), download the ZIP file and extract the CSV. | `postcode-LSOA look-up/Data/ONSPD_MAY_2025_UK.csv` |
| English Indices of Deprivation 2025 | Open the [government release page](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025) and download **File 7** in CSV format. | `IMD/File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv` |
| Census 2021 household tenure | Open [ONS dataset TS054](https://www.ons.gov.uk/datasets/TS054/editions/2021/versions/2), select 2021 LSOAs covering Greater Manchester and download the CSV. Rename it if necessary. | `census/4169201263056497.csv` |

The exact EPC extract is not redistributed because access may be restricted. An authorised extract with the same fields can be used, but record counts may differ if it was downloaded at a different time. A different EPC filename can be supplied with `--epc`.

Running the pipeline creates the analytical dataset and other results in `data/processed/` and `outputs/`. The submitted aggregate results are available in `verification_outputs/` for comparison.

Do not publish property-level EPC data or files whose licences restrict redistribution.
