# Local data directory

This directory stores generated analytical and spatial data. Its contents are ignored by Git apart from this file.

After a successful run, the expected files are:

```text
data/
|-- processed/
|   |-- lsoa_analytical_dataset.csv
|   `-- lsoa_spatial_analysis.gpkg
`-- spatial/
    |-- gm_lsoa_2021_bgc_v5.gpkg
    `-- gm_lsoa_2021_bgc_v5_metadata.json
```

The four source datasets are placed in the repository-root folders described in `docs/data_access.md`; none should be committed.

