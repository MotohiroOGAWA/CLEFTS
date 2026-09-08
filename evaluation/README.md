# CLEFTS Evaluation

Column-oriented similarity evaluation used by CLEFTS Workbench. An msentity
`.mssim` file is joined to one `.msds` (or CSV, TSV, JSON, JSONL/NDJSON, or
Parquet) metadata source. The join column is configurable and defaults to
`SpecID` when that column exists in both files.

```bash
python -m evaluation inspect --input result.mssim --metadata observed.msds
python -m evaluation summarize --input result.mssim --metadata observed.msds \
  --group-column AdductType --output-image adduct-boxplot.svg
python -m evaluation summarize --input result.mssim --metadata observed.msds \
  --join-column IDENTIFIER --group-column CollisionEnergy \
  --mode numeric --bins 0,10,20 --output-image collision-energy-boxplot.svg
```

Each group is represented as a Tukey box plot of the `.mssim`
`cosine_similarity` values. SVG output is transparent by default. Use
`--opaque` for a white background and `--width`, `--height`, and `--color` to
customize its appearance.
