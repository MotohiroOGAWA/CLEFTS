# PubChem-sourced mol training data

These workflows build mol training datasets from a local PubChem CID/SMILES
table. The code is PubChem-sourced, but its purpose is mol training data
preparation.

```bash
PUBCHEM_CID_SMILES=pubchem_cid_smiles.parquet
```

## Input Preview

`pubchem_cid_smiles.parquet` is a parquet table with PubChem compound IDs and
their SMILES strings. The workflows expect at least these columns:

| cid | smiles |
| ---: | --- |
| 1 | `CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C` |
| 2 | `CC(=O)OC(CC(=O)O)C[N+](C)(C)C` |
| 3 | `C1=CC(C(C(=C1)C(=O)O)O)O` |
| 4 | `CC(CN)O` |
| 5 | `C(C(=O)COP(=O)(O)O)N` |

## Descriptor Coverage Sample

This sample searches PubChem for molecules that cover descriptor-value bins
used by the mol training descriptor targets. It writes a selected parquet file,
a newline-delimited SMILES file for training, and a JSON report describing the
covered bins.

```bash
python workflows/mol_training_data/pubchem/sample_pubchem_descriptor_coverage.py \
  --input "$PUBCHEM_CID_SMILES" \
  --output-dir data/processed/pubchem/descriptor_coverage_sample \
  --min-count 100 \
  --max-count 300 \
  --discovery-rows 1000000 \
  --batch-size 1000 \
  --workers 30 \
  --center mean
```

Add `--yes` to overwrite an existing output directory non-interactively.

Primary outputs:

```text
data/processed/pubchem/descriptor_coverage_sample/pubchem_descriptor_coverage_sample.parquet
data/processed/pubchem/descriptor_coverage_sample/pubchem_descriptor_coverage_sample.smiles.txt
data/processed/pubchem/descriptor_coverage_sample/pubchem_descriptor_coverage_sample_report.json
```

## Mol Encoder Feature Coverage Sample

This sample searches PubChem for molecules that cover atom and bond feature
values used by the mol encoder. It is useful for making the pretraining set
include uncommon symbols, valences, hybridizations, bond types, and related
feature targets.

```bash
python workflows/mol_training_data/pubchem/sample_pubchem_mol_encoder_coverage.py \
  --input "$PUBCHEM_CID_SMILES" \
  --output-dir data/processed/pubchem/mol_encoder_coverage_sample \
  --symbols "C,N,O,P,S,F,Cl,Br,I" \
  --min-count 100 \
  --max-count 300 \
  --batch-size 10000 \
  --workers 10
```

Add `--yes` to overwrite an existing output directory non-interactively.

Primary outputs:

```text
data/processed/pubchem/mol_encoder_coverage_sample/pubchem_mol_encoder_coverage_sample.parquet
data/processed/pubchem/mol_encoder_coverage_sample/pubchem_mol_encoder_coverage_sample.smiles.txt
data/processed/pubchem/mol_encoder_coverage_sample/pubchem_mol_encoder_coverage_sample_report.json
```

## Morgan Sphere Exclusion Sample

This sample builds a chemically diverse subset by selecting molecules whose
Morgan fingerprints are not too similar to previously selected molecules. It is
useful for adding broad structural diversity to mol encoder pretraining.

```bash
python workflows/mol_training_data/pubchem/sphere_exclusion/run.py \
  --input "$PUBCHEM_CID_SMILES" \
  --output data/processed/pubchem/sphere_exclusion_sample/pubchem_morgan2048_sphere_exclusion.parquet \
  --report data/processed/pubchem/sphere_exclusion_sample/pubchem_morgan2048_sphere_exclusion_report.json \
  --threshold 0.5 \
  --radius 2 \
  --n-bits 2048 \
  --max-selected 500000 \
  --batch-size 50 \
  --fingerprint-workers 30
```

Add `--yes` to overwrite the output directory non-interactively.

Primary outputs:

```text
data/processed/pubchem/sphere_exclusion_sample/pubchem_morgan2048_sphere_exclusion.parquet
data/processed/pubchem/sphere_exclusion_sample/pubchem_morgan2048_sphere_exclusion.smiles.txt
data/processed/pubchem/sphere_exclusion_sample/pubchem_morgan2048_sphere_exclusion_report.json
```
