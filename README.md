# CLEFTS (CLeavage-Expressive Fragmentation Tree Spectrum generator)

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)
![RDKit](https://img.shields.io/badge/RDKit-2025.03.6-green)

---

## Batch MS/MS spectrum prediction

Predict every compatible record in an MSDataset with one model load and bounded
unique-SMILES chunks:

```bash
msentity-predict-clefts \
  --input source.msds \
  --output source_clefts.msds \
  --model model.pt \
  --db MoNA \
  --batch-size 32 \
  --device cuda
```

The input must provide unique, non-empty `SpecID`, `SMILES`, `PrecursorMZ`,
`AdductType`, and `CollisionEnergy` values (column names are configurable).
Output records retain the input metadata, move the immediate input identifier to
`SourceSpecID`, and receive `clefts-<DB>-<sequence>` identifiers. Provenance and
prediction conditions are recorded in `PredictionTool`, `PredictionToolVersion`,
`PredictionModel`, `PredictionTimestamp`, `PredictedPrecursorMZ`,
`PredictionCollisionEnergy`, `PredictionCollisionEnergyUnit`, and
`PredictionAdductType`. Input `PrecursorMZ`, `ExactMass`, and collision-energy
metadata are not overwritten. Dataset-level attributes and tags also identify
the generated spectra.

Prediction structures are processed in chunks while the checkpoint remains
loaded. A failed multi-SMILES chunk is retried per molecule so valid records can
still be saved. `<output>.run/run.json` records the execution and
`<output>.run/failures.tsv` contains record-level failures. Existing output is
replaced only with `--overwrite`; output publication uses an atomic rename.

The same operation is available in **Workbench → Predict Spectrum → Batch
MSDataset prediction**. `clefts-predict-spectrum` is an equivalent command name.



---
