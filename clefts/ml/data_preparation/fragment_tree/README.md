# Preparing fragment-tree training data

Run preparation from the application root:

```bash
python -m clefts.cli train create-fragment-tree-data \
  --input train.msds --validation-input validation.msds \
  --output-dir prepared --params model_config.json \
  --num-workers 4 --chunk-size 4 \
  --max-node 1000 --max-edge -1 \
  --normalize-intensities 1 --overwrite 1
```

Preparation is the chemistry boundary. It parses molecules, creates primitive
actions, assigns peaks to pathways, materializes fragments, constructs graph
tensors, enumerates ion/hydrogen states, computes final formulas and masses,
groups equivalent physical ions, and stores target assignments.

Spectra with the same compound and normalized main adduct share one branch
group. Their positive pathways are unioned across collision energies before
weak negatives are created. Packed data includes:

- source graphs and primitive-action features;
- teacher action sets at multiple depths;
- valid continuation actions and explicit next-state indices;
- positive paths grouped by experimental peak and weak-negative actions;
- shared fragment-tree topology and prepared fragment graph tensors;
- physical-ion candidates and packed `(ion, unsaturation, radical)` explanations;
- final formula tensors, exact m/z, charge, targets, and peak assignments.

The prepared payload uses one current contract and contains no data-format
version or migration layer. Regenerate `.preft.pt` files after a contract change.
Training loads these tensors directly and must not invoke RDKit or other
chemistry operations.

Training details are in
[`fragment_tree_training/README.md`](../../training/fragment_tree_training/README.md).
