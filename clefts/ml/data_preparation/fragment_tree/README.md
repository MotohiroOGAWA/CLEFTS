# Creating branching teacher data

Run the existing Preparation CLI from `mnt/app`:

```bash
python -m clefts.cli train create-fragment-tree-data \
  --input train.msds --validation-input validation.msds \
  --output-dir prepared --params model_config.json \
  --num-workers 4 --chunk-size 4 \
  --max-node 1000 --max-edge -1 \
  --normalize-intensities 1 --overwrite 1
```

All Preparation argument names, defaults and choices are preserved. There are no
new branching Preparation flags. The Workbench uses this same command.

Each Source SMILES produces a schema-v6 `SourceActionStructure` `.preft.pt`.
`fragmentation_schema` is `source-anchored-branching-v1`. v5 structures require
regeneration from the original spectra.

The chemistry tree assigns measured peaks to pathways. Each supported pathway
retains its actual transition order and precursor seed. Only MS2 suffixes become
teacher nodes and positive transitions, including their intermediate ancestors.
Ambiguous pathways are retained. Multiple precursor candidates are independent
rows; Source itself has an empty row. Teacher MS2 depth starts at zero.

The action generator deterministically deduplicates exact actions and removes
unsupported chemistry, empty fragments and true no-ops. Preparation never uses
neural scores or random rankings to remove teachers. Every positive action must
remain in the Source action universe.

Stored fields include:

- `teacher_node_sample_index`, `teacher_node_precursor_row_index`,
  `teacher_node_parent_index`, `teacher_node_added_action_index`,
  `teacher_node_ms2_depth` and CSR `teacher_node_action_ptr/index`.
- CSR `teacher_positive_action_ptr/index/weight`; weights are maximum descendant
  observed intensities. `teacher_node_observed` is observation metadata, not EOS.
- CSR `sample_positive_action_ptr/index` containing only learned MS2 branch actions.
- CSR `sample_precursor_row_ptr`, `precursor_row_action_ptr/index` for seed alternatives.
- Positive transition tensors, teacher-to-materialized-node mapping, prepared
  post-model graphs, formula intensities, and original peak annotations.

Full possible action states, negative transitions and dense node/action labels
are not stored. Valid candidates and weak negatives are generated during forward
through the same chemistry compatibility component used at inference.

Manifests and statistics report teacher node count, positive transitions,
precursor candidates and maximum MS2 depth in addition to the existing fields.
The result viewer displays schema-v6 summaries and branching action targets.
Collation offsets sample, action, precursor, teacher and materialized node indexes.

Training details and validation outputs are described in
[`fragment_tree_training/README.md`](../../training/fragment_tree_training/README.md).
