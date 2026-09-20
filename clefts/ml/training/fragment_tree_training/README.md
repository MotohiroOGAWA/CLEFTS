# Branching cleavage training (schema v6)

Training uses a pretrained molecular encoder, Source-anchored concrete cleavage
actions, a `BranchingCleavageDecoder`, and the post-materialization intensity
model. Each valid action receives an independent sigmoid probability. Several
cleavages can be positive at one fragment node; there is no action/EOS softmax.

```bash
python -m clefts.cli train fragment-tree \
  --train-dir prepared/train_structures \
  --val-dir prepared/validation_structures \
  --output-dir training/run \
  --mol-encoder-checkpoint mol_encoder_pretrained.pt \
  --epochs 20 --device cuda \
  --next-weight 1 --negative-weight 0.2 \
  --minimum-positive-weight 0.05 --branch-threshold 0.5
```

`--next-weight` remains an alias in the existing CLI vocabulary for the branching
loss weight. The Workbench calls this **Branching Action Loss**. The prediction
threshold belongs to training/model configuration (`prediction_threshold`), not
to the Preparation CLI. Existing preparation arguments remain unchanged.

## Teacher semantics

A teacher node is `(sample, precursor candidate row, normalized action state)`.
Source precursors have an empty seed. Non-root precursor preparation is stored
separately and contributes no MS2 branch target. MS2 depth starts at zero at each
seed. Candidates have independent teacher provenance even if their structures
later merge during materialization.

Preparation retains actual peak-assigned pathway transitions, unions ambiguous
pathways, and closes each supported path back to its precursor. It does not
enumerate all combinations or permutations for teacher creation. The chemistry
FragmentTree still explores chemistry under the existing resource limits.

`teacher_positive_action_ptr/index/weight` is sparse CSR. Raw weight is the
maximum observed descendant intensity, including direct observations. Thus an
unobserved intermediate cleavage receives the salience of the observed fragments
it supports. Sample absolute positives are the union of MS2 branch actions;
precursor preparation is only mandatory pool context.

## Loss and compatibility

For valid positive logits, use weighted `softplus(-logit)`, normalized by the sum
of positive weights. Effective weight is `raw_salience + minimum_positive_weight`.
For valid negatives, use mean `softplus(logit)` multiplied by `negative_weight`.
Unobserved valid candidates are weak negatives, including candidates at teacher
leaves. Invalid actions have no loss. Positives are forced into the training pool;
validation generation uses the naturally filtered pool.

Training and inference use `ActionCompatibilityEngine` via the same
`score_states` method: selected actions, conflicts, reactant invalidation,
precedence violations, empty retained fragments, no-ops and capacity violations
are masked. No RDKit runs in training forward. Selected inference states are
materialized through the explicit chemistry boundary before intensity prediction.

`beam_size` is retained as a per-sample bound on states at each generation level.
`max_decode_steps` limits MS2 expansions from the precursor. A branch ends when
no accepted child remains or a limit is reached. Structure deduplication is
separate from action-state provenance.

## Validation and outputs

Every epoch and every enabled intermediate validation calls the real generator
in eval/no-grad mode using only source molecules, adducts and collision energies.
It performs natural action filtering, branching decoding, materialization and
post-model spectrum generation. Teacher-forced loss remains a separate diagnostic
and controls scheduling/checkpoint selection.

`spectrum_validation/epoch_N.json` and `step_N.json` contain every generated
spectrum and original peak set, one-to-one mass-tolerance matched cosine per
sample, histogram, mean, standard deviation and quantiles. Unassigned original
peaks and unmatched predictions remain in the cosine norms. Empty predictions
score zero and contribute to the nonempty-spectrum fraction. These small-run
artifacts can be large for production validation sets.

Metrics include branching precision/recall, positive/valid/predicted actions per
node, recall by MS2 depth, generated/unique fragment counts, natural prefilter
recall, teacher-spectrum cosine, and free-running spectrum cosine. TensorBoard,
JSON/TSV reports, last/best checkpoints and intermediate subset provenance are
written to the training directory.

Schema v5 structures are rejected with a regeneration message. Regenerate from
original MSDataset files. Old autoregressive configuration names are accepted as
configuration aliases; old decoder checkpoints are not compatible with v6 weights.

## Verification

```bash
OMP_NUM_THREADS=2 python -m pytest tests/ml/action/test_branching_v6.py -q
PYTHONPATH=. OMP_NUM_THREADS=2 python evaluation/branching/prepare_smoke.py
```

The real-data smoke recipe selects small molecule-disjoint subsets from the two
files in `data/train_preprocessing/test/preparation_config.json`, retaining source
record indexes. See `evaluation/branching/IMPLEMENTATION_REPORT.md` for commands,
measured results, and limitations.
