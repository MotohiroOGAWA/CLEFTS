# Fragment-tree training

Training has two objectives. The branch model predicts cleavage paths once per
`(compound, main adduct)` group, without collision energy. The intensity model
then predicts each prepared physical-ion candidate independently for every
spectrum, using main adduct and collision energy as separate inputs.

```bash
python -m clefts.cli train fragment-tree \
  --train-dir prepared/train_structures \
  --val-dir prepared/validation_structures \
  --output-dir training/run \
  --mol-encoder-checkpoint mol_encoder_pretrained.pt \
  --epochs 20 --device cuda \
  --branch-weight 1 --negative-weight 0.2 \
  --branch-mil-temperature 0.1 --intensity-weight 1 \
  --branch-path-threshold 0 --max-fragment-nodes 100
```

`max_action_count` is inherited from the prepared dataset. A zero branch path
threshold disables score pruning. Search always deduplicates normalized states,
keeps the highest cumulative log probability, and applies the fragment-node
budget once to the shared branch group.

## Branch objective

Every transition log probability is `logsigmoid(logit)`. A path score is the
sum of its transition log probabilities. Alternative explanations for one peak
use a normalized smooth maximum:

```text
T * (logsumexp(path_score / T) - log(number_of_paths))
```

Preparation unions positive paths over every collision energy in the branch
group. At each teacher-supported depth, a chemically valid continuation absent
from that union is stored as a weak negative. The branch loss combines positive
MIL with `negative_weight * mean(softplus(negative_logit))`.

## Physical-ion intensity

Preparation groups equivalent final formula/adduct/charge states into one
physical candidate while preserving every `(ion, unsaturation, radical)`
explanation. The model embeds those three fields and aggregates equivalent
explanations with normalized attention. It never sums one intensity per
explanation. Physical candidates use independent softplus intensities, so
several ion states of one fragment can be non-zero simultaneously.

Targets are max-normalized per spectrum. `intensity_power` is used only inside
the full-spectrum and precursor-free losses; inference max-normalizes raw model
outputs without applying that power. Ion-assignment supervision remains an
auxiliary term, and there is no independent hydrogen-shift loss.

## Prepared-tensor boundary

Training consumes source/fragment graphs, action/state transitions, MIL paths,
physical candidates, explanation indices, masses, formulas, and targets already
stored by preparation. Forward, loss, backward, and optimizer steps perform
only tensor and neural-network operations. RDKit parsing, fragmentation,
materialization, ion chemistry, formula calculation, and graph construction are
not permitted in the training hot path.

Prepared data is intentionally versionless. When its contract changes,
regenerate it from the original spectra; no migration or old-format converter
is provided.

## Metrics and outputs

Reports include branch positive/negative loss, recall and state counts by depth,
generated-tree search statistics, physical candidate/explanation counts, full
and precursor-free cosine, and intensity MAE. Validation also generates spectra
with the inference path. TensorBoard, JSON/TSV reports, last/best checkpoints,
and detailed spectrum-validation artifacts are written to the run directory.

Run the focused regression suite with:

```bash
pytest -q tests/ml/action/test_fragment_tree_redesign.py
```
