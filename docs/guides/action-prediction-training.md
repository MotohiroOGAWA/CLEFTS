# Fragment-tree branch and intensity training

## Prediction workflow

1. Group spectra by source molecule and normalized main adduct. Collision
   energies in a group share one branch search.
2. Encode the source graph and all primitive cleavage actions once.
3. Starting at Source, score valid next actions with the current action-set
   representation and main-adduct embedding.
4. Accumulate branch probability in log space, deduplicate normalized states,
   and stop at the path threshold, action limit, node budget, or a state with no
   valid continuation.
5. Materialize the selected fragment tree outside neural training.
6. Generate physical-ion candidates with current CLEFTS ion rules. Equivalent
   final ions share one candidate while retaining all `(ion, unsaturation,
   radical)` explanations.
7. Encode the fragment tree and aggregate equivalent explanations with
   normalized attention. Predict independent physical-ion intensities using
   separate main-adduct and collision-energy features.
8. Max-normalize raw intensities for the final spectrum.

## Main settings

| Argument | Default | Purpose |
|---|---:|---|
| `--action-main-adduct-dim` | 128 | Width of the Branch Scorer's main-adduct input |
| `--main-adduct-embedding-dim` | 128 | Width of the dedicated main-adduct conditioning path |
| `--collision-energy-feature-dim` | 128 | Width of the intensity-only collision-energy features |
| `--branch-path-threshold` | 0 | Minimum cumulative path probability; 0 disables pruning |
| `--max-fragment-nodes` | 100 | Shared node budget per branch group |
| `--branch-weight` | 1 | Top-level branch-loss weight |
| `--negative-weight` | 0.2 | Weak-negative term inside branch loss |
| `--branch-mil-temperature` | 0.1 | Smooth-max temperature for alternative positive paths |
| `--intensity-weight` | 1 | Top-level physical-ion intensity-loss weight |
| `--post-intensity-power` | 0.5 | Target/prediction power used only during loss calculation |
| `--post-precursor-free-weight` | 0.5 | Precursor-free spectrum-loss weight |
| `--post-ion-threshold` | 0.5 | Inference-only physical-ion confidence threshold |
| `--post-peak-intensity-threshold` | 0 | Inference-only normalized-intensity threshold |

`max_action_count` and chemistry configuration are inherited from prepared data.
The Workbench parameter inspector labels architecture, training-only,
inference-only, dataset-inherited, and search parameters separately.

## Supervision

Preparation unions teacher pathways across every collision energy in a branch
group. For one peak, alternative paths use normalized smooth-max MIL. At every
teacher-supported depth, chemically valid actions absent from all supported
continuations become weak negatives. Thus a fragment missing at low collision
energy is not marked negative when another spectrum in the group supports it.

The intensity objective uses one output for both full-spectrum and
precursor-free losses, plus physical-ion assignment supervision. Several ion
states can have non-zero intensity simultaneously. There is no hydrogen-shift
classification loss and no sum over equivalent explanation intensities.

## Prepared-data boundary

Preparation stores source and fragment graphs, primitive actions, transition
states, valid/positive/weak-negative actions, peak-to-path MIL indices, tree
connectivity, formulas, masses, physical candidates, explanation indices, and
targets. Training performs embedding lookup, gather/scatter, attention,
transformers, loss, and backpropagation only. Prepared files are regenerated
when this contract changes; no format migration layer is provided.

Regression coverage includes CE-shared branch trees, cumulative probability,
multi-depth negatives, cross-CE protection, equivalent-candidate grouping,
candidate-count invariance, simultaneous ion peaks, CE-dependent intensities,
and a strict forward/loss/backward test with chemistry entry points disabled.
