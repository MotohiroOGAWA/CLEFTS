# Action Prediction and Fragment Tree Training

## Prediction workflow

1. Group input molecules and their measurement conditions by SMILES. Limit the number of samples and molecular graphs processed together to `max_samples`. Conditions for the same molecule share molecular and action features.
2. The Mol Encoder produces molecular and atom features without measurement conditions. Enumerate primitive actions from all Cleavage Patterns on the source molecule, then construct action embeddings from atom positions, SMARTS atom roles, cleavage types, retained regions, and related features. Action embeddings do not include adduct type or collision energy (CE).
3. Evaluate combinations containing at most `precursor_candidate_max_action_count` actions against the chemical rules. Record each precursor combination matching a sample's adduct as a separate alternative state. Record an empty combination when the source molecule itself is a valid precursor. Keep alternative combinations separate. If precursor resolution fails, raise an explicit error rather than silently falling back to the source molecule.
4. Embed each precursor state with a Set Transformer. Combine it with condition embeddings to compute absolute logits for `precursor × valid next action` and `terminate at precursor`. Aggregate next-action scores across precursor alternatives by taking the maximum for each sample.
5. Remove next actions whose scores are at or below the threshold, then apply the candidate limit. Always retain actions required to construct a precursor, regardless of score. The total, including required actions, must not exceed `action-top-k` during prediction or `action-max-k` during training. Raise an error if required precursor actions or training positives cannot fit within the limit.
6. Start decoding from precursor states. Predict the next action or end-of-sequence (EOS) using the Set Transformer state embedding, conditions, and attention over action tokens. Retain candidates with beam search and deduplicate identical action sets. Terminal states have scores; the initial precursor termination score also contributes to the EOS prior.
7. Apply compatibility masks on the GPU to reject conflicting bond edits, precedence cycles, duplicates, empty retained regions, invalid normalization, and action-count violations. Also mask a next action when previously selected actions invalidate its original reactant atoms or bonds. Reject child states that lose their precursor anchor. Normalize redundant edits using the original source atom maps.
8. During prediction, use RDKit to generate molecules only for selected states. Cache identical source/edit effects across sample chunks. Include the connection from the source molecule to a non-root precursor as a seed edge in the tree. For a seed edge containing multiple actions, use the mean embedding of all seed actions.
9. Generate ion, formula, and m/z candidates using the Fragmenter's ion rules and reachability from the precursor. Encode shared fragment molecules once with the Mol Encoder, then distribute their features to each sample's tree. Reuse source-molecule features as well.
10. Pass fragment nodes, fragment edges, and conditions to the Post Model's Graphormer. Combine tree features and ion states nonlinearly to predict ion scores, then aggregate them by formula. Bound the aggregation correction to preserve gradients from the scores, and normalize each sample's maximum intensity to 1.

`predict_batches()` yields outputs incrementally. Use `sample_input_index` to restore the original input order. The MSDataset prediction CLI and Workbench Batch Prediction use this API. Use the single-output `predict()` API for inputs within the sample limit.

## Model and search settings

| Argument | Default | Purpose |
|---|---:|---|
| `--mol-encoder-checkpoint` | Required for new training | Inherit architecture and pretrained weights |
| `--action-hidden-dim` | 128 | Action feature dimension |
| `--action-condition-dim` | 128 | Condition feature dimension |
| `--action-num-heads` | 4 | Number of action/state attention heads |
| `--action-max-roles` | Automatic, at least 64 | Capacity for SMARTS atom roles |
| `--action-state-layers` | 2 | Transformer layers encoding the current action set |
| `--action-state-dropout` | 0 | State Transformer dropout |
| `--action-top-k` | 64 | Total action limit during prediction, including precursor actions |
| `--action-max-k` | 128 | Total action limit during training, including forced positives |
| `--action-threshold` | 1 | Absolute next-action logit threshold; not an intensity unit |
| `--beam-size` | 32 | Beam width per sample |
| `--max-decode-steps` | 16 | Maximum decoding steps |
| `--post-hidden-dim` | 128 | Post Model feature dimension |
| `--post-num-layers` | 2 | Fragment-tree Graphormer layers |
| `--post-num-heads` | 4 | Fragment-tree attention heads |
| `--post-cosine-loss-weight` | 0.5 | Weight of the cosine term in the intensity loss |
| `--max-samples` | 128 | Maximum measurement conditions processed together |
| `--validation-interval-steps` | 0 | Evaluate a subset every N optimizer updates; 0 disables intermediate checks |
| `--validation-fraction` | 0.1 | Fraction of validation samples used for intermediate checks; must be in (0, 1] |

Cleavage patterns, adducts, maximum cleavage counts, the precursor cleavage limit, and mass tolerance are inherited from the dataset's Fragmenter. The training CLI does not expose configuration JSON arguments or generic override arguments.

## Training

Neural forward/backward passes, compatibility masks, and validation run on CUDA. `--device` defaults to `cuda`; CPU training is rejected. Initial file loading, CPU tensor collation, and logging still run on the host, but the training loop does not perform molecular cleavage or RDKit reconstruction. Molecular SMILES and shared-graph mappings are prepared at startup.

Training uses teacher states, transitions, fragment graphs, and intensities stored in `.preft.pt` files. Newly generated structures connect non-root precursors with whole seed edges, matching prediction. Missing precursor masks in older schema-v5 structures can be reconstructed from stored tensor relations. Teacher positives that violate the new reactant-order constraints cause an explicit failure. Regenerate those structures from the original data.

The Mol Encoder is frozen by default, including ordinary training. Specify `--train-mol-encoder` to update it. Pattern-expansion fine-tuning uses a frozen base with added adapters.

Map observed peak intensities to stored states, then propagate salience to actions that reach those states. Use maximum aggregation to avoid counting an ambiguously assigned peak multiple times.

- **Absolute classification loss:** Learn positive and negative precursor/next-action and precursor/EOS scores relative to the configured threshold.
- **Absolute intensity ranking loss:** Apply cross entropy against an intensity-derived candidate distribution to prioritize candidates associated with stronger peaks.
- **Next-action/EOS loss:** Use teacher forcing on stored states and weight valid positive transitions by intensity. Preserve supervision for low-intensity positives. Prefix states preceding a non-root precursor are excluded from MS2 decoder supervision.
- **Post intensity loss:** Combine `log1p` MSE on within-sample relative intensities with spectral cosine loss. This discourages hiding incorrect spectral shapes by uniformly reducing intensities.

```text
L = absolute_weight × (L_absolute + absolute_intensity_weight × L_rank)
  + next_weight × L_next
  + intensity_weight × (L_log_intensity + post_cosine_loss_weight × L_cosine)
```

Discrete top-k selection, beam search, and RDKit operations are not differentiated directly. Candidate scores, positive transitions, and stored-graph intensities receive separate supervision, with shared gradients reaching action and condition features. Measure retention before forced-positive insertion to avoid overstating filter performance.

## Training stability and outputs

Training uses AdamW, gradient clipping of 1 by default, and a warmup of 100 optimizer steps. When validation loss plateaus, halve the learning rate with patience 3 and a minimum of 1e-6. Early stopping defaults to patience 10; set it to 0 to disable it. The default seed is 42. Non-finite losses or gradients raise errors.

The output directory contains:

- At startup: `training_args.json`, `training_config.json` including inherited configuration, and `dataset_summary.json`
- Each epoch: `metrics.json`, `metrics.tsv`, `last.pt`, and `best.pt` when validation improves
- TensorBoard: `tensorboard/events.out.tfevents.*`
- At completion: `training_report.json` and `training_report.md`
- On failure inside the training loop: `training_failure.json`

TensorBoard records loss components, action/intensity retention before forced insertion, next-action/EOS accuracy, spectral cosine similarity, MAE, learning rate, gradient norm, epoch duration, and peak CUDA memory.

`actions_after_filter` counts candidates retained before forced insertion. `training_pool_actions` counts the training pool after retaining precursor actions and positives. MAE compares intensities after normalizing each sample's maximum target intensity to 1.

```bash
tensorboard --logdir /path/to/output/tensorboard
```

Resume restores the model, optimizer, scheduler, step, history, and random states, and inherits the model configuration and loss definitions. Standard validation evaluates teacher forcing on stored graphs; it is distinct from free-running prediction accuracy on unseen compounds.

To inspect progress within large epochs, set `--validation-interval-steps 1000 --validation-fraction 0.1`. Intermediate checks use a fixed, uniformly sampled 10% of validation records, rounded up to at least one record. They run on CUDA without gradients and preserve training mode. The interval counts optimizer updates across epochs and Resume. Full epoch validation remains responsible for scheduling, early stopping, and best-checkpoint selection. See [Intermediate validation](training.md#intermediate-validation) for Workbench controls, output files, and TensorBoard details.

## Implementation verification and limitations

A 120-epoch capacity check ran on an RTX A6000 using two synthetic spectra of CCCO: one with the ordinary precursor and one with a dehydrated precursor. The same records were used for training and validation. The frozen Mol Encoder had randomly initialized test weights; this was not an accuracy evaluation of an encoder pretrained on real data.

| Metric | Result |
|---|---:|
| Train loss | 4.584904 → 1.087841 |
| Final validation loss | 1.089695 |
| Intensity-weighted filter retention | 1.000000 |
| Mean cosine similarity on stored graphs | 0.850601 |
| Free-running cosine similarity, ordinary precursor | 0.984251 |
| Free-running cosine similarity, dehydrated precursor | 0.716094 |
| Mean free-running cosine similarity | 0.850172 |

Free-running comparisons include both extra and missing peaks. These results confirm learning capacity on a small dataset; they do not establish generalization to unseen compounds.

Reproduce the check with:

```bash
python clefts_workbench/scripts/check-action-training.py \
  --output-dir /tmp/new-action-verification
```

The script saves a detailed report, learning curves in PNG/SVG format, TensorBoard events, checkpoints, and free-running prediction metrics.

Regression tests cover prediction agreement across sample chunking, restoration of input order, multi-action precursor retention and seed edges, capacity violations, GPU masking of changed reactant bonds, CUDA forward/backward without RDKit, finite gradients, TensorBoard output, Resume, and reconstruction of masks from legacy tensor metadata.

All 48 action tests and 42 chemical-rule/Fragment Tree Builder tests passed. Workbench settings, the training UI, Resume, Batch Prediction, and explicit CLI arguments also passed their checks. Boundary checks confirm that scores equal to the threshold are excluded while required precursor actions remain retained.

## Design references

- [Set Transformer (Lee et al., ICML 2019)](https://proceedings.mlr.press/v97/lee19d.html): Permutation-invariant attention representations of action sets. The implementation reuses the existing Set Transformer and exposes state-layer count and dropout settings.
- [Attention, Learn to Solve Routing Problems! (Kool et al., ICLR 2019)](https://arxiv.org/abs/1803.08475): Candidate embeddings, state- and condition-dependent selection, and dynamic feasibility masks informed the design. This implementation uses supervised targets and intensities rather than the paper's REINFORCE method.

Neither paper directly establishes molecular-cleavage performance. Their application here is a design choice.
