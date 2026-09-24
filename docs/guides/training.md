# Train and fine-tune

```bash
python -m clefts.cli train fragment-tree \
  --train-dir data/prepared/train_structures \
  --val-dir data/prepared/validation_structures \
  --output-dir models/my-model \
  --mol-encoder-checkpoint models/mol-encoder/checkpoint.pt \
  --epochs 100 --batch-size 4 --lr 0.0001
```

The model learns CE-independent fragment branches with normalized positive MIL and depth-diverse weak negatives, then predicts CE-dependent physical-ion intensities. Their weights are configurable with `--branch-weight`, `--negative-weight`, `--branch-mil-temperature`, and `--intensity-weight`. AdamW uses `--weight-decay`; `--gradient-clip 0` disables gradient clipping.

In **Workbench → Fragment Tree Training**, **Load Configuration** and **Save Configuration** are in the page header; the load control also accepts a dropped `*.pfttrain.json` file. Saved training configurations contain only run settings and the trainable Branch / Fragment Transformer / Ion-State / Intensity parameters. Fragmenter settings, Cleavage Patterns, ion-adduct rules, Symbols, molecular-encoder parameters, and adduct ordering are not user inputs to training. They are inherited from the prepared train/validation datasets and the molecular-encoder or model checkpoint.

After selecting both prepared datasets and the applicable checkpoint, Workbench checks that the train and validation Fragmenter settings, adduct ordering, and Symbols agree, and that the molecular encoder is compatible with those Symbols. The inherited configuration is then shown read-only. Cleavage Pattern Sets and Ion Adduct Rule Sets can be collapsed as a set and one Pattern or Rule at a time; Symbols use a compact horizontal layout.

## Compatible pretrained checkpoint

Use `--initialize-from models/base/last.pt` to fine-tune compatible model weights with a fresh optimizer and epoch counter. Configuration and prepared structures must match the checkpoint architecture. Use `--resume` to continue an interrupted training run with its optimizer state instead.

## Expand a frozen base

Advanced fine-tuning uses `--fine-tune-checkpoint` and `--adapter-width`. Select a complete expanded cleavage set and regenerate both prepared splits with its model configuration before training. The expanded Fragmenter is inherited from those datasets. Existing base parameters remain frozen while the newly added parameters are trained.

## Monitoring

Training writes `last.pt`, `metrics.json` and `metrics.tsv` and emits `epoch_end` JSON Lines. Workbench shows epoch progress, train/validation losses, action metrics and bounded searchable stdout/stderr logs. Training processes continue after their panel closes. Restored job metadata distinguishes a missing process from a known successful exit.

### Intermediate validation

For large training datasets, evaluate a validation subset between epoch boundaries:

```bash
python -m clefts.cli train fragment-tree \
  --train-dir data/prepared/train_structures \
  --val-dir data/prepared/validation_structures \
  --output-dir models/my-model \
  --mol-encoder-checkpoint models/mol-encoder/checkpoint.pt \
  --epochs 100 \
  --validation-interval-steps 1000 \
  --validation-fraction 0.1
```

`--validation-interval-steps` counts optimizer updates across epochs and Resume. Its default is `0`, which disables intermediate checks. `--validation-fraction` defaults to `0.1` and must be greater than 0 and at most 1. It selects individual validation samples, rather than a fraction of structure files or batches. The sample count is rounded up, with a minimum of one sample.

A uniformly sampled subset is fixed using the training seed so successive checks are comparable. Resume retains the subset seed and previous intermediate results. The interval and fraction remain editable when resuming. CUDA evaluation runs without gradients, then restores training mode. `--max-samples` continues to limit each validation batch.

In **Workbench → New Training → Optimization and Loss**, set **Validation interval (optimizer steps, 0 disables)** to `1000` and **Intermediate validation fraction** to `0.1`. These values are included in Copy Command, Start Training, and saved training settings. New-form defaults can also be set through `clefts.workbench.trainingDefaults` using `validationIntervalSteps` and `validationFraction`.

Each check emits an `intermediate_validation_end` JSON event with step, sample count, loss components, and action/spectrum metrics. Workbench Training Jobs shows the latest result and a loss chart against optimizer steps. Results are saved in `intermediate_validation.json` and `intermediate_validation.tsv`; `intermediate_validation_subset.json` records the selected sample indices. TensorBoard step metrics are stored under `tensorboard/iterations` and are included when opening the parent `tensorboard` directory.

Full validation still runs at each epoch end. Learning-rate scheduling, early stopping, and best-checkpoint selection use full-validation results; intermediate checks are for monitoring.
