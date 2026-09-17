# Train and fine-tune

```bash
python -m clefts.cli train fragment-tree \
  --train-dir data/prepared/train_structures \
  --val-dir data/prepared/validation_structures \
  --output-dir models/my-model \
  --params model.json --epochs 100 --batch-size 4 --lr 0.0001
```

The model learns absolute primitive-action filtering, multi-positive next actions and EOS, and materialized fragment formula intensities. Their weights are configurable with `--absolute-weight`, `--next-weight`, `--negative-weight` and `--intensity-weight`. AdamW uses `--weight-decay`; `--gradient-clip 0` disables gradient clipping.

## Compatible pretrained checkpoint

Use `--initialize-from models/base/last.pt` to fine-tune compatible model weights with a fresh optimizer and epoch counter. Configuration and prepared structures must match the checkpoint architecture. Use `--resume` to continue an interrupted training run with its optimizer state instead.

## Expand a frozen base

Advanced fine-tuning uses `--fine-tune-checkpoint`, `--fine-tune-pattern-set` and `--adapter-width`. Select a complete expanded cleavage set and regenerate both prepared splits with its model configuration before training. Existing base parameters remain frozen while the newly added parameters are trained.

## Monitoring

Training writes `last.pt`, `metrics.json` and `metrics.tsv` and emits `epoch_end` JSON Lines. Workbench shows epoch progress, train/validation losses, action metrics and bounded searchable stdout/stderr logs. Training processes continue after their panel closes. Restored job metadata distinguishes a missing process from a known successful exit.
