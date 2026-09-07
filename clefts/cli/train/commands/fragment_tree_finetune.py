"""CLI for expanding a frozen fragment-tree checkpoint."""
import json
import math

from ...base import CLICommand


class FragmentTreeFineTuneCommand(CLICommand):
    name = 'fragment-tree-finetune'
    help = 'Fine-tune new cleavage patterns with a frozen base and small per-layer expansions.'

    def configure(self, parser):
        parser.add_argument('--checkpoint', required=True, help='Base fragment-tree training model.pt (not a MolEncoder checkpoint).')
        parser.add_argument('--cleavage-pattern-set', required=True, help='Complete new .clevageset.json including all old patterns.')
        parser.add_argument('--train-dir', required=True, help='Structures regenerated using the complete new pattern set.')
        parser.add_argument('--val-dir', required=True, help='Validation structures regenerated with the same settings.')
        parser.add_argument('--output-dir', required=True, help='Fine-tuning output project directory.')
        parser.add_argument('--adapter-width', type=int, default=8, help='Extra low-rank nodes per linear/attention projection (default: 8).')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--weight-decay', type=float, default=0.0)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch-size', type=int, default=1)
        parser.add_argument('--max-samples', type=int, default=100)
        parser.add_argument('--num-workers', type=int, default=0)
        parser.add_argument('--device', default='cpu')
        parser.add_argument('--assignment-score-threshold', type=float, default=0.8)
        parser.add_argument('--experiment-name', default='exp_finetune')
        parser.add_argument('--ckpt-id', default=None, help='Resume a checkpoint within this fine-tuning experiment.')
        parser.add_argument('--dry-run', action='store_true', help='Validate data/config/weights and report frozen/trainable parameters without training or writing files.')

    def run(self, args):
        from pathlib import Path
        import torch
        from clefts.ml.training.fragment_tree_training.training_model import (
            load_and_validate_split_preprocessing, validate_preprocessing_compatibility,
            build_training_model, build_train_config, run_training,
        )
        from clefts.ml.training.fragment_tree_training.fine_tuning import (
            prepare_model_config, initialize_from_base, parameter_report,
        )
        if not all(math.isfinite(value) for value in (args.lr, args.weight_decay, args.assignment_score_threshold)):
            raise SystemExit('Learning rate, weight decay and assignment threshold must be finite.')
        if args.lr <= 0 or args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.weight_decay < 0:
            raise SystemExit('LR, epochs and batch size must be positive; workers and weight decay nonnegative.')
        if args.dry_run and args.ckpt_id:
            raise SystemExit('--dry-run validates initialization from the base; omit --ckpt-id.')
        for folder in (args.train_dir, args.val_dir):
            if not Path(folder).is_dir():
                raise SystemExit(f'Structure directory not found: {folder}')
        preprocessing, config_path = load_and_validate_split_preprocessing(args.train_dir, args.val_dir)
        config = prepare_model_config(args.checkpoint, args.cleavage_pattern_set, preprocessing, args.adapter_width)
        validate_preprocessing_compatibility(project_dir=args.output_dir, model_config=config, preprocessing_config_path=config_path)
        train_path, val_path = Path(args.train_dir).resolve(), Path(args.val_dir).resolve()
        train_data = train_path / 'data' if (train_path / 'data').is_dir() else train_path
        val_data = val_path / 'data' if (val_path / 'data').is_dir() else val_path
        valid_records = next((path for path in (val_data / 'valid_records.msds', val_data.parent / 'valid_records.msds') if path.is_file()), None)
        if valid_records is None:
            raise SystemExit('Validation split requires valid_records.msds; regenerate with validation record output enabled.')
        train_config = build_train_config(
            project_dir=args.output_dir, experiment_name=args.experiment_name, ckpt_id=args.ckpt_id,
            training_structure_dir=str(train_data), validation_structure_dir=str(val_data),
            lr=args.lr, weight_decay=args.weight_decay, epoch=args.epochs, batch_size=args.batch_size,
            max_samples=args.max_samples, device=args.device, assignment_score_threshold=args.assignment_score_threshold,
        )
        train_config['validation_valid_records_file'] = str(valid_records)
        if args.dry_run:
            model = build_training_model(config, torch.device(args.device))
            initialize_from_base(model, config)
            print(json.dumps(parameter_report(model), indent=2))
            return
        run_training(project_dir=args.output_dir, model_config_inline=config, train_config_inline=train_config,
                     preprocessing_config_path=config_path, num_workers=args.num_workers)
