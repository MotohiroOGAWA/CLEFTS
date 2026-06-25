from __future__ import annotations

import argparse

from ...base import CLICommand
from clefts.ml.training.fragment_tree_training.training_model import (
    DEFAULT_EXPERIMENT_NAME,
    build_train_config,
    run_training_from_config,
)


class FragmentTreeTrainCommand(CLICommand):
    name = "fragment-tree"
    aliases = ("tree",)
    help = "Train FragmentTreeTrainingModel."
    description = "Train FragmentTreeTrainingModel from command-line options."
    order = 10

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.usage = "%(prog)s PROJECT_DIR [options]"
        parser.add_argument(
            "project_dir",
            metavar="PROJECT_DIR",
            help="Training project directory, e.g. data/training/project_single_bond_pos.",
        )
        parser.add_argument(
            "-m",
            "--model-config",
            default=None,
            help="Model config path or name under PROJECT_DIR/config. Defaults to PROJECT_DIR/config/model_config.json.",
        )
        parser.add_argument(
            "--experiment-name",
            default=DEFAULT_EXPERIMENT_NAME,
            help="Experiment name under PROJECT_DIR/experiments. Default: exp_main.",
        )
        parser.add_argument(
            "--ckpt-id",
            default=None,
            help="Checkpoint id to resume from. Default: null/new run.",
        )
        parser.add_argument("--batch-size", type=int, default=1, help="default: %(default)s")
        parser.add_argument("--device", default="cpu", help="default: %(default)s")
        parser.add_argument(
            "--epoch",
            "--epochs",
            dest="epoch",
            type=int,
            default=10,
            help="default: %(default)s",
        )
        parser.add_argument(
            "--validation-interval-steps",
            type=int,
            default=100,
            help="Run validation every N successful training steps. Use 0 to disable.",
        )
        parser.add_argument(
            "--save-interval-epochs",
            "--save-interval",
            dest="save_interval_epochs",
            type=int,
            default=1,
            help="Save a checkpoint every N epochs. Use 0 to disable.",
        )
        parser.add_argument(
            "--save-interval-steps",
            "--save-interval-iters",
            dest="save_interval_steps",
            type=int,
            default=100,
            help="Save a checkpoint every N successful training steps. Use 0 to disable.",
        )
        parser.add_argument(
            "--training-structure-dir",
            default=None,
            help="Training structure directory. Defaults to PROJECT_DIR/train_structures/data.",
        )
        parser.add_argument(
            "--validation-structure-dir",
            default=None,
            help="Validation structure directory. Defaults to PROJECT_DIR/validation_structures/data.",
        )

    def run(self, args: argparse.Namespace) -> None:
        train_config = build_train_config(
            project_dir=args.project_dir,
            experiment_name=args.experiment_name,
            ckpt_id=args.ckpt_id,
            batch_size=args.batch_size,
            device=args.device,
            epoch=args.epoch,
            validation_interval_steps=(
                None
                if args.validation_interval_steps is None
                or args.validation_interval_steps <= 0
                else args.validation_interval_steps
            ),
            save_interval=args.save_interval_epochs,
            save_interval_steps=(
                None
                if args.save_interval_steps is None or args.save_interval_steps <= 0
                else args.save_interval_steps
            ),
            training_structure_dir=args.training_structure_dir,
            validation_structure_dir=args.validation_structure_dir,
        )
        run_training_from_config(
            project_dir=args.project_dir,
            model_config_path=args.model_config,
            train_config=train_config,
        )
