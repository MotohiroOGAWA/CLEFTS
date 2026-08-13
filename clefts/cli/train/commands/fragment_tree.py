from __future__ import annotations

import argparse

from ...base import CLICommand
from clefts.ml.training.fragment_tree_training.training_model import (
    DEFAULT_EXPERIMENT_NAME,
    build_model_config_from_pretrained,
    build_train_config,
    load_and_validate_split_preprocessing,
    run_training,
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
        parser.add_argument("--mol-encoder-checkpoint", required=True)
        parser.add_argument("--cleavage-edge-fnet-checkpoint", required=True)
        parser.add_argument("--condition-adduct-embedding-dim", type=int, default=16)
        parser.add_argument("--condition-ce-feature-dim", type=int, default=16)
        parser.add_argument("--condition-ce-fc-dims", default="32")
        parser.add_argument("--condition-feature-dim", type=int, default=64)
        parser.add_argument("--condition-fc-dims", default="128,64")
        parser.add_argument("--tree-hidden-dim", type=int, default=128)
        parser.add_argument("--tree-num-layers", type=int, default=2)
        parser.add_argument("--tree-num-heads", type=int, default=8)
        parser.add_argument("--tree-max-degree", type=int, default=16)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--max-edges-per-step", type=int, default=128)
        parser.add_argument("--max-retained-edges", type=int, default=30)
        parser.add_argument("--max-next-cleavage-candidates", type=int, default=3)
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
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--grad-clip-norm", type=float, default=1.0)
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
            "--validate-at-start",
            action="store_true",
            help="Run validation before the first training epoch.",
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
        csv_ints = lambda value: tuple(
            int(part.strip()) for part in value.split(",") if part.strip()
        )
        train_config = build_train_config(
            project_dir=args.project_dir,
            experiment_name=args.experiment_name,
            ckpt_id=args.ckpt_id,
            batch_size=args.batch_size,
            device=args.device,
            epoch=args.epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            validation_interval_steps=(
                None
                if args.validation_interval_steps is None
                or args.validation_interval_steps <= 0
                else args.validation_interval_steps
            ),
            validate_at_start=args.validate_at_start,
            save_interval=args.save_interval_epochs,
            save_interval_steps=(
                None
                if args.save_interval_steps is None or args.save_interval_steps <= 0
                else args.save_interval_steps
            ),
            training_structure_dir=args.training_structure_dir,
            validation_structure_dir=args.validation_structure_dir,
        )
        preprocessing, preprocessing_config_path = load_and_validate_split_preprocessing(
            train_config["training_structure_dir"],
            train_config["validation_structure_dir"],
        )
        model_config = build_model_config_from_pretrained(
            mol_encoder_checkpoint=args.mol_encoder_checkpoint,
            cleavage_edge_fnet_checkpoint=args.cleavage_edge_fnet_checkpoint,
            fragmenter_params=dict(preprocessing["fragmenter_params"]),
            condition_encoder_params={
                "adduct_embedding_dim": args.condition_adduct_embedding_dim,
                "ce_feature_dim": args.condition_ce_feature_dim,
                "ce_fc_dims": csv_ints(args.condition_ce_fc_dims),
                "feature_dim": args.condition_feature_dim,
                "fc_dims": csv_ints(args.condition_fc_dims),
            },
            tree_encoder_params={
                "hidden_dim": args.tree_hidden_dim,
                "num_layers": args.tree_num_layers,
                "num_heads": args.tree_num_heads,
                "max_degree": args.tree_max_degree,
            },
            dropout=args.dropout,
            generator_params={
                "max_edges_per_step": args.max_edges_per_step,
                "max_retained_edges": args.max_retained_edges,
                "max_next_cleavage_candidates": args.max_next_cleavage_candidates,
            },
        )
        run_training(
            project_dir=args.project_dir,
            model_config_inline=model_config,
            train_config_inline=train_config,
            preprocessing_config_path=preprocessing_config_path,
        )
