from __future__ import annotations

import argparse
from pathlib import Path

from ...base import CLICommand
from clefts.ml.input.create_fragment_tree_training_data import main as create_main
from clefts.ml.input.create_fragment_tree_training_data import parse_args as create_parse_args


class CreateFragmentTreeDataCommand(CLICommand):
    name = "create-fragment-tree-data"
    aliases = ("create-tree-data",)
    help = "Create FragmentTree training structure files."
    description = "Create FragmentTree training data under a project directory."
    order = 20

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.usage = "%(prog)s PROJECT_DIR MODEL_CONFIG TRAIN_INPUT [options]"
        parser.add_argument(
            "project_dir",
            metavar="PROJECT_DIR",
            help="Training project directory.",
        )
        parser.add_argument(
            "model_config",
            metavar="MODEL_CONFIG",
            help="FragmentSpectrumGenerator config path or name under PROJECT_DIR/config.",
        )
        parser.add_argument(
            "train_input",
            metavar="TRAIN_INPUT",
            help="Training input MSDataset path.",
        )
        parser.add_argument(
            "--validation-input",
            default=None,
            help="Optional validation input MSDataset path.",
        )
        parser.add_argument("--smiles-column", default="SMILES")
        parser.add_argument("--precursor-mz-column", default="PrecursorMZ")
        parser.add_argument("--adduct-type-column", default="AdductType")
        parser.add_argument("--collision-energy-column", default="CollisionEnergy")
        parser.add_argument("--instrument-column", default=None)
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--max-node", type=int, default=-1)
        parser.add_argument("--max-edge", type=int, default=-1)
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument(
            "--overwrite-model-config",
            action="store_true",
            help="Overwrite PROJECT_DIR/config/model_config.json without prompting.",
        )
        parser.add_argument("--save-train-valid-records", action="store_true")
        parser.add_argument(
            "--no-save-validation-valid-records",
            dest="save_validation_valid_records",
            action="store_false",
        )
        parser.set_defaults(save_validation_valid_records=True)
        parser.add_argument("--train-valid-output", default=None)
        parser.add_argument("--validation-valid-output", default=None)
        parser.add_argument("--train-assignment-score-output", default=None)
        parser.add_argument("--validation-assignment-score-output", default=None)
        parser.add_argument("--num-workers", type=int, default=1)
        parser.add_argument("--chunk-size", type=int, default=32)
        parser.add_argument("--parallel-temp-dir", default=None)
        parser.add_argument("--keep-parallel-temp", action="store_true")

    def run(self, args: argparse.Namespace) -> None:
        project_dir = Path(args.project_dir)
        model_config = Path(args.model_config)
        if not model_config.exists() and not model_config.is_absolute():
            candidate = project_dir / "config" / model_config
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".json")
            model_config = candidate

        argv = [
            "--train-input",
            str(args.train_input),
            "--output-dir",
            str(project_dir),
            "--params",
            str(model_config),
            "--model-config-output",
            str(project_dir / "config" / "model_config.json"),
            "--smiles-column",
            str(args.smiles_column),
            "--precursor-mz-column",
            str(args.precursor_mz_column),
            "--adduct-type-column",
            str(args.adduct_type_column),
            "--collision-energy-column",
            str(args.collision_energy_column),
            "--device",
            str(args.device),
            "--max-node",
            str(args.max_node),
            "--max-edge",
            str(args.max_edge),
            "--num-workers",
            str(args.num_workers),
            "--chunk-size",
            str(args.chunk_size),
        ]
        if args.validation_input is not None:
            argv.extend(["--validation-input", str(args.validation_input)])
        if args.instrument_column is not None:
            argv.extend(["--instrument-column", str(args.instrument_column)])
        if args.overwrite:
            argv.append("--overwrite")
        if args.overwrite_model_config:
            argv.append("--overwrite-model-config")
        if args.save_train_valid_records:
            argv.append("--save-train-valid-records")
        if not args.save_validation_valid_records:
            argv.append("--no-save-validation-valid-records")
        if args.train_valid_output is not None:
            argv.extend(["--train-valid-output", str(args.train_valid_output)])
        if args.validation_valid_output is not None:
            argv.extend(["--validation-valid-output", str(args.validation_valid_output)])
        if args.train_assignment_score_output is not None:
            argv.extend(["--train-assignment-score-output", str(args.train_assignment_score_output)])
        if args.validation_assignment_score_output is not None:
            argv.extend(["--validation-assignment-score-output", str(args.validation_assignment_score_output)])
        if args.parallel_temp_dir is not None:
            argv.extend(["--parallel-temp-dir", str(args.parallel_temp_dir)])
        if args.keep_parallel_temp:
            argv.append("--keep-parallel-temp")

        create_main(create_parse_args(argv))
