from __future__ import annotations

import argparse

from ...base import CLICommand
from clefts.ml.data_preparation.fragment_tree.create_training_data import (
    build_arg_parser as build_create_training_data_arg_parser,
    main as run_create_training_data,
)


class CreateFragmentTreeDataCommand(CLICommand):
    name = "create-fragment-tree-data"
    aliases = ("create-tree-data",)
    help = "Create Source-anchored action training structure files (schema v5)."
    description = "Create schema-v5 action training data directly from an original MSDataset."
    order = 20

    def configure(self, parser: argparse.ArgumentParser) -> None:
        source = build_create_training_data_arg_parser()
        parser.usage = "%(prog)s [options]"
        for action in source._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            parser._add_action(action)

    def run(self, args: argparse.Namespace) -> None:
        from clefts.ml.specgen.config_options import namespace_argv
        run_create_training_data(namespace_argv(build_create_training_data_arg_parser(), args))
