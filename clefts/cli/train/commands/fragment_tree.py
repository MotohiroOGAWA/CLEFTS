from __future__ import annotations

import argparse

from ...base import CLICommand
from clefts.ml.training.fragment_tree_training.training import (
    build_arg_parser as build_training_arg_parser,
    main as run_training,
)


class FragmentTreeTrainCommand(CLICommand):
    name = "fragment-tree"
    aliases = ("tree",)
    help = "Train (or fine-tune) the Source-anchored action fragment-tree model."
    description = (
        "Train ActionFragmentTreeTrainingModel from schema-v5 .preft.pt structures. "
        "Pass --fine-tune-checkpoint and --fine-tune-pattern-set together to expand "
        "a frozen base checkpoint with a new cleavage pattern set instead."
    )
    order = 30

    def configure(self, parser: argparse.ArgumentParser) -> None:
        source = build_training_arg_parser()
        parser.usage = "%(prog)s [options]"
        for action in source._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            parser._add_action(action)

    def run(self, args: argparse.Namespace) -> None:
        from clefts.ml.specgen.config_options import namespace_argv
        run_training(namespace_argv(build_training_arg_parser(), args))
