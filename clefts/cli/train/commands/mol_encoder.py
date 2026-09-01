from __future__ import annotations

import argparse

from ...base import CLICommand
from clefts.ml.training.mol_training.training_model import (
    build_arg_parser as build_mol_pretraining_arg_parser,
    main as run_mol_pretraining,
)


class MolEncoderPretrainingCommand(CLICommand):
    name = "mol-encoder"
    aliases = ("mol", "mol-pretrain")
    help = "Pretrain MolEncoder with node, edge, and graph-level tasks."
    description = "Pretrain MolEncoder from training and validation MSDS files."
    order = 30

    def configure(self, parser: argparse.ArgumentParser) -> None:
        source = build_mol_pretraining_arg_parser()
        parser.usage = "%(prog)s [options]"
        for action in source._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            parser._add_action(action)

    def run(self, args: argparse.Namespace) -> None:
        argv = []
        for action in build_mol_pretraining_arg_parser()._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            value = getattr(args, action.dest, None)
            if isinstance(action, argparse._StoreTrueAction):
                if value:
                    argv.append(action.option_strings[0])
                continue
            if value is None:
                continue
            option = action.option_strings[0] if action.option_strings else None
            if option is None:
                continue
            if isinstance(value, (list, tuple)):
                if not value:
                    continue
                argv.append(option)
                argv.extend(str(item) for item in value)
            else:
                argv.extend([option, str(value)])
        run_mol_pretraining(argv)
