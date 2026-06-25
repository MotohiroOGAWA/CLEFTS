from __future__ import annotations

from ..base import CLIGroup


class TrainGroup(CLIGroup):
    name = "train"
    help = "Train CLEFTS models."
    description = "Train CLEFTS machine-learning models."
    order = 30

    @property
    def commands_package(self) -> str:
        return "clefts.cli.train.commands"
