from __future__ import annotations

from ..base import CLIGroup


class CleavageGroup(CLIGroup):
    name = "cleavage"
    help = "Run cleavage-pattern based fragmentation."
    description = (
        "Build fragment trees from molecules using user-defined "
        "cleavage patterns."
    )
    order = 10

    @property
    def commands_package(self) -> str:
        return "clefts.cli.cleavage.commands"