from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
import json

from .CleavageStep import CleavageStep


@dataclass(frozen=True)
class FragmentPathwayEdge:
    steps: Tuple[CleavageStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple.")

        if not all(isinstance(step, CleavageStep) for step in self.steps):
            raise TypeError("steps must contain only CleavageStep instances.")

    def __repr__(self) -> str:
        return (
            "FragmentPathwayEdge("
            f"steps={self.steps}"
            ")"
        )

    def __str__(self) -> str:
        return self.to_json_str()

    def to_list(self) -> list[Any]:
        return [
            step.to_list()
            for step in self.steps
        ]

    @classmethod
    def from_list(cls, data: list[Any]) -> FragmentPathwayEdge:
        if not isinstance(data, list):
            raise TypeError("FragmentPathwayEdge data must be a list.")

        return cls(
            steps=tuple(
                CleavageStep.from_list(step_data)
                for step_data in data
            )
        )

    def to_json_str(self) -> str:
        return json.dumps(
            self.to_list(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def parse(cls, text: str) -> FragmentPathwayEdge:
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                f"FragmentPathwayEdge JSON must be a list: {text}"
            )

        return cls.from_list(data)

    @classmethod
    def from_json_str(cls, text: str) -> FragmentPathwayEdge:
        return cls.parse(text)