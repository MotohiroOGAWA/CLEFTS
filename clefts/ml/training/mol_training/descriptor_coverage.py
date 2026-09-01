from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class DescriptorBinSpec:
    name: str
    lower: float
    upper: float
    width: float

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"Descriptor bin width must be positive: {self.name}")
        if self.upper <= self.lower:
            raise ValueError(f"Descriptor bin upper bound must be greater than lower bound: {self.name}")

    @property
    def bin_count(self) -> int:
        return max(1, int(math.ceil((self.upper - self.lower) / self.width)))


DEFAULT_DESCRIPTOR_BIN_SPECS: Tuple[DescriptorBinSpec, ...] = (
    DescriptorBinSpec(name="ExactMolWt", lower=0.0, upper=2000.0, width=25.0),
    DescriptorBinSpec(name="HeavyAtomCount", lower=0.0, upper=120.0, width=2.0),
    DescriptorBinSpec(name="TPSA", lower=0.0, upper=400.0, width=10.0),
    DescriptorBinSpec(name="MolLogP", lower=-10.0, upper=15.0, width=0.5),
    DescriptorBinSpec(name="NumHAcceptors", lower=0.0, upper=30.0, width=1.0),
    DescriptorBinSpec(name="NumHDonors", lower=0.0, upper=20.0, width=1.0),
    DescriptorBinSpec(name="NumRotatableBonds", lower=0.0, upper=40.0, width=1.0),
    DescriptorBinSpec(name="RingCount", lower=0.0, upper=20.0, width=1.0),
    DescriptorBinSpec(name="NumAromaticRings", lower=0.0, upper=12.0, width=1.0),
    DescriptorBinSpec(name="NumAliphaticRings", lower=0.0, upper=12.0, width=1.0),
    DescriptorBinSpec(name="FractionCSP3", lower=0.0, upper=1.0, width=0.05),
    DescriptorBinSpec(name="NumHeteroatoms", lower=0.0, upper=50.0, width=1.0),
    DescriptorBinSpec(name="FormalCharge", lower=-6.0, upper=6.0, width=1.0),
    DescriptorBinSpec(name="BertzCT", lower=0.0, upper=4000.0, width=50.0),
)

DEFAULT_DESCRIPTOR_BIN_SPECS_BY_NAME: Dict[str, DescriptorBinSpec] = {
    spec.name: spec for spec in DEFAULT_DESCRIPTOR_BIN_SPECS
}


def descriptor_bin_label(bin_index: int) -> str:
    return f"bin_{int(bin_index):03d}"


def descriptor_bin_bounds(spec: DescriptorBinSpec, bin_index: int) -> Tuple[float, float]:
    start = spec.lower + (spec.width * float(bin_index))
    stop = min(start + spec.width, spec.upper)
    return start, stop


def descriptor_bin_index(value: float, spec: DescriptorBinSpec) -> int | None:
    if math.isnan(value):
        return None
    if value < spec.lower or value > spec.upper:
        return None
    if value == spec.upper:
        return spec.bin_count - 1
    index = int(math.floor(((value - spec.lower) / spec.width) + 1e-12))
    if index < 0:
        return None
    return min(index, spec.bin_count - 1)


def build_descriptor_record_index(
    descriptor_rows: Sequence[Sequence[float]],
    descriptor_names: Sequence[str],
    *,
    min_record_count: int,
    specs_by_name: Mapping[str, DescriptorBinSpec] | None = None,
) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, object]]]:
    specs = dict(DEFAULT_DESCRIPTOR_BIN_SPECS_BY_NAME if specs_by_name is None else specs_by_name)
    missing = [name for name in descriptor_names if name not in specs]
    if missing:
        raise ValueError(f"Missing descriptor bin specs for: {', '.join(missing)}")

    index: Dict[str, Dict[str, List[int]]] = {}
    summary: Dict[str, Dict[str, object]] = {}
    counts_by_name: Dict[str, List[int]] = {}
    records_by_name: Dict[str, List[List[int]]] = {}

    for name in descriptor_names:
        spec = specs[name]
        counts_by_name[name] = [0 for _ in range(spec.bin_count)]
        records_by_name[name] = [[] for _ in range(spec.bin_count)]

    for row_index, row in enumerate(descriptor_rows):
        if len(row) != len(descriptor_names):
            raise ValueError(
                f"Descriptor row length ({len(row)}) does not match descriptor_names length ({len(descriptor_names)})."
            )
        for descriptor_index, name in enumerate(descriptor_names):
            spec = specs[name]
            bin_index = descriptor_bin_index(float(row[descriptor_index]), spec)
            if bin_index is None:
                continue
            counts_by_name[name][bin_index] += 1
            records_by_name[name][bin_index].append(row_index)

    descriptor_name_to_index = {name: index for index, name in enumerate(descriptor_names)}
    for name in descriptor_names:
        spec = specs[name]
        counts = counts_by_name[name]
        records = records_by_name[name]
        descriptor_column_index = descriptor_name_to_index[name]
        finite_values = []
        for row in descriptor_rows:
            value = float(row[descriptor_column_index])
            if math.isfinite(value):
                finite_values.append(value)
        mean_value = sum(finite_values) / len(finite_values) if finite_values else float("nan")
        eligible_bin_indices = {
            bin_index
            for bin_index, count in enumerate(counts)
            if int(count) >= int(min_record_count)
        }
        center_bin_index = descriptor_bin_index(mean_value, spec) if finite_values else None
        if center_bin_index is not None and center_bin_index not in eligible_bin_indices:
            center_bin_index = None
        if center_bin_index is None and eligible_bin_indices:
            target = mean_value if finite_values else 0.0
            center_bin_index = min(
                eligible_bin_indices,
                key=lambda bin_index: abs(sum(descriptor_bin_bounds(spec, bin_index)) / 2.0 - target),
            )

        active_bin_indices = set()
        if center_bin_index is not None:
            left = int(center_bin_index)
            while left - 1 in eligible_bin_indices:
                left -= 1
            right = int(center_bin_index)
            while right + 1 in eligible_bin_indices:
                right += 1
            active_bin_indices = set(range(left, right + 1))

        bins: Dict[str, List[int]] = {}
        bin_summary: List[Dict[str, object]] = []
        for bin_index, (count, row_indices) in enumerate(zip(counts, records)):
            label = descriptor_bin_label(bin_index)
            lower, upper = descriptor_bin_bounds(spec, bin_index)
            eligible = bin_index in eligible_bin_indices
            included = bin_index in active_bin_indices
            bin_summary.append(
                {
                    "label": label,
                    "bin_index": int(bin_index),
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(count),
                    "eligible": bool(eligible),
                    "included": bool(included),
                }
            )
            if included:
                bins[label] = list(row_indices)

        index[name] = bins
        summary[name] = {
            "name": name,
            "lower": float(spec.lower),
            "upper": float(spec.upper),
            "width": float(spec.width),
            "bin_count": int(spec.bin_count),
            "min_record_count": int(min_record_count),
            "center": "mean",
            "center_value": float(mean_value),
            "center_bin_index": int(center_bin_index) if center_bin_index is not None else None,
            "eligible_bin_count": int(len(eligible_bin_indices)),
            "valid_bin_count": int(len(active_bin_indices)),
            "ignored_bin_count": int(sum(1 for count in counts if 0 < count < int(min_record_count))),
            "bin_summary": bin_summary,
        }

    return index, summary
