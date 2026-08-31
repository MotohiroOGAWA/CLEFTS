from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


def balanced_sample_chunks(sample_count: int, max_samples: int) -> List[Tuple[int, int]]:
    """Split one compound into near-equal contiguous chunks."""
    if sample_count < 0 or max_samples < 1:
        raise ValueError("sample_count must be non-negative and max_samples positive.")
    if sample_count == 0:
        return []
    chunk_count = math.ceil(sample_count / max_samples)
    base, remainder = divmod(sample_count, chunk_count)
    sizes = [base + (index < remainder) for index in range(chunk_count)]
    result: List[Tuple[int, int]] = []
    start = 0
    for size in sizes:
        stop = start + int(size)
        result.append((start, stop))
        start = stop
    return result


def pack_compound_chunks(
    sample_counts: Sequence[int], max_samples: int
) -> List[List[Tuple[int, int, int]]]:
    """Pack compounds efficiently; split oversized compounds evenly."""
    pieces = [
        (compound, start, stop)
        for compound, count in enumerate(sample_counts)
        for start, stop in balanced_sample_chunks(int(count), max_samples)
    ]
    pieces.sort(key=lambda item: item[2] - item[1], reverse=True)
    batches: List[List[Tuple[int, int, int]]] = []
    loads: List[int] = []
    for piece in pieces:
        size = piece[2] - piece[1]
        destination = next(
            (i for i, load in enumerate(loads) if load + size <= max_samples), None
        )
        if destination is None:
            batches.append([piece])
            loads.append(size)
        else:
            batches[destination].append(piece)
            loads[destination] += size
    return batches


def run_shape_preflight(
    *, max_samples: int, max_edges_per_step: int, feature_dim: int,
    device: torch.device, output_file: str | Path,
) -> Dict[str, float | int | str]:
    """Measure the configured worst edge×sample tensor before real work."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        edge = torch.empty(
            (max_samples, max_edges_per_step, feature_dim), device=device
        ).normal_()
        condition = torch.empty((max_samples, 1, feature_dim), device=device).normal_()
        result = torch.nn.functional.gelu(edge + condition)
        _ = float(result.square().mean().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    report: Dict[str, float | int | str] = {
        "device": str(device), "max_samples": int(max_samples),
        "max_edges_per_step": int(max_edges_per_step), "feature_dim": int(feature_dim),
        "edge_sample_pairs": int(max_samples * max_edges_per_step),
        "elapsed_seconds": float(time.perf_counter() - started),
        "cuda_peak_memory_bytes": peak,
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
