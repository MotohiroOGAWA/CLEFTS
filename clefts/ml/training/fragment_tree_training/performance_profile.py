from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import torch


def _structure_shape(batch: Dict[str, Any]) -> Dict[str, int]:
    structure = batch["structure"]
    valid_sample_edges = structure.sample_edge_index[1] >= 0
    if valid_sample_edges.any():
        sample_counts = torch.bincount(
            structure.sample_edge_index[0, valid_sample_edges].long(),
            minlength=int(structure.num_samples),
        )
        max_sample_edges = int(sample_counts.max().item())
    else:
        max_sample_edges = 0
    return {
        "samples": int(structure.num_samples),
        "nodes": int(structure.num_nodes),
        "edges": int(structure.edge_index.size(1)),
        "sample_edge_pairs": int(
            valid_sample_edges.sum().item()
            if structure.sample_edge_index.numel()
            else 0
        ),
        "max_sample_edges": max_sample_edges,
        "estimated_max_sample_tree_nodes": max_sample_edges + 1,
    }


def _module_report(model: torch.nn.Module) -> list[Dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        parameters = list(module.parameters(recurse=False))
        buffers = list(module.buffers(recurse=False))
        rows.append(
            {
                "name": name or "<root>",
                "type": type(module).__name__,
                "parameter_count": int(sum(value.numel() for value in parameters)),
                "trainable_parameter_count": int(
                    sum(value.numel() for value in parameters if value.requires_grad)
                ),
                "parameter_bytes": int(
                    sum(value.numel() * value.element_size() for value in parameters)
                ),
                "buffer_bytes": int(
                    sum(value.numel() * value.element_size() for value in buffers)
                ),
            }
        )
    return rows


def _event_value(event: Any, *names: str) -> float:
    for name in names:
        if hasattr(event, name):
            return float(getattr(event, name))
    return 0.0


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}.")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def run_training_performance_profile(
    *,
    model: torch.nn.Module,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    output_dir: str | Path,
    row_limit: int = 200,
) -> Dict[str, Any]:
    """Profile one real forward/backward batch before optimizer updates."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    modules = _module_report(model)
    (output_path / "modules.json").write_text(
        json.dumps(modules, indent=2), encoding="utf-8"
    )

    batch = None
    batch_count = 0
    largest_score = -1
    for candidate in loader:
        candidate_shape = _structure_shape(candidate)
        # The tree Graphormer is quadratic in the number of sample-tree nodes.
        estimated_tree_nodes = candidate_shape["estimated_max_sample_tree_nodes"]
        score = candidate_shape["samples"] * estimated_tree_nodes * estimated_tree_nodes
        batch_count += 1
        if score > largest_score:
            batch = candidate
            largest_score = score
    if batch is None:
        raise ValueError("Cannot profile an empty training loader.")
    shape = _structure_shape(batch)
    structure = batch["structure"].to(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda_device = None
    if device.type == "cuda":
        cuda_device = _cuda_device_index(device)
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_device)
        free_before, total_memory = torch.cuda.mem_get_info(cuda_device)
    else:
        free_before, total_memory = 0, 0

    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.train(True)
    started = time.perf_counter()
    report: Dict[str, Any] = {
        "status": "running",
        "device": str(device),
        "batch_shape": shape,
        "batches_scanned": batch_count,
        "largest_estimated_tree_nodes": int(
            shape["estimated_max_sample_tree_nodes"]
        ),
        "graphormer_quadratic_score": int(largest_score),
        "cuda_total_bytes": int(total_memory),
        "cuda_free_before_bytes": int(free_before),
        "module_parameter_bytes": int(sum(row["parameter_bytes"] for row in modules)),
        "module_buffer_bytes": int(sum(row["buffer_bytes"] for row in modules)),
    }
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as profiler:
            with torch.profiler.record_function("fragment_tree_train_step"):
                output = model(structure)
                output["loss"].backward()
        if device.type == "cuda":
            torch.cuda.synchronize(cuda_device)
        profiler.export_chrome_trace(str(output_path / "trace.json"))
        events = profiler.key_averages(group_by_input_shape=True)
        table = events.table(
            sort_by="self_cuda_memory_usage" if device.type == "cuda" else "self_cpu_memory_usage",
            row_limit=row_limit,
        )
        (output_path / "operator_table.txt").write_text(table, encoding="utf-8")
        operator_rows = []
        for event in events:
            operator_rows.append(
                {
                    "name": event.key,
                    "calls": int(event.count),
                    "input_shapes": str(event.input_shapes),
                    "self_cpu_time_us": _event_value(event, "self_cpu_time_total"),
                    "cpu_time_us": _event_value(event, "cpu_time_total"),
                    "self_cuda_time_us": _event_value(
                        event, "self_cuda_time_total", "self_device_time_total"
                    ),
                    "cuda_time_us": _event_value(
                        event, "cuda_time_total", "device_time_total"
                    ),
                    "self_cpu_memory_bytes": int(
                        _event_value(event, "self_cpu_memory_usage")
                    ),
                    "self_cuda_memory_bytes": int(
                        _event_value(
                            event, "self_cuda_memory_usage", "self_device_memory_usage"
                        )
                    ),
                }
            )
        operator_rows.sort(
            key=lambda row: (row["self_cuda_memory_bytes"], row["cuda_time_us"]),
            reverse=True,
        )
        (output_path / "operators.json").write_text(
            json.dumps(operator_rows[:row_limit], indent=2), encoding="utf-8"
        )
        report["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        report["status"] = "cuda_oom"
        report["error"] = str(exc)
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        if device.type == "cuda":
            report["cuda_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(cuda_device)
            )
            report["cuda_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(cuda_device)
            )
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        (output_path / "summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    if report["status"] != "ok":
        gib = 1024 ** 3
        detail = str(report.get("error", "unknown profiling error"))
        raise RuntimeError(
            "Training performance preflight failed before optimizer updates. "
            f"status={report['status']}; batch_shape={shape}; "
            f"peak_allocated={report.get('cuda_peak_allocated_bytes', 0) / gib:.2f} GiB; "
            f"peak_reserved={report.get('cuda_peak_reserved_bytes', 0) / gib:.2f} GiB; "
            f"detail={detail}; report={output_path / 'summary.json'}"
        )
    return report
