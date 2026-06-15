from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, List

from tqdm import tqdm


def generate_task_arguments(
    input_files: List[str],
    output_files: List[str],
    input_flag: str = "-i",
    output_flag: str = "-o",
) -> Generator[List[str], None, None]:
    """Generate arguments for subprocess tasks.

    Parameters
    ----------
    input_files:
        Input file paths.

    output_files:
        Output file paths.

    input_flag:
        Command-line flag for input.

    output_flag:
        Command-line flag for output.

    Yields
    ------
    List[str]
        Argument list for one subprocess.
    """

    for in_file, out_file in zip(input_files, output_files):
        yield [input_flag, in_file, output_flag, out_file]


def to_module_name(path: str) -> str:
    """Convert file path to module name if needed."""

    if path.endswith(".py"):
        path = os.path.splitext(path)[0]
        path = path.replace(os.sep, ".")

    return path


def run_in_subprocess(
    commands: List[str],
    print_output: bool = False,
) -> None:
    """Run a single task in a subprocess.

    Parameters
    ----------
    commands:
        Full command list to execute.

    print_output:
        If True, show subprocess stdout/stderr.
        If False, suppress stdout/stderr.
    """

    if print_output:
        subprocess.run(
            commands,
            check=True,
        )
    else:
        subprocess.run(
            commands,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def run_parallel_subprocesses(
    commands_list: List[List[str]],
    max_workers: int = 4,
    print_output: bool = False,
) -> None:
    """Run multiple subprocesses in parallel.

    Parameters
    ----------
    commands_list:
        List of full subprocess commands.

    max_workers:
        Number of parallel workers.

    print_output:
        If True, show subprocess stdout/stderr.
        If False, suppress stdout/stderr.
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_in_subprocess,
                commands,
                print_output,
            )
            for commands in commands_list
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Parallel tasks",
        ):
            future.result()