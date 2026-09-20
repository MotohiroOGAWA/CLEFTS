"""Precompute RDKit-heavy per-compound fragmentation data for spectrum
prediction (cleavage actions and candidate precursor sequences), in an
isolated process. RDKit does not parallelize well across Python threads, so
real OS subprocess isolation is required for a speedup, not a thread pool.

This module only does the isolated per-chunk work; the dispatch/merge side
(deciding whether to run serially or fan out via
clefts.utils.parallel_subprocess, and consuming the result) lives in
build_prediction_cache() in source_action_msdataset.py, mirroring the
create_training_data.py / subprocess_worker.py split.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from clefts.libs.mmkit.mmkit import Adduct, Compound
from clefts.ml.data_preparation.fragment_tree.context import create_preparation_context


def prepare_compound_cache(smiles: str, adduct_strs, fragmenter) -> dict:
    """Cleavage actions and, per requested adduct, the candidate precursor
    action sequences -- both plain picklable dataclasses (no live RDKit Mol
    references), reused verbatim (no RDKit call) by
    SourceAnchoredFragmentSpectrumGenerator.prepare().

    Every adduct in adduct_strs always gets an entry in precursor_sequences,
    even an empty tuple: either this source has no valid pathway to that
    adduct under the model's ion rules, or resolving it raised. Either way,
    the caller (predict_source_msdataset) treats "no sequences" as a normal
    per-record failure to skip -- not something to silently drop here and
    let prepare_source_actions raise UnresolvedPrecursorError over later,
    deep inside the batched prediction loop where it takes every other
    sample sharing this compound down with it."""
    source = Compound.from_smiles(smiles)
    actions = fragmenter.fragment_ion_tree_builder.create_cleavage_actions(source)
    precursor_tree = None
    precursor_sequences = {}
    for adduct_str in adduct_strs:
        try:
            adduct = Adduct.parse(adduct_str)
            if precursor_tree is None:
                # Bounded by precursor_candidate_max_action_count, built at
                # most once per compound regardless of how many adducts it
                # was requested with.
                precursor_tree = fragmenter.build_fragment_ion_tree(
                    source, max_action_count=fragmenter.precursor_candidate_max_action_count,
                    _include_fragment_compound_cache=True)
            sequences = {pa.action_sequence for pa in fragmenter.resolve_precursor_actions(precursor_tree, adduct)}
            precursor_sequences[adduct_str] = tuple(sorted(sequences, key=lambda seq: (seq is not None, seq.key if seq else ())))
        except Exception:
            precursor_sequences[adduct_str] = ()
    return {"actions": actions, "precursor_sequences": precursor_sequences}


def prepare_chunk(groups: dict[str, list[str]], prep_config: dict) -> dict[str, dict]:
    """groups: {smiles: [adduct_str, ...]}. Returns {smiles: result}, where
    result is {"ok": True, "actions":..., "precursor_sequences":...} or
    {"ok": False, "error": "..."} for a compound RDKit could not fragment."""
    fragmenter = create_preparation_context(prep_config).fragmenter
    results = {}
    for smiles, adduct_strs in groups.items():
        try:
            results[smiles] = {"ok": True, **prepare_compound_cache(smiles, adduct_strs, fragmenter)}
        except Exception as error:
            results[smiles] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    with Path(args.task).open("rb") as stream:
        payload = pickle.load(stream)
    results = prepare_chunk(payload["groups"], payload["prep_config"])
    with Path(args.result).open("wb") as stream:
        pickle.dump(results, stream, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
