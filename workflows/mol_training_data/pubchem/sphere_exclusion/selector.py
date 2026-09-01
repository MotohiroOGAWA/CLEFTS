from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict

from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


class SphereExclusionIndex:
    """Exact leader-style sphere exclusion over selected fingerprints.

    A candidate is excluded if its Tanimoto similarity to any selected
    fingerprint is >= threshold. Fingerprints are bucketed by on-bit count
    so only buckets that can possibly reach the threshold are compared.
    """

    def __init__(self, threshold: float) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError('threshold must be in (0, 1].')
        self.threshold = threshold
        self._buckets: DefaultDict[int, list[ExplicitBitVect]] = defaultdict(list)
        self.selected_count = 0
        self.comparison_count = 0

    def _candidate_bucket_range(self, on_bits: int) -> range:
        min_bits = max(0, math.ceil(self.threshold * on_bits))
        max_bits = math.floor(on_bits / self.threshold) if self.threshold > 0 else on_bits
        return range(min_bits, max_bits + 1)

    def max_similarity(self, fp: ExplicitBitVect) -> float:
        on_bits = int(fp.GetNumOnBits())
        best = 0.0
        for bucket_id in self._candidate_bucket_range(on_bits):
            bucket = self._buckets.get(bucket_id)
            if not bucket:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, bucket)
            self.comparison_count += len(bucket)
            if sims:
                best = max(best, max(sims))
                if best >= self.threshold:
                    return best
        return best

    def is_excluded(self, fp: ExplicitBitVect) -> tuple[bool, float]:
        similarity = self.max_similarity(fp)
        return similarity >= self.threshold, similarity

    def add(self, fp: ExplicitBitVect) -> None:
        self._buckets[int(fp.GetNumOnBits())].append(fp)
        self.selected_count += 1

    def fingerprints(self) -> list[ExplicitBitVect]:
        return [fp for bucket in self._buckets.values() for fp in bucket]

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)
