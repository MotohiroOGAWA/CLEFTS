from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _FragmentIonStateStore:
    """Array-backed storage for node-level ion state candidates.

    ion_states:
        2D int array with shape (num_states, 2).

        Column 0:
            unsaturation count.

        Column 1:
            radical flag as 0 or 1.

    node_state_indptr:
        Pointer array from node index to ion state range.
        Shape is (num_nodes + 1,).

    Notes
    -----
    Hydrogen delta can be calculated as:

        delta_h = -2 * unsaturation - radical
    """

    ion_states: np.ndarray
    node_state_indptr: np.ndarray

    def __post_init__(self) -> None:
        ion_states = np.asarray(self.ion_states, dtype=np.int16)
        node_state_indptr = np.asarray(self.node_state_indptr, dtype=np.int64)

        if ion_states.ndim != 2:
            raise ValueError("ion_states must be a 2D array.")

        if ion_states.shape[1] != 2:
            raise ValueError("ion_states must have shape (num_states, 2).")

        if node_state_indptr.ndim != 1:
            raise ValueError("node_state_indptr must be a 1D array.")

        if len(node_state_indptr) == 0:
            raise ValueError("node_state_indptr must not be empty.")

        if node_state_indptr[0] != 0:
            raise ValueError("node_state_indptr[0] must be 0.")

        if np.any(node_state_indptr[1:] < node_state_indptr[:-1]):
            raise ValueError("node_state_indptr must be non-decreasing.")

        if node_state_indptr[-1] != len(ion_states):
            raise ValueError(
                "node_state_indptr[-1] must equal number of ion states."
            )

        unsaturations = ion_states[:, 0]
        radicals = ion_states[:, 1]

        if np.any(unsaturations < 0):
            raise ValueError("unsaturation values must be non-negative.")

        if not np.all((radicals == 0) | (radicals == 1)):
            raise ValueError("radical values must be 0 or 1.")

        object.__setattr__(self, "ion_states", ion_states)
        object.__setattr__(self, "node_state_indptr", node_state_indptr)

    @staticmethod
    def empty(num_nodes: int = 0) -> "_FragmentIonStateStore":
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")

        return _FragmentIonStateStore(
            ion_states=np.empty((0, 2), dtype=np.int16),
            node_state_indptr=np.zeros(num_nodes + 1, dtype=np.int64),
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_state_indptr) - 1

    @property
    def num_states(self) -> int:
        return len(self.ion_states)

    @property
    def unsaturations(self) -> np.ndarray:
        return self.ion_states[:, 0]

    @property
    def radicals(self) -> np.ndarray:
        return self.ion_states[:, 1].astype(bool)

    @property
    def delta_h(self) -> np.ndarray:
        """Return hydrogen deltas for all ion states."""
        return -2 * self.unsaturations - self.ion_states[:, 1]

    def get_state_range(self, node_index: int) -> slice:
        if not 0 <= node_index < self.num_nodes:
            raise IndexError(f"Invalid node index: {node_index}")

        start = int(self.node_state_indptr[node_index])
        end = int(self.node_state_indptr[node_index + 1])
        return slice(start, end)

    def get_node_states(self, node_index: int) -> np.ndarray:
        return self.ion_states[self.get_state_range(node_index)]

    def get_node_delta_h(self, node_index: int) -> np.ndarray:
        return self.delta_h[self.get_state_range(node_index)]

    def copy(self) -> "_FragmentIonStateStore":
        return _FragmentIonStateStore(
            ion_states=self.ion_states.copy(),
            node_state_indptr=self.node_state_indptr.copy(),
        )