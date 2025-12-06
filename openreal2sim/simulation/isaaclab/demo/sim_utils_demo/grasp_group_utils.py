"""
Light-weight GraspGroup fallback.

If graspnetAPI is installed we reuse its GraspGroup. Otherwise we provide a very
small compatible subset that can load npy files saved by graspnetAPI and expose
rotation/translation information for the manipulation scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

try:  # pragma: no cover - only executed when dependency exists.
    from graspnetAPI.grasp import GraspGroup as _ExternalGraspGroup  # type: ignore

    GraspGroup = _ExternalGraspGroup
except ImportError:

    @dataclass
    class SimpleGrasp:
        rotation_matrix: np.ndarray  # (3, 3)
        translation: np.ndarray  # (3,)

    class GraspGroup:
        """Minimal GraspGroup replacement supporting npy loading and indexing."""

        def __init__(self, array: np.ndarray | None = None):
            if array is None:
                self._array = np.zeros((0, 17), dtype=np.float32)
            else:
                self._array = np.asarray(array, dtype=np.float32)

        @property
        def grasp_group_array(self) -> np.ndarray:
            return self._array

        def __len__(self) -> int:
            return len(self._array)

        def __getitem__(
            self, index: Union[int, slice, Sequence[int], np.ndarray]
        ) -> Union[SimpleGrasp, "GraspGroup"]:
            if isinstance(index, int):
                row = self._array[index]
                return SimpleGrasp(
                    rotation_matrix=row[4:13].reshape(3, 3),
                    translation=row[13:16],
                )
            if isinstance(index, slice):
                return GraspGroup(self._array[index])
            if isinstance(index, (list, tuple, np.ndarray)):
                return GraspGroup(self._array[index])
            raise TypeError(f"Unsupported index type {type(index)!r} for GraspGroup")

        def from_npy(self, npy_file_path: str) -> "GraspGroup":
            data = np.load(npy_file_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            self._array = data.astype(np.float32, copy=False)
            return self

        # convenience for legacy code expecting classmethod behaviour
        @classmethod
        def load(cls, npy_file_path: str) -> "GraspGroup":
            return cls().from_npy(npy_file_path)

__all__ = ["GraspGroup"]


