"""Load the PPG-DaLiA pickles without executing arbitrary code.

The public files are Python pickles. A normal ``pickle.load`` can run any
callable the file names, so we only allow the numpy reconstruction helpers
and plain containers that numeric arrays need.
"""

from __future__ import annotations

import pickle
from pathlib import Path

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("collections", "OrderedDict"),
    ("_codecs", "encode"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        if module.startswith("numpy.core"):
            module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


def load(path: str | Path):
    with open(path, "rb") as f:
        return _RestrictedUnpickler(f, encoding="latin1").load()
