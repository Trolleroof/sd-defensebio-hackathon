from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TaskConfig:
    raw: dict
    path: Path

    def __getattr__(self, k):
        try:
            return self.raw[k]
        except KeyError as e:
            raise AttributeError(k) from e

    @property
    def hist_n(self) -> int:
        return int(self.history_s // self.step_s)

    @property
    def hor_n(self) -> int:
        return int(self.horizon_s // self.step_s)

    @property
    def stride_n(self) -> int:
        return int(self.window_stride_s // self.step_s)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def load_task(path: str | Path) -> TaskConfig:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return TaskConfig(yaml.safe_load(p.read_text()), p)


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return yaml.safe_load(p.read_text())


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
