"""Drowsiness detection package."""

from .config import DetectorConfig


def run(config: DetectorConfig) -> None:
    from .app import run as _run

    _run(config)


__all__ = ["DetectorConfig", "run"]
