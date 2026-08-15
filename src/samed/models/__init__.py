"""Thin adapters over upstream segmentation models.

Importing this package registers every wrapper whose upstream dependency is
installed.  A missing back-end is not an error - the laptop that runs the
analysis and unit tests has no GPU stack, while the cluster has all of them - so
each wrapper module is imported defensively and simply stays unregistered when
its dependency is absent.
"""

from __future__ import annotations

from samed.models.base import MaskSet, PromptableSegmenter, available, create, register

__all__ = ["MaskSet", "PromptableSegmenter", "available", "create", "register"]


def _load_backends() -> None:
    for module in ("samed.models.sam",):
        try:
            __import__(module)
        except ImportError:
            continue


_load_backends()
