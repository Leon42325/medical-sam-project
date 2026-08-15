"""Dataset adapters: the only place that knows how a given dataset is laid out.

An adapter's whole job is to locate files and name what is in them. It does not
preprocess: normalisation, slice filtering and PNG export are the paper's
protocol (Sec. 2.2) and live once in :mod:`samed.data.preprocess`, applied
identically to every dataset. Keeping the split sharp is what stops the
reproduction from quietly diverging per dataset.

Adding a dataset therefore means describing a layout, not writing a pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

__all__ = ["Series", "Adapter", "register", "create", "available"]


@dataclass(frozen=True)
class Series:
    """One acquisition: an ordered run of slices sharing a label encoding.

    A series is the unit of intensity normalisation (one min-max over the whole
    run, so slices stay comparable) and the unit of subject accounting (one
    patient's scan must not be spread across subjects in the sampling protocol).

    ``images`` and ``labels`` are parallel and equal in length: ``labels[i]``
    annotates ``images[i]``. ``targets`` maps a target name to the pixel value
    that denotes it in the label files.
    """

    dataset: str
    modality: str
    subject: str
    images: list[Path]
    labels: list[Path]
    targets: dict[str, int] = field(default_factory=dict)
    #: The person scanned. Distinct from ``subject`` because one patient can
    #: contribute several series - CHAOS images every MR patient with both a
    #: T1DUAL and a T2SPIR sequence - and those are not independent
    #: observations. Confidence intervals cluster on this; sampling spreads
    #: over ``subject``, since the two sequences are genuinely different images.
    patient: str = ""

    def __post_init__(self) -> None:
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"{self.dataset}/{self.subject} ({self.modality}): "
                f"{len(self.images)} images but {len(self.labels)} labels - "
                "the slice-to-annotation pairing is not one-to-one"
            )

    @property
    def key(self) -> str:
        return f"{self.dataset}_{self.modality}_{self.subject}".replace(" ", "")

    @property
    def patient_id(self) -> str:
        return self.patient or self.subject


class Adapter(ABC):
    """Enumerates the series of one dataset."""

    name: str

    @abstractmethod
    def series(self, root: Path) -> Iterator[Series]:
        """Yield every series found under ``root``, in a deterministic order."""


_REGISTRY: dict[str, Callable[[], Adapter]] = {}


def register(name: str) -> Callable[[Callable[[], Adapter]], Callable[[], Adapter]]:
    def decorator(factory: Callable[[], Adapter]) -> Callable[[], Adapter]:
        if name in _REGISTRY:
            raise ValueError(f"adapter {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def create(name: str) -> Adapter:
    if name not in _REGISTRY:
        raise KeyError(f"no adapter for {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)


from samed.data.adapters import chaos as _chaos  # noqa: E402,F401  (registers itself)
