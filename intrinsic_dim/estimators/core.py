from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DimEstimator(Protocol):
    """
    A callable that estimates the intrinsic dimension of patch-structured data.

    Input
    -----
    samples : torch.Tensor, shape (Ph, Pw, N, D)

    Output
    ------
    dims : torch.Tensor, shape (Ph, Pw)
    """

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator that registers an estimator under *name*."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_estimator(name: str, **kwargs) -> DimEstimator:
    """
    Instantiate a registered estimator by name.

    Example
    -------
    >>> est  = get_estimator("mle", k=12)
    >>> dims = est(samples)   # (Ph, Pw)
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown estimator '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_estimators() -> list[str]:
    """Return the names of all registered estimators."""
    return sorted(_REGISTRY.keys())