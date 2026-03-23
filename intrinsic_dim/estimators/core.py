"""
core.py — protocols and registries for dim estimators and variance estimators.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import torch


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class DimEstimator(Protocol):
    """
    samples : (Ph, Pw, N, D)  →  dims : (Ph, Pw)
    """
    def __call__(self, samples: torch.Tensor) -> torch.Tensor: ...
    def variance_of(self, samples: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class DimVarianceEstimator(Protocol):
    """
    Black-box variance estimator — wraps a DimEstimator and calls it
    repeatedly on subsamples.

    samples : (Ph, Pw, N, D)  →  variance : (Ph, Pw)
    """
    def __call__(self, samples: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Dim estimator registry
# ---------------------------------------------------------------------------

_DIM_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator that registers a dim estimator under *name*."""
    def decorator(cls):
        _DIM_REGISTRY[name] = cls
        return cls
    return decorator


def get_estimator(name: str, **kwargs) -> DimEstimator:
    """
    Instantiate a registered dim estimator by name.

    Example
    -------
    >>> est  = get_estimator("mle", k=12, variance="bootstrap", n_trials=50)
    >>> dims = est(samples)              # (Ph, Pw)
    >>> var  = est.variance_of(samples)  # (Ph, Pw)
    """
    if name not in _DIM_REGISTRY:
        available = ", ".join(sorted(_DIM_REGISTRY))
        raise KeyError(f"Unknown estimator '{name}'. Available: {available}")
    return _DIM_REGISTRY[name](**kwargs)


def list_estimators() -> list[str]:
    return sorted(_DIM_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Variance estimator registry
# ---------------------------------------------------------------------------

_VAR_REGISTRY: dict[str, type] = {}


def register_variance(name: str):
    """Class decorator that registers a variance estimator under *name*."""
    def decorator(cls):
        _VAR_REGISTRY[name] = cls
        return cls
    return decorator


def get_variance_estimator(name: str, **kwargs) -> DimVarianceEstimator:
    """
    Instantiate a registered variance estimator by name.

    The variance estimator receives the dim estimator instance via kwargs
    so it can call it repeatedly on subsamples.
    """
    if name not in _VAR_REGISTRY:
        available = ", ".join(sorted(_VAR_REGISTRY))
        raise KeyError(f"Unknown variance estimator '{name}'. Available: {available}")
    return _VAR_REGISTRY[name](**kwargs)


def list_variance_estimators() -> list[str]:
    return sorted(_VAR_REGISTRY.keys())