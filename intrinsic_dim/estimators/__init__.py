"""
dim_estimators
==============
Plug-in collection of manifold dimension estimators.

Usage
-----
    from dim_estimators import get_estimator, list_estimators

    est  = get_estimator("mle", k=12, n_anchors=500)
    dims = est(samples)   # (Ph, Pw) tensor

Adding a new estimator
----------------------
1. Create `dim_estimators/my_estimator.py` with pure computation functions.
2. In `registry.py`, import, define the wrapper class, and call register().
"""

from .core import (          # noqa: F401
    DimEstimator,
    get_estimator,
    list_estimators,
)

# Trigger all registrations defined in registry.py.
from . import registry       # noqa: F401