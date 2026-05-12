"""
Random number generator helpers.
"""

from numbers import Integral

from numpy.random import default_rng


def coerce_rng(rng=None, seed=None):
    if rng is None:
        return default_rng(seed)
    if isinstance(rng, Integral):
        return default_rng(rng)
    return rng
