"""Compatibility wrapper for plotting helpers.

The plotting implementation currently lives in ``figures.py``. Importing from
this module keeps notebook calls like ``import plot`` working.
"""

from figures import *  # noqa: F403
