"""
Utility functions for global drift test.
"""

import numpy as np


METRICS = {
    "single_mean": {"call": lambda x: float(np.mean(x.values)), "is_singlecol": True},
    "multicol_mean": {"call": lambda x: float(np.mean(x.values)), "is_singlecol": False},
}


def string_to_float(value):
    """
    Преобразование строки в число с обработкой ошибок.
    P3-2: специфичные исключения вместо bare except.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
