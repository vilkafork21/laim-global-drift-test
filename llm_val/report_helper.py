"""
Report helper functions.

Изменение vs baseline: `worst_semaphore` теперь корректно обращается с gray —
он считается «хуже всех» (P1-8), что соответствует семантике «нет данных».
"""

import typing as tp

import numpy as np
import pandas as pd


# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ СВЕТОФОРА
# =============================================================================

def semaphore_by_threshold(
    value: float,
    threshold: tp.Tuple[float, float],
    greater_is_better: bool = True,
    left_border_is_bigger: bool = True,
    right_border_is_bigger: bool = True,
) -> str:
    """
    Расчёт цвета светофора по двум порогам.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "gray"
    semaphore = {0: "red", 1: "yellow", 2: "green"}
    if (value < threshold[0]) or (not left_border_is_bigger and value == threshold[0]):
        interval = 0
    elif (value < threshold[1]) or (not right_border_is_bigger and value == threshold[1]):
        interval = 1
    else:
        interval = 2
    if not greater_is_better:
        interval = abs(interval - 2)
    return semaphore[interval]


def semaphore_by_threshold_without_yellow(
    value, threshold, greater_is_better=True, border_is_bigger=True,
):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "gray"
    semaphore = {0: "red", 1: "green"}
    if (value < threshold) or (not border_is_bigger and value == threshold):
        interval = 0
    else:
        interval = 1
    if not greater_is_better:
        interval = abs(interval - 1)
    return semaphore[interval]


def semaphore_by_threshold_without_red(
    value, threshold, greater_is_better=True, border_is_bigger=True,
):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "gray"
    semaphore = {0: "yellow", 1: "green"}
    if (value < threshold) or (not border_is_bigger and value == threshold):
        interval = 0
    else:
        interval = 1
    if not greater_is_better:
        interval = abs(interval - 1)
    return semaphore[interval]


# =============================================================================
# АГРЕГИРУЮЩИЕ ФУНКЦИИ СВЕТОФОРА
# =============================================================================

def worst_semaphore(semaphore_list: tp.List[str]) -> str:
    """
    Расчёт «худшего» светофора из списка.

    Семантика gray (P1-8): «нет данных» считается худшим состоянием — если хотя бы
    один источник вернул gray, итог gray. Затем по убыванию: red < yellow < green.
    """
    if not semaphore_list:
        return "gray"
    if "gray" in semaphore_list:
        return "gray"
    severity = {"red": 0, "yellow": 1, "green": 2}
    return min(semaphore_list, key=lambda c: severity.get(c, 99))


def tricky_semaphore(
    value, semaphore_list, greater_is_better=True,
    threshold_metric=(0.4, 0.6),
    red_freq_threshold=0.2, yellow_freq_threshold=0.2,
):
    color = "yellow"
    value_color = semaphore_by_threshold(value, threshold_metric, greater_is_better)
    freq_color = pd.Series(semaphore_list, dtype="str").value_counts(normalize=True).to_dict()
    if (value_color == "red") or (freq_color.get("red", 0) > red_freq_threshold):
        color = "red"
    elif (
        value_color == "green"
        and freq_color.get("yellow", 0) < yellow_freq_threshold
        and freq_color.get("red", 0) <= 0
    ):
        color = "green"
    return color


# =============================================================================
# КВАНТИЛЬНЫЕ / ЧАСТОТНЫЕ ФУНКЦИИ (без изменений в логике, копия из baseline)
# =============================================================================

def quantile_semaphore(value, threshold, inner_is_better=True):
    semaphore = {0: "red", 1: "yellow", 2: "green"}
    if (1 - threshold[0]) / 2 < value < (1 + threshold[0]) / 2:
        interval = 2
    elif (1 - threshold[1]) / 2 <= value <= (1 + threshold[1]) / 2:
        interval = 1
    else:
        interval = 0
    if not inner_is_better:
        interval = abs(interval - 2)
    return semaphore[interval]


def shuffle_ci_semaphore(mean_metric, std_metric, interval, std_threshold):
    if (mean_metric < interval[0]) or (mean_metric > interval[1]):
        return "red"
    if std_metric > std_threshold:
        return "yellow"
    return "green"


def quantile_semaphore_without_yellow(value, threshold, inner_is_better=True):
    if inner_is_better:
        return "green" if threshold[0] <= value <= threshold[1] else "red"
    return "green" if (value < threshold[0] or value > threshold[1]) else "red"


def local_frequency_semaphore(value, ideal_responses, threshold):
    yellow_bound, red_bound = threshold[0], threshold[1]
    if (value >= red_bound * ideal_responses) or (value <= ideal_responses / red_bound):
        return "red"
    elif (value >= yellow_bound * ideal_responses) or (value <= ideal_responses / yellow_bound):
        return "yellow"
    return "green"


def frequency_semaphore(values, threshold):
    ideal = np.mean(values)
    return [local_frequency_semaphore(v, ideal, threshold) for v in values]


def proportion_semaphore(semaphores, freq_threshold):
    s = pd.Series(semaphores)
    yellows = (s == "yellow").sum() / len(s)
    reds = (s == "red").sum() / len(s)
    if reds > freq_threshold[1]:
        return "red"
    if yellows + reds > freq_threshold[0]:
        return "yellow"
    return "green"


def quanity_semaphore(semaphore_list, threshold):
    yellow_count = sum(1 for c in semaphore_list if c == "yellow")
    red_count = sum(1 for c in semaphore_list if c == "red")
    if red_count >= threshold[0]:
        return "red"
    if yellow_count >= threshold[1]:
        return "yellow"
    return "green"
