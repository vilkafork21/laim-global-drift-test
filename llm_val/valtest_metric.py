"""
Модуль теста метрик для базовой оценки качества.

Содержит функции для расчета метрик качества и формирования
отчета с определением цвета светофора.
"""

import typing as tp
from copy import deepcopy

import pandas as pd
from llm_val.report_helper import semaphore_by_threshold, tricky_semaphore
from llm_val.sampler import Sampler
from llm_val.scorer import Scorer


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТОВ
# =============================================================================


def report_valtest_metric(
    metric_value: tp.Dict[str, float],
    main_metric: str,
    data_type: str = "test",
    semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]] = {"default": (0.4, 0.6)},
    main_semaphore_threshold: tp.Tuple[float, float] = (0.4, 0.6),
    red_freq_threshold: float = 0.2,
    yellow_freq_threshold: float = 0.2,
    greater_is_better: bool = True,
    is_info: bool = False,
) -> tp.Dict[str, tp.Any]:
    """
    Создание отчета по результатам теста метрик.

    Формирует структуру результатов с определением цвета светофора
    на основе значений метрик и заданных порогов.

    Args:
        metric_value: Словарь со значениями метрик
        main_metric: Название ключевой метрики качества
        data_type: Тип данных, на которых производится расчет метрики
        semaphore_threshold: Словарь с границами светофора для каждой метрики
        main_semaphore_threshold: Границы светофора для ключевой метрики
        red_freq_threshold: Доля допустимых красных светофоров
        yellow_freq_threshold: Доля допустимых желтых светофоров
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный

    Returns:
        Словарь с результатами отчета в формате llm_val
    """
    color_list = {}
    metric_value = deepcopy(metric_value)
    main_value = metric_value.pop(main_metric)
    for name, value in metric_value.items():
        color_list[name] = semaphore_by_threshold(
            value,
            threshold=semaphore_threshold.get(name, semaphore_threshold["default"]),
            greater_is_better=greater_is_better,
        )
    color = (
        "gray"
        if is_info
        else tricky_semaphore(
            value=main_value,
            semaphore_list=list(color_list.values()),
            greater_is_better=greater_is_better,
            threshold_metric=main_semaphore_threshold,
            red_freq_threshold=red_freq_threshold,
            yellow_freq_threshold=yellow_freq_threshold,
        )
    )
    result_dataframes = []
    if len(color_list) > 0:
        result_dataframes.append(
            pd.DataFrame(
                [metric_value, color_list],
                index=[f"Значение метрики на {data_type}", "Результат теста"],
            ).T
        )
    result_dataframes.insert(
        0,
        pd.DataFrame(
            {
                f'Значение ключевой метрики "{main_metric}" на {data_type}': [
                    main_value
                ],
                "Результат теста": [color],
            }
        ),
    )
    res_dict = {}
    res_dict["semaphore"] = color
    res_dict["result_plots"] = []
    res_dict["result_dataframes"] = result_dataframes
    return res_dict


# =============================================================================
# ОСНОВНОЙ ТЕСТ
# =============================================================================


def valtest_metric(
    sampler: Sampler,
    scorer: Scorer,
    main_metric: str,
    data_type: str = "test",
    metric_agg: str = "single_mean",
    semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]] = {"default": (0.4, 0.6)},
    main_semaphore_threshold: tp.Tuple[float, float] = (0.4, 0.6),
    red_freq_threshold: float = 0.2,
    yellow_freq_threshold: float = 0.2,
    greater_is_better: bool = True,
    is_info: bool = False,
    metric_value: tp.Optional[tp.Dict[str, float]] = None,
    **kwargs,
) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    """
    Тест подсчета метрик качества.

    Рассчитывает метрики на данных семплера с использованием скорера
    и формирует отчет с определением цвета светофора.

    Args:
        sampler: Объект класса Sampler с данными
        scorer: Объект класса Scorer для расчета метрик
        main_metric: Название ключевой метрики качества
        data_type: Тип данных для расчета метрики
        metric_agg: Название агрегации метрики
        semaphore_threshold: Словарь с границами светофора для метрик
        main_semaphore_threshold: Границы светофора для ключевой метрики
        red_freq_threshold: Доля допустимых красных светофоров
        yellow_freq_threshold: Доля допустимых желтых светофоров
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        metric_value: Опциональное значение метрики (если уже рассчитано)

    Returns:
        Словарь с результатами в формате llm_val
    """
    if metric_value is None:
        metric_value = scorer.calc(
            sampler=sampler, data_type=data_type, metric_name=metric_agg
        )
    precomputed = {"metric_value": deepcopy(metric_value)}
    report = report_valtest_metric(
        metric_value=metric_value,
        main_metric=main_metric,
        data_type=data_type,
        semaphore_threshold=semaphore_threshold,
        main_semaphore_threshold=main_semaphore_threshold,
        red_freq_threshold=red_freq_threshold,
        yellow_freq_threshold=yellow_freq_threshold,
        greater_is_better=greater_is_better,
        is_info=is_info,
    )
    return {"report": report, "precomputed": precomputed}