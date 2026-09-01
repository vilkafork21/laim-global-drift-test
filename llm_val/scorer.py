"""
Модуль скореров для подсчета метрик качества.

Содержит базовый абстрактный класс Scorer и его реализацию
AutoAsessorScorer для расчета метрик на данных асессоров.
"""

import typing as tp
from abc import ABC, abstractmethod

from llm_val.sampler import Sampler


# =============================================================================
# БАЗОВЫЕ КЛАССЫ
# =============================================================================


class Scorer(ABC):
    """
    Абстрактный класс скорера.

    Все скореры должны наследоваться от данного класса и
    реализовывать метод calc() для подсчета метрик.
    """

    def __init__(self, metrics: tp.Dict[str, tp.Any]):
        """
        Инициализация скорера.

        Args:
            metrics: Словарь с определениями метрик
        """
        self.metrics = metrics

    @abstractmethod
    def calc(
        self, sampler: Sampler, data_type: str, metric_name: str
    ) -> tp.Dict[str, float]:
        """
        Метод подсчета метрик.

        Args:
            sampler: Объект семплера с данными
            data_type: Тип данных ('train', 'test', 'val', 'oot')
            metric_name: Название метрики для подсчета

        Returns:
            Словарь со значениями метрик
        """
        raise NotImplementedError(f"Определите calc в {self.__class__.__name__}")


# =============================================================================
# РЕАЛИЗАЦИИ СКОРЕРОВ
# =============================================================================


class AutoAsessorScorer(Scorer):
    """
    Скорер для данных асессоров.

    Рассчитывает метрики качества на основе столбца 'target'
    в данных семплера. Поддерживает агрегацию по одной колонке
    или по всем колонкам.
    """

    def calc(
        self, sampler: Sampler, data_type: str, metric_name: tp.Optional[str] = None
    ) -> tp.Dict[str, float]:
        """
        Подсчёт метрик для указанного типа данных.

        Args:
            sampler: Объект семплера с данными
            data_type: Тип данных ('train', 'test', 'val', 'oot')
            metric_name: Название метрики (опционально)

        Returns:
            Словарь со значениями метрик
        """
        metrics = self.metrics
        data = getattr(sampler, data_type)["y"]
        if metric_name is not None:
            metrics = {metric_name: self.metrics[metric_name]}
        metric_res = {}

        for func_name, func_dict in metrics.items():
            if func_dict["is_singlecol"]:
                for name_col in data:
                    name = f"{name_col}"
                    metric_res[name] = func_dict["call"](data[name_col])
            else:
                metric_res[func_name] = func_dict["call"](data)
        return metric_res