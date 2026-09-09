"""
Главный модуль теста на глобальный дрифт запросов.
"""

import logging
import math
from ast import literal_eval

import pandas as pd

from config import Config
from giga_wraper import GigaEmbed
from llm_val.sampler import AutoAsessorSampler
from llm_val.scorer import AutoAsessorScorer
from llm_val.utils import METRICS  # единый источник (P3-4)
from llm_val.valtest_global_drift_stability import (
    MIN_CHUNKS,
    valtest_global_drift_stability,
)
from laim_monitoring import prepare_drift_frames

from html_report import format_report_number, render_test_report


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТОВ
# =============================================================================




def html_report_valtest_global_drift(res: dict, semaphore_title: str) -> str:
    pre = res["precomputed"]
    baseline, current = pre.get("metric_value"), pre.get("metric_value_estimate")
    delta = baseline - current if baseline is not None and current is not None else None
    correlations = pre.get("feature_correlations", {})
    features = "; ".join(f"{name} ({format_report_number(correlations.get(name))})"
                         for name in pre.get("selected_features", []))
    rows = [
        ("Значение КМ на эталонной корзине (OOS)", format_report_number(baseline)),
        ("Прогноз КМ по глобальному дрифту (OOT)", format_report_number(current)),
        ("Источник прогноза", "Расстояния между запросами и метки эталонной корзины"),
        ("Абсолютное снижение D = КМ OOS − КМ OOT", format_report_number(delta)),
        ("Пороги снижения D (жёлтый / красный)", " / ".join(format_report_number(v) for v in pre.get("semaphore_threshold", (0.15, 0.25)))),
        ("Отобранные признаки (корреляция Пирсона)", features or "Не отобраны"),
        ("Число подвыборок OOS add / объектов в подвыборке",
         f"{format_report_number(pre.get('n_chunks'), 0)} / {format_report_number(pre.get('chunk_size'), 0)}"),
    ]
    if pre.get("selection_low_confidence"):
        rows.append(("Надёжность отбора признаков", "Пониженная: статистическая значимость не подтверждена"))
    return render_test_report(
        "6.3.5", "Глобальный дрифт запросов",
        "Оценить изменение распределения запросов текущего потока относительно эталонной корзины "
        "и его возможную связь с качеством ответов.",
        rows, res["report"]["semaphore"],
        "Это прогноз качества по семантике запросов и оценкам эталонной корзины. "
        "Ответы решения и оценки Автоасессора за отчётный период в прогнозе не используются. "
        "Положительное D означает ожидаемое снижение, отрицательное — рост. "
        "Отсутствие значимых признаков не доказывает ухудшение: в этом случае результат серый.",
        f"СЗ выше E; не менее 200 объектов OOS и {MIN_CHUNKS} подвыборок не менее чем по 20 объектов. "
        "Число подвыборок определяется объёмом эталона, максимум 10. Отбор: |r| > 0,3 "
        "и p < 0,05 / 11 (поправка Бонферрони); прогноз — RidgeCV.",
        "Пороги по умолчанию: зелёный — D < 0,15 и зелёная КМ на эталоне; красный — "
        "D ≥ 0,25 или красная КМ на эталоне; иначе жёлтый. Серый: нет значимых признаков, "
        "недостаточно объектов, невалидные данные или информационный режим.",
        reason=pre.get("reason") or ("Оценка недоступна или выбран информационный режим."
                                    if res["report"]["semaphore"] in ("gray", "grey") else ""),
    )


# Цвет, отдаваемый ПЛАТФОРМЕ и АГРЕГАТОРУ, должен быть в их словаре
# (red/amber/green/gray). Внутри теста используется "yellow"/"grey" —
# нормализуем на границе вывода, иначе светофор на узле не отрисуется,
# а agg-master не засчитает жёлтый (он считает color == "amber").
_PLATFORM_COLOR = {"yellow": "amber", "grey": "gray"}


def report_valtest_global_drift(res, semaphore_title):
    semaphore_color = res["report"]["semaphore"]
    platform_color = _PLATFORM_COLOR.get(semaphore_color, semaphore_color)
    html_report = html_report_valtest_global_drift(res, semaphore_title)
    precomputed = res.get("precomputed", {})

    def number(value):
        return float(value) if value is not None and math.isfinite(float(value)) else None

    return {
        "all_results": {
            "calculated_traffic_lights": {
                "test_light": platform_color,
                "semaphore_title": semaphore_title,
            },
            "color": platform_color,
            "status": precomputed.get(
                "status", "not_computable" if platform_color == "gray" else "computed"
            ),
            "reason_code": precomputed.get("reason_code"),
            "reason": precomputed.get("reason"),
            "n_oos": precomputed.get("n_oos"),
            "n_oot": precomputed.get("n_oot"),
            "metric_value_reference": number(precomputed.get("metric_value")),
            "metric_value_monitoring": number(precomputed.get("metric_value_estimate")),
            "metric_value_source": precomputed.get("metric_value_source"),
            "selected_features": precomputed.get("selected_features", []),
            "feature_correlations": precomputed.get("feature_correlations", {}),
            "selection_low_confidence": precomputed.get("selection_low_confidence"),
            "n_chunks": precomputed.get("n_chunks"),
        },
        "hidden_port": html_report,
    }


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

# P0-4: ключи унифицированы на "gray" во всех словарях.
_REASON_BY_COLOR = {
    "red": "Выборки на валидации и мониторинге значимо различаются, дрифт влияет на метрику",
    "green": "Выборки обладают схожей семантикой, дрифт не обнаружен",
    "yellow": "Выборки обладают схожей семантикой; рекомендуются дополнительные тесты",
    "gray": "Не удалось оценить семантическую схожесть запросов между выборками",
}

_SEMAPHORE_TITLE = {
    "red": "Результат теста глобального дрифта соответствует красному светофору",
    "green": "Результат теста глобального дрифта соответствует зелёному светофору",
    "yellow": "Результат теста глобального дрифта соответствует жёлтому светофору",
    "gray": "Результат теста глобального дрифта не может быть оценён",
}


def main(
    reference_umr: pd.DataFrame,
    monitoring_umr: pd.DataFrame,
    monitoring_metric: dict,
    n_chunks: int = MIN_CHUNKS,
    p_value: float = 0.05,
    corr_threshold: float = 0.3,
    metric_agg: str = "single_mean",
    data_types: tuple = ("train", "test"),
    red_threshold: float = 0.25,
    green_threshold: float = 0.15,
    greater_is_better: bool = True,
    is_info: bool = False,
    random_state: int = 42,
):
    """
    Прогноз КМ по запросам OOT и размеченному эталону OOS.

    Изменения относительно baseline:
    - n_chunks default = MIN_CHUNKS=5 (P0-2)
    - Defaults p_value/corr_threshold/red/green согласованы с HTML (P1-1)
    - random_state в UI (P1-6)
    - dropna subset как список (P1-9)
    - main_metric перезаписывается на "target" после rename (P1-10)
    - Удалены устаревшие `top_distance_features` (заменены фиксированным набором scale-aware фичей в valtest)
    """
    # Защитный literal_eval (P1-6 в local-аналогии)
    if isinstance(data_types, str):
        data_types = literal_eval(data_types)

    # P1-1: пороги в правильном порядке
    semaphore_threshold = (
        min(red_threshold, green_threshold),
        max(red_threshold, green_threshold),
    )

    reference_frame, monitoring_frame = prepare_drift_frames(
        reference_umr, monitoring_umr, monitoring_metric
    )
    main_metric = "target"

    sampler = AutoAsessorSampler(agent_df=monitoring_frame, real_df=reference_frame)
    scorer = AutoAsessorScorer(metrics=METRICS)

    config = Config()
    embedding_model = GigaEmbed(**config.contour_configs)

    logging.info("Тест на глобальный дрифт запущен")
    res = valtest_global_drift_stability(
        sampler=sampler,
        scorer=scorer,
        main_metric=main_metric,
        model=embedding_model,
        n_chunks=n_chunks,
        p_value=p_value,
        corr_threshold=corr_threshold,
        metric_binarizer=None,
        metric_agg=metric_agg,
        data_types=data_types,
        semaphore_threshold=semaphore_threshold,
        greater_is_better=greater_is_better,
        is_info=is_info,
        random_state=random_state,
        metric_scale=monitoring_metric.get("baseline", {}).get("scale"),
        test_color=None,
        metric_value_estimate=None,
    )

    semaphore_color = res["report"]["semaphore"]
    semaphore_title = _SEMAPHORE_TITLE[semaphore_color]

    report_result = report_valtest_global_drift(res, semaphore_title)
    report_result["all_results"]["test_name"] = "global_drift"
    return {
        "all_results": report_result["all_results"],
        "test_description": report_result["hidden_port"],
    }
