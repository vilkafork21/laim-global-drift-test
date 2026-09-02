"""
Главный модуль теста на глобальный дрифт запросов.
"""

import logging
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

from html_report_helper import display_semaphore, show_criteria_semaphore

logger = logging.getLogger(__name__)


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТОВ
# =============================================================================

def _table_styles():
    return [
        {"selector": "th", "props": [
            ("background-color", "#f5f5f5"),
            ("text-align", "center"),
            ("border", "1px solid #ddd"),
            ("padding", "5px"),
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"),
            ("border", "1px solid #ddd"),
            ("padding", "5px"),
        ]},
        {"selector": "", "props": [
            ("border-collapse", "collapse"),
            ("border", "1px solid black"),
        ]},
    ]


def html_report_valtest_global_drift(res, semaphore_title):
    table_styles = _table_styles()

    green_criterion = (
        "Абсолютное снижение ключевой метрики менее 15 п.п. "
        "И светофор OOT — «Зелёный»"
    )
    yellow_criterion = "Иначе"
    red_criterion = (
        "Абсолютное снижение ключевой метрики качества более 25 п.п. "
        "ИЛИ светофор OOT — «Красный»"
    )
    grey_criterion = (
        "Нет ни одной оценённой единицы мониторинга "
        "либо эталон слишком мал для разбиения"
    )

    criterion_df = show_criteria_semaphore(
        green_criterion, yellow_criterion, red_criterion, grey_criterion, table_styles
    )
    criterion_df_html = criterion_df.to_html(border=0, classes="table")

    semaphore_color = res["report"]["semaphore"]
    semaphore_html = display_semaphore(semaphore_color, return_html=True)

    pre = res["precomputed"]
    mv = pre["metric_value"]
    mve = pre["metric_value_estimate"]

    def _fmt(v):
        if v is None:
            return "n/a"
        try:
            if pd.isna(v):
                return "n/a"
        except (TypeError, ValueError):
            pass
        return f"{float(v):.3f}"

    res_df = pd.DataFrame(
        {
            "Показатель": [
                "Значение метрики на валидации",
                "Оценка метрики на мониторинге",
                "Абсолютное снижение",
                "Отобранных фичей",
                "Результат теста",
            ],
            "Значение": [
                _fmt(mv),
                _fmt(mve),
                _fmt(abs(mv - mve)) if (mve is not None and mv is not None and not pd.isna(mve)) else "n/a",
                str(len(pre.get("selected_features", []))),
                semaphore_html,
            ],
        }
    )

    try:
        res_df_to_html = res_df.style.hide().set_table_styles(table_styles)
    except AttributeError:
        res_df_to_html = res_df.style.hide_index().set_table_styles(table_styles)
    res_df_html = res_df_to_html.to_html(border=0, classes="table")

    # Таблица отобранных фичей и их корреляций (P2-5)
    selected = pre.get("selected_features", [])
    corrs = pre.get("feature_correlations", {})
    if selected:
        feat_df = pd.DataFrame(
            {
                "Фича": selected,
                "Корреляция Пирсона": [round(float(corrs.get(name, float('nan'))), 4) for name in selected],
            }
        )
        try:
            feat_df_html = feat_df.style.hide().set_table_styles(table_styles).to_html(border=0, classes="table")
        except AttributeError:
            feat_df_html = feat_df.style.hide_index().set_table_styles(table_styles).to_html(border=0, classes="table")
    else:
        feat_df_html = "<p style=\"text-align: left;\">Статистически значимых фичей не обнаружено.</p>"

    html_report = f"""
<h2 style="text-align: center;">Тест на глобальный drift запросов</h2>
<p style="text-align: left;"><b>Цель теста</b></p>
<p style="text-align: left;">Оценить ключевую метрику качества агента исходя из степени изменения запросов к ней.</p>
<p style="text-align: left;"><b>Условия проведения</b></p>
<ul style="text-align: left; margin-left: 20px; padding-left: 20px;">
    <li>Тест применяется только при наличии достаточного количества данных в OOS.</li>
    <li>Минимум {MIN_CHUNKS} подвыборок OOSadd для статистической значимости.</li>
</ul>
<p style="text-align: left;"><b>Алгоритм расчёта</b></p>
<ol style="text-align: left; margin-left: 20px; padding-left: 20px;">
    <li>OOS делится на OOSbase и OOSadd поровну.</li>
    <li>Из OOSadd выделяются n_chunks подвыборок OOSadd_i.</li>
    <li>Между эмбеддингами OOSadd_i и OOSbase считаются cosine_distances.</li>
    <li>Для каждой OOSadd_i извлекаются 11 scale-aware фичей: перцентили (q25/q50/q75/q95), mean, std матрицы D и аналогичные статистики per-query средних ADi (q50/q75/q95, mean, std).</li>
    <li>Отбор фичей: Пирсон-корреляция с поправкой Бенджамини—Хохберга (FDR); при пустом отборе — top-K по |r| с пометкой низкой значимости.</li>
    <li>Обучается RidgeCV: features → метрика.</li>
    <li>Эмбеддинги OOT нарезаются на чанки того же размера → предсказание метрики чанк-за-чанком → усреднение.</li>
</ol>
<p style="text-align: left;"><b>Критерии выставления светофора</b></p>
<div style="text-align: left; width: 100%;">{criterion_df_html}</div><br>
<p style="text-align: left;"><b>Результаты теста</b></p>
<div style="text-align: left; width: 100%;">{res_df_html}</div><br>
<p style="text-align: left;"><b>Отобранные фичи</b></p>
<div style="text-align: left; width: 100%;">{feat_df_html}</div><br>
"""
    return html_report


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
            "metric_value_reference": precomputed.get("metric_value"),
            "metric_value_monitoring": precomputed.get("metric_value_estimate"),
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

_SEMAPHORE_TITLE = {
    "red": "Результат теста динамики ключевой метрики соответствует красному светофору",
    "green": "Результат теста динамики ключевой метрики соответствует зелёному светофору",
    "yellow": "Результат теста динамики ключевой метрики соответствует жёлтому светофору",
    "gray": "Результат теста динамики ключевой метрики не может быть оценён",
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
    Запуск теста на глобальный дрифт по размеченному scored_data ассессора.

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

    logger.info("Тест на глобальный дрифт запущен")
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
        test_color=None,
        metric_value_estimate=None,
    )

    pre = res["precomputed"]
    if pre.get("metric_value") is not None:
        pre["metric_value"] = round(pre["metric_value"], 3)
    if pre.get("metric_value_estimate") is not None and not pd.isna(pre["metric_value_estimate"]):
        pre["metric_value_estimate"] = round(pre["metric_value_estimate"], 3)

    semaphore_color = res["report"]["semaphore"]
    semaphore_title = _SEMAPHORE_TITLE[semaphore_color]

    report_result = report_valtest_global_drift(res, semaphore_title)
    report_result["all_results"]["test_name"] = "global_drift"
    return {
        "all_results": report_result["all_results"],
        "test_description": report_result["hidden_port"],
    }
