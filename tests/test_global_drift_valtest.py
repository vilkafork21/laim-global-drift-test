"""Серый светофор допустим только когда мерить нечего: отбор фич и отказы судьи."""

from __future__ import annotations

import numpy as np
import pandas as pd

from laim_monitoring import prepare_drift_frames
from llm_val.sampler import AutoAsessorSampler
from llm_val.scorer import AutoAsessorScorer
from llm_val.utils import METRICS
from llm_val.valtest_global_drift_stability import valtest_global_drift_stability


class _StubEmbeddings:
    """Детерминированные эмбеддинги из статистик текста, без сети."""

    def get_embedding(self, texts):
        vectors = []
        for text in texts:
            codes = np.frombuffer(str(text).encode("utf-8"), dtype=np.uint8).astype(float)
            vectors.append([
                float(codes.mean()),
                float(codes.std()),
                float(len(codes) % 7),
                float(len(codes) % 13),
            ])
        return vectors


def _reference_frame(rows: int = 160) -> pd.DataFrame:
    return pd.DataFrame({
        "question": [f"вопрос {i} " + "слово " * (i % 5 + 1) for i in range(rows)],
        "answer": ["ответ"] * rows,
        "target": [float(i % 3 != 0) for i in range(rows)],
    })


def _monitoring_frame(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        "question": [f"запрос {i} " + "текст " * (i % 4 + 1) for i in range(rows)],
        "answer": ["ответ"] * rows,
        "target": [float(i % 4 != 0) for i in range(rows)],
    })


def _run_valtest(reference: pd.DataFrame, monitoring: pd.DataFrame) -> dict:
    return valtest_global_drift_stability(
        sampler=AutoAsessorSampler(agent_df=monitoring, real_df=reference),
        scorer=AutoAsessorScorer(metrics=METRICS),
        main_metric="target",
        model=_StubEmbeddings(),
        n_chunks=5,
        metric_binarizer=None,
        data_types=("train", "test"),
        test_color=None,
        metric_value_estimate=None,
    )


def test_selection_never_empty_and_color_from_actual_metric():
    """Адаптивные чанки + FDR + фолбэк: фичи отобраны, цвет не серый."""
    res = _run_valtest(_reference_frame(), _monitoring_frame())

    pre = res["precomputed"]
    assert pre["selected_features"], "отбор фич не должен быть пустым"
    assert pre["n_chunks"] > 5, "n_chunks должен масштабироваться от объёма add"
    assert pre["metric_value_source"] == "assessor_scored_data"
    assert res["report"]["semaphore"] != "gray"


def test_empty_scored_monitoring_is_reasoned_not_computable():
    """Ни одной оценённой единицы мониторинга — честный серый с причиной."""
    res = _run_valtest(_reference_frame(), _monitoring_frame(rows=0))

    pre = res["precomputed"]
    assert pre["status"] == "not_computable"
    assert pre["reason_code"] == "no_scored_monitoring_units"
    assert res["report"]["semaphore"] == "gray"


def test_judge_refusals_do_not_fail_drift_frames():
    """Отказ судьи (NaN main_metric) исключается, а не роняет тест при policy=fail."""
    contract = {
        "contract_version": "laim-monitoring-metric.v2", "umr_version": "laim-umr.v2",
        "status": "computed", "basket_id": "CI1", "name": "quality",
        "score_column": "main_metric", "assessment_mode": "qa",
        "scoring": {
            "method": "identity",
            "sources": [{
                "source_id": "source_1", "column_name": "score_metric",
                "role": "final_score", "normalization": "numeric", "polarity": "direct",
            }],
            "missing_policy": "fail", "majority_denominator": None,
        },
        "aggregation": {"method": "mean", "weight_column": None},
        "baseline": {"value": 0.9, "scale": "ratio", "recomputed_value": 0.9},
        "primary_validation": {"affects_monitoring": False},
    }
    from measurement_fixture import reviewed_metric
    contract = reviewed_metric(contract)
    reference = pd.DataFrame({
        "query_id": ["r1", "r2", "r3"],
        "input_query_count": [1, 1, 1],
        "input_query": ["a", "b", "c"],
        "output_answer": ["x", "y", "z"],
        "score_metric": [1.0, 0.0, 1.0],
        "main_metric": [1.0, 0.0, 1.0],
    })
    monitoring = pd.DataFrame({
        "query_id": ["m1", "m2", "m3"],
        "input_query_count": [1, 1, 1],
        "input_query": ["d", "e", "f"],
        "output_answer": ["x", "y", "z"],
        "score_metric": [1.0, None, 0.0],
        "main_metric": [1.0, None, 0.0],
    })

    reference_frame, monitoring_frame = prepare_drift_frames(reference.assign(definition_id=contract["definition_id"], dataset_role="reference"), monitoring.assign(definition_id=contract["definition_id"], dataset_role="monitoring"), contract)

    assert len(reference_frame) == 3
    assert monitoring_frame["target"].tolist() == [1.0, 0.0]


def test_computed_verdict_carries_population_sizes():
    # n_oos/n_oot публикуются и на успешном пути: без них вердикт нельзя
    # соотнести с популяцией, на которой он посчитан.
    res = _run_valtest(_reference_frame(), _monitoring_frame())
    pre = res["precomputed"]
    assert pre["n_oos"] == 160 and pre["n_oot"] == 60
