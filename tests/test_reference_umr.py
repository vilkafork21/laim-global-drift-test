"""Reference UMR в формате тестового датасета: packed dialogue и flat с session_id."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from laim_monitoring import MonitoringContractError, unitize


def _contract(mode: str) -> dict:
    return {
        "contract_version": "laim-monitoring-metric.v2", "umr_version": "laim-umr.v2",
        "status": "computed", "basket_id": "CI1", "name": "quality", "score_column": "main_metric",
        "assessment_mode": mode,
        "scoring": {
            "method": "identity",
            "sources": [{
                "source_id": "source_1", "column_name": "score_metric", "role": "final_score",
                "normalization": "numeric", "polarity": "direct",
            }],
            "missing_policy": "fail", "majority_denominator": None,
        },
        "aggregation": {"method": "mean", "weight_column": None},
        "baseline": {
            "value": 0.5, "scale": "ratio", "value_source": "validation_report",
            "reported_value": 0.5, "reported_scale": "ratio", "recomputed_value": 0.5,
            "reconciliation": "match",
        },
        "primary_validation": {
            "threshold": None, "comparator": None, "scale": "ratio", "verdict": None,
            "affects_monitoring": False,
        },
        "evidence": {},
    }


def test_packed_dialogue_reference_is_unitized_per_session():
    frame = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "dialogue": ["[('q1', 'hi', 'hello'), ('q2', 'bye', 'see you')]", "[('q3', 'x', 'y')]"],
        "input_query_count": [1, 1],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    units = unitize(frame, _contract("dialogue"))
    assert len(units) == 2
    assert [turn["input_query"] for turn in units["dialogue"].iloc[0]] == ["hi", "bye"]
    assert units["source_1"].tolist() == [1.0, 0.0]
    assert units["main_metric"].tolist() == [1.0, 0.0]


def test_flat_reference_with_session_id_keeps_turn_history():
    frame = pd.DataFrame({
        "session_id": ["s1", "s1", "s2"],
        "query_id": ["q1", "q2", "q3"],
        "input_query_count": [1, 1, 1],
        "input_query": ["hi", "bye", "x"],
        "output_answer": ["hello", "see you", "y"],
        "score_metric": [1.0, 0.0, 1.0],
        "main_metric": [1.0, 0.0, 1.0],
    })
    units = unitize(frame, _contract("turn_with_history"))
    assert len(units) == 3
    assert [turn["input_query"] for turn in units["assessment_context"].iloc[1]["history"]] == ["hi"]


def test_flat_reference_without_canonical_columns_is_rejected():
    with pytest.raises(MonitoringContractError):
        unitize(pd.DataFrame({"question": ["q"], "answer": ["a"]}), _contract("qa"))


def test_drift_frames_from_packed_reference_and_scored_dialogue():
    """Packed reference и flat scored_data ассессора дают единицы-сессии."""
    from laim_monitoring import prepare_drift_frames

    contract = _contract("dialogue")
    reference = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "dialogue": [
            "[('t1', 'вопрос один', 'ответ один'), ('t2', 'вопрос два', 'ответ два')]",
            "[('t3', 'вопрос три', 'ответ три')]",
        ],
        "input_query_count": [1, 1],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    monitoring = pd.DataFrame({
        "session_id": ["m1", "m2", "m2"],
        "query_id": ["mt1", "mt2", "mt3"],
        "input_query": ["наблюдённый вопрос", "ещё вопрос", "и ещё"],
        "output_answer": ["наблюдённый ответ", "ещё ответ", "и ответ"],
        "assessment_unit_id": ["m1", "m2", "m2"],
        "main_metric": [1.0, 0.0, 0.0],
    })

    ref_frame, mon_frame = prepare_drift_frames(reference, monitoring, contract)

    assert len(ref_frame) == 2  # единица drift — диалог
    assert len(mon_frame) == 2
    assert ref_frame["target"].tolist() == [1.0, 0.0]
    assert mon_frame["target"].tolist() == [1.0, 0.0]
    assert "вопрос один" in ref_frame["question"].iloc[0]


def test_drift_frames_from_flat_monitoring_with_session_id():
    """qa/turn_with_history: flat monitoring TDC без служебных колонок."""
    from laim_monitoring import prepare_drift_frames

    contract = _contract("turn_with_history")
    reference = pd.DataFrame({
        "session_id": ["s1", "s1"],
        "query_id": ["q1", "q2"],
        "input_query_count": [1, 1],
        "input_query": ["в1", "в2"],
        "output_answer": ["о1", "о2"],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    monitoring = pd.DataFrame({
        "scenario": ["r", "r"],
        "session_id": ["m1", "m1"],
        "query_id": ["mq1", "mq2"],
        "input_query_count": [1, 1],
        "input_query": ["нв1", "нв2"],
        "output_answer": ["но1", "но2"],
        "main_metric": [1.0, 0.0],
    })

    ref_frame, mon_frame = prepare_drift_frames(reference, monitoring, contract)

    assert len(ref_frame) == 2
    assert len(mon_frame) == 2
    # История реплик сессии входит в question drift-фрейма
    assert "в1" in ref_frame["question"].iloc[1]


@pytest.mark.parametrize("mode", ["qa", "dialogue"])
def test_small_basket_returns_structured_not_computable(mode, monkeypatch):
    import main as drift

    if mode == "qa":
        reference = pd.DataFrame({
            "query_id": [f"r{index}" for index in range(8)],
            "input_query": [f"вопрос {index}" for index in range(8)],
            "output_answer": [f"ответ {index}" for index in range(8)],
            "score_metric": [float(index % 2) for index in range(8)],
            "main_metric": [float(index % 2) for index in range(8)],
        })
        monitoring = reference.iloc[:4].drop(columns=["score_metric"])
    else:
        reference = pd.DataFrame({
            "session_id": [f"r{index}" for index in range(8)],
            "dialogue": [
                repr([(f"r{index}-1", f"вопрос {index}", f"ответ {index}")])
                for index in range(8)
            ],
            "input_query_count": [1] * 8,
            "score_metric": [float(index % 2) for index in range(8)],
            "main_metric": [float(index % 2) for index in range(8)],
        })
        monitoring = reference.iloc[:4].drop(columns=["score_metric"])

    monkeypatch.setattr(
        drift, "Config", lambda: SimpleNamespace(contour_configs={})
    )
    monkeypatch.setattr(drift, "GigaEmbed", lambda **_kwargs: object())

    result = drift.main(reference, monitoring, _contract(mode), n_chunks=5)

    assert result["all_results"]["color"] == "gray"
    assert result["all_results"]["status"] == "not_computable"
    assert result["all_results"]["reason_code"] == "insufficient_reference_units"


def test_not_computable_metric_skips_drift_computation(monkeypatch):
    import main as drift

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Вычислительный путь не должен запускаться")

    monkeypatch.setattr(drift, "prepare_drift_frames", forbidden)
    monkeypatch.setattr(drift, "Config", forbidden)
    monkeypatch.setattr(drift, "GigaEmbed", forbidden)
    monkeypatch.setattr(drift, "valtest_global_drift_stability", forbidden)

    result = drift.main(
        object(),
        object(),
        {
            "contract_version": "laim-monitoring-metric.v2",
            "umr_version": "laim-umr.v2",
            "status": "not_computable",
            "reason_code": "ambiguous_baseline",
            "reason": "baseline нельзя определить однозначно",
        },
    )

    light = result["all_results"]
    assert light["color"] == "gray"
    assert light["status"] == "not_computable"
    assert light["reason_code"] == "ambiguous_baseline"
    assert light["reason"] == "baseline нельзя определить однозначно"
    assert light["test_name"] == "global_drift"


@pytest.mark.parametrize("mode", ["turn_with_history", "dialogue"])
def test_no_significant_features_returns_structured_reason(mode, monkeypatch):
    import main as drift

    if mode == "turn_with_history":
        reference = pd.DataFrame({
            "session_id": [f"r{index}" for index in range(10)],
            "query_id": [f"rq{index}" for index in range(10)],
            "input_query": [f"вопрос {index}" for index in range(10)],
            "output_answer": [f"ответ {index}" for index in range(10)],
            "score_metric": [float(index % 2) for index in range(10)],
            "main_metric": [float(index % 2) for index in range(10)],
        })
        monitoring = reference.iloc[:4].drop(columns=["score_metric"])
    else:
        reference = pd.DataFrame({
            "session_id": [f"r{index}" for index in range(10)],
            "dialogue": [
                repr([(f"rq{index}", f"вопрос {index}", f"ответ {index}")])
                for index in range(10)
            ],
            "score_metric": [float(index % 2) for index in range(10)],
            "main_metric": [float(index % 2) for index in range(10)],
        })
        monitoring = reference.iloc[:4].drop(columns=["score_metric"])

    class ConstantEmbedding:
        def get_embedding(self, texts):
            return np.ones((len(texts), 2), dtype=float)

    monkeypatch.setattr(
        drift, "Config", lambda: SimpleNamespace(contour_configs={})
    )
    monkeypatch.setattr(
        drift, "GigaEmbed", lambda **_kwargs: ConstantEmbedding()
    )

    result = drift.main(reference, monitoring, _contract(mode), n_chunks=5)

    # Карточка 6.3.8: без истории размеченных периодов прогноз не оценивается;
    # диагностика публикуется, цвет по среднему судьи не выставляется.
    assert result["all_results"]["color"] == "gray"
    assert result["all_results"]["status"] == "not_computable"
    assert result["all_results"]["reason_code"] == "no_labeled_history"
    assert result["all_results"]["informative"] is True
    assert result["all_results"]["reason"]
    assert result["all_results"]["n_oos"] == 10
    assert result["all_results"]["n_oot"] == 4
    assert result["all_results"]["metric_value_reference"] == 0.5
    assert result["all_results"]["metric_value_monitoring"] == 0.5
    assert result["all_results"]["metric_value_source"] == "assessor_scored_data"
    assert result["all_results"]["selected_features"] == []


def test_descriptor_requires_assessor_scored_data():
    descriptor = json.loads(
        (Path(__file__).resolve().parents[1] / "descriptor.json").read_text()
    )
    monitoring_port = next(
        port for port in descriptor["ports"] if port["name"] == "monitoring_umr"
    )
    assert "scored_data" in monitoring_port["description"]
    assert "main_metric" in monitoring_port["description"]
    assert "parquet_test_dataset" not in monitoring_port["description"]
