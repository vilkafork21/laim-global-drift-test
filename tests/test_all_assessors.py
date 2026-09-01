"""Контракт all_assessors (единогласие): принимается и считается как min голосов."""

from __future__ import annotations

import pandas as pd

from laim_monitoring import score_units, unitize, validate_monitoring_metric


def _contract() -> dict:
    return {
        "contract_version": "laim-monitoring-metric.v2", "umr_version": "laim-umr.v2",
        "status": "computed", "basket_id": "CI1", "name": "quality", "score_column": "main_metric",
        "assessment_mode": "qa",
        "scoring": {
            "method": "all_assessors",
            "sources": [
                {"source_id": f"source_{index}", "column_name": f"mark{index}_metric",
                 "role": "assessor_vote", "normalization": "numeric", "polarity": "direct"}
                for index in (1, 2)
            ],
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


def test_all_assessors_contract_is_accepted_and_scored_as_unanimity():
    contract = validate_monitoring_metric(_contract())
    frame = pd.DataFrame({
        "query_id": ["q1", "q2"],
        "input_query": ["a", "b"],
        "output_answer": ["x", "y"],
        "mark1_metric": [1, 1],
        "mark2_metric": [1, 0],
    })
    units = unitize(frame, contract)
    assert score_units(units, _contract()).tolist() == [1.0, 0.0]
