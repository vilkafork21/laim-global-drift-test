"""Проверка источника КМ глобального дрифта без сети и изменения ноды."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
NODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE))

from llm_val.sampler import AutoAsessorSampler  # noqa: E402
from llm_val.scorer import AutoAsessorScorer  # noqa: E402
from llm_val.utils import METRICS  # noqa: E402
from llm_val.valtest_global_drift_stability import valtest_global_drift_stability  # noqa: E402


class Embeddings:
    """Детерминированные векторы вместо обращения к сервису эмбеддингов."""

    def get_embedding(self, texts: list[str]) -> np.ndarray:
        indices = np.array([int(text) for text in texts], dtype=float)
        return np.column_stack([
            np.ones(len(indices)), indices / 200, np.sin(indices), np.cos(indices),
        ])


def run(reference_scores: np.ndarray, monitoring_score: float) -> dict:
    reference = pd.DataFrame({
        "question": [str(index) for index in range(len(reference_scores))],
        "answer": "", "target": reference_scores,
    })
    monitoring = pd.DataFrame({
        "question": [str(index) for index in range(30, 70)],
        "answer": "", "target": monitoring_score,
    })
    return valtest_global_drift_stability(
        AutoAsessorSampler(monitoring, reference), AutoAsessorScorer(METRICS),
        "target", Embeddings(),
    )


def test_no_features_is_gray_independent_of_monitoring_scores():
    for score in [0., 1., np.nan]:
        result = run(np.ones(200), score)
        assert result['report']['semaphore'] == 'gray'
        assert result['precomputed']['status'] == 'not_computable'
        assert result['precomputed']['selected_features'] == []
        assert np.isnan(result['precomputed']['metric_value_estimate'])
    assert run(np.ones(199), 1.)['report']['semaphore'] == 'gray'


def test_prediction_and_adaptive_chunks_are_independent_of_oot_labels():
    scores = np.linspace(1., 0., 400)
    results = [run(scores, score) for score in [0., 1., np.nan]]
    values = [result['precomputed']['metric_value_estimate'] for result in results]
    assert np.isfinite(values).all(), results[0]['precomputed']
    assert values[0] == values[1] == values[2]
    for result in results:
        assert result['precomputed']['metric_value_source'] == 'query_distance_prediction'
        assert result['precomputed']['n_chunks'] == 10
        assert result['precomputed']['chunk_size'] >= 20


def test_bonferroni_does_not_fall_back(monkeypatch):
    from llm_val import valtest_global_drift_stability as drift
    from collections import namedtuple
    result_type = namedtuple('Correlation', 'statistic pvalue')
    monkeypatch.setattr(drift.stats, 'pearsonr', lambda *args: result_type(.9, .01))
    result = run(np.linspace(1., 0., 400), 1.)
    assert result['report']['semaphore'] == 'gray'
    assert result['precomputed']['selected_features'] == []


def test_query_only_transport_and_json_gray_result():
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location('global_main', NODE / 'main.py')
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    result = run(np.ones(200), np.nan)
    output = node.report_valtest_global_drift(result, 'Результат')
    json.dumps(output['all_results'], allow_nan=False)
    assert output['all_results']['metric_value_monitoring'] is None
    from laim_monitoring.core import _drift_frame
    data = pd.DataFrame({'query_id': ['1', '2'], 'input_query': ['первый', 'второй'],
                         'output_answer': ['', ''], 'session_id': ['s', 's'], 'main_metric': [0., 1.]})
    contract = {'assessment_mode': 'dialogue'}
    first = _drift_frame(data, contract, require_target=False)
    second = _drift_frame(data.drop(columns='main_metric'), contract, require_target=False)
    pd.testing.assert_frame_equal(first, second)


def test_global_green_and_reference_red_scenarios():
    assert run(np.linspace(1., .6, 400), np.nan)['report']['semaphore'] == 'green'
    assert run(np.linspace(.5, .1, 400), np.nan)['report']['semaphore'] == 'red'
    from llm_val.valtest_global_drift_stability import report_valtest_global_drift_stability
    for forecast, color in [(.66, 'green'), (.65, 'yellow'), (.55, 'red')]:
        result = report_valtest_global_drift_stability({'target': .8}, forecast,
                    'target', 'green', ['D_mean'], {'D_mean': -.99})
        assert result['semaphore'] == color
