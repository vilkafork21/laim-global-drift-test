"""Проверки HTML: светофор, пропуски, экранирование и сохранность результата."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("node_report", ROOT / "main.py")
NODE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODE)


@pytest.mark.parametrize("color", ["green", "yellow", "red", "gray"])
def test_report_preserves_result_and_explains_color(color):
    payload = {
        "report": {"semaphore": color},
        "precomputed": {
            "metric_value": 0.69, "metric_value_estimate": 0.62,
            "reliability": {"mean": 0.83, "median": 0.85, "q05": 0.71,
                            "share_below_threshold": 0.042},
            "gini_value": 0.52, "gini_std": 0.06,
            "gini_ci_lower": 0.49, "gini_ci_upper": 0.55,
            "n_oos_groups": 312, "n_oot_groups": 1240,
            "selected_features": ["<script>alert(1)</script>"],
            "reason": "Нет данных <script>alert(1)</script>",
        },
    }
    before = copy.deepcopy(payload)
    html = NODE.html_report_valtest_global_drift(payload, "Результат теста")
    assert payload == before
    assert 'class="laim-test-report"' in html
    assert "Тест 6.3.5" in html
    assert "Цель теста" in html and "Алгоритм расчёта" in html
    assert html.index("Критерии выставления светофора") < html.index('<table class="results-table"')
    assert "<details>" not in html and 'class="test-number"' not in html
    assert '<th scope="col">Показатель</th>' in html
    assert "Результат теста" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert {"green": "Зелёный", "yellow": "Жёлтый", "red": "Красный", "gray": "Не оценено"}[color] in html
    assert "4,2 %" in html if "local" in "global_drift" else True


def test_missing_numbers_are_not_zero():
    from html_report import format_report_number
    assert format_report_number(None) == "Не рассчитано"
    assert format_report_number(float("nan")) == "Не рассчитано"
    assert format_report_number(float("inf")) == "Не рассчитано"
    assert format_report_number(0.0) == "0,000"


def test_html_does_not_change_machine_output(monkeypatch):
    payload = {"report": {"semaphore": "green"}, "precomputed": {
        "metric_value": 0.69, "metric_value_estimate": 0.62,
        "gini_value": 0.52, "reliability": {"mean": 0.83, "share_below_threshold": 0.042},
    }}
    before = copy.deepcopy(payload)
    actual = NODE.report_valtest_global_drift(payload, "Результат")
    monkeypatch.setattr(NODE, "html_report_valtest_global_drift", lambda *args: "")
    expected = NODE.report_valtest_global_drift(payload, "Результат")
    assert actual["all_results"] == expected["all_results"]
    assert payload == before
