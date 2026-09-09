"""Представление тестов в формате исходного HTML-отчёта."""

from __future__ import annotations


def format_report_number(value: float | int | None, decimals: int = 3, *, percent: bool = False) -> str:
    """Пропуски остаются пропусками; доли отображаются в процентах только явно."""
    import math

    if value is None or not math.isfinite(float(value)):
        return "Не рассчитано"
    number = float(value) * 100 if percent else float(value)
    text = f"{number:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return text + " %" if percent else text


def render_test_report(
    test_id: str,
    title: str,
    purpose: str,
    rows: list[tuple[str, str]],
    color: str,
    interpretation: str,
    conditions: tuple[str, ...],
    criteria: list[tuple[str, str]],
    *,
    algorithm: tuple[str, ...],
    reason: str = "",
    chart_html: str = "",
) -> str:
    """HTML-фрагмент; светофор поступает из расчёта без переоценки."""
    from html import escape

    def signal(value: str) -> str:
        value = {"amber": "yellow", "grey": "gray"}.get(value, value)
        labels = {"green": "Зелёный", "yellow": "Жёлтый", "red": "Красный", "gray": "Не оценено"}
        label = labels.get(value, "Неизвестный результат")
        lights = "".join(
            f'<span class="lamp {lamp if value == lamp else "inactive"}"></span>'
            for lamp in ("red", "yellow", "green")
        )
        return f'<span class="signal" role="img" aria-label="{label}" title="{label}">{lights}</span>'

    table_rows = "".join(
        f'<tr><th scope="row">{escape(str(key))}</th><td>{escape(str(value))}</td></tr>'
        for key, value in rows
    )
    criteria_rows = "".join(
        f'<tr><td>{escape(text)}</td><td>{signal(value)}</td></tr>' for value, text in criteria
    )
    steps = "".join(f"<li>{escape(step)}</li>" for step in algorithm)
    conditions_html = (
        "<p><b>Условия проведения</b></p><ul>" +
        "".join(f"<li>{escape(condition)}</li>" for condition in conditions) + "</ul>"
        if conditions else ""
    )
    reason_html = f"<p>{escape(str(reason))}</p>" if reason else ""
    return f'''<article class="laim-test-report" aria-label="Тест {escape(test_id)}">
<style>
.laim-test-report {{box-sizing:border-box;width:100%;min-width:0;max-width:100%;margin:0;padding:0;
  background:transparent;color:#1c1c1c;font:14px/1.48 Inter,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  text-align:left;overflow-wrap:anywhere}}
.laim-test-report * {{box-sizing:border-box}}
.laim-test-report h2 {{font-family:inherit;font-size:18px;font-weight:680;line-height:1.48;
  text-align:center;margin:28px 0 5px;color:#111827;letter-spacing:-.01em}}
.laim-test-report p {{font-size:14px;line-height:1.48;margin:14px 0;white-space:normal}}
.laim-test-report ol,.laim-test-report ul {{margin:14px 0 14px 20px;padding:0 0 0 20px}}
.laim-test-report li {{margin:0;padding:0}}
.laim-test-report table {{border-collapse:collapse;width:100%;min-width:0;max-width:100%;table-layout:fixed;
  margin:10px 0;border:1px solid #ddd;background:#fff}}
.laim-test-report th,.laim-test-report td {{font-family:inherit;font-size:12.75px !important;
  line-height:1.48;font-weight:400 !important;padding:5px;border:1px solid #ddd;
  text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere;word-break:normal;
  color:#1f2937;background:transparent;letter-spacing:normal}}
.laim-test-report thead th {{font-size:18.2px !important;font-weight:700 !important;text-align:center;
  background:#f5f5f5;color:#64748b;position:static}}
.laim-test-report tbody tr {{background:#fff}}
.laim-test-report tbody tr:nth-child(even) {{background:#f9f9f9}}
.laim-test-report .results-table th:first-child {{width:58%}}
.laim-test-report .criteria-table th:first-child {{width:74%}}
.laim-test-report td {{font-variant-numeric:tabular-nums}}
.laim-test-report .signal {{display:inline-flex;gap:5px;align-items:center;vertical-align:middle;
  margin:0;padding:5px 10px;border:2px solid #ccc;border-radius:100px;background:transparent;box-shadow:none}}
.laim-test-report .lamp {{display:block;width:10px;height:10px;border-radius:50%}}
.laim-test-report .red {{background:#ca1d1d}} .laim-test-report .yellow {{background:#ffd600}}
.laim-test-report .green {{background:#04d930}} .laim-test-report .inactive {{background:#ccc}}
.laim-test-report .interpretation {{font-size:12.75px}}
.laim-test-report img {{max-width:100%;height:auto}}
@media(max-width:600px) {{.laim-test-report .criteria-table th:first-child {{width:68%}}}}
@media print {{.laim-test-report tr {{break-inside:avoid}}}}
</style>
<h2>{escape(title)}</h2>
<p><b>Цель теста</b></p><p>{escape(purpose)}</p>
{conditions_html}
<p><b>Алгоритм расчёта</b></p><ol>{steps}</ol>
<p><b>Критерии выставления светофора</b></p>
<table class="criteria-table" aria-label="Критерии теста {escape(test_id)}">
<thead><tr><th scope="col">Критерий</th><th scope="col">Результат</th></tr></thead>
<tbody>{criteria_rows}</tbody></table><br>
<p><b>Результаты теста</b></p>
<table class="results-table" aria-label="Результаты теста {escape(test_id)}">
<thead><tr><th scope="col">Показатель</th><th scope="col">Значение</th></tr></thead>
<tbody>{table_rows}<tr><th scope="row">Результат теста</th><td>{signal(color)}</td></tr></tbody></table><br>
{reason_html}{chart_html}
<p class="interpretation">{escape(interpretation)}</p>
</article>'''
