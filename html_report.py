"""Единое представление результатов тестов мониторинга."""

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
    conditions: str,
    criteria: str,
    *,
    reason: str = "",
    chart_html: str = "",
) -> str:
    """HTML-фрагмент отчёта; светофор поступает из расчёта без переоценки."""
    from html import escape

    color = {"amber": "yellow", "grey": "gray"}.get(color, color)
    labels = {"green": "Зелёный · В пределах допуска", "yellow": "Жёлтый · Требует внимания",
              "red": "Красный · Выявлено отклонение", "gray": "Не оценено · Серый"}
    label = labels.get(color, "Неизвестный результат")
    lights = "".join(
        f'<span class="lamp {lamp if color == lamp else "inactive"}"></span>'
        for lamp in ("red", "yellow", "green")
    )
    status = f'<span class="signal" aria-hidden="true">{lights}</span><strong>{label}</strong>'
    table_rows = "".join(
        f'<tr><th scope="row">{escape(str(key))}</th><td>{escape(str(value))}</td></tr>'
        for key, value in rows
    )
    reason_html = f'<p class="report-reason">{escape(str(reason))}</p>' if reason else ""
    return f'''<article class="laim-test-report">
<style>
.laim-test-report {{box-sizing:border-box;width:100%;max-width:100%;min-width:0;margin:12px 0;padding:0;
  background:#fff;color:#1f2937;font:13px/1.48 Inter,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  text-align:left;overflow-wrap:anywhere}}
.laim-test-report * {{box-sizing:border-box}}
.laim-test-report .test-number {{font-size:11.5px;color:#64748b;margin:0 0 4px}}
.laim-test-report h2 {{font-family:inherit;font-size:18px;font-weight:600;line-height:1.4;
  margin:0 0 12px;color:#111827;letter-spacing:normal}}
.laim-test-report h3 {{font-size:13px;font-weight:600;margin:14px 0 5px;color:#374151}}
.laim-test-report p {{margin:5px 0 10px;white-space:normal}}
.laim-test-report table {{border-collapse:collapse;width:100%;min-width:0;max-width:100%;
  table-layout:fixed;margin:8px 0 12px;background:#fff}}
.laim-test-report th,.laim-test-report td {{padding:7px 10px;border:1px solid #ddd;
  font-family:inherit;font-size:12.75px !important;line-height:1.48;font-weight:400 !important;
  text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere;word-break:normal;
  color:#1f2937;background:#fff;letter-spacing:normal}}
.laim-test-report thead th {{background:#f5f5f5;font-weight:600 !important;position:static}}
.laim-test-report th:first-child {{width:58%}}
.laim-test-report td {{font-variant-numeric:tabular-nums;color:#1f2937}}
.laim-test-report tbody tr {{background:#fff}}
.laim-test-report .result-row th,.laim-test-report .result-row td {{background:#fafafa}}
.laim-test-report .signal {{display:inline-flex;gap:3px;vertical-align:middle;margin:0 7px 0 0;
  padding:0;border:0;border-radius:0;background:transparent;box-shadow:none}}
.laim-test-report .lamp {{width:8px;height:8px;border-radius:50%;display:block}}
.laim-test-report .red {{background:#b83c3c}} .laim-test-report .yellow {{background:#d69c1b}}
.laim-test-report .green {{background:#23825b}} .laim-test-report .inactive {{background:#ddd}}
.laim-test-report .report-reason {{color:#475569}}
.laim-test-report details {{border:0;border-radius:0;box-shadow:none;margin:12px 0 0;padding:0;background:#fff}}
.laim-test-report summary {{cursor:pointer;color:#475569;font-size:12px;font-weight:400}}
.laim-test-report summary:focus-visible {{outline:2px solid #334e60;outline-offset:4px}}
.laim-test-report img {{max-width:100%;height:auto}}
@media(max-width:600px) {{.laim-test-report th,.laim-test-report td {{padding:6px 7px}}}}
@media print {{.laim-test-report tr {{break-inside:avoid}}}}
</style>
<p class="test-number">Тест {escape(test_id)} · Этап 4. Контроль качества при эксплуатации</p>
<h2>{escape(title)}</h2>
<h3>Цель теста</h3><p>{escape(purpose)}</p>
<h3>Результаты теста</h3>
<table aria-label="Результаты теста {escape(test_id)}"><thead><tr><th scope="col">Показатель</th>
<th scope="col">Значение</th></tr></thead><tbody>{table_rows}
<tr class="result-row"><th scope="row">Результат теста</th><td>{status}</td></tr></tbody></table>
{reason_html}<h3>Интерпретация результатов</h3><p>{escape(interpretation)}</p>
{chart_html}
<details><summary>Условия проведения и критерии оценки</summary>
<h3>Условия проведения</h3><p>{escape(conditions)}</p>
<h3>Критерии выставления светофора</h3><p>{escape(criteria)}</p></details>
</article>'''
