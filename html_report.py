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
.laim-test-report {{box-sizing:border-box;max-width:980px;margin:24px auto;padding:32px 40px;
  background:#fff;color:#24323b;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
  border:1px solid #dce2e5;border-radius:6px;text-align:left}}
.laim-test-report * {{box-sizing:border-box}}
.laim-test-report .test-number {{font-size:13px;color:#586b77;margin:0 0 6px}}
.laim-test-report h2 {{font-size:27px;line-height:1.25;letter-spacing:-.4px;margin:0 0 12px;color:#192b36}}
.laim-test-report h3 {{font-size:16px;margin:24px 0 8px;color:#192b36}}
.laim-test-report p {{margin:8px 0 14px}}
.laim-test-report .test-type {{font-size:13px;color:#586b77}}
.laim-test-report table {{border-collapse:collapse;width:100%;table-layout:fixed;margin:10px 0 20px}}
.laim-test-report th,.laim-test-report td {{padding:11px 16px;border-bottom:1px solid #dce2e5;
  text-align:left;vertical-align:top;overflow-wrap:anywhere;font-weight:400}}
.laim-test-report thead th {{background:#eaf0f3;font-weight:600}}
.laim-test-report th:first-child {{width:60%}}
.laim-test-report td {{font-variant-numeric:tabular-nums;color:#192b36}}
.laim-test-report .result-row th,.laim-test-report .result-row td {{background:#f5f8f9;font-weight:600}}
.laim-test-report .signal {{display:inline-flex;gap:4px;padding:5px 7px;border:1px solid #b7c3c9;
  border-radius:16px;vertical-align:middle;margin-right:9px;background:#fff}}
.laim-test-report .lamp {{width:10px;height:10px;border-radius:50%;display:block}}
.laim-test-report .red {{background:#b83c3c}} .laim-test-report .yellow {{background:#d69c1b}}
.laim-test-report .green {{background:#23825b}} .laim-test-report .inactive {{background:#dce2e5}}
.laim-test-report .report-reason {{padding:10px 14px;background:#f5f8f9;border-left:3px solid #82939e}}
.laim-test-report details {{border-top:1px solid #dce2e5;margin-top:24px;padding-top:14px}}
.laim-test-report summary {{cursor:pointer;color:#334e60;font-weight:600}}
.laim-test-report summary:focus-visible {{outline:2px solid #334e60;outline-offset:4px}}
.laim-test-report img {{max-width:100%;height:auto}}
@media(max-width:600px) {{.laim-test-report {{margin:8px auto;padding:20px 16px}}
  .laim-test-report h2 {{font-size:23px}} .laim-test-report th,.laim-test-report td {{padding:9px 8px}}
  .laim-test-report th:first-child {{width:56%}}}}
@media print {{.laim-test-report {{border:0;margin:0;padding:0;max-width:none}}
  .laim-test-report tr {{break-inside:avoid}}}}
</style>
<p class="test-number">Тест {escape(test_id)} · Этап 4. Контроль качества при эксплуатации</p>
<h2>{escape(title)}</h2>
<p class="test-type">Базовый · Количественный</p>
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
