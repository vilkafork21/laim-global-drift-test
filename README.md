# laim-global-drift-test

Нода мониторингового контура LAIM: тест на глобальный дрифт запросов. Принимает
эталонную корзину (`reference_umr`), размеченный автоассессором мониторинг
(`monitoring_umr` с `main_metric`) и валидированный контракт метрики
(`monitoring_metric`); отдаёт в агрегатор светофор с фактическим значением
ключевой метрики (КМ) на мониторинге и диагностикой влияния семантического
дрифта на КМ.

## Зачем нода нужна

Методика требует ответить на два вопроса: снизилась ли КМ на новых запросах
относительно корзины валидации и связано ли это с тем, что запросы семантически
«уехали». Нода считает КМ по готовым оценкам ассессора и строит регрессию
«удалённость подвыборки от корзины → КМ» по эмбеддингам запросов.

Ключевое проектное решение: **цвет решает факт, а не прогноз**. Светофор
выставляется по абсолютному снижению фактической КМ на мониторинге относительно
корзины (`report_valtest_global_drift_stability`); регрессия — диагностика, её
предсказание в порты не публикуется, публикуются отобранные признаки и
корреляции. Второе решение: серый только когда мерить нечего (нет оценённых
единиц мониторинга, эталон слишком мал); пустой отбор признаков серого не даёт.

## Место в контуре

```text
laim-baskets-adapter.reference_umr ──────────────► reference_umr
laim-asessor-agent.scored_data ──────────────────► monitoring_umr      laim-global-drift-test
laim-kriteria-selector.validated_monitoring_metric ► monitoring_metric
                                                     │
                                                     ├─► all_results      ─► laim-agg.in
                                                     └─► test_description ─► HTML на карточке ноды
                                                                             (в port_wiring не подключён)
```

Схема — `monitoring/shared/port_wiring.json` (`laim-sberds-wiring.v7`) в LAIM.

## Порты и настройки

### Входы (все обязательные, `descriptor.json`)

| Порт | Тип | Что приходит |
|---|---|---|
| `reference_umr` | dataframe | Корзина в формате тестового датасета (`laim-umr.v2`): flat или packed `dialogue`, с колонкой `main_metric` |
| `monitoring_umr` | dataframe | `scored_data` автоассессора: flat UMR с обязательной колонкой `main_metric` (оценка судьи, `NaN` — отказ); принимается также parquet bytes или путь к parquet |
| `monitoring_metric` | default | Контракт `laim-monitoring-metric.v2` (`umr_version: laim-umr.v2`); `assessment_mode` обязателен — по нему определяется единица наблюдения |

### Выходы

| Порт | Тип | Что отдаёт |
|---|---|---|
| `all_results` | default | JSON с цветом, статусом и числами теста (см. «Форматы выхода»); платформа читает `color` и `calculated_traffic_lights.test_light` |
| `test_description` | hidden | HTML-отчёт: цель, алгоритм, критерии светофора, таблицы результата и отобранных признаков |

### Настройки

| Настройка | По умолчанию | Зачем менять |
|---|---|---|
| `n_chunks` | `10` в `descriptor.json`; `MIN_CHUNKS = 5` — значение по умолчанию `main()` при прямом вызове без настройки и жёсткий минимум | Число подвыборок OOSadd. Значение `< 5` — `ValueError`. Это нижняя граница: код поднимает число чанков адаптивно (не меньше 10 строк на чанк, не больше 120, не больше `|OOSadd| // 2`). От переданного значения зависит порог серого: `n_oos < 2 * n_chunks - 1` |
| `p_value` | `0.05` | Уровень FDR Бенджамини—Хохберга при отборе признаков |
| `corr_threshold` | `0.3` | Порог `|r|` Пирсона для «сильных» признаков |
| `metric_agg` | `single_mean` | Не менять: `multicol_mean` возвращает ключ `multicol_mean` вместо `target`, и тест падает с `KeyError: 'target'` |
| `data_types` | `('train', 'test')` | Не менять: семплер заполняет только `train` (OOS) и `test` (OOT) |
| `green_threshold` | `0.15` | Снижение КМ меньше порога — зелёный |
| `red_threshold` | `0.25` | Снижение КМ от порога и выше — красный; пороги нормализуются `min`/`max` |
| `greater_is_better` | `true` | `false` меняет знак: рост КМ считается ухудшением, гейт по OOS тоже переворачивается |
| `random_state` | `42` | Разбиение OOS на base/add (`train_test_split`) |
| `is_info` | `false` | `true` — цвет всегда серый, `status: not_computable`, `reason_code: null` |

## Как проходит прогон

```text
1. Контракт     validate_monitoring_metric(require_computed=False); нет assessment_mode → ошибка
2. Единицы      normalize_umr + _unitize обоих UMR; question/target по assessment_mode; NaN main_metric отброшен
3. Серые гейты  n_oot == 0 → no_scored_monitoring_units; n_oos < 2*n_chunks-1 → insufficient_reference_units
4. Разбиение    OOS → OOSbase / OOSadd (50/50, random_state); адаптивный n_chunks; OOSadd → n_chunks чанков
5. Признаки     эмбеддинги GigaEmbed; D = cosine_distances(chunk, base); 11 признаков: D_q25/q50/q75/q95,
                D_mean, D_std, ADi_q50/q75/q95, ADi_mean, ADi_std (ADi = D.mean(axis=1)); КМ чанка = mean(target)
6. Отбор        Пирсон по неконстантным признакам; BH-FDR по всем тестам; cap = n_chunks // 4;
                пусто → top-cap из |r| > corr_threshold, затем top-cap по |r| (selection_low_confidence)
7. Регрессия    RidgeCV(alphas 0.01..100); OOT по чанкам медианного размера, предсказание усредняется —
                диагностика, в порты не уходит
8. Факт и цвет  metric_value_reference/monitoring = mean(target) по OOS/OOT; цвет по снижению (0.15 / 0.25)
                и гейту по OOS (0.4 / 0.6), худший из двух
9. Публикация   all_results (yellow → amber) + HTML в test_description
```

Пример лога успешного прогона (формат строк — из кода; значения условные):

```text
INFO root: Размер OOS (real): 320
INFO root: Размер OOT (agent): 80
INFO root: Тест на глобальный дрифт запущен
WARNING root: n_chunks=10 мало для 11 фичей; рекомендуется ≥ 44. Регрессия будет регуляризована (Ridge), но статистическая значимость низкая.
INFO root: n_chunks 10 → 16 (адаптив от |add|=160, запрошено 16, потолок 80)
INFO root: Получение эмбеддингов base (n=160)
INFO root: Деление add на 16 чанков
INFO root: Медианный размер чанка: 10
WARNING root: BH-FDR пуст — top-3 из |r|>0.3 (низкая значимость)
INFO root: Отобрано 3 фичей: ['ADi_std', 'ADi_q95', 'ADi_q75'] (low_confidence=True)
INFO root: Выставление светофора по метрике на OOS
```

Предупреждение о 44 чанках исчезает лишь при `n_chunks >= 44`, то есть при
`|OOSadd| >= 440`. Сбой и деградация шлюза эмбеддингов выглядят так:

```text
WARNING root: GigaEmbed: 3 из 160 текстов длиннее 1000 символов — разбиты на части
WARNING root: GigaEmbed batch failed (attempt 1/3): <текст ошибки>; повтор через 1.0s
WARNING root: GigaEmbed: лимит токенов — разбиваю 100 текстов на части
```

## Форматы выхода и контракты

Единица наблюдения задаётся `assessment_mode` контракта (`laim_monitoring/core.py`,
`_drift_frame`): `qa` — строка, `question` = `input_query`; `dialogue` — сессия,
`question` = JSON-список `input_query` всех реплик, `main_metric` константен
внутри сессии; `turn_with_history` — реплика, `question` = история плюс текущий
запрос. `answer` в дрифт не входит. Единица без числового `main_metric` (отказ
судьи) исключается из обоих фреймов до расчётов независимо от `missing_policy`.

`all_results` (`main.py`, `report_valtest_global_drift`):

| Ключ | Значение |
|---|---|
| `color`, `calculated_traffic_lights.test_light` | `red` / `amber` / `green` / `gray` — внутренний `yellow` переведён в `amber` через `_PLATFORM_COLOR` |
| `calculated_traffic_lights.semaphore_title` | текст из `_SEMAPHORE_TITLE` по цвету |
| `status` | `computed`; `not_computable` при сером |
| `reason_code`, `reason` | `null` в обычном прогоне; иначе см. «Падение против деградации» |
| `n_oos`, `n_oot` | число единиц эталона и мониторинга после отброса `NaN` `main_metric`; заполняются на любом исходе |
| `metric_value_reference` | `mean(target)` по OOS, округлено до 3 знаков |
| `metric_value_monitoring` | `mean(target)` по OOT, округлено до 3 знаков; `NaN` при сером |
| `metric_value_source` | `assessor_scored_data` |
| `selected_features`, `feature_correlations` | отобранные признаки; корреляции Пирсона по всем неконстантным признакам |
| `selection_low_confidence` | `true`, если отбор прошёл через фолбэк мимо BH-FDR |
| `n_chunks` | фактическое число чанков после адаптации |
| `test_name` | `global_drift` |

Цвет: `drop = metric_value_reference - metric_value_monitoring`; `drop < 0.15` —
зелёный, `0.15 <= drop < 0.25` — жёлтый, `drop >= 0.25` — красный. Вторая
компонента — цвет самой КМ на OOS из `llm_val/valtest_metric.py` с зашитыми
порогами `(0.4, 0.6)`: `< 0.4` красный, `[0.4, 0.6)` жёлтый, `>= 0.6` зелёный.
Итог — `worst_semaphore` из двух; серый побеждает всё. Корзина с КМ ниже 0.6
никогда не даст зелёный, даже без снижения.

## Падение против деградации

Нода падает (исключение уходит платформе):

| Причина | Исключение |
|---|---|
| Контракт не `laim-monitoring-metric.v2`/`v1`, `umr_version` не `laim-umr.v2`, нет `assessment_mode`, `score_column` не `main_metric` | `MonitoringContractError` |
| UMR пуст; flat UMR с пустым `query_id`; flat и `dialogue` одновременно; нет ни `query_id`/`input_query`/`output_answer`, ни `dialogue`; контекстный режим без `session_id`; turn не из трёх элементов | `MonitoringContractError` |
| Нет колонки `main_metric` в эталоне или мониторинге; `main_metric` не константен внутри сессии | `MonitoringContractError` |
| `n_chunks < 5` | `ValueError` |
| Все запросы OOSbase пустые | `ValueError` |
| `metric_agg = multicol_mean` | `KeyError: 'target'` |
| Шлюз эмбеддингов недоступен после ретраев | `RuntimeError` из `GigaEmbed` |

Деградация (серый или пометка, прогон завершается):

| Событие | Реакция |
|---|---|
| Ни одной единицы мониторинга с числовым `main_metric` | `gray`, `not_computable`, `no_scored_monitoring_units` |
| `n_oos < 2 * n_chunks - 1` (19 при `n_chunks = 10`) | `gray`, `not_computable`, `insufficient_reference_units`, в `reason` — `n_oos` и порог |
| Все признаки константны (эмбеддинги одинаковы) | цвет по факту, `status: computed`, `reason_code: no_significant_features`, регрессия пропущена |
| BH-FDR не отобрал ни одного признака | фолбэк top-K, `selection_low_confidence: true`, WARNING |
| `is_info = true` | `gray`, `not_computable`, `reason_code: null` |
| Часть строк без `main_metric` | отброшены молча; число видно по «Размер OOS/OOT» в логе против исходного датафрейма |

## Внешние сервисы

Единственный сервис — эмбеддинги GigaChat через `giga_wraper.GigaEmbed`
(наследник `gigachat.GigaChat`, вызов `embeddings(batch)`). Конфигурация —
`config.py`, переменные окружения (читается и `.env` через `python-dotenv`):

- `AI_GATEWAY_URL` задан — контур `sds`: клиент получает только
  `base_url = AI_GATEWAY_URL + "/api/v1"`;
- иначе контур `sigma`: `BASE_URL`, `AUTH_URL`, `CREDENTIALS`, `SCOPE`,
  `VERIFY_SSL_CERTS` (строка `True`).

Модель эмбеддингов нода не задаёт — используется значение по умолчанию
библиотеки `gigachat`. Поведение: тексты длиннее 1000 символов режутся на части
по 1000, части идут батчами по 100, вектор текста — среднее векторов его частей;
ошибка с `413` / `tokens limit exceeded` / `payload too large` делит тексты
батча пополам рекурсивно до 200 символов; иные ошибки — до 3 попыток с
backoff 1 с, 2 с, затем `RuntimeError: GigaEmbed: исчерпаны попытки (3)` и
падение ноды. За прогон запрашивается `n_oos + n_oot` векторов в `2 + n_chunks`
обращениях к `get_embedding`. Детерминированность разбиений — `random_state`;
за воспроизводимость векторов отвечает шлюз.

## Наблюдаемость

Порта журнала нет. В лог платформы уходят строки корневого логгера (см. пример
выше): размеры OOS/OOT, итоговое `n_chunks`, отбор признаков, предупреждения
`GigaEmbed`. Источник истины о прогоне — `all_results`: при сотне прогонов
агрегируйте по `color`, `status`, `reason_code`, `selection_low_confidence`,
`n_oos`, `n_oot`, `n_chunks`; пару `metric_value_reference` / `metric_value_monitoring`
сравнивайте с порогами 0.15/0.25 и гейтом 0.6. HTML дублирует числа таблицей.

## Карта кода

```text
main.py                                   порты, настройки, нормализация цвета, сборка all_results и HTML
config.py                                 переменные окружения, выбор контура sigma/sds
giga_wraper.py                            GigaEmbed: чанкование, батчи, ретраи, mean-pooling
html_report_helper.py                     светофоры и таблицы критериев для HTML
laim_monitoring/core.py                   контракт v2, normalize_umr, unitize, prepare_drift_frames
llm_val/valtest_global_drift_stability.py тест: гейты, разбиение, признаки, отбор, RidgeCV, отчёт
llm_val/valtest_metric.py                 цвет КМ на OOS (пороги 0.4/0.6)
llm_val/report_helper.py                  semaphore_by_threshold, worst_semaphore, tricky_semaphore
llm_val/sampler.py                        AutoAsessorSampler: question/answer/target → train/test
llm_val/scorer.py                         AutoAsessorScorer: mean по target
llm_val/utils.py                          METRICS (single_mean, multicol_mean), string_to_float
tests/                                    гейты, contract dialogue/turn_with_history, all_assessors, GigaEmbed
```

## Что делать, если

- **`gray` с `insufficient_reference_units`** — в эталоне меньше `2 * n_chunks - 1`
  единиц с `main_metric` (`n_oos` в `reason`): корзина мала либо `main_metric`
  в `reference_umr` пуст. Уменьшать `n_chunks` ниже 5 нельзя.
- **`gray` с `no_scored_monitoring_units`** — у ассессора нет ни одной
  числовой оценки: смотрите его `assessment_result`, дрифт здесь ни при чём.
- **`amber` при равных `metric_value_reference` и `metric_value_monitoring`** —
  сработал гейт по OOS: КМ корзины ниже 0.6. Это свойство корзины, не дрифт.
- **`selection_low_confidence: true` в каждом прогоне** — норма при малом
  `|OOSadd|`: чанков меньше 44, BH-FDR не проходит. На цвет не влияет.
- **Нода упала с `MonitoringContractError` или `RuntimeError: GigaEmbed`** —
  первое: текст называет поле (чаще `monitoring_metric` без `assessment_mode`
  или UMR без `main_metric`); второе: шлюз недоступен, проверьте
  `AI_GATEWAY_URL` (или `BASE_URL`/`CREDENTIALS` для sigma).

## Деплой

База — `py312-simple`; точка входа — функция `main` в `main.py`. `sourceFiles`
в `descriptor.json` перечисляет 12 файлов: `main.py`, `html_report_helper.py`,
`config.py`, `giga_wraper.py`, шесть модулей `llm_val/` и два `laim_monitoring/`.
Автоматического теста соответствия `sourceFiles` диску в `tests/` нет — сверять
вручную при добавлении файлов. Зависимости `requirements.txt`:
`scikit-learn>=1.0`, `pandas>=1.5`, `scipy>=1.10`, `python-dotenv`, `gigachat`,
`numpy`, `ipython`, `jinja2` (`jinja2` нужен `pandas.Styler`; `ipython`
импортируется с фолбэком). CI (`.github/workflows/ci.yml`): Python 3.12,
`ruff check .`, `python -m pytest -q`. Нода самодостаточна: `laim_monitoring/` —
вендорная копия контракта, импортов из соседних каталогов нет. ZIP собирается
из файлов `sourceFiles` плюс `descriptor.json` и `requirements.txt` с ветки
`dev` данного репозитория.

## Глоссарий

- **OOS / OOT** — out-of-sample (эталонная корзина, `train` семплера) и
  out-of-time (мониторинг, `test` семплера).
- **OOSbase / OOSadd / чанк** — половины OOS: относительно base считаются
  расстояния, add режется на чанки — подвыборки, каждая одна точка регрессии.
- **D / ADi** — матрица косинусных расстояний «чанк × base» и средние
  расстояния каждого запроса чанка до base.
- **BH-FDR** — поправка Бенджамини—Хохберга на множественные сравнения.
- **КМ / `main_metric`** — ключевая метрика качества; в дрифт-фрейме — колонка
  `target`, среднее по ней и есть значение КМ.
- **Гейт по OOS** — цвет самой КМ на корзине с порогами 0.4/0.6 (см. «Форматы выхода»).
