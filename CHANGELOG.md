# Global Drift test — CHANGELOG of corrections

Применены правки из `02_global_drift_test_analysis.md`. ID соответствуют приоритетам в анализе.

## P0 — критические

| ID | Что | Файл |
|----|-----|------|
| P0-1 | Полностью убраны lambda-замыкания — заменены на фиксированный, дедуплицированный, scale-aware набор фичей через `_make_feature_extractors()` (11 фичей: 6 от полной D, 5 от ADi). Закрывает и P1-3 одновременно | `llm_val/valtest_global_drift_stability.py` |
| P0-2 | `n_chunks` приведён к единому значению (default 10 в дескрипторе), минимум `MIN_CHUNKS=5` через жёсткий `raise ValueError`. Согласовано между `main.py` и `descriptor.json` | `valtest_global_drift_stability.py`, `main.py`, `descriptor.json` |
| P0-3 | Регуляризованная регрессия (RidgeCV вместо LinearRegression) + Bonferroni-поправка на pvalue (`α/n_features`) — устраняет переобучение и multiple-testing | `valtest_global_drift_stability.py` |
| P0-4 | Все словари переписаны на ключ `"gray"`; baseline `"grey"` → `"gray"` | `main.py` |
| P0-5 | `pearsonr` вызывается только если `np.std(features[:, j]) > 0`; константные столбцы скипаются; результат хеджируется `np.isfinite` | `valtest_global_drift_stability.py` |

## P1 — серьёзные

| ID | Что | Файл |
|----|-----|------|
| P1-1 | Defaults согласованы с HTML: `red_threshold=0.25`, `green_threshold=0.15`, `p_value=0.05`, `corr_threshold=0.3`; в `main()` пороги нормализуются `min/max` | `main.py`, `descriptor.json` |
| P1-2 | Предсказание OOT идёт **по чанкам того же размера**, что в тренировке (`_predict_oot_by_chunks`); устраняет scale-bias фичей | `valtest_global_drift_stability.py` |
| P1-3 | Закрыто вместе с P0-1: дубликаты `mean(D)` vs `mean(D.mean(1))` устранены явной фиксированной схемой фичей | `valtest_global_drift_stability.py` |
| P1-4 | `np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)` перед регрессией | `valtest_global_drift_stability.py` |
| P1-5 | Не используем `sampler_copy.test["y"]` как буфер для chunk_Y; написан `_scorer_calc_on_y` через лёгкий локальный sampler | `valtest_global_drift_stability.py` |
| P1-6 | `random_state` выведен в UI | `main.py`, `descriptor.json` |
| P1-7 | `GigaEmbed`: батчинг по `DEFAULT_BATCH_SIZE=100`, retry до 3 раз с экспоненциальным backoff | `giga_wraper.py` |
| P1-8 | `worst_semaphore` обрабатывает gray как «худший» сценарий — если в списке есть gray, возвращает gray | `llm_val/report_helper.py` |
| P1-9 | `dropna(subset=[main_metric])` (список вместо строки) | `main.py` |
| P1-10 | После `.rename(columns={main_metric: "target"})` переменная `main_metric` явно перезаписывается на `"target"` | `main.py` |

## P2 — улучшения

| ID | Что | Файл |
|----|-----|------|
| P2-1 | `LinearRegression` → `RidgeCV(alphas=(0.01, 0.1, 1, 10, 100))`. Предсказание клиппуется в [0, 1] если метрика в этом диапазоне | `valtest_global_drift_stability.py` |
| P2-4 | Валидация эмбеддингов: фильтр пустых question, `np.nan_to_num` на эмбеддингах | `valtest_global_drift_stability.py` |
| P2-5 | В отчёт добавлена таблица «Отобранные фичи + корреляции Пирсона»; список фичей и корреляций пробрасывается через `precomputed` | `valtest_global_drift_stability.py`, `main.py` |

## P3 — гигиена

| ID | Что | Файл |
|----|-----|------|
| P3-1 | Опечатка «вычеслена» → «вычислена» | `valtest_global_drift_stability.py` |
| P3-2 | `string_to_float` ловит специфичные исключения; `_convert_history_to_qa` логирует пропуски | `llm_val/utils.py`, `llm_val/sampler.py` |
| P3-4 | `METRICS` определён только в `utils.py`; `main.py` импортирует оттуда | `main.py`, `llm_val/utils.py` |

## Не реализовано (R&D-роадмап)

- **P2-2**: альтернативная архитектура через MMD/Sliced Wasserstein — крупная задача, заявка в R&D-роадмап LAIM.
- **P2-3**: абстракция `EmbeddingModel` для поддержки множества провайдеров — отдельная задача.
- **P2-6**: расширение `METRICS` (precision/recall/F1/RMSE) — отдельная задача в коннекте с автоассессором.
- **P2-7**: time-based split OOSadd — требует наличия колонки даты в данных, нужно скоординировать с владельцами data contract.
- **P3-3**: унификация сигнатуры `show_criteria_semaphore` — в текущем тесте уже 5 аргументов (включая grey_criterion), что соответствует обновлённой схеме.
- **P3-5**: упаковка `test_example` в Python-пакет — не блокер, оставлен `sys.path.insert` как минимально инвазивный костыль.

## Поведенческие изменения, требующие внимания

1. **`n_chunks < 5` теперь поднимает `ValueError`** — раньше тест выдавал NaN и серый светофор молча. Это намеренно: пользователь должен исправить конфигурацию.
2. **Параметр `top_distance_features` удалён** — фичи фиксированы. Существующие конфигурации в SberDS потребуют пересохранения.
3. **`metric_value` всегда округляется до 3 знаков** для отображения; в `precomputed` остаётся исходное значение.
4. **При отсутствии отобранных фичей** тест возвращает gray (как и раньше), но теперь явно с пометкой в отчёте.
5. **GigaEmbed теперь делает retry** — при сетевом сбое запуск займёт дольше, но не упадёт сразу.
