"""
Модуль теста на глобальный дрифт запросов.

Анализирует семантический дрифт между OOS и OOT через статистики
эмбеддингов и предсказывает ключевую метрику качества с помощью
регуляризованной регрессии (Ridge).
"""

import logging
import typing as tp
from copy import deepcopy

import numpy as np
import pandas as pd
from llm_val.report_helper import semaphore_by_threshold, worst_semaphore
from llm_val.sampler import Sampler
from llm_val.scorer import Scorer
from llm_val.valtest_metric import valtest_metric
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split


# Минимальное число чанков для статистически осмысленной корреляции (P0-5).
MIN_CHUNKS = 5

# Если задано n_chunks < MIN_CHUNKS, тест поднимет ошибку.
# Verify против переобучения: n_chunks должно быть в ~4 раза больше числа фичей.
MIN_CHUNKS_PER_FEATURE = 4

# Адаптивный n_chunks: фиксированные 5 чанков делали отбор фичей статистически
# невозможным (Bonferroni на 5 точках требует |r|≈0.96) — «вечный серый» отбора.
# Число чанков масштабируется от объёма add-выборки: на чанк ≥ MIN_CHUNK_ROWS
# строк, всего не более MAX_CHUNKS.
MIN_CHUNK_ROWS = 10
MAX_CHUNKS = 120


# =============================================================================
# ФИЧИ: единый, дедуплицированный набор scale-aware агрегатов
# =============================================================================

def _make_feature_extractors() -> tp.List[tp.Tuple[str, tp.Callable]]:
    """
    Набор фичей для извлечения из матрицы расстояний D (chunk_size × base_size).

    Половина фичей — статистики D как целого (глобальный «градус» удалённости
    выборки), половина — статистики ADi = D.mean(axis=1) (per-query средние
    расстояния, «локальные» характеристики).

    Все фичи перцентильные или mean/std — устойчивые при изменении размера выборки
    (P1-2, P1-3 — нет дубликатов и зашкаливающих max/topK).
    """
    extractors: tp.List[tp.Tuple[str, tp.Callable]] = []

    # Статистики полной матрицы
    for q in (25, 50, 75, 95):
        extractors.append((f"D_q{q}", lambda D, q=q: float(np.percentile(D, q))))
    extractors.append(("D_mean", lambda D: float(np.mean(D))))
    extractors.append(("D_std", lambda D: float(np.std(D))))

    # Статистики per-query средних (ADi)
    def _adi(D):
        return D.mean(axis=1)

    for q in (50, 75, 95):
        extractors.append((f"ADi_q{q}", lambda D, q=q: float(np.percentile(_adi(D), q))))
    extractors.append(("ADi_mean", lambda D: float(np.mean(_adi(D)))))
    extractors.append(("ADi_std", lambda D: float(np.std(_adi(D)))))

    return extractors


def _extract_features(D: np.ndarray, extractors) -> np.ndarray:
    return np.asarray([fn(D) for _, fn in extractors], dtype=float)


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТОВ
# =============================================================================

def report_valtest_global_drift_stability(
    metric_value: tp.Dict[str, float],
    metric_value_estimate: float,
    main_metric: str,
    test_color: str,
    selected_features: tp.List[str],
    feature_correlations: tp.Dict[str, float],
    data_types: tp.Tuple[str, str] = ("train", "test"),
    semaphore_threshold: tp.Tuple[float, float] = (0.15, 0.25),
    greater_is_better: bool = True,
    is_info: bool = False,
) -> tp.Dict[str, tp.Any]:
    """
    Создание отчёта по результатам теста на глобальный дрифт.
    """
    metric_value_scalar = metric_value[main_metric]

    if is_info or np.isnan(metric_value_estimate):
        color = "gray"
    else:
        abs_diff = metric_value_estimate - metric_value_scalar
        if greater_is_better:
            abs_diff = -abs_diff
        color_by_metric = semaphore_by_threshold(
            abs_diff, semaphore_threshold, greater_is_better=False
        )
        color = worst_semaphore([test_color, color_by_metric])

    df = pd.DataFrame(
        {
            f"Значение метрики на {data_types[0]}": [round(float(metric_value_scalar), 4)],
            f"Фактическое значение метрики на {data_types[1]}": [
                round(float(metric_value_estimate), 4) if not np.isnan(metric_value_estimate) else None
            ],
            "Абсолютная разница": [
                round(float(metric_value_estimate - metric_value_scalar), 4)
                if not np.isnan(metric_value_estimate) else None
            ],
            "Результат теста": [color],
        }
    )

    # Таблица отобранных фичей (P2-5)
    features_df = pd.DataFrame(
        {
            "Фича": selected_features,
            "Корреляция Пирсона": [
                round(float(feature_correlations.get(name, np.nan)), 4)
                for name in selected_features
            ],
        }
    )

    return {
        "semaphore": color,
        "result_plots": [],
        "result_dataframes": [df, features_df],
    }


def _not_computable_result(
    *,
    main_metric: str,
    data_types: tp.Tuple[str, str],
    semaphore_threshold: tp.Tuple[float, float],
    greater_is_better: bool,
    reason_code: str,
    reason: str,
    n_oos: int,
    n_oot: int,
) -> tp.Dict[str, tp.Any]:
    metric_value = {main_metric: np.nan}
    report = report_valtest_global_drift_stability(
        metric_value=metric_value,
        metric_value_estimate=np.nan,
        main_metric=main_metric,
        test_color="gray",
        selected_features=[],
        feature_correlations={},
        data_types=data_types,
        semaphore_threshold=semaphore_threshold,
        greater_is_better=greater_is_better,
        is_info=True,
    )
    return {
        "report": report,
        "precomputed": {
            "status": "not_computable",
            "reason_code": reason_code,
            "reason": reason,
            "n_oos": n_oos,
            "n_oot": n_oot,
            "metric_value": np.nan,
            "metric_value_estimate": np.nan,
            "selected_features": [],
            "feature_correlations": {},
        },
    }


# =============================================================================
# ОСНОВНОЙ ТЕСТ
# =============================================================================

def valtest_global_drift_stability(
    sampler: Sampler,
    scorer: Scorer,
    main_metric: str,
    model: tp.Any,
    n_chunks: int = MIN_CHUNKS,
    distance_func: tp.Callable = cosine_distances,
    p_value: float = 0.05,
    corr_threshold: float = 0.3,
    random_state: int = 42,
    metric_binarizer: tp.Optional[tp.Callable] = None,
    metric_agg: str = "single_mean",
    data_types: tp.Tuple[str, str] = ("train", "test"),
    semaphore_threshold: tp.Tuple[float, float] = (0.15, 0.25),
    greater_is_better: bool = True,
    is_info: bool = False,
    metric_value: tp.Optional[tp.Dict[str, float]] = None,
    test_color: tp.Optional[str] = None,
    metric_value_estimate: tp.Optional[float] = None,
    **kwargs,
) -> tp.Dict[str, tp.Any]:
    """
    Тест на анализ качества ответа модели в зависимости от глобального
    семантического дрифта запросов.

    Алгоритм:
    1. Делим OOS на base / add. На base считаем эмбеддинги.
    2. Делим add на n_chunks подвыборок. Для каждой:
       — считаем матрицу cosine_distances до base;
       — извлекаем scale-aware фичи;
       — считаем ключевую метрику на этом чанке.
    3. Отбираем фичи: корреляция Пирсона + BH-FDR, при пустом отборе —
       top-K по |r| с пометкой низкой значимости.
    4. Обучаем RidgeCV: features → metric.
    5. На OOT извлекаем фичи **по чанкам того же размера** (P1-2) →
       предсказываем метрику чанк-за-чанком → усредняем.
    6. Сравниваем metric_estimate с metric на OOS → итоговый светофор.
    """
    extractors = _make_feature_extractors()
    n_features = len(extractors)
    feature_names = [name for name, _ in extractors]

    # P0-2/P0-5: жёсткое требование на n_chunks
    if n_chunks < MIN_CHUNKS:
        raise ValueError(
            f"n_chunks={n_chunks} < {MIN_CHUNKS}: корреляционный тест статистически неинформативен"
        )
    if n_chunks < MIN_CHUNKS_PER_FEATURE * n_features:
        logging.warning(
            f"n_chunks={n_chunks} мало для {n_features} фичей; "
            f"рекомендуется ≥ {MIN_CHUNKS_PER_FEATURE * n_features}. "
            "Регрессия будет регуляризована (Ridge), но статистическая значимость низкая."
        )

    selected_features: tp.List[str] = []
    feature_correlations: tp.Dict[str, float] = {}
    selection_low_confidence = False
    drift_metric_value_estimate = metric_value_estimate
    metric_value_source = "precomputed"

    if metric_value_estimate is None:
        logging.info("Метрика на новых данных не вычислена заранее — начало вычисления")
        sampler_copy = deepcopy(sampler)
        if metric_binarizer is not None:
            train_data = getattr(sampler_copy, data_types[0])
            test_data = getattr(sampler_copy, data_types[1])
            setattr(sampler_copy, data_types[0], {
                "X": train_data["X"],
                "y": pd.DataFrame({"metric_value": metric_binarizer(train_data["y"])}),
            })
            setattr(sampler_copy, data_types[1], {
                "X": test_data["X"],
                "y": pd.DataFrame({"metric_value": metric_binarizer(test_data["y"])}),
            })

        data_train = getattr(sampler_copy, data_types[0])
        data_test = getattr(sampler_copy, data_types[1])
        if len(data_test["X"]) == 0:
            return _not_computable_result(
                main_metric=main_metric,
                data_types=data_types,
                semaphore_threshold=semaphore_threshold,
                greater_is_better=greater_is_better,
                reason_code="no_scored_monitoring_units",
                reason=(
                    "scored_data не содержит ни одной оценённой единицы "
                    "мониторинга (main_metric пуст)"
                ),
                n_oos=len(data_train["X"]),
                n_oot=0,
            )
        minimum_oos = 2 * n_chunks - 1
        if len(data_train["X"]) < minimum_oos:
            return _not_computable_result(
                main_metric=main_metric,
                data_types=data_types,
                semaphore_threshold=semaphore_threshold,
                greater_is_better=greater_is_better,
                reason_code="insufficient_reference_units",
                reason=(
                    "Недостаточно независимых единиц OOS для разбиения: "
                    f"n_oos={len(data_train['X'])} < {minimum_oos} "
                    f"при n_chunks={n_chunks}"
                ),
                n_oos=len(data_train["X"]),
                n_oot=len(data_test["X"]),
            )
        base_X, add_X, base_Y, add_Y = train_test_split(
            data_train["X"], data_train["y"], test_size=0.5, random_state=random_state
        )

        # Адаптивный n_chunks от объёма add: больше данных → больше точек для
        # корреляции. Явно заданное БОЛЬШЕЕ n_chunks уважается, но клампится
        # так, чтобы в чанке было ≥ 2 строк.
        adaptive = max(MIN_CHUNKS, min(MAX_CHUNKS, len(add_X) // MIN_CHUNK_ROWS))
        requested = max(n_chunks, adaptive)
        feasible_max = max(MIN_CHUNKS, min(MAX_CHUNKS, len(add_X) // 2))
        final_chunks = min(requested, feasible_max)
        if final_chunks != n_chunks:
            logging.info(
                f"n_chunks {n_chunks} → {final_chunks} (адаптив от |add|={len(add_X)}, "
                f"запрошено {requested}, потолок {feasible_max})"
            )
            n_chunks = final_chunks

        # P2-4: фильтр пустых question
        base_questions = [
            (i, q) for i, q in zip(base_X.index, base_X["question"].astype(str).tolist())
            if q and not q.isspace()
        ]
        if not base_questions:
            raise ValueError("Все base-вопросы пусты после фильтрации")
        base_idx_kept, base_q_list = zip(*base_questions)
        base_X = base_X.loc[list(base_idx_kept)]
        base_Y = base_Y.loc[list(base_idx_kept)]

        logging.info(f"Получение эмбеддингов base (n={len(base_q_list)})")
        base_embeddings = np.asarray(model.get_embedding(list(base_q_list)), dtype=float)
        if np.isnan(base_embeddings).any():
            logging.warning("В base_embeddings есть NaN — заменяю на 0")
            base_embeddings = np.nan_to_num(base_embeddings, nan=0.0)

        logging.info(f"Деление add на {n_chunks} чанков")
        chunk_idx_splits = np.array_split(add_X.index.values, n_chunks)
        chunk_size_avg = int(np.median([len(s) for s in chunk_idx_splits]))
        logging.info(f"Медианный размер чанка: {chunk_size_avg}")

        features = np.zeros((n_chunks, n_features), dtype=float)
        test_metric_values = np.zeros(n_chunks, dtype=float)

        for k, chunk_idx in enumerate(chunk_idx_splits):
            chunk_X = add_X.loc[chunk_idx]
            chunk_Y = add_Y.loc[chunk_idx]
            chunk_questions = chunk_X["question"].astype(str).tolist()
            chunk_embeddings = np.asarray(model.get_embedding(chunk_questions), dtype=float)
            if np.isnan(chunk_embeddings).any():
                chunk_embeddings = np.nan_to_num(chunk_embeddings, nan=0.0)

            D = distance_func(chunk_embeddings, base_embeddings)
            features[k] = _extract_features(D, extractors)

            # P1-5: считаем метрику напрямую, без подмены sampler_copy.test
            test_metric_values[k] = _scorer_calc_on_y(scorer, chunk_Y, metric_agg, main_metric)

        # P0-1: устранён lambda-closure баг (никаких лишних замыканий)
        # P2-4: NaN-safety
        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Отбор фичей: BH-FDR + гарантированный top-K фолбэк. Пустой отбор не
        # нужен: светофор считается по фактической метрике scored_data, фичи —
        # диагностика дрифта. cap — анти-переобучение регрессии.
        logging.info("Отбор фичей: Пирсон + BH-FDR (+ top-K фолбэк)")
        all_corr = []
        for j in range(n_features):
            if np.std(features[:, j]) == 0.0:
                continue  # константный столбец — корреляция не определена
            corr_res = stats.pearsonr(features[:, j], test_metric_values)
            if not (np.isfinite(corr_res.statistic) and np.isfinite(corr_res.pvalue)):
                continue
            feature_correlations[feature_names[j]] = float(corr_res.statistic)
            all_corr.append(
                (feature_names[j], float(corr_res.statistic), float(corr_res.pvalue))
            )

        cap = max(1, min(n_chunks // MIN_CHUNKS_PER_FEATURE, len(all_corr)))
        strong = [c for c in all_corr if abs(c[1]) > corr_threshold]
        if all_corr:
            # BH по всем протестированным фичам: m = все тесты, иначе поправка
            # анти-консервативна («double dipping»).
            tested_sorted = sorted(all_corr, key=lambda c: c[2])
            m = len(tested_sorted)
            bh_k = 0
            for i, (_, _, p) in enumerate(tested_sorted, start=1):
                if p <= (i / m) * p_value:
                    bh_k = i
            bh_passed = {c[0] for c in tested_sorted[:bh_k]}
            significant = [c for c in strong if c[0] in bh_passed]
            selected_features = [
                c[0] for c in sorted(significant, key=lambda c: -abs(c[1]))[:cap]
            ]
        if not selected_features and strong:
            selection_low_confidence = True
            selected_features = [
                c[0] for c in sorted(strong, key=lambda c: -abs(c[1]))[:cap]
            ]
            logging.warning(
                f"BH-FDR пуст — top-{len(selected_features)} из |r|>{corr_threshold} "
                "(низкая значимость)"
            )
        if not selected_features and all_corr:
            selection_low_confidence = True
            selected_features = [
                c[0] for c in sorted(all_corr, key=lambda c: -abs(c[1]))[:cap]
            ]
            logging.warning(
                f"Ни одна фича не прошла |r|>{corr_threshold} — гарантированный "
                f"top-{len(selected_features)} по |r| (низкая значимость)"
            )

        logging.info(
            f"Отобрано {len(selected_features)} фичей: {selected_features} "
            f"(low_confidence={selection_low_confidence})"
        )

        # P2-1: Ridge с CV alpha вместо LinearRegression
        if selected_features:
            sel_idx = [feature_names.index(n) for n in selected_features]
            X_train = features[:, sel_idx]
            # alphas покрывают диапазон от слабой к сильной регуляризации
            ridge = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0)).fit(X_train, test_metric_values)

            # P1-2: предсказание OOT по чанкам того же размера
            metric_value_estimate = _predict_oot_by_chunks(
                data_test["X"], model, base_embeddings, distance_func,
                extractors, sel_idx, ridge, chunk_size_avg
            )

            # Если метрика в [0, 1], клиппуем
            if 0.0 <= test_metric_values.min() and test_metric_values.max() <= 1.0:
                metric_value_estimate = float(np.clip(metric_value_estimate, 0.0, 1.0))
        else:
            metric_value_estimate = float("nan")

        drift_metric_value_estimate = metric_value_estimate
        metric_value_estimate = _scorer_calc_on_y(
            scorer, data_test["y"], metric_agg, main_metric
        )
        metric_value_source = "assessor_scored_data"

        if metric_value is None or test_color is None:
            logging.info("Выставление светофора по метрике на OOS")
            metric_result = valtest_metric(
                sampler=sampler_copy,
                scorer=scorer,
                main_metric=main_metric,
                data_type=data_types[0],
                metric_agg=metric_agg,
                greater_is_better=greater_is_better,
            )
            test_color = metric_result["report"]["semaphore"]
            metric_value = metric_result["precomputed"]["metric_value"]

    precomputed = {
        "metric_value": metric_value[main_metric],
        "metric_value_estimate": metric_value_estimate,
        "drift_metric_value_estimate": drift_metric_value_estimate,
        "metric_value_source": metric_value_source,
        "selected_features": selected_features,
        "feature_correlations": feature_correlations,
        "selection_low_confidence": selection_low_confidence,
        "n_chunks": int(n_chunks),
        # Знаменатели вердикта нужны потребителю на любом исходе, не только
        # при отказе от отбора признаков.
        "n_oos": len(getattr(sampler, data_types[0])["X"]),
        "n_oot": len(getattr(sampler, data_types[1])["X"]),
    }
    if not selected_features:
        precomputed.update({
            "status": "computed",
            "reason_code": "no_significant_features",
            "reason": (
                "Метрика мониторинга рассчитана по scored_data ассессора; "
                "признаки расстояний константны — диагностика влияния "
                "дрифта недоступна"
            ),
        })
    logging.info("Начало составления отчёта")
    report = report_valtest_global_drift_stability(
        metric_value=metric_value,
        metric_value_estimate=metric_value_estimate,
        main_metric=main_metric,
        test_color=test_color,
        selected_features=selected_features,
        feature_correlations=feature_correlations,
        data_types=data_types,
        semaphore_threshold=semaphore_threshold,
        greater_is_better=greater_is_better,
        is_info=is_info,
    )
    return {"report": report, "precomputed": precomputed}


# =============================================================================
# ВНУТРЕННИЕ ХЕЛПЕРЫ
# =============================================================================

def _scorer_calc_on_y(scorer: Scorer, y_chunk: pd.DataFrame, metric_agg: str, main_metric: str) -> float:
    """
    Вычисление метрики напрямую из y_chunk без подмены sampler'а (P1-5).
    Используем minimal API совместимый со scorer.calc (через временный одиночный
    sampler).
    """
    # scorer.calc принимает sampler — создадим лёгкий
    class _LocalSampler:
        train = {"X": None, "y": y_chunk.reset_index(drop=True).rename(columns={y_chunk.columns[0]: main_metric})}
    local_sampler = _LocalSampler()
    res = scorer.calc(sampler=local_sampler, data_type="train", metric_name=metric_agg)
    return float(res[main_metric])


def _predict_oot_by_chunks(
    oot_X: pd.DataFrame,
    model,
    base_embeddings: np.ndarray,
    distance_func,
    extractors,
    selected_indices: tp.List[int],
    regressor,
    chunk_size: int,
) -> float:
    """
    Предсказание метрики на OOT по чанкам того же размера, что в тренировке (P1-2).
    Усредняем предсказания по чанкам — устраняет scale-bias фичей.
    """
    if chunk_size <= 0:
        chunk_size = max(1, len(oot_X))
    n_oot = len(oot_X)
    if n_oot == 0:
        return float("nan")

    questions = oot_X["question"].astype(str).tolist()
    # P2-4: эмбеддинги одним батчем (батчинг — внутри GigaEmbed после P1-7)
    embeddings = np.asarray(model.get_embedding(questions), dtype=float)
    if np.isnan(embeddings).any():
        embeddings = np.nan_to_num(embeddings, nan=0.0)

    predictions = []
    for start in range(0, n_oot, chunk_size):
        chunk_emb = embeddings[start:start + chunk_size]
        if len(chunk_emb) == 0:
            continue
        D = distance_func(chunk_emb, base_embeddings)
        feats = _extract_features(D, extractors)
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        pred = regressor.predict(feats[selected_indices].reshape(1, -1))[0]
        predictions.append(pred)

    if not predictions:
        return float("nan")
    return float(np.mean(predictions))
