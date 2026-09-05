"""Валидация контракта, единицы наблюдения и Decimal-агрегация."""

from __future__ import annotations

from measurement import MeasurementError, validate_measurement

import ast
import json
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path

import pandas as pd

_ASSESSMENT_MODES = {"qa", "turn_with_history", "dialogue"}


class MonitoringContractError(ValueError):
    """Контракт или данные не позволяют получить однозначный результат."""


def _require(mapping: dict, name: str, expected=None):
    if name not in mapping:
        raise MonitoringContractError(f"monitoring_metric не содержит {name}")
    value = mapping[name]
    if expected is not None and value not in expected:
        raise MonitoringContractError(f"Недопустимое {name}: {value!r}")
    return value


def _decimal(value: object, name: str = "value") -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MonitoringContractError(f"{name} не является Decimal: {value!r}") from exc
    if not result.is_finite():
        raise MonitoringContractError(f"{name} должно быть конечным")
    return result


def validate_monitoring_metric(payload: object, *, require_computed: bool = True) -> dict:
    if not isinstance(payload, dict):
        raise MonitoringContractError("monitoring_metric должен быть JSON object")
    try:
        contract = validate_measurement(payload, require_computed=require_computed)
    except MeasurementError as exc:
        raise MonitoringContractError(str(exc)) from exc
    return contract


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _key(value: object) -> tuple[str, str]:
    if isinstance(value, bool):
        return "bool", str(value)
    if isinstance(value, (int, float, Decimal)) and not _blank(value):
        return "number", str(_decimal(value).normalize())
    return "text", str(value).strip().casefold()


def _constant(values: list[object], name: str) -> object | None:
    present = [value for value in values if not _blank(value)]
    if not present:
        return None
    if len({_key(value) for value in present}) != 1:
        raise MonitoringContractError(f"{name} не константен внутри dialogue")
    return present[0]


def _source_values(row, sources: list[dict]) -> dict[str, object]:
    result = {}
    for source in sources:
        column = source["column_name"]
        if column in row.index:
            result[source["source_id"]] = row[column]
    return result


def _turn_order(value: object) -> int | None:
    """Привести turn_index из любого транспорта (int/float/str) к int >= 1."""
    if _blank(value) or isinstance(value, bool):
        return None
    try:
        decimal_index = _decimal(value)
    except MonitoringContractError:
        return None
    if decimal_index != decimal_index.to_integral_value() or decimal_index < 1:
        return None
    return int(decimal_index)


def _ordered_groups(frame: pd.DataFrame) -> list[tuple[object, list[int]]]:
    """Группы в порядке появления; внутри группы — по turn_index.

    Транспорт между портами меняет типы и теряет колонки, поэтому данные
    приводятся и достраиваются: непригодный или отсутствующий turn_index
    заменяется порядком строк, строка без группы становится своей группой.
    """
    group_values = (
        frame["reference_group_id"].tolist()
        if "reference_group_id" in frame else [None] * len(frame)
    )
    order_values = (
        frame["turn_index"].tolist()
        if "turn_index" in frame else [None] * len(frame)
    )
    groups: dict[object, tuple[object, list[tuple[int | None, int]]]] = {}
    for position, (group, raw_index) in enumerate(zip(group_values, order_values)):
        key = ("solo", position) if _blank(group) else (
            ("text", group.strip()) if isinstance(group, str) else _key(group)
        )
        label = f"row-{position}" if _blank(group) else group
        groups.setdefault(key, (label, []))[1].append((_turn_order(raw_index), position))
    result = []
    for label, indexed_positions in groups.values():
        indexes = [index for index, _position in indexed_positions]
        usable = (
            all(index is not None for index in indexes)
            and sorted(indexes) == list(range(1, len(indexes) + 1))
        )
        positions = (
            [position for _index, position in sorted(indexed_positions)]
            if usable else [position for _index, position in indexed_positions]
        )
        result.append((label, positions))
    return result


def _turn(row, order: int) -> dict[str, object]:
    return {
        "turn_index": order,
        "input_query": "" if _blank(row["input_query"]) else str(row["input_query"]),
        "output_answer": "" if _blank(row["output_answer"]) else str(row["output_answer"]),
    }


def _restore_array_delimiters(text: str) -> str:
    """str() многоходового ndarray разделяет внешние array(...) пробелом или
    переводом строки без запятой — вставляем запятые, не трогая содержимое
    строковых литералов (текст реплики может содержать ") array(")."""
    result = []
    quote = None
    escaped = False
    for position, char in enumerate(text):
        result.append(char)
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == ")":
            tail = position + 1
            while tail < len(text) and text[tail].isspace():
                tail += 1
            if text.startswith("array(", tail) and tail > position + 1:
                result.append(",")
    return "".join(result)


def _parse_sequence_text(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    for candidate in dict.fromkeys((text, _restore_array_delimiters(text))):
        try:
            return _numpy_literal(ast.parse(candidate, mode="eval"))
        except (SyntaxError, ValueError):
            continue
    return text


def _sequence(value: object, name: str) -> list[object]:
    if isinstance(value, str) and value.strip():
        value = _parse_sequence_text(value.strip())
    if not isinstance(value, (list, tuple)):
        tolist = getattr(value, "tolist", None)
        value = tolist() if callable(tolist) else value
    if not isinstance(value, (list, tuple)):
        raise MonitoringContractError(f"{name} должен быть последовательностью")
    return list(value)


def _numpy_literal(node: ast.AST) -> object:
    """Безопасно разбирает repr вложенных numpy array из parquet transport."""
    if isinstance(node, ast.Expression):
        return _numpy_literal(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_numpy_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_numpy_literal(item) for item in node.elts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "array"
        and len(node.args) == 1
        and all(
            keyword.arg == "dtype"
            and isinstance(keyword.value, (ast.Name, ast.Constant))
            for keyword in node.keywords
        )
    ):
        return _numpy_literal(node.args[0])
    raise ValueError("неподдерживаемый строковый literal")


def _unique_labels() -> "callable":
    """Счётчик меток: пустое значение получает fallback, повтор — суффикс #N."""
    seen: dict[tuple[str, str], int] = {}

    def label(value: object, fallback: str) -> object:
        result = fallback if _blank(value) else value
        key = _key(result)
        seen[key] = seen.get(key, 0) + 1
        return result if seen[key] == 1 else f"{result}#{seen[key]}"

    return label


def _unpack_dialogue_tdc(frame: pd.DataFrame) -> pd.DataFrame:
    """Развернуть packed dialogue в плоские строки с группами и порядком."""
    extra_columns = [
        name for name in frame.columns
        if name not in {"dialogue", "input_query_count", "reference_group_id", "turn_index"}
    ]
    group_label = _unique_labels()
    records = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        group = group_label(
            row["session_id"] if "session_id" in row.index else None,
            f"session-{row_number}",
        )
        turns = _sequence(row["dialogue"], f"dialogue в строке {row_number}")
        if not turns:
            raise MonitoringContractError(
                f"dialogue в строке {row_number} не содержит turns"
            )
        common = {name: row[name] for name in extra_columns}
        input_query_count = (
            row["input_query_count"]
            if "input_query_count" in row.index and not _blank(row["input_query_count"])
            else 1
        )
        for turn_index, turn in enumerate(turns, start=1):
            values = _sequence(
                turn,
                f"turn {turn_index} dialogue в строке {row_number}",
            )
            if len(values) != 3:
                raise MonitoringContractError(
                    f"turn {turn_index} dialogue в строке {row_number} должен содержать "
                    "query_id, input_query, output_answer"
                )
            query_id, input_query, output_answer = values
            record = dict(common)
            record.update({
                "query_id": f"{group}-t{turn_index}" if _blank(query_id) else query_id,
                "input_query": input_query,
                "output_answer": output_answer,
                "reference_group_id": group,
                "turn_index": turn_index,
                "input_query_count": input_query_count,
            })
            records.append(record)
    return pd.DataFrame(records)


def normalize_umr(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Канонизировать UMR в формате тестового датасета (reference из корзины или
    monitoring из TDC) к плоской форме с reference_group_id/turn_index: строгость —
    к контракту метрики, к данным — коэрсия и достройка (транспорт между портами
    меняет типы и теряет колонки)."""
    if not isinstance(frame, pd.DataFrame):
        raise MonitoringContractError("UMR должен быть pandas.DataFrame")
    if frame.empty:
        raise MonitoringContractError("UMR пуст")

    mode = _require(contract, "assessment_mode", _ASSESSMENT_MODES)
    flat_columns = ("query_id", "input_query", "output_answer")
    if all(name in frame for name in flat_columns):
        if "dialogue" in frame:
            raise MonitoringContractError(
                "UMR смешан: плоские колонки и dialogue одновременно; "
                "транспорт обязан быть либо flat, либо packed"
            )
        result = frame.copy()
        if any(_blank(value) for value in result["query_id"]):
            raise MonitoringContractError(
                "Плоский UMR содержит пустой query_id: NA/NaT/пустые "
                "идентификаторы не материализуются текстом"
            )
    elif "dialogue" in frame:
        result = _unpack_dialogue_tdc(frame)
    else:
        missing = [name for name in flat_columns if name not in frame]
        raise MonitoringContractError(
            f"UMR не соответствует ни packed dialogue, ни плоской форме: "
            f"плоская форма неполна, нет колонок {missing}"
        )
    if mode == "qa":
        return result

    if "reference_group_id" not in result:
        # Контекстные режимы без явного порядка обязаны нести session_id:
        # порядок строк сам по себе хронологию не доказывает.
        if "session_id" not in result:
            raise MonitoringContractError(
                "Контекстный плоский UMR требует session_id либо явные "
                "reference_group_id + turn_index"
            )
        if any(_blank(value) for value in result["session_id"]):
            raise MonitoringContractError(
                "Контекстный плоский UMR содержит пустой session_id"
            )
        result["reference_group_id"] = result["session_id"]
    group_column: list[object] = [None] * len(result)
    order_column: list[int] = [1] * len(result)
    for group, positions in _ordered_groups(result):
        for order, position in enumerate(positions, start=1):
            group_column[position] = group
            order_column[position] = order
    result["reference_group_id"] = group_column
    result["turn_index"] = order_column
    if "input_query_count" not in result:
        result["input_query_count"] = 1
    return result


def _load_tdc_monitoring(
    value: pd.DataFrame | bytes | bytearray | str | Path,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (bytes, bytearray)):
        source = BytesIO(value)
    elif isinstance(value, (str, Path)):
        source = Path(value)
        if not source.is_file():
            raise MonitoringContractError(
                f"Файл Monitoring-входа TDC не найден: {source}"
            )
    else:
        raise MonitoringContractError(
            "Monitoring-вход TDC должен быть pandas.DataFrame, parquet bytes "
            "или путём к parquet"
        )
    try:
        return pd.read_parquet(source)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise MonitoringContractError(
            "Monitoring-вход TDC не содержит читаемый parquet"
        ) from exc


def _current_turn(row) -> dict[str, object]:
    return {
        "input_query": "" if _blank(row["input_query"]) else str(row["input_query"]),
        "output_answer": "" if _blank(row["output_answer"]) else str(row["output_answer"]),
    }


def _turn_record(
    frame,
    position,
    sources,
    score_column,
    *,
    mode,
    group=None,
    history=None,
):
    row = frame.iloc[position]
    record = {
        "_unit_id": row["query_id"],
        "_row_positions": (position,),
        "_group_id": group,
        "input_query": row["input_query"],
        "output_answer": row["output_answer"],
        "input_query_count": row.get("input_query_count", 1),
        **_source_values(row, sources),
    }
    current = (
        _turn(row, len(history or ()) + 1)
        if mode == "turn_with_history" else _current_turn(row)
    )
    record["assessment_context"] = {"mode": mode, "current_turn": current}
    if mode == "turn_with_history":
        record["assessment_context"]["history"] = list(history or ())
    if score_column in frame:
        record[score_column] = row[score_column]
    return record


def _unitize(frame: pd.DataFrame, contract: dict, *, include_sources: bool = True) -> pd.DataFrame:
    frame = normalize_umr(frame, contract)
    query_label = _unique_labels()
    frame["query_id"] = [
        query_label(value, f"row-{position}")
        for position, value in enumerate(frame["query_id"].tolist())
    ]
    sources = contract.get("scoring", {}).get("sources", []) if include_sources else []
    score_column = contract.get("score_column", "main_metric")
    weighted = (
        contract.get("aggregation", {}).get("method")
        == "frequency_weighted_mean"
    )
    mode = _require(contract, "assessment_mode", _ASSESSMENT_MODES)

    records = []
    if mode == "qa":
        for position in range(len(frame)):
            records.append(_turn_record(
                frame, position, sources, score_column,
                mode=mode,
                group=None,
            ))
        return pd.DataFrame(records)

    groups = _ordered_groups(frame)
    if mode == "turn_with_history":
        for group, positions in groups:
            history = []
            for position in positions:
                records.append(_turn_record(
                    frame, position, sources, score_column,
                    mode=mode, group=group, history=history,
                ))
                history.append(_turn(frame.iloc[position], len(history) + 1))
        return pd.DataFrame(records)

    for group, positions in groups:
        part = frame.iloc[positions]
        turns = [
            _turn(frame.iloc[position], order)
            for order, position in enumerate(positions, start=1)
        ]
        record = {
            "_unit_id": group,
            "_row_positions": tuple(positions),
            "_group_id": group,
            "dialogue": turns,
            "assessment_context": {"mode": mode, "turns": turns},
            "input_query_count": _constant(
                part["input_query_count"].tolist(), "input_query_count"
            ) if weighted else 1,
        }
        for source in sources:
            column = source["column_name"]
            if column in part:
                record[source["source_id"]] = _constant(
                    part[column].tolist(), column
                )
        if score_column in part:
            record[score_column] = _constant(part[score_column].tolist(), score_column)
        records.append(record)
    return pd.DataFrame(records)


def unitize(frame: pd.DataFrame, payload: dict, *, include_sources: bool = True) -> pd.DataFrame:
    return _unitize(frame, validate_monitoring_metric(payload), include_sources=include_sources)


def _normalize(value: object, source: dict) -> object | None:
    if _blank(value):
        return None
    normalization = source["normalization"]
    if normalization == "label":
        result: object = str(value).strip().casefold()
    elif normalization == "numeric":
        text = str(value).strip()
        percent = text.endswith("%")
        result = _decimal(text.rstrip("%").strip())
        if percent:
            result /= Decimal(100)
    else:
        lookup = {str(key).strip().casefold(): _decimal(mapped) for key, mapped in normalization.items()}
        key = str(value).strip().casefold()
        if key not in lookup:
            raise MonitoringContractError(
                f"value_map источника {source['source_id']} не покрывает {value!r}"
            )
        result = lookup[key]
    if source["polarity"] == "inverted":
        if result not in {Decimal(0), Decimal(1)}:
            raise MonitoringContractError("inverted polarity требует бинарное значение")
        result = Decimal(1) - result
    return result


def _missing(policy: str, reason: str) -> Decimal | None:
    if policy == "fail":
        raise MonitoringContractError(reason)
    if policy in {"exclude_unit", "exclude_value"}:
        return None
    return Decimal(0)


def _present_values(
    values: list[object], policy: str, reason: str
) -> list[Decimal] | None:
    """Применить missing_policy к значениям единицы: None означает «единица не оценивается»."""
    if policy == "exclude_value":
        values = [value for value in values if value is not None]
        return values or None
    if any(value is None for value in values):
        replacement = _missing(policy, reason)
        if replacement is None:
            return None
        values = [replacement if value is None else value for value in values]
    return values


def _min_binary(values: list[Decimal], method: str) -> Decimal:
    if any(value not in {Decimal(0), Decimal(1)} for value in values):
        raise MonitoringContractError(f"{method} требует 0/1")
    return min(values)


def _unanimous(votes: list[object], policy: str, reason: str) -> Decimal | None:
    values = _present_values(votes, policy, reason)
    return None if values is None else _min_binary(values, "all_assessors")


def _score_row(row, contract: dict) -> Decimal | None:
    scoring = contract["scoring"]
    by_role: dict[str, list[object]] = {}
    for source in scoring["sources"]:
        if source["source_id"] not in row.index:
            raise MonitoringContractError(f"Нет source value {source['source_id']}")
        by_role.setdefault(source["role"], []).append(_normalize(row[source["source_id"]], source))
    method = scoring["method"]
    policy = scoring["missing_policy"]
    if method == "identity":
        value = by_role["final_score"][0]
        return _missing(policy, "Отсутствует final score") if value is None else value
    if method == "accuracy":
        prediction, target = by_role["prediction"][0], by_role["target"][0]
        if prediction is None or target is None:
            return _missing(policy, "Отсутствует prediction или target")
        return Decimal(int(prediction == target))
    if method in {"mean_criteria", "all_criteria"}:
        values = _present_values(by_role["criterion"], policy, "Отсутствует criterion score")
        if values is None:
            return None
        if method == "mean_criteria":
            return sum(values, Decimal(0)) / len(values)
        return _min_binary(values, "all_criteria")
    votes = by_role["assessor_vote"]
    if method == "all_assessors":
        # Единогласие: та же политика пропусков, что у all_criteria, но по голосам.
        return _unanimous(votes, policy, "Отсутствует голос assessor")
    present = [vote for vote in votes if vote is not None]
    if any(vote not in {Decimal(0), Decimal(1)} for vote in present):
        raise MonitoringContractError("majority требует 0/1")
    if not present:
        return _missing(policy, "Нет assessor votes")
    denominator = len(votes) if scoring["majority_denominator"] == "declared" else len(present)
    positives = sum(present, Decimal(0))
    if positives * 2 == denominator:
        return _missing(policy, "Majority завершился ничьей")
    return Decimal(int(positives * 2 > denominator))


def score_units(units: pd.DataFrame, payload: dict) -> pd.Series:
    contract = validate_monitoring_metric(payload)
    return pd.Series(
        [float(score) if (score := _score_row(row, contract)) is not None else None for _, row in units.iterrows()],
        index=units.index,
        dtype="float64",
    )


def broadcast_scores(frame: pd.DataFrame, units: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    if len(units) != len(scores):
        raise MonitoringContractError("Число scores не совпадает с числом units")
    result = frame.copy()
    result["main_metric"] = None
    result["assessment_unit_id"] = None
    score_column = result.columns.get_loc("main_metric")
    unit_column = result.columns.get_loc("assessment_unit_id")
    for unit_id, positions, score in zip(
        units["_unit_id"], units["_row_positions"], scores.tolist()
    ):
        for position in positions:
            result.iat[position, score_column] = score
            result.iat[position, unit_column] = unit_id
    result["main_metric"] = pd.to_numeric(result["main_metric"], errors="coerce")
    return result


def _drift_frame(frame: pd.DataFrame, contract: dict, *, require_target: bool) -> pd.DataFrame:
    units = _unitize(frame, contract)
    if require_target and "main_metric" not in units:
        raise MonitoringContractError("Для drift отсутствует main_metric")
    target = pd.to_numeric(
        units["main_metric"] if "main_metric" in units else pd.Series([None] * len(units)),
        errors="coerce",
    )
    # Пустой main_metric — отказ судьи либо неразмеченная единица: исключается
    # из drift независимо от missing_policy (политика описывает вычисление
    # main_metric адаптером, а не потребление готовых оценок).
    mode = contract["assessment_mode"]
    if mode == "dialogue":
        questions = [
            json.dumps(
                [turn["input_query"] for turn in dialogue],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for dialogue in units["dialogue"]
        ]
    elif mode == "turn_with_history":
        questions = [
            json.dumps(
                [
                    *(turn["input_query"] for turn in context["history"]),
                    context["current_turn"]["input_query"],
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for context in units["assessment_context"]
        ]
    else:
        questions = units["input_query"].astype(str).tolist()
    group_ids = [
        unit_id if _blank(group_id) else group_id
        for group_id, unit_id in zip(units["_group_id"], units["_unit_id"])
    ]
    result = pd.DataFrame({
        "question": questions,
        "answer": [""] * len(units),
        "target": target,
        "reference_group_id": group_ids,
    })
    return result.dropna(subset=["target"]).reset_index(drop=True) if require_target else result


def prepare_drift_frames(
    reference_umr: pd.DataFrame,
    monitoring_umr: pd.DataFrame | bytes | bytearray | str | Path,
    payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = validate_monitoring_metric(payload, require_computed=False)
    if "assessment_mode" not in contract:
        # Контракт без MeasurementPlan (plan-less not_computable) не несёт
        # режим оценки — drift честно отказывается вместо невнятного падения
        # внутри нормализации.
        raise MonitoringContractError(
            "Drift не вычисляется: monitoring_metric не содержит assessment_mode "
            f"(status={contract.get('status')!r}, reason={contract.get('reason')!r})"
        )
    # normalize_umr выполнит _unitize внутри _drift_frame — второй проход не нужен
    monitoring_umr = _load_tdc_monitoring(monitoring_umr)
    for name, data in (("reference", reference_umr), ("monitoring", monitoring_umr)):
        identities = data.get("definition_id")
        if identities is None or not identities.eq(contract.get("definition_id")).all():
            raise MonitoringContractError(f"{name}.definition_id не соответствует определению КМ")
        roles = data.get("dataset_role")
        if roles is None or not roles.eq(name).all():
            raise MonitoringContractError(f"{name}.dataset_role не соответствует назначению данных")
    return (
        _drift_frame(reference_umr, contract, require_target=True),
        _drift_frame(monitoring_umr, contract, require_target=True),
    )


def aggregate_main_metric(frame: pd.DataFrame, payload: dict) -> dict[str, object]:
    contract = validate_monitoring_metric(payload)
    units = unitize(frame, contract)
    if "main_metric" not in units:
        raise MonitoringContractError("Нет main_metric для агрегации")
    records: list[tuple[Decimal, Decimal]] = []
    excluded = 0
    policy = contract["scoring"]["missing_policy"]
    weighted = contract["aggregation"]["method"] == "frequency_weighted_mean"
    for _, row in units.iterrows():
        if _blank(row["main_metric"]):
            if policy == "fail":
                raise MonitoringContractError("Пустой main_metric")
            if policy == "zero":
                score = Decimal(0)
            else:
                excluded += 1
                continue
        else:
            score = _decimal(row["main_metric"], "main_metric")
        weight = _decimal(row["input_query_count"], "input_query_count") if weighted else Decimal(1)
        if weight <= 0:
            raise MonitoringContractError("Вес должен быть положительным")
        records.append((score, weight))
    if not records:
        raise MonitoringContractError("Нет оцененных единиц")
    total_weight = sum((weight for _score, weight in records), Decimal(0))
    value = sum((score * weight for score, weight in records), Decimal(0)) / total_weight
    return {
        "name": contract["name"],
        "value": float(value),
        "total_units": len(units),
        "scored_units": len(records),
        "excluded_units": excluded,
        "weight_sum": float(total_weight),
    }
