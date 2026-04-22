from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def prepare_dataframe(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    feature_columns: list[str],
    group_columns: list[str],
    require_target: bool,
) -> pd.DataFrame:
    """
    Приводит входной DataFrame к стабильному виду для fit/predict.

    На этом этапе мы еще не строим ML-признаки.
    Здесь задача другая: сделать так, чтобы все дальнейшие шаги пайплайна
    работали на предсказуемой и чистой таблице.

    Что делает функция:
    1. проверяет наличие обязательных колонок;
    2. создает отсутствующие необязательные feature/group колонки;
    3. приводит дату к datetime;
    4. на train приводит target к числу;
    5. удаляет явно невалидные строки;
    6. агрегирует дубликаты по ключу `date + groups`;
    7. сортирует результат по времени.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_columns = [date_column]
    if require_target:
        required_columns.append(target_column)

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = df.copy()

    # Все дополнительные признаки и группировки должны существовать,
    # даже если пользователь не передал их на predict.
    # Тогда downstream-код может опираться на единый интерфейс колонок.
    for column in [*feature_columns, *group_columns]:
        if column not in frame.columns:
            frame[column] = np.nan

    # Дата нужна в нормальном datetime-виде.
    # Невалидные даты превращаются в NaT и затем удаляются.
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).copy()

    # На обучении target обязателен и должен быть числовым.
    if require_target:
        frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
        frame = frame.dropna(subset=[target_column]).copy()

    # Если на одну дату и одну группу пришло несколько строк,
    # склеиваем их в одну запись.
    keys = [date_column, *group_columns]
    frame = aggregate_duplicates(frame, keys, target_column, require_target)

    # Временной ряд всегда приводим к возрастающему порядку по времени.
    frame = frame.sort_values([date_column, *group_columns]).reset_index(drop=True)
    return frame


def aggregate_duplicates(
    frame: pd.DataFrame,
    keys: list[str],
    target_column: str,
    require_target: bool,
) -> pd.DataFrame:
    """
    Агрегирует дубли по ключу `date + group_columns`.

    Логика агрегации намеренно простая и читаемая:
    - target суммируется;
    - числовые внешние признаки усредняются;
    - категориальные признаки берут первое непустое значение.
    """
    if frame.duplicated(keys).sum() == 0:
        return frame

    agg_rules: dict[str, Any] = {}

    for column in frame.columns:
        if column in keys:
            continue

        if column == target_column and require_target:
            agg_rules[column] = "sum"
        elif pd.api.types.is_numeric_dtype(frame[column]):
            agg_rules[column] = "mean"
        else:
            agg_rules[column] = lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan

    return frame.groupby(keys, as_index=False).agg(agg_rules)


def build_training_frame(
    frame: pd.DataFrame,
    date_column: str,
    target_column: str,
    feature_columns: list[str],
    group_columns: list[str],
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Превращает временной ряд в supervised-таблицу признаков для ML-моделей.

    Для каждой даты строим признаки только из прошлого.
    Это защищает обучение от утечки будущего в train.
    """
    rows: list[dict[str, Any]] = []
    targets: list[float] = []
    max_lag = max(lags)

    for i in range(len(frame)):
        row = frame.iloc[i]
        history = frame.iloc[:i]
        group_history = select_group_history(history, row, date_column, group_columns)

        # Пока истории меньше самого большого лага,
        # строить строку признаков бессмысленно.
        if len(group_history) < max_lag:
            continue

        feature_row = build_feature_row(
            row=row,
            history=history,
            date_column=date_column,
            target_column=target_column,
            feature_columns=feature_columns,
            group_columns=group_columns,
            lags=lags,
            rolling_windows=rolling_windows,
        )
        rows.append(feature_row)
        targets.append(float(row[target_column]))

    return pd.DataFrame(rows), pd.Series(targets, dtype=float)


def build_feature_row(
    row: pd.Series,
    history: pd.DataFrame,
    date_column: str,
    target_column: str,
    feature_columns: list[str],
    group_columns: list[str],
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> dict[str, Any]:
    """
    Строит один набор признаков для конкретной строки временного ряда.

    Типы признаков:
    - календарные;
    - лаговые по target;
    - rolling-статистики по target;
    - внешние признаки пользователя;
    - идентификаторы групп, если рядов несколько.
    """
    row_date = pd.Timestamp(row[date_column])
    group_history = select_group_history(history, row, date_column, group_columns)
    target_values = group_history[target_column].astype(float).to_numpy()

    # Календарные признаки помогают модели улавливать недельность,
    # месячность и сезонный контекст даты.
    result: dict[str, Any] = {
        "day_of_week": row_date.dayofweek,
        "day_of_month": row_date.day,
        "month": row_date.month,
        "quarter": row_date.quarter,
        "week_of_year": int(row_date.isocalendar().week),
        "is_weekend": int(row_date.dayofweek >= 5),
        "day_of_year_sin": np.sin(2 * np.pi * row_date.dayofyear / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * row_date.dayofyear / 365.25),
    }

    # Лаги показывают модели прошлые значения целевой переменной.
    for lag in lags:
        if len(target_values) >= lag:
            result[f"lag_{lag}"] = target_values[-lag]
        else:
            result[f"lag_{lag}"] = np.nan

    # Rolling-статистики добавляют более сглаженную историю.
    for window in rolling_windows:
        window_values = target_values[-window:] if len(target_values) >= window else target_values

        if len(window_values) == 0:
            result[f"rolling_mean_{window}"] = np.nan
            result[f"rolling_std_{window}"] = np.nan
        else:
            result[f"rolling_mean_{window}"] = float(np.mean(window_values))
            result[f"rolling_std_{window}"] = float(np.std(window_values))

    # Внешние признаки пользователя копируем как есть.
    # Их дальнейшая обработка произойдет на этапе transform_features.
    for column in feature_columns:
        result[column] = row.get(column, np.nan)

    # Группы держим в feature-space, чтобы модель различала ряды.
    for column in group_columns:
        result[column] = row.get(column, np.nan)

    return result


def select_group_history(
    history: pd.DataFrame,
    row: pd.Series,
    date_column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Возвращает исторические точки только для той же группы, что и текущая строка.

    Если групп нет, возвращается вся прошлую история ряда.
    """
    if history.empty:
        return history

    mask = history[date_column] < row[date_column]
    for column in group_columns:
        mask &= history[column].eq(row.get(column))

    return history.loc[mask].sort_values(date_column)


def fit_encoder(raw_frame: pd.DataFrame) -> dict[str, Any]:
    """
    Запоминает правила преобразования сырых признаков в числовую матрицу.

    Сохраняем:
    - порядок исходных колонок;
    - медианы числовых признаков для заполнения пропусков;
    - список категориальных колонок;
    - набор one-hot dummy-колонок, увиденных на train.
    """
    state: dict[str, Any] = {}
    state["raw_feature_columns"] = raw_frame.columns.tolist()

    numeric_columns = [
        column for column in raw_frame.columns if pd.api.types.is_numeric_dtype(raw_frame[column])
    ]

    numeric_fill_values: dict[str, float] = {}
    for column in numeric_columns:
        series = pd.to_numeric(raw_frame[column], errors="coerce").dropna()
        numeric_fill_values[column] = float(series.median()) if not series.empty else 0.0

    state["numeric_fill_values"] = numeric_fill_values
    state["categorical_columns"] = [
        column for column in raw_frame.columns if column not in numeric_fill_values
    ]

    # Во время fit один раз создаем dummy-колонки и запоминаем их схему.
    encoded = transform_features(raw_frame, state, fit=True)
    state["feature_names"] = encoded.columns.tolist()
    return state


def transform_features(raw_frame: pd.DataFrame, state: dict[str, Any], fit: bool = False) -> pd.DataFrame:
    """
    Превращает сырые признаки в числовую матрицу для моделей.

    Числовые признаки:
    - приводим к numeric;
    - заполняем пропуски медианами из train.

    Категориальные признаки:
    - приводим к строкам;
    - кодируем one-hot;
    - на predict переиспользуем ровно ту же схему dummy-колонок, что и на fit.
    """
    frame = raw_frame.copy()

    # Восстанавливаем полный набор сырых колонок из train-схемы.
    for column in state["raw_feature_columns"]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[state["raw_feature_columns"]]

    numeric_frame = pd.DataFrame(index=frame.index)
    for column, fill_value in state["numeric_fill_values"].items():
        numeric_frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill_value)

    categorical_frame = pd.DataFrame(index=frame.index)
    for column in state["categorical_columns"]:
        categorical_frame[column] = frame[column].fillna("__missing__").astype(str)

    encoded = numeric_frame.copy()

    if not categorical_frame.empty:
        dummies = pd.get_dummies(categorical_frame, prefix=categorical_frame.columns, dtype=float)

        if fit:
            state["known_dummy_columns"] = dummies.columns.tolist()

        # Любые новые категории на predict не ломают код:
        # просто сохраняется train-схема dummy-колонок.
        dummies = dummies.reindex(columns=state.get("known_dummy_columns", []), fill_value=0.0)
        encoded = pd.concat([numeric_frame, dummies], axis=1)

    # На predict и на fit порядок колонок должен быть идентичным.
    return encoded.reindex(columns=state.get("feature_names", encoded.columns.tolist()), fill_value=0.0)


def get_last_known_feature_values(
    history: pd.DataFrame | None,
    date_column: str,
    feature_columns: list[str],
) -> dict[str, Any]:
    """
    Возвращает последние известные значения внешних признаков.

    Это fallback для режима `forecast(horizon)`, когда будущий DataFrame
    пользователь не передает.
    """
    if history is None or history.empty:
        return {}

    last_row = history.sort_values(date_column).iloc[-1]
    result: dict[str, Any] = {}

    for column in feature_columns:
        result[column] = last_row.get(column, np.nan)

    return result
