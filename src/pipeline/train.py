from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model import DemandForecastModel, ForecastModelConfig


def parse_args() -> argparse.Namespace:
    """Описывает CLI-параметры для скрипта обучения."""
    parser = argparse.ArgumentParser(description="Train demand forecasting model.")
    parser.add_argument("--data", required=True, help="Path to train CSV file.")
    parser.add_argument("--model-path", required=True, help="Where to save model zip.")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--target-column", default="quantity")
    parser.add_argument("--feature-columns", default="")
    parser.add_argument("--group-columns", default="")
    return parser.parse_args()


def split_columns(raw_value: str) -> list[str]:
    """Преобразует строку формата `a,b,c` в список имен колонок."""
    if not raw_value.strip():
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()

    # Читаем CSV в DataFrame.
    df = pd.read_csv(args.data)

    # Собираем модель с явным указанием схемы данных.
    model = DemandForecastModel(
        date_column=args.date_column,
        target_column=args.target_column,
        feature_columns=split_columns(args.feature_columns),
        group_columns=split_columns(args.group_columns),
        config=ForecastModelConfig(),
    )

    # Обучаем ансамбль LightGBM + XGBoost.
    model.fit(df)

    # Сохраняем готовый артефакт, который потом можно загрузить без retrain.
    model.save(args.model_path)

    print("Model trained successfully.")
    print("Quality metrics:", model.quality_metrics_)
    print("Quality score:", model.quality_score_)


if __name__ == "__main__":
    main()
