from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model import DemandForecastModel


def parse_args() -> argparse.Namespace:
    """Описывает CLI-параметры для скрипта оценки качества."""
    parser = argparse.ArgumentParser(description="Evaluate saved demand forecasting model.")
    parser.add_argument("--model-path", required=True, help="Path to saved model zip.")
    parser.add_argument("--data", required=True, help="CSV with future dates, features and real target.")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--target-column", default="quantity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Загружаем тестовую или отложенную выборку.
    df = pd.read_csv(args.data)

    # Загружаем модель и строим прогноз на тех же датах.
    model = DemandForecastModel.load(args.model_path)
    forecast = model.predict(df)

    # Сравниваем предсказание с реальным target.
    y_true = pd.to_numeric(df[args.target_column], errors="coerce").to_numpy()
    y_pred = forecast[model.prediction_column].to_numpy()
    metrics = model._calculate_metrics(y_true, y_pred)

    print("Evaluation metrics:", metrics)
    print(forecast.head(10))


if __name__ == "__main__":
    main()
