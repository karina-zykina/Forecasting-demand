from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model import DemandForecastModel


def parse_args() -> argparse.Namespace:
    """Описывает CLI-параметры для прогнозного скрипта."""
    parser = argparse.ArgumentParser(description="Build forecast from saved demand model.")
    parser.add_argument("--model-path", required=True, help="Path to saved model zip.")
    parser.add_argument("--future-data", default="", help="CSV with future dates and optional features.")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon if future-data is not passed.")
    parser.add_argument("--output-path", default="", help="Optional path to save forecast CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Загружаем ранее обученный артефакт.
    model = DemandForecastModel.load(args.model_path)

    # Если есть future CSV, используем его.
    # Если нет, строим рекурсивный прогноз только по горизонту.
    if args.future_data:
        future_df = pd.read_csv(args.future_data)
        forecast = model.predict(future_df=future_df, horizon=args.horizon)
    else:
        forecast = model.forecast(args.horizon)

    if args.output_path:
        forecast.to_csv(args.output_path, index=False)
        print(f"Forecast saved to {args.output_path}")
    else:
        print(forecast.head(20))


if __name__ == "__main__":
    main()
