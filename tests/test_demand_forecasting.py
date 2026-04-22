from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecasting import DemandForecastModel, ForecastModelConfig


def build_dataset(rows: int = 180) -> pd.DataFrame:
    """
    Строит синтетический временной ряд для smoke-теста.

    Датасет специально сделан простым и устойчивым:
    у спроса есть тренд, погодный эффект, выходные и промо.
    """
    rng = np.random.default_rng(123)
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")

    frame = pd.DataFrame(
        {
            "date": dates,
            "SKU": ["SKU_A" if i % 2 == 0 else "SKU_B" for i in range(rows)],
            "temperature": rng.normal(8, 10, size=rows).round(1),
            "promo_type": rng.choice(["none", "discount"], size=rows),
        }
    )

    day_of_week = frame["date"].dt.dayofweek
    weekend = day_of_week.isin([5, 6]).astype(int)
    promo_effect = frame["promo_type"].map({"none": 0, "discount": 7})
    trend = np.linspace(0, 8, rows)
    noise = rng.normal(0, 1.5, size=rows)

    frame["quantity"] = (
        40 + trend + np.maximum(frame["temperature"], 0) * 0.4 + weekend * 5 + promo_effect + noise
    ).clip(lower=0).round(2)
    return frame


class DemandForecastModelTestCase(unittest.TestCase):
    def test_fit_predict_save_and_load(self) -> None:
        """
        Проверяет основной пользовательский сценарий:
        fit -> predict -> forecast -> save -> load -> predict.
        """
        train_df = build_dataset()
        future_dates = pd.date_range(train_df["date"].max() + pd.Timedelta(days=1), periods=20, freq="D")
        future_df = pd.DataFrame(
            {
                "date": future_dates,
                "SKU": ["SKU_A" if i % 2 == 0 else "SKU_B" for i in range(20)],
                "temperature": np.linspace(5, 15, 20),
                "promo_type": ["discount" if i % 3 == 0 else "none" for i in range(20)],
            }
        )

        config = ForecastModelConfig(lags=(1, 7, 14), rolling_windows=(7, 14), min_history=30)
        model = DemandForecastModel(
            date_column="date",
            target_column="quantity",
            feature_columns=["temperature", "promo_type"],
            group_columns=["SKU"],
            prediction_column="predicted_quantity",
            config=config,
        )

        # Обучение должно завершиться без ошибок и заполнить fitted-state.
        model.fit(train_df)

        # Прогноз по future DataFrame.
        forecast = model.predict(future_df)
        predictions = forecast["predicted_quantity"].to_numpy()
        sku_mapping = model.predict_by_sku(sku_column="SKU", future_df=future_df.head(1).copy())

        # Дополнительная проверка на вычисление важностей.
        importance = model.get_feature_importance()

        self.assertEqual(len(predictions), len(future_df))
        self.assertIn("predicted_quantity", forecast.columns)
        self.assertTrue((predictions >= 0).all())
        self.assertIsNotNone(model.quality_score_)
        self.assertIn("lag_1", importance)
        self.assertIn("temperature", importance)
        self.assertIn("SKU_A", sku_mapping)
        self.assertTrue(np.isfinite(sku_mapping["SKU_A"]))

        # Прогноз только по горизонту тоже должен работать.
        horizon_only = model.forecast(7)
        self.assertEqual(len(horizon_only), 7)

        # Проверяем сохранение и повторную загрузку артефакта.
        artifact_path = Path("artifacts") / "test_model.zip"
        model.save(str(artifact_path))
        loaded_model = DemandForecastModel.load(str(artifact_path))
        loaded_forecast = loaded_model.predict(future_df.copy())
        loaded_predictions = loaded_forecast["predicted_quantity"].to_numpy()

        self.assertEqual(len(loaded_predictions), len(predictions))
        self.assertTrue(np.isfinite(loaded_predictions).all())


if __name__ == "__main__":
    unittest.main()
