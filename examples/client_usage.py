from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_forecasting import DemandForecastModel, ForecastModelConfig


def build_train_dataset(rows: int = 220) -> pd.DataFrame:
    """
    Генерирует небольшой синтетический датасет.

    Это не production-данные, а просто наглядный пример того,
    какие данные можно передать в проект.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")

    frame = pd.DataFrame(
        {
            "date": dates,
            "SKU": ["SKU_A" if i % 2 == 0 else "SKU_B" for i in range(rows)],
            "temperature": rng.normal(10, 8, size=rows).round(1),
            "promo_type": rng.choice(["none", "discount", "display"], size=rows),
        }
    )

    # Ниже искусственно задаем зависимость спроса от температуры,
    # промо-активности, выходных и тренда, чтобы у модели был сигнал.
    weekend = frame["date"].dt.dayofweek.isin([5, 6]).astype(int)
    promo_effect = frame["promo_type"].map({"none": 0, "discount": 8, "display": 4})
    trend = np.linspace(0, 12, rows)
    noise = rng.normal(0, 2, size=rows)
    frame["quantity"] = (
        40
        + trend
        + np.maximum(frame["temperature"], 0) * 0.5
        + weekend * 6
        + promo_effect
        + noise
    ).clip(lower=0).round(2)
    return frame


def build_future_features(train_df: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    """
    Строит будущие внешние признаки для прогноза.

    В реальном сценарии сюда обычно попадают:
    - промо-календарь;
    - цены;
    - погода;
    - региональные события;
    - другие экзогенные факторы.
    """
    rng = np.random.default_rng(7)
    future_dates = pd.date_range(
        train_df["date"].max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )
    return pd.DataFrame(
        {
            "date": future_dates,
            "SKU": ["SKU_A" if i % 2 == 0 else "SKU_B" for i in range(horizon)],
            "temperature": rng.normal(12, 5, size=horizon).round(1),
            "promo_type": rng.choice(["none", "discount", "display"], size=horizon),
        }
    )


if __name__ == "__main__":
    train_df = build_train_dataset()
    future_df = build_future_features(train_df, horizon=14)

    # Конфиг ансамбля можно кастомизировать точечно.
    config = ForecastModelConfig(
        lags=(1, 7, 14, 28),
        rolling_windows=(7, 14),
        default_horizon=14,
    )

    # Пользователь описывает только схему данных.
    # Все остальное пайплайн делает внутри сам.
    model = DemandForecastModel(
        date_column="date",
        target_column="quantity",
        feature_columns=["temperature", "promo_type"],
        group_columns=["SKU"],
        prediction_column="predicted_quantity",
        config=config,
    )

    # Обучение на исторических данных.
    model.fit(train_df)
    print(f"Quality score: {model.quality_score_:.3f}")

    # Смотрим, какие признаки больше влияли на деревья ансамбля.
    print("Feature importance:")
    for column, weight in list(model.get_feature_importance().items())[:10]:
        print(f"  {column}: {weight:.1%}")

    # Прогноз по future-таблице.
    forecast = model.predict(future_df)
    print("\nForecast sample:")
    print(forecast[["date", "SKU", "predicted_quantity"]].head(10))

    # Компактный клиентский формат, если нужен ответ как `SKU: количество`.
    # Для одной даты вернется плоский словарь, для нескольких дат - словарь по датам.
    sku_forecast = model.predict_by_sku(sku_column="SKU", future_df=future_df.head(2).copy())
    print("\nForecast as SKU mapping:")
    print(sku_forecast)

    # Прогноз только по горизонту, без отдельного future DataFrame.
    horizon_only = model.forecast(7)
    print("\nForecast without future exogenous features:")
    print(horizon_only[["date", "SKU", "predicted_quantity"]])

    # Сохранение и повторная загрузка модели.
    artifact_path = Path("artifacts") / "demand_forecast_model.zip"
    model.save(str(artifact_path))

    loaded_model = DemandForecastModel.load(str(artifact_path))
    loaded_forecast = loaded_model.predict(future_df.copy())
    print(f"\nLoaded model predictions: {len(loaded_forecast)} rows")
