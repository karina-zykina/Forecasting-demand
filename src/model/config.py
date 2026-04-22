from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ForecastModelConfig:
    """Конфиг ансамбля `LightGBM + XGBoost`."""

    lags: tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 14, 28)
    seasonal_period: int = 7
    min_history: int = 40
    validation_fraction: float = 0.2
    min_validation_size: int = 14
    default_horizon: int = 14
    clip_predictions: bool = True

    lightgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbosity": -1,
        }
    )

    xgboost_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": 42,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
    )

    ensemble_weights: dict = field(
        default_factory=lambda: {
            "lightgbm": 0.50,
            "xgboost": 0.50,
        }
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ForecastModelConfig":
        if payload is None:
            return cls()
        return cls(**payload)
