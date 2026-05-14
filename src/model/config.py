from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ForecastModelConfig:
    """Config for the LightGBM + CatBoost ensemble."""

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

    catboost_params: dict = field(
        default_factory=lambda: {
            "iterations": 250,
            "learning_rate": 0.05,
            "depth": 6,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.9,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "loss_function": "RMSE",
            "verbose": False,
            "allow_writing_files": False,
        }
    )

    ensemble_weights: dict = field(
        default_factory=lambda: {
            "lightgbm": 0.50,
            "catboost": 0.50,
        }
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ForecastModelConfig":
        if payload is None:
            return cls()

        payload = payload.copy()
        xgboost_params = payload.pop("xgboost_params", None)
        if xgboost_params is not None and "catboost_params" not in payload:
            payload["catboost_params"] = cls._catboost_params_from_xgboost(xgboost_params)

        weights = payload.get("ensemble_weights")
        if isinstance(weights, dict) and "xgboost" in weights and "catboost" not in weights:
            weights = weights.copy()
            weights["catboost"] = weights.pop("xgboost")
            payload["ensemble_weights"] = weights

        return cls(**payload)

    @staticmethod
    def _catboost_params_from_xgboost(xgboost_params: dict) -> dict:
        return {
            "iterations": xgboost_params.get("n_estimators", 250),
            "learning_rate": xgboost_params.get("learning_rate", 0.05),
            "depth": xgboost_params.get("max_depth", 6),
            "bootstrap_type": "Bernoulli",
            "subsample": xgboost_params.get("subsample", 0.9),
            "random_seed": xgboost_params.get("random_state", 42),
            "loss_function": "RMSE",
            "verbose": False,
            "allow_writing_files": False,
        }
