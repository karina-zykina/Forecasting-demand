from __future__ import annotations

import pickle
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import ForecastModelConfig
from .preprocessing import (
    build_feature_row,
    build_training_frame,
    fit_encoder,
    get_last_known_feature_values,
    prepare_dataframe,
    transform_features,
)


class DemandForecastModel:

    def __init__(
        self,
        date_column: str = "date",
        target_column: str = "quantity",
        feature_columns: list[str] | None = None,
        group_columns: list[str] | None = None,
        prediction_column: str = "predicted_quantity",
        config: ForecastModelConfig | dict | None = None,
        lags: tuple[int, ...] | list[int] | None = None,
        rolling_windows: tuple[int, ...] | list[int] | None = None,
        seasonal_period: int | None = None,
        min_history: int | None = None,
        validation_fraction: float | None = None,
    ) -> None:
        self.date_column = date_column
        self.target_column = target_column
        self.feature_columns = list(feature_columns or [])
        self.group_columns = list(group_columns or [])
        self.prediction_column = prediction_column

        self.config = self._build_config(
            config=config,
            lags=lags,
            rolling_windows=rolling_windows,
            seasonal_period=seasonal_period,
            min_history=min_history,
            validation_fraction=validation_fraction,
        )

        self.encoder_state_: dict[str, Any] = {}
        self.lightgbm_model_: lgb.Booster | None = None
        self.catboost_model_: CatBoostRegressor | None = None
        self.history_: pd.DataFrame | None = None
        self.group_frame_: pd.DataFrame | None = None
        self.last_date_: pd.Timestamp | None = None
        self.target_mean_: float = 0.0
        self.quality_metrics_: dict[str, float] = {}
        self.quality_score_: float | None = None

    def fit(self, df: pd.DataFrame) -> "DemandForecastModel":
        prepared = prepare_dataframe(
            df=df,
            date_column=self.date_column,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            group_columns=self.group_columns,
            require_target=True,
        )

        if len(prepared) < self.config.min_history:
            raise ValueError(f"Need at least {self.config.min_history} rows, got {len(prepared)}.")

        raw_features, targets = build_training_frame(
            frame=prepared,
            date_column=self.date_column,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            group_columns=self.group_columns,
            lags=self.config.lags,
            rolling_windows=self.config.rolling_windows,
        )

        if raw_features.empty:
            raise ValueError("Not enough history to build lag features.")

        self.encoder_state_ = fit_encoder(raw_features)
        x = transform_features(raw_features, self.encoder_state_)
        y = targets.to_numpy(dtype=float)

        split_index = self._get_split_index(len(x))

        if split_index is not None:
            x_train = x.iloc[:split_index]
            x_valid = x.iloc[split_index:]
            y_train = y[:split_index]
            y_valid = y[split_index:]

            temp_models = self._fit_all_models(x_train.to_numpy(), y_train)
            valid_pred = self._predict_with_models(temp_models, x_valid.to_numpy())
            self.quality_metrics_ = self._calculate_metrics(y_valid, valid_pred)
        else:
            self.quality_metrics_ = {}

        final_models = self._fit_all_models(x.to_numpy(), y)
        self.lightgbm_model_ = final_models["lightgbm"]
        self.catboost_model_ = final_models["catboost"]

        if not self.quality_metrics_:
            train_pred = self._predict_with_models(final_models, x.to_numpy())
            self.quality_metrics_ = self._calculate_metrics(y, train_pred)

        self.quality_score_ = max(0.0, min(1.0, 1.0 - self.quality_metrics_["smape"] / 100.0))

        self.history_ = prepared.copy()
        self.group_frame_ = prepared[self.group_columns].drop_duplicates().reset_index(drop=True)
        self.last_date_ = prepared[self.date_column].max()
        self.target_mean_ = float(prepared[self.target_column].mean())
        return self

    def predict(self, future_df: pd.DataFrame | None = None, horizon: int | None = None) -> pd.DataFrame:
        self._check_is_fitted()

        future = self._prepare_future_frame(future_df, horizon)
        future = future.sort_values([self.date_column, *self.group_columns]).reset_index(drop=True)

        history = self.history_.copy()
        predictions: list[float] = []

        for _, row in future.iterrows():
            feature_row = build_feature_row(
                row=row,
                history=history,
                date_column=self.date_column,
                target_column=self.target_column,
                feature_columns=self.feature_columns,
                group_columns=self.group_columns,
                lags=self.config.lags,
                rolling_windows=self.config.rolling_windows,
            )

            raw_frame = pd.DataFrame([feature_row])
            x = transform_features(raw_frame, self.encoder_state_)
            pred = float(self._predict_one_row(x.to_numpy())[0])

            if self.config.clip_predictions:
                pred = max(0.0, pred)

            predictions.append(pred)

            new_row = row.copy()
            new_row[self.target_column] = pred
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        result = future.copy()
        result[self.prediction_column] = predictions
        return result

    def forecast(self, horizon: int, future_features: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.predict(future_df=future_features, horizon=horizon)

    def predict_by_sku(
        self,
        sku_column: str = "SKU",
        future_df: pd.DataFrame | None = None,
        horizon: int | None = None,
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """
        Возвращает прогноз в формате `SKU: количество`.

        Если в прогнозе только одна дата, вернется плоский словарь:
        {
            "SKU_1": 120.5,
            "SKU_2": 87.0,
        }

        Если дат несколько, вернется вложенный словарь по датам:
        {
            "2025-01-01": {"SKU_1": 120.5, "SKU_2": 87.0},
            "2025-01-02": {"SKU_1": 118.2, "SKU_2": 90.1},
        }
        """
        forecast = self.predict(future_df=future_df, horizon=horizon)
        return self._forecast_to_sku_mapping(forecast, sku_column=sku_column)

    def forecast_by_sku(
        self,
        horizon: int,
        sku_column: str = "SKU",
        future_features: pd.DataFrame | None = None,
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """Удобный алиас для прогнозирования по горизонту в формате `SKU: количество`."""
        forecast = self.forecast(horizon=horizon, future_features=future_features)
        return self._forecast_to_sku_mapping(forecast, sku_column=sku_column)

    def get_feature_importance(
        self,
        plot: bool = True,
        top_n: int | None = None,
        show: bool = True,
        ax: Any | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        """
        Return ensemble feature importances and optionally draw a bar chart.

        Importance is calculated for each base model, normalized inside that
        model, and then combined with the same weights that are used for
        ensemble predictions.
        """
        self._check_is_fitted()

        feature_names = self.encoder_state_["feature_names"]
        weights = self._normalized_ensemble_weights()
        lgb_importance = self._get_lightgbm_feature_importance()
        catboost_importance = self._get_catboost_feature_importance()

        result: dict[str, float] = {}
        for index, feature_name in enumerate(feature_names):
            result[feature_name] = float(
                weights["lightgbm"] * lgb_importance[index]
                + weights["catboost"] * catboost_importance[index]
            )

        sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

        if plot:
            self._plot_feature_importance(
                sorted_result,
                top_n=top_n,
                show=show,
                ax=ax,
                figsize=figsize,
            )

        return sorted_result

    def save(self, path: str) -> None:
        self._check_is_fitted()

        payload = {
            "init_params": {
                "date_column": self.date_column,
                "target_column": self.target_column,
                "feature_columns": self.feature_columns,
                "group_columns": self.group_columns,
                "prediction_column": self.prediction_column,
                "config": self.config.to_dict(),
            },
            "state": {
                "encoder_state_": self.encoder_state_,
                "lightgbm_model_": self.lightgbm_model_,
                "catboost_model_": self.catboost_model_,
                "history_": self.history_,
                "group_frame_": self.group_frame_,
                "last_date_": self.last_date_,
                "target_mean_": self.target_mean_,
                "quality_metrics_": self.quality_metrics_,
                "quality_score_": self.quality_score_,
            },
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path_obj, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("model.pkl", pickle.dumps(payload))

    @classmethod
    def load(cls, path: str) -> "DemandForecastModel":
        with zipfile.ZipFile(path, "r") as archive:
            payload = pickle.loads(archive.read("model.pkl"))

        model = cls(**payload["init_params"])
        for key, value in payload["state"].items():
            setattr(model, key, value)
        return model

    def summary(self) -> dict[str, Any]:
        self._check_is_fitted()
        return {
            "config": asdict(self.config),
            "date_column": self.date_column,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "group_columns": self.group_columns,
            "quality_score": self.quality_score_,
            "quality_metrics": self.quality_metrics_,
            "n_history_rows": len(self.history_),
            "n_model_features": len(self.encoder_state_["feature_names"]),
        }

    def _build_config(
        self,
        config: ForecastModelConfig | dict | None,
        lags: tuple[int, ...] | list[int] | None,
        rolling_windows: tuple[int, ...] | list[int] | None,
        seasonal_period: int | None,
        min_history: int | None,
        validation_fraction: float | None,
    ) -> ForecastModelConfig:
        if isinstance(config, dict):
            config = ForecastModelConfig.from_dict(config)

        if config is None:
            config = ForecastModelConfig()

        payload = config.to_dict()

        if lags is not None:
            payload["lags"] = tuple(lags)
        if rolling_windows is not None:
            payload["rolling_windows"] = tuple(rolling_windows)
        if seasonal_period is not None:
            payload["seasonal_period"] = seasonal_period
        if min_history is not None:
            payload["min_history"] = min_history
        if validation_fraction is not None:
            payload["validation_fraction"] = validation_fraction

        return ForecastModelConfig.from_dict(payload)

    def _fit_all_models(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        lightgbm_dataset = lgb.Dataset(x, label=y)
        lightgbm_rounds = int(self.config.lightgbm_params.get("n_estimators", 250))
        lightgbm_params = self.config.lightgbm_params.copy()
        lightgbm_params.pop("n_estimators", None)
        lightgbm_params.setdefault("objective", "regression")
        lightgbm_params.setdefault("metric", "l2")
        lightgbm_model = lgb.train(lightgbm_params, lightgbm_dataset, num_boost_round=lightgbm_rounds)

        catboost_params = self.config.catboost_params.copy()
        catboost_params.setdefault("loss_function", "RMSE")
        catboost_params.setdefault("verbose", False)
        catboost_params.setdefault("allow_writing_files", False)
        catboost_model = CatBoostRegressor(**catboost_params)
        catboost_model.fit(x, y)

        return {
            "lightgbm": lightgbm_model,
            "catboost": catboost_model,
        }

    def _predict_with_models(self, models: dict[str, Any], x: np.ndarray) -> np.ndarray:
        weights = self._normalized_ensemble_weights()

        lightgbm_pred = models["lightgbm"].predict(x)
        catboost_pred = models["catboost"].predict(x)

        final_pred = (
            weights["lightgbm"] * lightgbm_pred
            + weights["catboost"] * catboost_pred
        )
        return np.asarray(final_pred, dtype=float)

    def _predict_one_row(self, x: np.ndarray) -> np.ndarray:
        models = {
            "lightgbm": self.lightgbm_model_,
            "catboost": self.catboost_model_,
        }
        return self._predict_with_models(models, x)

    def _normalized_ensemble_weights(self) -> dict[str, float]:
        weights = {
            "lightgbm": float(self.config.ensemble_weights.get("lightgbm", 0.0)),
            "catboost": float(self.config.ensemble_weights.get("catboost", 0.0)),
        }
        total = sum(weights.values())

        if total <= 0:
            return {"lightgbm": 0.5, "catboost": 0.5}

        for key in weights:
            weights[key] = float(weights[key] / total)

        return weights

    def _get_split_index(self, n_rows: int) -> int | None:
        valid_size = max(
            self.config.min_validation_size,
            int(round(n_rows * self.config.validation_fraction)),
        )

        if n_rows - valid_size < 20:
            return None

        return n_rows - valid_size

    def _prepare_future_frame(self, future_df: pd.DataFrame | None, horizon: int | None) -> pd.DataFrame:
        if future_df is None:
            future_horizon = horizon or self.config.default_horizon
            return self._build_empty_future_frame(future_horizon)

        prepared = prepare_dataframe(
            df=future_df,
            date_column=self.date_column,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            group_columns=self.group_columns,
            require_target=False,
        )

        if horizon is not None:
            prepared = prepared.head(horizon).copy()

        return prepared

    def _build_empty_future_frame(self, horizon: int) -> pd.DataFrame:
        dates = pd.date_range(self.last_date_ + pd.Timedelta(days=1), periods=horizon, freq="D")

        if not self.group_columns:
            frame = pd.DataFrame({self.date_column: dates})
            defaults = get_last_known_feature_values(
                self.history_, self.date_column, self.feature_columns
            )
            for column in self.feature_columns:
                frame[column] = defaults.get(column, np.nan)
            return frame

        parts: list[pd.DataFrame] = []
        for _, group_row in self.group_frame_.iterrows():
            part = pd.DataFrame({self.date_column: dates})

            for column in self.group_columns:
                part[column] = group_row[column]

            group_history = self.history_.copy()
            for column in self.group_columns:
                group_history = group_history[group_history[column].eq(group_row[column])]

            defaults = get_last_known_feature_values(
                group_history, self.date_column, self.feature_columns
            )
            for column in self.feature_columns:
                part[column] = defaults.get(column, np.nan)

            parts.append(part)

        return pd.concat(parts, ignore_index=True)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae = float(np.mean(np.abs(y_true - y_pred)))

        denominator = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
        mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)

        smape_denominator = np.where(
            (np.abs(y_true) + np.abs(y_pred)) < 1e-8,
            1.0,
            (np.abs(y_true) + np.abs(y_pred)) / 2.0,
        )
        smape = float(np.mean(np.abs(y_true - y_pred) / smape_denominator) * 100.0)

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 0.0 if ss_tot < 1e-8 else float(1.0 - ss_res / ss_tot)

        return {"mae": mae, "mape": mape, "smape": smape, "r2": r2}

    def _normalize_importance(self, values: Any) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        total = values.sum()
        if total <= 0:
            return np.zeros_like(values, dtype=float)
        return values / total

    def _get_lightgbm_feature_importance(self) -> np.ndarray:
        importance = self.lightgbm_model_.feature_importance(importance_type="gain")
        return self._normalize_importance(importance)

    def _get_catboost_feature_importance(self) -> np.ndarray:
        importance = self.catboost_model_.get_feature_importance()
        return self._normalize_importance(importance)

    def _plot_feature_importance(
        self,
        importance: dict[str, float],
        top_n: int | None,
        show: bool,
        ax: Any | None,
        figsize: tuple[float, float] | None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required to draw feature importance. "
                "Install it or call get_feature_importance(plot=False)."
            ) from exc

        items = list(importance.items())
        if top_n is not None:
            if top_n <= 0:
                raise ValueError("top_n must be a positive integer or None.")
            items = items[:top_n]

        plot_frame = pd.DataFrame(items, columns=["feature", "importance"])
        plot_frame = plot_frame.sort_values("importance", ascending=True)

        if ax is None:
            if figsize is None:
                height = max(4.0, min(18.0, 0.35 * len(plot_frame) + 1.5))
                figsize = (10.0, height)
            _, ax = plt.subplots(figsize=figsize)

        ax.barh(plot_frame["feature"], plot_frame["importance"], color="#2F80ED")
        ax.set_title("Ensemble feature importance")
        ax.set_xlabel("Normalized importance")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)

        if ax.figure is not None:
            ax.figure.tight_layout()

        if show:
            plt.show()

    def _forecast_to_sku_mapping(
        self,
        forecast: pd.DataFrame,
        sku_column: str,
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """
        Преобразует DataFrame прогноза в компактный SKU-словарь.
        """
        if sku_column not in forecast.columns:
            raise ValueError(
                f"SKU column '{sku_column}' is missing in forecast result. "
                f"Pass it in group_columns or include it in future data."
            )

        output = forecast[[self.date_column, sku_column, self.prediction_column]].copy()
        output[sku_column] = output[sku_column].astype(str)
        output[self.prediction_column] = output[self.prediction_column].astype(float)

        unique_dates = output[self.date_column].drop_duplicates().tolist()
        if len(unique_dates) == 1:
            single_day = output.sort_values(sku_column)
            return {
                row[sku_column]: float(row[self.prediction_column])
                for _, row in single_day.iterrows()
            }

        nested: dict[str, dict[str, float]] = {}
        for forecast_date, date_frame in output.groupby(self.date_column, sort=True):
            nested[pd.Timestamp(forecast_date).strftime("%Y-%m-%d")] = {
                row[sku_column]: float(row[self.prediction_column])
                for _, row in date_frame.sort_values(sku_column).iterrows()
            }

        return nested

    def _check_is_fitted(self) -> None:
        if self.lightgbm_model_ is None or self.catboost_model_ is None:
            raise ValueError("Model is not fitted yet. Call fit(...) first.")
        if self.history_ is None:
            raise ValueError("Model history is empty. Call fit(...) first.")
