from __future__ import annotations

import warnings

import numpy as np


class RidgeRegressor:
    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha
        self.x_mean = None
        self.x_std = None
        self.y_mean = 0.0
        self.coef = None

    def fit(self, x, y, sample_weight=None) -> "RidgeRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        sample_weight = np.asarray(sample_weight, dtype=float)
        sample_weight = sample_weight / sample_weight.mean()
        self.x_mean = np.average(x, axis=0, weights=sample_weight)
        self.x_std = np.sqrt(np.average((x - self.x_mean) ** 2, axis=0, weights=sample_weight))
        self.x_std[self.x_std == 0] = 1.0
        self.y_mean = float(np.average(y, weights=sample_weight))

        x_scaled = (x - self.x_mean) / self.x_std
        y_centered = y - self.y_mean
        weighted_x = x_scaled * np.sqrt(sample_weight)[:, None]
        weighted_y = y_centered * np.sqrt(sample_weight)
        xtx = weighted_x.T @ weighted_x
        penalty = self.alpha * np.eye(xtx.shape[0])
        self.coef = np.linalg.solve(xtx + penalty, weighted_x.T @ weighted_y)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        x_scaled = (x - self.x_mean) / self.x_std
        preds = self.y_mean + x_scaled @ self.coef
        return np.maximum(preds, 0.0)


class NonNegativeRegressor:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y, sample_weight=None) -> "NonNegativeRegressor":
        if sample_weight is None:
            self.estimator.fit(x, y)
        else:
            self.estimator.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return np.maximum(self.estimator.predict(x), 0.0)


def make_regressor(name: str, alpha: float = 20.0, random_state: int = 42):
    name = name.lower()
    if name == "ridge":
        return RidgeRegressor(alpha=alpha)
    if name == "extra":
        from sklearn.ensemble import ExtraTreesRegressor

        return NonNegativeRegressor(
            ExtraTreesRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=12,
                random_state=random_state,
                n_jobs=-1,
            )
        )
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return NonNegativeRegressor(
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.04,
                l2_regularization=0.1,
                min_samples_leaf=8,
                random_state=random_state,
            )
        )
    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return NonNegativeRegressor(
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=8,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                verbosity=-1,
            )
        )
    raise ValueError(f"unknown model: {name}")


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))
