from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from sklearn import metrics

def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    if n_samples <= n_features + 1:
        return float("nan")
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

@dataclass
class RegressionReport:
    r2: float
    r2_adjusted: float
    mse: float
    rmse: float
    mae: float
    mape: float
    median_ae: float
    explained_variance: float

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> "RegressionReport":
        r2 = metrics.r2_score(y_true, y_pred)
        r2_adj = adjusted_r2(r2, n_samples=len(y_true), n_features=n_features)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = metrics.mean_absolute_error(y_true, y_pred)
        # MAPE can be inf if y_true has zeros; handle safely
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.abs((y_true - y_pred) / y_true)
            ape = ape[~np.isinf(ape) & ~np.isnan(ape)]
            mape = float(np.mean(ape)) if ape.size > 0 else float("nan")
        medae = metrics.median_absolute_error(y_true, y_pred)
        evs = metrics.explained_variance_score(y_true, y_pred)
        return cls(
            r2=float(r2),
            r2_adjusted=float(r2_adj),
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape),
            median_ae=float(medae),
            explained_variance=float(evs),
        )

    def to_dict(self):
        return asdict(self)
