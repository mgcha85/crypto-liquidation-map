"""XGBoost baseline model for liquidation map trading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class XGBConfig:
    objective: Literal["binary:logistic", "multi:softmax", "reg:squarederror"] = "multi:softmax"
    num_class: int = 3
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    early_stopping_rounds: int = 10
    eval_metric: str = "mlogloss"
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    device: str = "cuda"


FEATURE_COLUMNS = [
    "total_intensity", "long_intensity", "short_intensity", "long_short_ratio",
    "above_below_ratio", "near_1pct_concentration", "near_2pct_concentration", 
    "near_5pct_concentration", "largest_long_cluster_distance", "largest_short_cluster_distance",
    "largest_long_cluster_volume", "largest_short_cluster_volume",
    "top3_long_dist_1", "top3_long_dist_2", "top3_long_dist_3",
    "top3_short_dist_1", "top3_short_dist_2", "top3_short_dist_3",
    "entropy", "skewness",
    "return_1h", "return_6h", "return_12h", "return_24h",
    "volatility_6h", "volatility_24h", "atr_24h", "volume_ma_ratio",
    "wick_ratio_upper", "wick_ratio_lower", "price_position",
]


class XGBoostModel:
    
    def __init__(self, config: XGBConfig | None = None):
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        
        self.config = config or XGBConfig()
        self.model: xgb.XGBClassifier | None = None
        self.feature_names = FEATURE_COLUMNS.copy()
    
    def _prepare_data(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df.select(available_cols).to_numpy()
        
        labels = df["label"].to_numpy()
        y = labels + 1
        
        return X, y
    
    def train(
        self,
        df_train: pl.DataFrame,
        df_val: pl.DataFrame | None = None,
    ) -> dict:
        X_train, y_train = self._prepare_data(df_train)
        
        model_params = {
            "objective": self.config.objective,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "n_estimators": self.config.n_estimators,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "min_child_weight": self.config.min_child_weight,
            "gamma": self.config.gamma,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            "tree_method": self.config.tree_method,
            "device": self.config.device,
            "eval_metric": self.config.eval_metric,
            "early_stopping_rounds": self.config.early_stopping_rounds if df_val is not None else None,
        }
        
        if self.config.objective.startswith("multi:"):
            model_params["num_class"] = self.config.num_class
        
        self.model = xgb.XGBClassifier(**model_params)
        
        if df_val is not None:
            X_val, y_val = self._prepare_data(df_val)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        else:
            self.model.fit(X_train, y_train, verbose=True)
        
        train_preds = self.model.predict(X_train)
        train_acc = (train_preds == y_train).mean()
        
        results = {
            "train_accuracy": train_acc,
            "train_samples": len(y_train),
            "n_features": X_train.shape[1],
        }
        
        if df_val is not None:
            val_preds = self.model.predict(X_val)
            val_acc = (val_preds == y_val).mean()
            results["val_accuracy"] = val_acc
            results["val_samples"] = len(y_val)
        
        return results
    
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df.select(available_cols).to_numpy()
        
        preds = self.model.predict(X)
        return preds - 1
    
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df.select(available_cols).to_numpy()
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pl.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importances_
        
        return pl.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        }).sort("importance", descending=True)
    
    def save(self, path: Path | str):
        if self.model is None:
            raise ValueError("Model not trained")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
    
    def load(self, path: Path | str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    from collections import Counter
    
    accuracy = (y_true == y_pred).mean()
    
    label_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "n_samples": len(y_true),
        "label_distribution": dict(label_counts),
        "pred_distribution": dict(pred_counts),
    }
    
    for label in [-1, 0, 1]:
        mask_true = y_true == label
        mask_pred = y_pred == label
        
        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        label_name = {-1: "sell", 0: "hold", 1: "buy"}[label]
        metrics[f"precision_{label_name}"] = precision
        metrics[f"recall_{label_name}"] = recall
        metrics[f"f1_{label_name}"] = f1
    
    return metrics
