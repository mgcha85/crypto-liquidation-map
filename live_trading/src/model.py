from pathlib import Path
from typing import Literal
import numpy as np
import polars as pl

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .config import get_xgb_params_dict, FEATURE_COLUMNS


Signal = Literal[-1, 0, 1]


class TradingModel:
    
    def __init__(self, model_path: Path | str | None = None):
        if not HAS_XGBOOST:
            raise ImportError("xgboost required: pip install xgboost")
        
        self.model: xgb.XGBClassifier | None = None
        self.feature_names = FEATURE_COLUMNS.copy()
        
        if model_path:
            self.load(model_path)
    
    def load(self, path: Path | str) -> None:
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
    
    def predict(self, df: pl.DataFrame) -> Signal:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df.select(available_cols).to_numpy()
        
        dmatrix = xgb.DMatrix(X, feature_names=available_cols)
        booster = self.model.get_booster()
        raw_preds = booster.predict(dmatrix)
        
        if len(raw_preds.shape) > 1:
            pred = int(raw_preds[0].argmax())
        else:
            pred = int(raw_preds[0] > 0.5)
        
        signal: Signal = pred - 1
        return signal
    
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df.select(available_cols).to_numpy()
        
        dmatrix = xgb.DMatrix(X, feature_names=available_cols)
        booster = self.model.get_booster()
        return booster.predict(dmatrix)
    
    def train_from_config(self, df_train: pl.DataFrame, df_val: pl.DataFrame | None = None) -> dict:
        params = get_xgb_params_dict()
        
        available_cols = [c for c in self.feature_names if c in df_train.columns]
        X_train = df_train.select(available_cols).to_numpy()
        y_train = df_train["label"].to_numpy() + 1
        
        self.model = xgb.XGBClassifier(**params)
        
        if df_val is not None:
            X_val = df_val.select(available_cols).to_numpy()
            y_val = df_val["label"].to_numpy() + 1
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        train_preds = self.model.predict(X_train)
        train_acc = (train_preds == y_train).mean()
        
        return {"train_accuracy": float(train_acc), "n_samples": len(y_train)}
    
    def save(self, path: Path | str) -> None:
        if self.model is None:
            raise ValueError("Model not trained")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
