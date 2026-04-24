#!/usr/bin/env python3
"""
Comprehensive Benchmark Pipeline
Phase 1-5: Baseline → Strategy Combination → Optimization → Final Benchmark
"""

import sys
sys.path.insert(0, "/mnt/data/projects/crypto-liquidation-map")

import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import polars as pl

from src.liquidation_map.ml.multi_timeframe import (
    MultiTimeframeStrategy,
    MultiTimeframeLoader,
    PRICE_BUCKETS,
)
from src.liquidation_map.ml.features import FeatureExtractor
from src.liquidation_map.ml.labeling import TripleBarrierLabeler, BarrierConfig
from src.liquidation_map.ml.backtest import Backtester, BacktestConfig
from src.liquidation_map.ml.models.xgboost_model import XGBoostModel, XGBConfig


@dataclass
class BenchmarkResult:
    name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    alpha: float
    
    def to_dict(self):
        return asdict(self)


class BenchmarkPipeline:
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        start_date: str = "2025-06-01",
        end_date: str = "2026-04-22",
        train_cutoff: str = "2026-01-01",
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.train_cutoff = train_cutoff
        self.cutoff_dt = datetime.strptime(train_cutoff, "%Y-%m-%d")
        
        self.loader = MultiTimeframeLoader()
        self.strategy = MultiTimeframeStrategy(symbol, train_cutoff=train_cutoff)
        self.price_bucket = PRICE_BUCKETS.get(symbol, 250.0)
        self.feature_extractor = FeatureExtractor(price_bucket_size=self.price_bucket)
        
        self.backtest_config = BacktestConfig(
            position_size_pct=0.1,
            taker_fee_pct=0.0004,
            slippage_bps=5.0,
        )
        self.backtester = Backtester(self.backtest_config)
        
        self.results: list[BenchmarkResult] = []
        self.df_1h: Optional[pl.DataFrame] = None
        self.df_5m: Optional[pl.DataFrame] = None
        self.df_oi: Optional[pl.DataFrame] = None
        self.bh_return: float = 0.0
    
    def load_data(self):
        print("Loading data...")
        self.df_1h = self.loader.get_candles(self.symbol, self.start_date, self.end_date, "1h")
        self.df_5m = self.loader.get_candles(self.symbol, self.start_date, self.end_date, "5m")
        self.df_oi = self.strategy.load_oi_data(self.start_date, self.end_date)
        
        prices_test = self.df_1h.filter(pl.col("timestamp") >= self.cutoff_dt)["close"].to_list()
        self.bh_return = (prices_test[-1] / prices_test[0] - 1) if prices_test else 0
        
        print(f"  1h: {len(self.df_1h)}, 5m: {len(self.df_5m)}, OI: {len(self.df_oi)}")
        print(f"  Buy & Hold: {self.bh_return*100:.2f}%")
    
    def _build_features(
        self,
        df_klines: pl.DataFrame,
        horizon: int,
        pt: float,
        sl: float,
        step: int = 6,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        barrier_config = BarrierConfig(profit_take=pt, stop_loss=sl, horizon=horizon)
        labeler = TripleBarrierLabeler(barrier_config)
        df_labels = labeler.compute_labels(df_klines, return_details=True)
        
        timestamps = df_klines["timestamp"].to_list()
        lookback_hours = 50
        lookback_td = timedelta(hours=lookback_hours)
        
        rows = []
        for i in range(lookback_hours * 12, len(timestamps) - horizon, step):
            ts = timestamps[i]
            window_start = ts - lookback_td
            
            klines_window = df_klines.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            oi_window = self.df_oi.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            
            if len(oi_window) < 50:
                continue
            
            label_row = df_labels.filter(pl.col("timestamp") == ts)
            if label_row.is_empty():
                continue
            
            current_price = float(df_klines.filter(pl.col("timestamp") == ts)["close"][0])
            
            liq_features = self.feature_extractor.extract_window_features(
                oi_window, klines_window, current_price
            )
            candle_features = self.strategy._extract_candle_features_scaled(klines_window, 60)
            
            row = {
                "timestamp": ts,
                "current_price": current_price,
                "label": int(label_row["label"][0]),
                **liq_features,
                **candle_features,
            }
            rows.append(row)
        
        if not rows:
            return pl.DataFrame(), pl.DataFrame()
        
        df_features = pl.DataFrame(rows)
        
        df_train = df_features.filter(pl.col("timestamp") < self.cutoff_dt)
        df_test = df_features.filter(pl.col("timestamp") >= self.cutoff_dt)
        
        df_train = df_train.filter(pl.col("label") != 0).with_columns([
            pl.when(pl.col("label") == -1).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("label")
        ])
        df_test = df_test.filter(pl.col("label") != 0).with_columns([
            pl.when(pl.col("label") == -1).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("label")
        ])
        
        return df_train, df_test
    
    def _train_model(self, df_train: pl.DataFrame) -> XGBoostModel:
        label_counts = df_train["label"].value_counts().to_dicts()
        sell_count = sum(r["count"] for r in label_counts if r["label"] == -1)
        buy_count = sum(r["count"] for r in label_counts if r["label"] == 0)
        scale_pos_weight = sell_count / max(buy_count, 1)
        
        xgb_config = XGBConfig(
            objective="binary:logistic",
            num_class=2,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        
        model = XGBoostModel(xgb_config)
        model.train(df_train)
        return model
    
    def _backtest(
        self, 
        model: XGBoostModel, 
        df_test: pl.DataFrame,
        name: str,
    ) -> BenchmarkResult:
        preds = model.predict(df_test)
        proba = model.predict_proba(df_test)
        signals = np.where(preds == 0, 1, -1)
        
        result = self.backtester.run(df_test, signals, proba)
        
        return BenchmarkResult(
            name=name,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            num_trades=result.num_trades,
            alpha=result.total_return - self.bh_return,
        )
    
    def phase1_baseline(self):
        print("\n" + "="*60)
        print("PHASE 1: BASELINE")
        print("="*60)
        
        print("\n[1] XGBoost 1h Baseline (PT=2%, SL=1%, H=24)")
        df_train, df_test = self._build_features(self.df_1h, horizon=24, pt=0.02, sl=0.01)
        if len(df_train) > 50 and len(df_test) > 20:
            model = self._train_model(df_train)
            result = self._backtest(model, df_test, "XGB 1h Baseline")
            self.results.append(result)
            print(f"    Return: {result.total_return*100:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        
        print("\n[2] XGBoost Long Horizon (PT=2%, SL=1%, H=48)")
        df_train, df_test = self._build_features(self.df_1h, horizon=48, pt=0.02, sl=0.01)
        if len(df_train) > 50 and len(df_test) > 20:
            model_48h = self._train_model(df_train)
            result = self._backtest(model_48h, df_test, "XGB Long Horizon (48h)")
            self.results.append(result)
            self._model_48h = model_48h
            self._df_test_48h = df_test
            print(f"    Return: {result.total_return*100:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
        
        print("\n[3] 5m→1h Ensemble")
        df_train_5m, df_test_5m = self._build_features(self.df_5m, horizon=288, pt=0.02, sl=0.01, step=12)
        df_train_1h, df_test_1h = self._build_features(self.df_1h, horizon=24, pt=0.02, sl=0.01)
        
        if len(df_train_5m) > 50 and len(df_train_1h) > 50:
            model_5m = self._train_model(df_train_5m)
            model_1h = self._train_model(df_train_1h)
            
            preds_5m = model_5m.predict(df_test_5m)
            signals_5m = np.where(preds_5m == 0, 1, -1)
            ts_5m = df_test_5m["timestamp"].to_list()
            
            preds_1h = model_1h.predict(df_test_1h)
            proba_1h = model_1h.predict_proba(df_test_1h)
            signals_1h = np.where(preds_1h == 0, 1, -1)
            ts_1h = df_test_1h["timestamp"].to_list()
            
            ensemble_signals = []
            ensemble_proba = []
            ensemble_ts = []
            ensemble_prices = []
            ensemble_labels = []
            
            for i, ts in enumerate(ts_1h):
                window_start = ts - timedelta(hours=6)
                window_end = ts + timedelta(minutes=30)
                
                recent_5m = [signals_5m[j] for j, t5 in enumerate(ts_5m) if window_start <= t5 <= window_end]
                
                if not recent_5m:
                    continue
                
                avg_5m = np.mean(recent_5m)
                sig_1h = signals_1h[i]
                
                if (sig_1h == 1 and avg_5m > 0) or (sig_1h == -1 and avg_5m < 0):
                    ensemble_signals.append(sig_1h)
                    ensemble_proba.append(proba_1h[i])
                    ensemble_ts.append(ts)
                    ensemble_prices.append(df_test_1h["current_price"][i])
                    ensemble_labels.append(df_test_1h["label"][i])
            
            if len(ensemble_signals) >= 5:
                df_ensemble = pl.DataFrame({
                    "timestamp": ensemble_ts,
                    "current_price": ensemble_prices,
                    "label": ensemble_labels,
                })
                res = self.backtester.run(df_ensemble, np.array(ensemble_signals), np.array(ensemble_proba))
                
                result = BenchmarkResult(
                    name="5m→1h Ensemble",
                    total_return=res.total_return,
                    sharpe_ratio=res.sharpe_ratio,
                    max_drawdown=res.max_drawdown,
                    win_rate=res.win_rate,
                    profit_factor=res.profit_factor,
                    num_trades=res.num_trades,
                    alpha=res.total_return - self.bh_return,
                )
                self.results.append(result)
                print(f"    Return: {result.total_return*100:.2f}%, Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.num_trades}")
                
                self._model_5m = model_5m
                self._model_1h = model_1h
                self._df_test_5m = df_test_5m
                self._df_test_1h = df_test_1h
    
    def phase2_combination(self):
        print("\n" + "="*60)
        print("PHASE 2: STRATEGY COMBINATION")
        print("="*60)
        
        if not hasattr(self, '_model_48h') or not hasattr(self, '_model_1h'):
            print("Models not trained. Run phase1 first.")
            return
        
        preds_5m = self._model_5m.predict(self._df_test_5m)
        signals_5m = np.where(preds_5m == 0, 1, -1)
        ts_5m = self._df_test_5m["timestamp"].to_list()
        
        preds_1h = self._model_1h.predict(self._df_test_1h)
        proba_1h = self._model_1h.predict_proba(self._df_test_1h)
        signals_1h = np.where(preds_1h == 0, 1, -1)
        ts_1h = self._df_test_1h["timestamp"].to_list()
        
        preds_48h = self._model_48h.predict(self._df_test_48h)
        proba_48h = self._model_48h.predict_proba(self._df_test_48h)
        signals_48h = np.where(preds_48h == 0, 1, -1)
        ts_48h = self._df_test_48h["timestamp"].to_list()
        
        sig_48h_dict = {ts: (sig, prob) for ts, sig, prob in zip(ts_48h, signals_48h, proba_48h)}
        
        print("\n[1] Agreement Filter: Ensemble + Long Horizon 동의시만")
        combo_signals = []
        combo_proba = []
        combo_ts = []
        combo_prices = []
        combo_labels = []
        
        for i, ts in enumerate(ts_1h):
            window_start = ts - timedelta(hours=6)
            window_end = ts + timedelta(minutes=30)
            
            recent_5m = [signals_5m[j] for j, t5 in enumerate(ts_5m) if window_start <= t5 <= window_end]
            if not recent_5m:
                continue
            
            avg_5m = np.mean(recent_5m)
            sig_1h = signals_1h[i]
            
            ensemble_agree = (sig_1h == 1 and avg_5m > 0) or (sig_1h == -1 and avg_5m < 0)
            
            nearest_48h = None
            min_diff = timedelta(hours=24)
            for t48 in ts_48h:
                diff = abs(ts - t48)
                if diff < min_diff:
                    min_diff = diff
                    nearest_48h = t48
            
            if nearest_48h is None or nearest_48h not in sig_48h_dict:
                continue
            
            sig_48, prob_48 = sig_48h_dict[nearest_48h]
            horizon_agree = (sig_1h == sig_48)
            
            if ensemble_agree and horizon_agree:
                combo_signals.append(sig_1h)
                avg_prob = (proba_1h[i] + prob_48) / 2
                combo_proba.append(avg_prob)
                combo_ts.append(ts)
                combo_prices.append(self._df_test_1h["current_price"][i])
                combo_labels.append(self._df_test_1h["label"][i])
        
        if len(combo_signals) >= 3:
            df_combo = pl.DataFrame({
                "timestamp": combo_ts,
                "current_price": combo_prices,
                "label": combo_labels,
            })
            res = self.backtester.run(df_combo, np.array(combo_signals), np.array(combo_proba))
            
            result = BenchmarkResult(
                name="Ensemble + 48h Agreement",
                total_return=res.total_return,
                sharpe_ratio=res.sharpe_ratio,
                max_drawdown=res.max_drawdown,
                win_rate=res.win_rate,
                profit_factor=res.profit_factor,
                num_trades=res.num_trades,
                alpha=res.total_return - self.bh_return,
            )
            self.results.append(result)
            print(f"    Return: {result.total_return*100:.2f}%, Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.num_trades}")
        else:
            print(f"    Only {len(combo_signals)} trades - insufficient")
        
        print("\n[2] Weighted Voting (0.6 Ensemble + 0.4 Long Horizon)")
        for w_ens in [0.7, 0.6, 0.5]:
            w_48h = 1.0 - w_ens
            
            weighted_signals = []
            weighted_proba = []
            weighted_ts = []
            weighted_prices = []
            weighted_labels = []
            
            for i, ts in enumerate(ts_1h):
                window_start = ts - timedelta(hours=6)
                window_end = ts + timedelta(minutes=30)
                
                recent_5m = [signals_5m[j] for j, t5 in enumerate(ts_5m) if window_start <= t5 <= window_end]
                if not recent_5m:
                    continue
                
                avg_5m = np.mean(recent_5m)
                sig_1h = signals_1h[i]
                
                ensemble_agree = (sig_1h == 1 and avg_5m > 0) or (sig_1h == -1 and avg_5m < 0)
                if not ensemble_agree:
                    continue
                
                nearest_48h = None
                min_diff = timedelta(hours=24)
                for t48 in ts_48h:
                    diff = abs(ts - t48)
                    if diff < min_diff:
                        min_diff = diff
                        nearest_48h = t48
                
                if nearest_48h is None or nearest_48h not in sig_48h_dict:
                    continue
                
                sig_48, prob_48 = sig_48h_dict[nearest_48h]
                
                weighted_sig = w_ens * sig_1h + w_48h * sig_48
                final_sig = 1 if weighted_sig > 0 else -1
                
                weighted_signals.append(final_sig)
                avg_prob = w_ens * proba_1h[i] + w_48h * prob_48
                weighted_proba.append(avg_prob)
                weighted_ts.append(ts)
                weighted_prices.append(self._df_test_1h["current_price"][i])
                weighted_labels.append(self._df_test_1h["label"][i])
            
            if len(weighted_signals) >= 5:
                df_weighted = pl.DataFrame({
                    "timestamp": weighted_ts,
                    "current_price": weighted_prices,
                    "label": weighted_labels,
                })
                res = self.backtester.run(df_weighted, np.array(weighted_signals), np.array(weighted_proba))
                
                result = BenchmarkResult(
                    name=f"Weighted ({w_ens:.1f}E+{w_48h:.1f}H)",
                    total_return=res.total_return,
                    sharpe_ratio=res.sharpe_ratio,
                    max_drawdown=res.max_drawdown,
                    win_rate=res.win_rate,
                    profit_factor=res.profit_factor,
                    num_trades=res.num_trades,
                    alpha=res.total_return - self.bh_return,
                )
                self.results.append(result)
                print(f"    [{w_ens:.1f}/{w_48h:.1f}] Return: {result.total_return*100:.2f}%, Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.num_trades}")
    
    def phase3_optimization(self):
        print("\n" + "="*60)
        print("PHASE 3: OPTIMIZATION ALGORITHMS")
        print("="*60)
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("Optuna not installed. Skipping.")
            return
        
        print("\n[1] Optuna: XGBoost Hyperparameter Optimization")
        
        df_train_base, df_test_base = self._build_features(self.df_1h, horizon=48, pt=0.02, sl=0.01)
        
        if len(df_train_base) < 50:
            print("    Insufficient training data")
            return
        
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 2),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            }
            
            label_counts = df_train_base["label"].value_counts().to_dicts()
            sell_count = sum(r["count"] for r in label_counts if r["label"] == -1)
            buy_count = sum(r["count"] for r in label_counts if r["label"] == 0)
            scale_pos_weight = sell_count / max(buy_count, 1)
            
            xgb_config = XGBConfig(
                objective="binary:logistic",
                num_class=2,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                **params,
            )
            
            model = XGBoostModel(xgb_config)
            model.train(df_train_base)
            
            preds = model.predict(df_test_base)
            proba = model.predict_proba(df_test_base)
            signals = np.where(preds == 0, 1, -1)
            
            result = self.backtester.run(df_test_base, signals, proba)
            
            return result.sharpe_ratio
        
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        print(f"    Best Sharpe: {study.best_value:.2f}")
        print(f"    Best params: {study.best_params}")
        
        best_params = study.best_params
        label_counts = df_train_base["label"].value_counts().to_dicts()
        sell_count = sum(r["count"] for r in label_counts if r["label"] == -1)
        buy_count = sum(r["count"] for r in label_counts if r["label"] == 0)
        
        xgb_config = XGBConfig(
            objective="binary:logistic",
            num_class=2,
            scale_pos_weight=sell_count / max(buy_count, 1),
            eval_metric="logloss",
            **best_params,
        )
        
        model = XGBoostModel(xgb_config)
        model.train(df_train_base)
        result = self._backtest(model, df_test_base, "Optuna-Optimized XGB")
        self.results.append(result)
        
        print("\n[2] DEAP: Trading Parameter Evolution (PT/SL/Horizon)")
        
        try:
            from deap import base, creator, tools, algorithms
        except ImportError:
            print("    DEAP not installed. Skipping genetic optimization.")
            return
        
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("pt", np.random.uniform, 0.015, 0.04)
        toolbox.register("sl", np.random.uniform, 0.005, 0.02)
        toolbox.register("horizon", np.random.randint, 12, 72)
        
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.pt, toolbox.sl, toolbox.horizon), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        eval_cache = {}
        eval_count = [0]
        
        def evaluate(individual):
            pt, sl, horizon = individual
            horizon = int(horizon)
            
            key = (round(pt, 3), round(sl, 3), horizon)
            if key in eval_cache:
                return eval_cache[key]
            
            eval_count[0] += 1
            if eval_count[0] % 5 == 0:
                print(f"    Evaluating {eval_count[0]}...", flush=True)
            
            try:
                df_train, df_test = self._build_features(self.df_1h, horizon=horizon, pt=pt, sl=sl, step=24)
                
                if len(df_train) < 30 or len(df_test) < 10:
                    return (-10.0,)
                
                model = self._train_model(df_train)
                preds = model.predict(df_test)
                proba = model.predict_proba(df_test)
                signals = np.where(preds == 0, 1, -1)
                
                result = self.backtester.run(df_test, signals, proba)
                
                fitness = result.sharpe_ratio - result.max_drawdown * 2
                
                eval_cache[key] = (fitness,)
                return (fitness,)
            except Exception as e:
                return (-10.0,)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        print("    Running 5 generations with population 10...")
        pop = toolbox.population(n=10)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        
        pop, log = algorithms.eaSimple(
            pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=5,
            stats=stats, halloffame=hof, verbose=False
        )
        
        best = hof[0]
        best_pt, best_sl, best_horizon = best[0], best[1], int(best[2])
        
        print(f"    Best: PT={best_pt:.3f}, SL={best_sl:.3f}, Horizon={best_horizon}")
        
        df_train, df_test = self._build_features(self.df_1h, horizon=best_horizon, pt=best_pt, sl=best_sl)
        if len(df_train) > 30 and len(df_test) > 10:
            model = self._train_model(df_train)
            result = self._backtest(model, df_test, f"DEAP-Evolved (PT={best_pt:.2f},SL={best_sl:.2f},H={best_horizon})")
            self.results.append(result)
    
    def phase5_summary(self):
        print("\n" + "="*60)
        print("PHASE 5: COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\nBuy & Hold: {self.bh_return*100:.2f}%")
        print("\n" + "-"*100)
        print(f"{'Strategy':<35} {'Return':>10} {'Alpha':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>7}")
        print("-"*100)
        
        sorted_results = sorted(self.results, key=lambda x: x.sharpe_ratio, reverse=True)
        
        for r in sorted_results:
            ret_str = f"{r.total_return*100:+.2f}%"
            alpha_str = f"{r.alpha*100:+.2f}%"
            sharpe_str = f"{r.sharpe_ratio:.2f}"
            dd_str = f"{r.max_drawdown*100:.2f}%"
            wr_str = f"{r.win_rate*100:.1f}%"
            pf_str = f"{r.profit_factor:.2f}"
            
            print(f"{r.name:<35} {ret_str:>10} {alpha_str:>10} {sharpe_str:>8} {dd_str:>8} {wr_str:>8} {pf_str:>6} {r.num_trades:>7}")
        
        print("-"*100)
        
        if sorted_results:
            best = sorted_results[0]
            print(f"\n🏆 Best Strategy (by Sharpe): {best.name}")
            print(f"   Return: {best.total_return*100:+.2f}%, Alpha: {best.alpha*100:+.2f}%, Sharpe: {best.sharpe_ratio:.2f}")
        
        output_dir = Path("data/train")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output = {
            "symbol": self.symbol,
            "test_period": f"{self.train_cutoff} to {self.end_date}",
            "buy_hold_return": self.bh_return,
            "generated_at": datetime.now().isoformat(),
            "results": [r.to_dict() for r in sorted_results],
        }
        
        with open(output_dir / f"{self.symbol.lower()}_benchmark.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved to {output_dir / f'{self.symbol.lower()}_benchmark.json'}")
        
        return sorted_results
    
    def run_all(self):
        self.load_data()
        self.phase1_baseline()
        self.phase2_combination()
        self.phase3_optimization()
        return self.phase5_summary()


if __name__ == "__main__":
    pipeline = BenchmarkPipeline()
    pipeline.run_all()
