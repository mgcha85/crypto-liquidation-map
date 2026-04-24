"""Walk-forward backtesting with purge and embargo."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator, Literal

import numpy as np
import polars as pl


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    position_size_pct: float = 0.1
    max_positions: int = 1
    
    taker_fee_pct: float = 0.0004
    slippage_bps: float = 5.0
    
    train_months: int = 12
    val_months: int = 3
    test_months: int = 1
    purge_hours: int = 24
    embargo_hours: int = 24
    
    min_train_samples: int = 1000
    
    long_only: bool = False


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    side: Literal["long", "short"]
    size: float
    pnl: float
    fees: float
    slippage: float
    signal: int
    confidence: float


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    num_trades: int
    exposure_pct: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    fold_results: list[dict] = field(default_factory=list)


class WalkForwardValidator:
    
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
    
    def create_folds(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> Iterator[tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
        """
        Create walk-forward folds with purge and embargo.
        
        Yields: (train_df, val_df, test_df) for each fold
        
        Timeline:
        |----TRAIN----|--PURGE--|----VAL----|--EMBARGO--|----TEST----|
        """
        timestamps = df[timestamp_col].sort().to_list()
        start_time = timestamps[0]
        end_time = timestamps[-1]
        
        train_delta = timedelta(days=30 * self.config.train_months)
        val_delta = timedelta(days=30 * self.config.val_months)
        test_delta = timedelta(days=30 * self.config.test_months)
        purge_delta = timedelta(hours=self.config.purge_hours)
        embargo_delta = timedelta(hours=self.config.embargo_hours)
        
        fold_start = start_time
        fold_num = 0
        
        while True:
            train_end = fold_start + train_delta
            val_start = train_end + purge_delta
            val_end = val_start + val_delta
            test_start = val_end + embargo_delta
            test_end = test_start + test_delta
            
            if test_end > end_time:
                break
            
            train_df = df.filter(
                (pl.col(timestamp_col) >= fold_start) & 
                (pl.col(timestamp_col) < train_end)
            )
            
            val_df = df.filter(
                (pl.col(timestamp_col) >= val_start) &
                (pl.col(timestamp_col) < val_end)
            )
            
            test_df = df.filter(
                (pl.col(timestamp_col) >= test_start) &
                (pl.col(timestamp_col) < test_end)
            )
            
            if len(train_df) >= self.config.min_train_samples:
                fold_num += 1
                print(f"Fold {fold_num}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
                yield train_df, val_df, test_df
            
            fold_start = fold_start + test_delta


class Backtester:
    
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
    
    def run(
        self,
        df: pl.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> BacktestResult:
        """
        Run backtest on predictions.
        
        df: must have timestamp, current_price, label columns
        predictions: array of -1, 0, 1 (sell, hold, buy)
        probabilities: optional (N, 3) array of class probabilities
        """
        capital = self.config.initial_capital
        equity = [capital]
        trades: list[Trade] = []
        
        timestamps = df["timestamp"].to_list()
        prices = df["current_price"].to_numpy()
        labels = df["label"].to_numpy()
        
        position = None
        entry_time = None
        entry_price = None
        position_size = 0.0
        
        for i in range(len(predictions) - 1):
            signal = predictions[i]
            confidence = probabilities[i].max() if probabilities is not None else 1.0
            current_price = prices[i]
            next_price = prices[i + 1]
            
            if position is None and signal != 0:
                if signal == 1:
                    side = "long"
                elif signal == -1 and not self.config.long_only:
                    side = "short"
                else:
                    continue
                
                position_value = capital * self.config.position_size_pct
                entry_price = current_price * (1 + self.config.slippage_bps / 10000)
                position_size = position_value / entry_price
                entry_fee = position_value * self.config.taker_fee_pct
                
                position = side
                entry_time = timestamps[i]
                capital -= entry_fee
            
            elif position is not None:
                should_exit = False
                
                if position == "long" and signal == -1:
                    should_exit = True
                elif position == "short" and signal == 1:
                    should_exit = True
                elif signal == 0:
                    should_exit = True
                
                if should_exit:
                    exit_price = next_price * (1 - self.config.slippage_bps / 10000 if position == "long" else 1 + self.config.slippage_bps / 10000)
                    
                    if position == "long":
                        pnl = (exit_price - entry_price) * position_size
                    else:
                        pnl = (entry_price - exit_price) * position_size
                    
                    exit_value = position_size * exit_price
                    exit_fee = exit_value * self.config.taker_fee_pct
                    
                    slippage_cost = position_size * current_price * self.config.slippage_bps / 10000 * 2
                    
                    capital += pnl - exit_fee
                    
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=timestamps[i + 1],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        side=position,
                        size=position_size,
                        pnl=pnl,
                        fees=entry_fee + exit_fee if 'entry_fee' in dir() else exit_fee,
                        slippage=slippage_cost,
                        signal=signal,
                        confidence=confidence,
                    ))
                    
                    position = None
                    entry_time = None
                    entry_price = None
                    position_size = 0.0
            
            equity.append(capital)
        
        equity_arr = np.array(equity)
        return self._compute_metrics(trades, equity_arr)
    
    def _compute_metrics(self, trades: list[Trade], equity: np.ndarray) -> BacktestResult:
        if len(trades) == 0:
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_trade_pnl=0.0,
                num_trades=0,
                exposure_pct=0.0,
                trades=[],
                equity_curve=equity,
            )
        
        total_return = (equity[-1] / equity[0]) - 1
        
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24)
        else:
            sharpe_ratio = 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252 * 24)
        else:
            sortino_ratio = 0.0
        
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        
        if max_drawdown > 0:
            annual_return = total_return * (365 * 24 / len(equity))
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = 0.0
        
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0.0
        
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_trade_pnl = np.mean(pnls)
        
        total_hours = len(equity)
        in_position_hours = sum(1 for _ in trades)
        exposure_pct = in_position_hours / total_hours if total_hours > 0 else 0.0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            num_trades=len(trades),
            exposure_pct=exposure_pct,
            trades=trades,
            equity_curve=equity,
        )


def run_walk_forward_backtest(
    df: pl.DataFrame,
    model_class,
    model_config,
    backtest_config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run full walk-forward backtest with model training per fold.
    
    1. Create folds with purge/embargo
    2. Train model on train set
    3. Validate on val set (early stopping)
    4. Test on test set
    5. Aggregate results
    """
    config = backtest_config or BacktestConfig()
    validator = WalkForwardValidator(config)
    backtester = Backtester(config)
    
    all_trades = []
    all_equity = [config.initial_capital]
    fold_results = []
    
    for fold_num, (train_df, val_df, test_df) in enumerate(validator.create_folds(df)):
        print(f"\n=== Fold {fold_num + 1} ===")
        
        model = model_class(model_config)
        train_result = model.train(train_df, val_df)
        print(f"Training: {train_result}")
        
        test_preds = model.predict(test_df)
        test_proba = model.predict_proba(test_df) if hasattr(model, 'predict_proba') else None
        
        fold_result = backtester.run(test_df, test_preds, test_proba)
        
        fold_results.append({
            "fold": fold_num + 1,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "total_return": fold_result.total_return,
            "sharpe_ratio": fold_result.sharpe_ratio,
            "max_drawdown": fold_result.max_drawdown,
            "num_trades": fold_result.num_trades,
            "win_rate": fold_result.win_rate,
        })
        
        all_trades.extend(fold_result.trades)
        
        if len(fold_result.equity_curve) > 1:
            scale = all_equity[-1] / fold_result.equity_curve[0]
            scaled_equity = fold_result.equity_curve[1:] * scale
            all_equity.extend(scaled_equity.tolist())
    
    final_equity = np.array(all_equity)
    final_result = backtester._compute_metrics(all_trades, final_equity)
    final_result.fold_results = fold_results
    
    return final_result
