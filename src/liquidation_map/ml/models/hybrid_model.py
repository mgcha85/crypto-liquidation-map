"""
Hybrid ML/DL model combining:
- 200 candle time series (1D conv or LSTM)
- 200 liquidation map time series (1D conv)
- 31 engineered ML features (MLP branch)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class HybridConfig:
    candle_length: int = 200
    candle_features: int = 5
    
    liq_map_length: int = 200
    liq_map_bins: int = 64
    
    ml_features: int = 31
    
    num_classes: int = 2
    
    conv_channels: list = None
    lstm_hidden: int = 64
    lstm_layers: int = 2
    
    fc_hidden: int = 128
    dropout: float = 0.3
    
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10
    
    device: str = "cuda"
    
    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128]


class HybridDataset:
    
    def __init__(
        self,
        candles: np.ndarray,
        liq_maps: np.ndarray,
        ml_features: np.ndarray,
        labels: np.ndarray,
    ):
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        
        self.candles = torch.FloatTensor(candles)
        self.liq_maps = torch.FloatTensor(liq_maps)
        self.ml_features = torch.FloatTensor(ml_features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.candles[idx],
            self.liq_maps[idx],
            self.ml_features[idx],
            self.labels[idx],
        )


class CandleBranch(nn.Module if HAS_TORCH else object):
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(config.candle_features, config.conv_channels[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(config.conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(config.conv_channels[0], config.conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(config.conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        self.lstm = nn.LSTM(
            input_size=config.conv_channels[1],
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
        )
        
        self.output_dim = config.lstm_hidden * 2
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        return torch.cat([h_forward, h_backward], dim=1)


class LiqMapBranch(nn.Module if HAS_TORCH else object):
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, config.conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(config.conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(config.conv_channels[0], config.conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(config.conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(config.conv_channels[1], config.conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(config.conv_channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.output_dim = config.conv_channels[2]
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(x.size(0), -1)


class MLFeatureBranch(nn.Module if HAS_TORCH else object):
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(config.ml_features, config.fc_hidden),
            nn.BatchNorm1d(config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden, config.fc_hidden // 2),
            nn.ReLU(),
        )
        
        self.output_dim = config.fc_hidden // 2
    
    def forward(self, x):
        return self.mlp(x)


class HybridModel(nn.Module if HAS_TORCH else object):
    
    def __init__(self, config: HybridConfig):
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        
        super().__init__()
        self.config = config
        
        self.candle_branch = CandleBranch(config)
        self.liq_map_branch = LiqMapBranch(config)
        self.ml_branch = MLFeatureBranch(config)
        
        combined_dim = (
            self.candle_branch.output_dim +
            self.liq_map_branch.output_dim +
            self.ml_branch.output_dim
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, config.fc_hidden),
            nn.BatchNorm1d(config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden, config.fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden // 2, config.num_classes),
        )
    
    def forward(self, candles, liq_maps, ml_features):
        candle_out = self.candle_branch(candles)
        liq_out = self.liq_map_branch(liq_maps)
        ml_out = self.ml_branch(ml_features)
        
        combined = torch.cat([candle_out, liq_out, ml_out], dim=1)
        return self.fusion(combined)


class HybridTrainer:
    
    def __init__(self, config: HybridConfig | None = None):
        if not HAS_TORCH:
            raise ImportError("torch not installed. Run: pip install torch")
        
        self.config = config or HybridConfig()
        self.model: HybridModel | None = None
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
    
    def train(
        self,
        train_candles: np.ndarray,
        train_liq_maps: np.ndarray,
        train_ml_features: np.ndarray,
        train_labels: np.ndarray,
        val_candles: np.ndarray | None = None,
        val_liq_maps: np.ndarray | None = None,
        val_ml_features: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
    ) -> dict:
        train_dataset = HybridDataset(train_candles, train_liq_maps, train_ml_features, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        val_loader = None
        if all(x is not None for x in [val_candles, val_liq_maps, val_ml_features, val_labels]):
            val_dataset = HybridDataset(val_candles, val_liq_maps, val_ml_features, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        
        self.model = HybridModel(self.config).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for candles, liq_maps, ml_features, labels in train_loader:
                candles = candles.to(self.device)
                liq_maps = liq_maps.to(self.device)
                ml_features = ml_features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(candles, liq_maps, ml_features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * candles.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            else:
                print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
        
        return {
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history["train_loss"]),
            "history": history,
        }
    
    def _evaluate(self, loader, criterion) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for candles, liq_maps, ml_features, labels in loader:
                candles = candles.to(self.device)
                liq_maps = liq_maps.to(self.device)
                ml_features = ml_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(candles, liq_maps, ml_features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * candles.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / total, correct / total
    
    def predict(
        self,
        candles: np.ndarray,
        liq_maps: np.ndarray,
        ml_features: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        candles_t = torch.FloatTensor(candles).to(self.device)
        liq_maps_t = torch.FloatTensor(liq_maps).to(self.device)
        ml_features_t = torch.FloatTensor(ml_features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(candles_t, liq_maps_t, ml_features_t)
            _, predicted = outputs.max(1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(
        self,
        candles: np.ndarray,
        liq_maps: np.ndarray,
        ml_features: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        candles_t = torch.FloatTensor(candles).to(self.device)
        liq_maps_t = torch.FloatTensor(liq_maps).to(self.device)
        ml_features_t = torch.FloatTensor(ml_features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(candles_t, liq_maps_t, ml_features_t)
            proba = F.softmax(outputs, dim=1)
        
        return proba.cpu().numpy()
    
    def save(self, path: Path | str):
        if self.model is None:
            raise ValueError("Model not trained")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, path)
    
    def load(self, path: Path | str):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint["config"]
        self.model = HybridModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
