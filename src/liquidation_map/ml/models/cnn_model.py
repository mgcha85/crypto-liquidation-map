"""CNN model for 2D liquidation heatmap classification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
class CNNConfig:
    input_height: int = 50
    input_width: int = 128
    num_classes: int = 3
    
    conv1_channels: int = 32
    conv2_channels: int = 64
    conv3_channels: int = 128
    
    kernel_size: int = 3
    pool_size: int = 2
    dropout: float = 0.3
    
    fc_hidden: int = 256
    
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10
    
    device: str = "cuda"


class LiquidationHeatmapDataset:
    
    def __init__(self, heatmaps: np.ndarray, labels: np.ndarray):
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        
        self.heatmaps = torch.FloatTensor(heatmaps).unsqueeze(1)
        self.labels = torch.LongTensor(labels + 1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.heatmaps[idx], self.labels[idx]


class LiquidationCNN(nn.Module if HAS_TORCH else object):
    """
    CNN architecture for 2D liquidation heatmap.
    
    Input: (batch, 1, height=50, width=128)
    Output: (batch, num_classes=3) logits
    
    Architecture:
    - 3 conv blocks with batch norm and max pooling
    - Global average pooling
    - FC layers with dropout
    """
    
    def __init__(self, config: CNNConfig):
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        
        super().__init__()
        self.config = config
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, config.conv1_channels, config.kernel_size, padding=1),
            nn.BatchNorm2d(config.conv1_channels),
            nn.ReLU(),
            nn.MaxPool2d(config.pool_size),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(config.conv1_channels, config.conv2_channels, config.kernel_size, padding=1),
            nn.BatchNorm2d(config.conv2_channels),
            nn.ReLU(),
            nn.MaxPool2d(config.pool_size),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(config.conv2_channels, config.conv3_channels, config.kernel_size, padding=1),
            nn.BatchNorm2d(config.conv3_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.conv3_channels, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden, config.num_classes),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


class CNNTrainer:
    
    def __init__(self, config: CNNConfig | None = None):
        if not HAS_TORCH:
            raise ImportError("torch not installed. Run: pip install torch")
        
        self.config = config or CNNConfig()
        self.model: LiquidationCNN | None = None
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
    
    def train(
        self,
        train_heatmaps: np.ndarray,
        train_labels: np.ndarray,
        val_heatmaps: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
    ) -> dict:
        train_dataset = LiquidationHeatmapDataset(train_heatmaps, train_labels)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        val_loader = None
        if val_heatmaps is not None and val_labels is not None:
            val_dataset = LiquidationHeatmapDataset(val_heatmaps, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        
        self.model = LiquidationCNN(self.config).to(self.device)
        
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
            
            for heatmaps, labels in train_loader:
                heatmaps = heatmaps.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(heatmaps)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * heatmaps.size(0)
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
            for heatmaps, labels in loader:
                heatmaps = heatmaps.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(heatmaps)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * heatmaps.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / total, correct / total
    
    def predict(self, heatmaps: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        X = torch.FloatTensor(heatmaps).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = outputs.max(1)
        
        return predicted.cpu().numpy() - 1
    
    def predict_proba(self, heatmaps: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        X = torch.FloatTensor(heatmaps).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
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
        self.model = LiquidationCNN(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
