import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Dict, Optional


class CNNBranch(nn.Module):
    
    def __init__(self, input_channels: int = 5, output_dim: int = 64):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=12, stride=12),
        )
        
        self.fc = nn.Linear(64 * 4, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class MLPBranch(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridTradingPolicy(nn.Module):
    
    def __init__(
        self,
        candle_shape: Tuple[int, int],
        ml_feature_dim: int,
        portfolio_dim: int = 2,
        action_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.candle_shape = candle_shape
        self.ml_feature_dim = ml_feature_dim
        self.portfolio_dim = portfolio_dim
        
        cnn_output_dim = 64
        mlp_output_dim = 64
        
        self.cnn_branch = CNNBranch(
            input_channels=candle_shape[1],
            output_dim=cnn_output_dim,
        )
        
        self.mlp_branch = MLPBranch(
            input_dim=ml_feature_dim + portfolio_dim,
            output_dim=mlp_output_dim,
        )
        
        fusion_input_dim = cnn_output_dim + mlp_output_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        candles: torch.Tensor,
        ml_features: torch.Tensor,
        portfolio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        candles = torch.clamp(candles, -10, 10)
        ml_features = torch.clamp(ml_features, -10, 10)
        ml_features = torch.nan_to_num(ml_features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        cnn_out = self.cnn_branch(candles)
        
        tabular = torch.cat([ml_features, portfolio], dim=-1)
        mlp_out = self.mlp_branch(tabular)
        
        fused = torch.cat([cnn_out, mlp_out], dim=-1)
        shared = self.fusion(fused)
        
        action_logits = self.actor(shared)
        action_logits = torch.clamp(action_logits, -20, 20)
        
        value = self.critic(shared)
        
        return action_logits, value
    
    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        candles = torch.FloatTensor(obs["candles"]).unsqueeze(0)
        ml_features = torch.FloatTensor(obs["ml_features"]).unsqueeze(0)
        portfolio = torch.FloatTensor(obs["portfolio"]).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = self.forward(candles, ml_features, portfolio)
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()
            
            log_prob = torch.log(probs[0, action] + 1e-8).item()
        
        return action, log_prob, value.item()
    
    def evaluate_actions(
        self,
        candles: torch.Tensor,
        ml_features: torch.Tensor,
        portfolio: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(candles, ml_features, portfolio)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class PPOTrainer:
    
    def __init__(
        self,
        policy: HybridTradingPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.9)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        batch: Dict[str, np.ndarray],
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        candles = torch.FloatTensor(batch["candles"])
        ml_features = torch.FloatTensor(batch["ml_features"])
        portfolio = torch.FloatTensor(batch["portfolio"])
        actions = torch.LongTensor(batch["actions"])
        old_log_probs = torch.FloatTensor(batch["log_probs"])
        advantages = torch.FloatTensor(batch["advantages"])
        returns = torch.FloatTensor(batch["returns"])
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_samples = len(actions)
        total_pg_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for _ in range(n_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]
                
                log_probs, values, entropy = self.policy.evaluate_actions(
                    candles[idx],
                    ml_features[idx],
                    portfolio[idx],
                    actions[idx],
                )
                
                ratio = torch.exp(log_probs - old_log_probs[idx])
                
                pg_loss1 = -advantages[idx] * ratio
                pg_loss2 = -advantages[idx] * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                value_loss = ((values - returns[idx]) ** 2).mean()
                
                entropy_loss = -entropy.mean()
                
                loss = pg_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                n_updates += 1
        
        self.scheduler.step()
        
        return {
            "pg_loss": total_pg_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "lr": self.scheduler.get_last_lr()[0],
        }
