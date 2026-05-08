#!/usr/bin/env python3
"""
Research-Grade Deep Learning Framework for Time Series Price Forecasting
=========================================================================

A comprehensive PyTorch implementation featuring:
- Multiple model architectures: CNN, RCNN, Temporal Attention CNN, Transformer-based models
- Advanced attention mechanisms: Multi-Head Self-Attention, Temporal Attention, Squeeze-Excitation
- Sophisticated training: Mixed precision, gradient clipping, learning rate scheduling
- Robust evaluation: Walk-forward validation, regime-conditional metrics, statistical tests
- Production features: Checkpointing, experiment tracking, reproducibility

Architecture Reference Papers:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Temporal Fusion Transformers" (Lim et al., 2021)
- "Stock Price Prediction Using CNN and LSTM" (Chen et al., 2020)
- "Deep Learning for Stock Prediction" (Fischer & Krauss, 2018)

Author: Research Implementation
License: MIT
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Tuple, Optional, Union, Callable, Any, Literal
)
from datetime import datetime
from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau,
    CosineAnnealingLR
)
from torch.cuda.amp import GradScaler, autocast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: Literal['cnn', 'rcnn', 'attention_cnn', 'attention_rcnn', 
                        'temporal_fusion', 'wavenet'] = 'attention_rcnn'
    
    # Input dimensions
    seq_len: int = 24
    n_features: int = 50
    
    # CNN parameters
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    cnn_dropout: float = 0.2
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    
    # RNN parameters
    rnn_type: Literal['lstm', 'gru', 'bilstm', 'bigru'] = 'bilstm'
    rnn_hidden_size: int = 128
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.3
    
    # Attention parameters
    num_attention_heads: int = 8
    attention_dim: int = 64
    attention_dropout: float = 0.1
    use_positional_encoding: bool = True
    
    # Dense layers
    fc_dims: List[int] = field(default_factory=lambda: [128, 64])
    fc_dropout: float = 0.3
    
    # Regularization
    weight_decay: float = 1e-4
    use_residual: bool = True
    use_squeeze_excitation: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    
    # Optimizer
    optimizer: Literal['adamw', 'adam', 'sgd'] = 'adamw'
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Scheduler
    scheduler: Literal['cosine', 'onecycle', 'plateau', 'cosine_warm'] = 'cosine_warm'
    warmup_epochs: int = 10
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    # Paths
    macro_path: str = ""
    price_path: str = ""
    
    # Sequence parameters
    lookback: int = 24
    horizon: int = 1
    target_col: str = "Price"
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Scaling
    scaler_type: Literal['standard', 'robust', 'minmax'] = 'robust'
    scale_target: bool = True
    
    # Feature engineering
    add_technical_features: bool = True
    add_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12])


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment tracking
    experiment_name: str = "price_forecast"
    output_dir: str = "./experiments"
    save_checkpoints: bool = True
    log_interval: int = 10
    
    def to_dict(self) -> Dict:
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir
        }
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class EarlyStopping:
    """Early stopping with patience and model checkpointing."""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-6,
        mode: str = 'min',
        checkpoint_path: Optional[str] = None
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(
        self,
        score: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        epoch: int = 0
    ) -> bool:
        if self.mode == 'min':
            score = -score
            
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch)
            self.counter = 0
            
        return self.early_stop
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ):
        self.best_model_state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'best_score': -self.best_score if self.mode == 'min' else self.best_score
        }
        if self.checkpoint_path:
            torch.save(self.best_model_state, self.checkpoint_path)
            
    def load_best_model(self, model: nn.Module):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state['model_state_dict'])


class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(self):
        self.history = defaultdict(list)
        self.current_epoch = defaultdict(list)
        
    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.current_epoch[k].append(v)
            
    def end_epoch(self):
        for k, v in self.current_epoch.items():
            self.history[k].append(np.mean(v))
        self.current_epoch.clear()
        
    def get_last(self, key: str) -> float:
        return self.history[key][-1] if self.history[key] else 0.0
    
    def get_history(self) -> Dict[str, List[float]]:
        return dict(self.history)


# ============================================================================
# DATA PROCESSING
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series forecasting."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        augment: bool = False,
        noise_std: float = 0.01
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.dates = dates
        self.augment = augment
        self.noise_std = noise_std
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment and self.training:
            # Add Gaussian noise for augmentation
            x = x + torch.randn_like(x) * self.noise_std
            
        return x, y


class DataProcessor:
    """Comprehensive data processing pipeline."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None
        
    def load_data(
        self,
        macro_path: Optional[str] = None,
        price_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Load and merge macro and price data."""
        macro_path = macro_path or self.config.macro_path
        price_path = price_path or self.config.price_path
        
        # Load macro features
        if macro_path and os.path.exists(macro_path):
            df_macro = pd.read_csv(macro_path)
            date_col = 'DATE' if 'DATE' in df_macro.columns else df_macro.columns[0]
            df_macro[date_col] = pd.to_datetime(df_macro[date_col])
            df_macro = df_macro.set_index(date_col).sort_index()
            df_macro = df_macro.dropna(axis=1, how='all')
        else:
            df_macro = pd.DataFrame()
            
        # Load price data
        if price_path and os.path.exists(price_path):
            df_price = pd.read_csv(price_path)
            date_col = 'DATE' if 'DATE' in df_price.columns else 'observation_date'
            val_col = [c for c in df_price.columns if c != date_col][0]
            df_price[date_col] = pd.to_datetime(df_price[date_col])
            df_price[val_col] = pd.to_numeric(df_price[val_col], errors='coerce')
            df_price = df_price.rename(columns={date_col: 'DATE', val_col: 'Price'})
            df_price = df_price.set_index('DATE').sort_index()
            df_price = df_price.dropna()
        else:
            raise FileNotFoundError("Price data file not found.")
            
        # Merge datasets
        if not df_macro.empty:
            df_all = df_macro.join(df_price, how='inner')
        else:
            df_all = df_price
            
        df_all = df_all.dropna()
        return df_all
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features."""
        df = df.copy()
        price_col = self.config.target_col
        
        if price_col not in df.columns:
            return df
            
        price = df[price_col]
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = price.rolling(window=window).mean()
            df[f'EMA_{window}'] = price.ewm(span=window, adjust=False).mean()
            
        # Momentum indicators
        df['ROC_10'] = price.pct_change(periods=10) * 100
        df['ROC_20'] = price.pct_change(periods=20) * 100
        
        # Volatility
        df['STD_10'] = price.rolling(window=10).std()
        df['STD_20'] = price.rolling(window=20).std()
        
        # RSI
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = price.ewm(span=12, adjust=False).mean()
        exp2 = price.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = price.rolling(window=20).mean()
        bb_std = price.rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (price - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
        
        # Price ratios
        df['Price_SMA20_Ratio'] = price / (df['SMA_20'] + 1e-10)
        df['Price_SMA50_Ratio'] = price / (df['SMA_50'] + 1e-10)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        df = df.copy()
        price_col = self.config.target_col
        
        if price_col not in df.columns:
            return df
            
        for lag in self.config.lag_periods:
            df[f'{price_col}_lag_{lag}'] = df[price_col].shift(lag)
            df[f'{price_col}_return_{lag}'] = df[price_col].pct_change(periods=lag)
            
        return df
    
    def build_sequences(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None,
        horizon: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Build sequences for supervised learning."""
        lookback = lookback or self.config.lookback
        horizon = horizon or self.config.horizon
        target_col = self.config.target_col
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
            
        data = df.values.astype(np.float32)
        target_idx = df.columns.get_loc(target_col)
        self.feature_names = list(df.columns)
        
        X, y, dates = [], [], []
        for t in range(lookback, len(df) - horizon + 1):
            X.append(data[t - lookback:t, :])
            y.append(data[t + horizon - 1, target_idx])
            dates.append(df.index[t + horizon - 1])
            
        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            pd.DatetimeIndex(dates)
        )
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
        """Chronological train/val/test split."""
        n = len(X)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        return {
            'train': (X[:train_end], y[:train_end], dates[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end], dates[train_end:val_end]),
            'test': (X[val_end:], y[val_end:], dates[val_end:])
        }
    
    def scale_data(
        self,
        splits: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
        """Scale features and target."""
        # Get scalers
        scaler_map = {
            'standard': StandardScaler,
            'robust': RobustScaler,
            'minmax': MinMaxScaler
        }
        ScalerClass = scaler_map.get(self.config.scaler_type, RobustScaler)
        
        X_train, y_train, dates_train = splits['train']
        n_train, L, D = X_train.shape
        
        # Fit feature scaler on training data
        self.feature_scaler = ScalerClass()
        X_train_flat = X_train.reshape(-1, D)
        self.feature_scaler.fit(X_train_flat)
        
        # Fit target scaler
        if self.config.scale_target:
            self.target_scaler = ScalerClass()
            self.target_scaler.fit(y_train.reshape(-1, 1))
        
        # Transform all splits
        scaled_splits = {}
        for split_name, (X, y, dates) in splits.items():
            n_samples = X.shape[0]
            X_scaled = self.feature_scaler.transform(X.reshape(-1, D)).reshape(n_samples, L, D)
            if self.config.scale_target:
                y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).ravel()
            else:
                y_scaled = y
            scaled_splits[split_name] = (X_scaled, y_scaled, dates)
            
        return scaled_splits
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform scaled target values."""
        if self.target_scaler is not None and self.config.scale_target:
            return self.target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        return y
    
    def process_pipeline(
        self,
        macro_path: Optional[str] = None,
        price_path: Optional[str] = None
    ) -> Tuple[Dict[str, DataLoader], Dict[str, pd.DatetimeIndex]]:
        """Complete data processing pipeline."""
        logger.info("Loading data...")
        df = self.load_data(macro_path, price_path)
        
        if self.config.add_technical_features:
            logger.info("Adding technical features...")
            df = self.add_technical_features(df)
            
        if self.config.add_lag_features:
            logger.info("Adding lag features...")
            df = self.add_lag_features(df)
            
        # Drop NaN values created by feature engineering
        df = df.dropna()
        logger.info(f"Dataset shape after processing: {df.shape}")
        
        logger.info("Building sequences...")
        X, y, dates = self.build_sequences(df)
        
        logger.info("Splitting data...")
        splits = self.split_data(X, y, dates)
        
        logger.info("Scaling data...")
        scaled_splits = self.scale_data(splits)
        
        return scaled_splits


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-like models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels)
        b, s, c = x.size()
        y = x.mean(dim=1)  # Global average pooling
        y = self.fc(y).unsqueeze(1)
        return x * y


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time series."""
    
    def __init__(self, hidden_size: int, attention_size: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, hidden_size)
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_size)
        
        return context, weights


class CausalConv1d(nn.Module):
    """Causal convolution for time series (prevents future information leakage)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    """Residual block with causal convolutions."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        
        self.norm1 = nn.BatchNorm1d(channels) if use_batch_norm else nn.Identity()
        self.norm2 = nn.BatchNorm1d(channels) if use_batch_norm else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        x = x + residual
        x = self.activation(x)
        
        return x


# ============================================================================
# MAIN MODEL ARCHITECTURES
# ============================================================================

class CNNModel(nn.Module):
    """Pure CNN model for time series forecasting."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.n_features, config.cnn_channels[0])
        
        # CNN layers with residual connections
        cnn_layers = []
        in_channels = config.cnn_channels[0]
        
        for i, out_channels in enumerate(config.cnn_channels):
            if i > 0:
                cnn_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1)
                )
            cnn_layers.append(
                ResidualBlock(
                    out_channels,
                    kernel_size=config.kernel_sizes[i] if i < len(config.kernel_sizes) else 3,
                    dilation=2 ** i,
                    dropout=config.cnn_dropout,
                    use_batch_norm=config.use_batch_norm
                )
            )
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Squeeze-Excitation
        if config.use_squeeze_excitation:
            self.se = SqueezeExcitation(config.cnn_channels[-1])
        else:
            self.se = nn.Identity()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        fc_layers = []
        fc_in = config.cnn_channels[-1]
        for fc_dim in config.fc_dims:
            fc_layers.extend([
                nn.Linear(fc_in, fc_dim),
                nn.GELU(),
                nn.Dropout(config.fc_dropout)
            ])
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, 1))
        
        self.fc = nn.Sequential(*fc_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = self.input_proj(x)  # (batch, seq_len, channels)
        x = x.transpose(1, 2)   # (batch, channels, seq_len)
        
        x = self.cnn(x)
        x = x.transpose(1, 2)   # (batch, seq_len, channels)
        
        x = self.se(x)
        x = x.transpose(1, 2)   # (batch, channels, seq_len)
        
        x = self.global_pool(x).squeeze(-1)  # (batch, channels)
        x = self.fc(x).squeeze(-1)
        
        return x


class RCNNModel(nn.Module):
    """CNN + RNN hybrid model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.n_features, config.cnn_channels[0])
        
        # CNN feature extractor
        cnn_layers = []
        in_channels = config.cnn_channels[0]
        
        for i, out_channels in enumerate(config.cnn_channels):
            if i > 0:
                cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            cnn_layers.append(
                ResidualBlock(
                    out_channels,
                    kernel_size=config.kernel_sizes[i] if i < len(config.kernel_sizes) else 3,
                    dropout=config.cnn_dropout,
                    use_batch_norm=config.use_batch_norm
                )
            )
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*cnn_layers)
        
        # RNN
        rnn_input_size = config.cnn_channels[-1]
        bidirectional = config.rnn_type.startswith('bi')
        rnn_class = nn.LSTM if 'lstm' in config.rnn_type.lower() else nn.GRU
        
        self.rnn = rnn_class(
            input_size=rnn_input_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            batch_first=True,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        rnn_output_size = config.rnn_hidden_size * (2 if bidirectional else 1)
        
        # Fully connected
        fc_layers = []
        fc_in = rnn_output_size
        for fc_dim in config.fc_dims:
            fc_layers.extend([
                nn.Linear(fc_in, fc_dim),
                nn.GELU(),
                nn.Dropout(config.fc_dropout)
            ])
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, 1))
        
        self.fc = nn.Sequential(*fc_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        batch_size = x.size(0)
        
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        # RNN
        rnn_out, _ = self.rnn(x)
        
        # Take last hidden state
        x = rnn_out[:, -1, :]
        
        x = self.fc(x).squeeze(-1)
        return x


class AttentionCNNModel(nn.Module):
    """CNN with Multi-Head Self-Attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.n_features, config.cnn_channels[0])
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                config.cnn_channels[0],
                max_len=config.seq_len,
                dropout=config.attention_dropout
            )
        else:
            self.pos_encoding = nn.Identity()
        
        # CNN
        cnn_layers = []
        in_channels = config.cnn_channels[0]
        
        for i, out_channels in enumerate(config.cnn_channels):
            if i > 0:
                cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            cnn_layers.append(
                ResidualBlock(
                    out_channels,
                    kernel_size=config.kernel_sizes[i] if i < len(config.kernel_sizes) else 3,
                    dropout=config.cnn_dropout,
                    use_batch_norm=config.use_batch_norm
                )
            )
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Multi-Head Self-Attention
        attn_dim = config.cnn_channels[-1]
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(attn_dim)
        
        # Temporal attention for aggregation
        self.temporal_attn = TemporalAttention(attn_dim, config.attention_dim)
        
        # Squeeze-Excitation
        if config.use_squeeze_excitation:
            self.se = SqueezeExcitation(attn_dim)
        else:
            self.se = nn.Identity()
        
        # Fully connected
        fc_layers = []
        fc_in = attn_dim
        for fc_dim in config.fc_dims:
            fc_layers.extend([
                nn.Linear(fc_in, fc_dim),
                nn.GELU(),
                nn.Dropout(config.fc_dropout)
            ])
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, 1))
        
        self.fc = nn.Sequential(*fc_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)
        
        # Squeeze-Excitation
        x = self.se(x)
        
        # Temporal attention for aggregation
        context, temporal_weights = self.temporal_attn(x)
        
        out = self.fc(context).squeeze(-1)
        
        return out, temporal_weights


class AttentionRCNNModel(nn.Module):
    """CNN + RNN + Attention (most sophisticated model)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.n_features, config.cnn_channels[0])
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                config.cnn_channels[0],
                max_len=config.seq_len,
                dropout=config.attention_dropout
            )
        else:
            self.pos_encoding = nn.Identity()
        
        # CNN
        cnn_layers = []
        in_channels = config.cnn_channels[0]
        
        for i, out_channels in enumerate(config.cnn_channels):
            if i > 0:
                cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            cnn_layers.append(
                ResidualBlock(
                    out_channels,
                    kernel_size=config.kernel_sizes[i] if i < len(config.kernel_sizes) else 3,
                    dilation=2 ** i,
                    dropout=config.cnn_dropout,
                    use_batch_norm=config.use_batch_norm
                )
            )
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Multi-Head Self-Attention (pre-RNN)
        attn_dim = config.cnn_channels[-1]
        self.pre_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.pre_attn_norm = nn.LayerNorm(attn_dim)
        
        # RNN
        bidirectional = config.rnn_type.startswith('bi')
        rnn_class = nn.LSTM if 'lstm' in config.rnn_type.lower() else nn.GRU
        
        self.rnn = rnn_class(
            input_size=attn_dim,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            batch_first=True,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        rnn_output_size = config.rnn_hidden_size * (2 if bidirectional else 1)
        
        # Post-RNN attention
        self.post_attn = nn.MultiheadAttention(
            embed_dim=rnn_output_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.post_attn_norm = nn.LayerNorm(rnn_output_size)
        
        # Temporal attention for aggregation
        self.temporal_attn = TemporalAttention(rnn_output_size, config.attention_dim)
        
        # Squeeze-Excitation
        if config.use_squeeze_excitation:
            self.se = SqueezeExcitation(rnn_output_size)
        else:
            self.se = nn.Identity()
        
        # Fully connected with skip connection from CNN
        fc_input_size = rnn_output_size + attn_dim  # Concatenate CNN and RNN outputs
        
        fc_layers = []
        fc_in = fc_input_size
        for fc_dim in config.fc_dims:
            fc_layers.extend([
                nn.Linear(fc_in, fc_dim),
                nn.GELU(),
                nn.Dropout(config.fc_dropout)
            ])
            fc_in = fc_dim
        fc_layers.append(nn.Linear(fc_in, 1))
        
        self.fc = nn.Sequential(*fc_layers)
        
        # CNN global pooling for skip connection
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Input projection + positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # CNN
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        # Save for skip connection
        cnn_out = self.cnn_pool(x.transpose(1, 2)).squeeze(-1)  # (batch, channels)
        
        # Pre-RNN attention
        attn_out, _ = self.pre_attn(x, x, x)
        x = self.pre_attn_norm(x + attn_out)
        
        # RNN
        rnn_out, _ = self.rnn(x)
        
        # Post-RNN attention
        attn_out, attn_weights = self.post_attn(rnn_out, rnn_out, rnn_out)
        x = self.post_attn_norm(rnn_out + attn_out)
        
        # Squeeze-Excitation
        x = self.se(x)
        
        # Temporal attention aggregation
        context, temporal_weights = self.temporal_attn(x)
        
        # Skip connection from CNN
        combined = torch.cat([context, cnn_out], dim=-1)
        
        out = self.fc(combined).squeeze(-1)
        
        return out, temporal_weights


class TemporalFusionModel(nn.Module):
    """Simplified Temporal Fusion Transformer-like architecture."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        hidden_size = config.cnn_channels[-1]
        
        # Input projection
        self.input_proj = nn.Linear(config.n_features, hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            hidden_size,
            max_len=config.seq_len,
            dropout=config.attention_dropout
        )
        
        # Variable selection network (simplified)
        self.var_select = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax(dim=-1)
        )
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            batch_first=True,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
            bidirectional=True
        )
        
        encoder_output_size = config.rnn_hidden_size * 2
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_output_size,
            nhead=config.num_attention_heads,
            dim_feedforward=encoder_output_size * 4,
            dropout=config.attention_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(encoder_output_size, config.attention_dim)
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(encoder_output_size, config.fc_dims[0]),
            nn.GELU(),
            nn.Dropout(config.fc_dropout),
            nn.Linear(config.fc_dims[0], 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input projection
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Variable selection (soft feature gating)
        var_weights = self.var_select(x.mean(dim=1, keepdim=True))
        x = x * var_weights
        
        # LSTM encoding
        x, _ = self.encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Temporal attention
        context, weights = self.temporal_attn(x)
        
        # Output
        out = self.output_net(context).squeeze(-1)
        
        return out, weights


def build_model(config: ModelConfig) -> nn.Module:
    """Factory function to build models."""
    model_map = {
        'cnn': CNNModel,
        'rcnn': RCNNModel,
        'attention_cnn': AttentionCNNModel,
        'attention_rcnn': AttentionRCNNModel,
        'temporal_fusion': TemporalFusionModel,
    }
    
    model_class = model_map.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model_type}")
        
    return model_class(config)


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Training orchestrator with advanced features."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device = None
    ):
        self.model = model
        self.config = config
        self.device = device or get_device()
        
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        
        # Setup scheduler
        self.scheduler = None  # Built after knowing total steps
        
        # Loss function
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Tracking
        self.metrics = MetricsTracker()
        self.early_stopping = None
        
    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.config.training
        
        if tc.optimizer == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay
            )
        elif tc.optimizer == 'adam':
            return Adam(
                self.model.parameters(),
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay
            )
        elif tc.optimizer == 'sgd':
            return SGD(
                self.model.parameters(),
                lr=tc.learning_rate,
                momentum=0.9,
                weight_decay=tc.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {tc.optimizer}")
            
    def _build_scheduler(self, total_steps: int):
        tc = self.config.training
        
        if tc.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=tc.epochs,
                eta_min=tc.min_lr
            )
        elif tc.scheduler == 'cosine_warm':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=tc.warmup_epochs,
                T_mult=2,
                eta_min=tc.min_lr
            )
        elif tc.scheduler == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=tc.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        elif tc.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=tc.min_lr
            )
            
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.training.use_amp and self.scaler is not None:
                with autocast():
                    output = self.model(X)
                    if isinstance(output, tuple):
                        output = output[0]
                    loss = self.criterion(output, y)
                    
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(X)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y)
                
                loss.backward()
                
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                    
                self.optimizer.step()
                
            total_loss += loss.item()
            n_batches += 1
            
            # Learning rate scheduling (per-step for OneCycleLR)
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
                
        avg_loss = total_loss / n_batches
        
        # Learning rate scheduling (per-epoch)
        if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
            if isinstance(self.scheduler, ReduceLROnPlateau):
                pass  # Will be called with val_loss
            else:
                self.scheduler.step()
                
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on data."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0
        
        for X, y in data_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            output = self.model(X)
            if isinstance(output, tuple):
                output = output[0]
                
            loss = self.criterion(output, y)
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
        avg_loss = total_loss / n_batches
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        results = {
            'loss': avg_loss,
            'mse': mean_squared_error(targets, preds),
            'rmse': np.sqrt(mean_squared_error(targets, preds)),
            'mae': mean_absolute_error(targets, preds),
            'r2': r2_score(targets, preds)
        }
        
        if return_predictions:
            results['predictions'] = preds
            results['targets'] = targets
            
        return results
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        tc = self.config.training
        
        # Build scheduler
        total_steps = len(train_loader) * tc.epochs
        self._build_scheduler(total_steps)
        
        # Setup early stopping
        checkpoint_path = None
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            
        self.early_stopping = EarlyStopping(
            patience=tc.patience,
            min_delta=tc.min_delta,
            mode='min',
            checkpoint_path=checkpoint_path
        )
        
        logger.info(f"Starting training for {tc.epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(tc.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Update scheduler (for ReduceLROnPlateau)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
                
            # Record metrics
            self.metrics.update({
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_metrics['loss'],
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2'],
                'lr': self.optimizer.param_groups[0]['lr']
            })
            self.metrics.end_epoch()
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{tc.epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.6f} | "
                    f"Val Loss: {val_metrics['loss']:.6f} | "
                    f"Val RMSE: {val_metrics['rmse']:.4f} | "
                    f"Val R²: {val_metrics['r2']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
                
            # Early stopping check
            if self.early_stopping(
                val_metrics['loss'],
                self.model,
                self.optimizer,
                epoch
            ):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # Load best model
        self.early_stopping.load_best_model(self.model)
        logger.info("Training complete. Loaded best model.")
        
        return self.metrics.get_history()


# ============================================================================
# EVALUATION & METRICS
# ============================================================================

class Evaluator:
    """Comprehensive model evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        data_processor: DataProcessor,
        device: torch.device = None
    ):
        self.model = model
        self.data_processor = data_processor
        self.device = device or get_device()
        self.model.to(self.device)
        
    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 64
    ) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        
        dataset = TimeSeriesDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        for X_batch, _ in loader:
            X_batch = X_batch.to(self.device)
            output = self.model(X_batch)
            if isinstance(output, tuple):
                output = output[0]
            predictions.append(output.cpu().numpy())
            
        return np.concatenate(predictions)
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        metrics = {
            f'{prefix}mse': mean_squared_error(y_true, y_pred),
            f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}mae': mean_absolute_error(y_true, y_pred),
            f'{prefix}r2': r2_score(y_true, y_pred),
        }
        
        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.any():
            metrics[f'{prefix}mape'] = mean_absolute_percentage_error(
                y_true[mask], y_pred[mask]
            ) * 100
            
        # Direction accuracy
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics[f'{prefix}direction_accuracy'] = (
                (actual_direction == pred_direction).mean() * 100
            )
            
        return metrics
    
    def regime_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        percentile_threshold: float = 80
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance in different market regimes."""
        threshold = np.percentile(y_true, percentile_threshold)
        
        # Normal regime
        normal_mask = y_true < threshold
        normal_metrics = self.compute_metrics(
            y_true[normal_mask], y_pred[normal_mask], 'normal_'
        )
        
        # Spike regime
        spike_mask = y_true >= threshold
        spike_metrics = self.compute_metrics(
            y_true[spike_mask], y_pred[spike_mask], 'spike_'
        )
        
        return {
            'normal': normal_metrics,
            'spike': spike_metrics,
            'threshold': threshold
        }
    
    def statistical_tests(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Perform statistical tests on predictions."""
        residuals = y_true - y_pred
        
        tests = {}
        
        # Normality of residuals (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        tests['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pvalue}
        
        # Autocorrelation of residuals (Durbin-Watson approximation)
        dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
        tests['durbin_watson'] = {'statistic': dw_stat}
        
        # Residual statistics
        tests['residual_stats'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        return tests
    
    def full_evaluation(
        self,
        X: np.ndarray,
        y_true_scaled: np.ndarray,
        dates: pd.DatetimeIndex,
        split_name: str = 'test'
    ) -> Dict[str, Any]:
        """Complete evaluation pipeline."""
        # Get predictions
        y_pred_scaled = self.predict(X)
        
        # Inverse transform
        y_true = self.data_processor.inverse_transform_target(y_true_scaled)
        y_pred = self.data_processor.inverse_transform_target(y_pred_scaled)
        
        results = {
            'split': split_name,
            'n_samples': len(y_true),
            'date_range': (dates[0], dates[-1]),
            'metrics': self.compute_metrics(y_true, y_pred),
            'regime_analysis': self.regime_analysis(y_true, y_pred),
            'statistical_tests': self.statistical_tests(y_true, y_pred),
            'predictions': y_pred,
            'actuals': y_true,
            'dates': dates
        }
        
        return results


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Comprehensive visualization utilities."""
    
    def __init__(self, output_dir: str = './figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.tab10.colors
        
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: str = 'training_history.png'
    ):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        ax = axes[0, 0]
        ax.plot(history['train_loss'], label='Train', color=self.colors[0])
        ax.plot(history['val_loss'], label='Validation', color=self.colors[1])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.set_yscale('log')
        
        # RMSE
        ax = axes[0, 1]
        ax.plot(history['val_rmse'], label='Val RMSE', color=self.colors[2])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Validation RMSE')
        ax.legend()
        
        # R²
        ax = axes[1, 0]
        ax.plot(history['val_r2'], label='Val R²', color=self.colors[3])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R²')
        ax.set_title('Validation R²')
        ax.legend()
        
        # Learning rate
        ax = axes[1, 1]
        ax.plot(history['lr'], label='Learning Rate', color=self.colors[4])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_predictions(
        self,
        dates: pd.DatetimeIndex,
        actuals: np.ndarray,
        predictions: np.ndarray,
        title: str = 'Actual vs Predicted',
        save_name: str = 'predictions.png'
    ):
        """Plot actual vs predicted values."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Time series plot
        ax = axes[0]
        ax.plot(dates, actuals, label='Actual', color=self.colors[0], linewidth=1.5)
        ax.plot(dates, predictions, label='Predicted', color=self.colors[1], 
                linewidth=1.5, alpha=0.8)
        ax.fill_between(dates, actuals, predictions, alpha=0.3, color=self.colors[2])
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter plot
        ax = axes[1]
        ax.scatter(actuals, predictions, alpha=0.5, s=20, color=self.colors[0])
        min_val, max_val = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted Scatter')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Residuals
        ax = axes[2]
        residuals = actuals - predictions
        ax.plot(dates, residuals, color=self.colors[3], linewidth=1)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.fill_between(dates, 0, residuals, alpha=0.3, color=self.colors[3])
        ax.set_xlabel('Date')
        ax.set_ylabel('Residual (Actual - Predicted)')
        ax.set_title('Prediction Residuals')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'test_rmse',
        save_name: str = 'model_comparison.png'
    ):
        """Compare multiple models."""
        models = list(results.keys())
        values = [results[m].get(metric, 0) for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, values, color=[self.colors[i % len(self.colors)] for i in range(len(models))])
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}')
        plt.xticks(rotation=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_attention_weights(
        self,
        weights: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        save_name: str = 'attention_weights.png'
    ):
        """Visualize attention weights."""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        if dates is not None and len(dates) == len(weights):
            ax.bar(range(len(weights)), weights, color=self.colors[0], alpha=0.7)
            ax.set_xlabel('Time Step')
        else:
            ax.bar(range(len(weights)), weights, color=self.colors[0], alpha=0.7)
            ax.set_xlabel('Time Step')
            
        ax.set_ylabel('Attention Weight')
        ax.set_title('Temporal Attention Weights')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Orchestrate complete experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = os.path.join(
            config.output_dir,
            f"{config.experiment_name}_{config.get_hash()}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
            
        # Initialize components
        self.data_processor = DataProcessor(config.data)
        self.visualizer = Visualizer(os.path.join(self.output_dir, 'figures'))
        self.device = get_device()
        
        # Set seed
        set_seed(config.training.seed, config.training.deterministic)
        
    def run(
        self,
        macro_path: Optional[str] = None,
        price_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete experiment."""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Process data
        logger.info("Processing data...")
        scaled_splits = self.data_processor.process_pipeline(macro_path, price_path)
        
        # Update model config with actual feature dimension
        X_train = scaled_splits['train'][0]
        self.config.model.seq_len = X_train.shape[1]
        self.config.model.n_features = X_train.shape[2]
        
        logger.info(f"Sequence length: {self.config.model.seq_len}")
        logger.info(f"Number of features: {self.config.model.n_features}")
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(
            scaled_splits['train'][0],
            scaled_splits['train'][1],
            scaled_splits['train'][2]
        )
        val_dataset = TimeSeriesDataset(
            scaled_splits['val'][0],
            scaled_splits['val'][1],
            scaled_splits['val'][2]
        )
        test_dataset = TimeSeriesDataset(
            scaled_splits['test'][0],
            scaled_splits['test'][1],
            scaled_splits['test'][2]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        
        # Build model
        logger.info(f"Building model: {self.config.model.model_type}")
        model = build_model(self.config.model)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        trainer = Trainer(model, self.config, self.device)
        history = trainer.fit(
            train_loader,
            val_loader,
            checkpoint_dir=os.path.join(self.output_dir, 'checkpoints')
        )
        
        # Visualize training
        self.visualizer.plot_training_history(history)
        
        # Evaluate
        logger.info("Evaluating model...")
        evaluator = Evaluator(model, self.data_processor, self.device)
        
        results = {}
        for split_name in ['train', 'val', 'test']:
            X, y, dates = scaled_splits[split_name]
            eval_results = evaluator.full_evaluation(X, y, dates, split_name)
            results[split_name] = eval_results
            
            logger.info(f"\n{split_name.upper()} Results:")
            for metric, value in eval_results['metrics'].items():
                logger.info(f"  {metric}: {value:.6f}")
                
            # Plot predictions
            self.visualizer.plot_predictions(
                eval_results['dates'],
                eval_results['actuals'],
                eval_results['predictions'],
                title=f'{split_name.title()} Set: Actual vs Predicted',
                save_name=f'predictions_{split_name}.png'
            )
            
        # Save results
        results_summary = {
            split: {
                'metrics': r['metrics'],
                'regime_analysis': r['regime_analysis'],
                'statistical_tests': r['statistical_tests']
            }
            for split, r in results.items()
        }
        
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
            
        logger.info(f"\nExperiment complete. Results saved to: {self.output_dir}")
        
        return results
    
    def run_model_comparison(
        self,
        model_types: List[str],
        macro_path: Optional[str] = None,
        price_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple model architectures."""
        logger.info("Running model comparison...")
        
        # Process data once
        scaled_splits = self.data_processor.process_pipeline(macro_path, price_path)
        
        X_train = scaled_splits['train'][0]
        self.config.model.seq_len = X_train.shape[1]
        self.config.model.n_features = X_train.shape[2]
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(
            scaled_splits['train'][0], scaled_splits['train'][1]
        )
        val_dataset = TimeSeriesDataset(
            scaled_splits['val'][0], scaled_splits['val'][1]
        )
        test_dataset = TimeSeriesDataset(
            scaled_splits['test'][0], scaled_splits['test'][1]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.training.batch_size,
            shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.training.batch_size, shuffle=False
        )
        
        all_results = {}
        
        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model: {model_type}")
            logger.info('='*60)
            
            # Update config
            self.config.model.model_type = model_type
            
            # Build and train model
            model = build_model(self.config.model)
            trainer = Trainer(model, self.config, self.device)
            
            history = trainer.fit(
                train_loader,
                val_loader,
                checkpoint_dir=os.path.join(self.output_dir, f'checkpoints_{model_type}')
            )
            
            # Evaluate
            evaluator = Evaluator(model, self.data_processor, self.device)
            
            test_results = evaluator.full_evaluation(
                scaled_splits['test'][0],
                scaled_splits['test'][1],
                scaled_splits['test'][2],
                'test'
            )
            
            all_results[model_type] = {
                'history': history,
                'test_metrics': test_results['metrics'],
                'predictions': test_results['predictions'],
                'actuals': test_results['actuals'],
                'dates': test_results['dates']
            }
            
            # Plot predictions for this model
            self.visualizer.plot_predictions(
                test_results['dates'],
                test_results['actuals'],
                test_results['predictions'],
                title=f'{model_type}: Actual vs Predicted (Test Set)',
                save_name=f'predictions_{model_type}.png'
            )
            
        # Model comparison plot
        comparison_metrics = {
            model: results['test_metrics']
            for model, results in all_results.items()
        }
        
        for metric in ['rmse', 'mae', 'r2']:
            self.visualizer.plot_model_comparison(
                comparison_metrics,
                metric=metric,
                save_name=f'comparison_{metric}.png'
            )
            
        # Summary table
        summary_df = pd.DataFrame({
            model: results['test_metrics']
            for model, results in all_results.items()
        }).T
        
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY (Test Set)")
        logger.info("="*60)
        logger.info(f"\n{summary_df.to_string()}")
        
        summary_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'))
        
        return all_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 50,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic data for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range('2000-01-01', periods=n_samples, freq='M')
    
    # Create macro features (random walk + noise)
    macro_data = {}
    for i in range(n_features - 1):
        trend = np.cumsum(np.random.randn(n_samples) * 0.1)
        noise = np.random.randn(n_samples) * 0.5
        macro_data[f'feature_{i}'] = trend + noise
        
    df_macro = pd.DataFrame(macro_data, index=dates)
    df_macro.index.name = 'DATE'
    
    # Create price (correlated with some features)
    price = (
        0.3 * df_macro['feature_0'] +
        0.2 * df_macro['feature_1'] +
        0.1 * df_macro['feature_2'] +
        np.cumsum(np.random.randn(n_samples) * 0.5) +
        100
    )
    price = np.abs(price)  # Ensure positive prices
    
    df_price = pd.DataFrame({'Price': price}, index=dates)
    df_price.index.name = 'DATE'
    
    return df_macro, df_price


def main():
    """Main entry point demonstrating usage."""
    
    # Create synthetic data for demonstration
    logger.info("Creating synthetic data for demonstration...")
    df_macro, df_price = create_synthetic_data(n_samples=500, n_features=20)
    
    # Save to temp files
    import tempfile
    temp_dir = tempfile.mkdtemp()
    macro_path = os.path.join(temp_dir, 'macro.csv')
    price_path = os.path.join(temp_dir, 'price.csv')
    
    df_macro.to_csv(macro_path)
    df_price.to_csv(price_path)
    
    # Configure experiment
    config = ExperimentConfig(
        experiment_name="price_forecast_demo",
        output_dir="./experiments",
        model=ModelConfig(
            model_type='attention_rcnn',
            seq_len=24,
            cnn_channels=[32, 64, 128],
            rnn_hidden_size=64,
            num_attention_heads=4,
        ),
        training=TrainingConfig(
            epochs=50,
            batch_size=16,
            learning_rate=1e-3,
            patience=15,
        ),
        data=DataConfig(
            macro_path=macro_path,
            price_path=price_path,
            lookback=24,
            horizon=1,
            add_technical_features=True,
            add_lag_features=True,
        )
    )
    
    # Single model training
    logger.info("\n" + "="*60)
    logger.info("SINGLE MODEL TRAINING")
    logger.info("="*60)
    
    runner = ExperimentRunner(config)
    results = runner.run(macro_path, price_path)
    
    # Model comparison
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    comparison_results = runner.run_model_comparison(
        model_types=['cnn', 'rcnn', 'attention_cnn', 'attention_rcnn'],
        macro_path=macro_path,
        price_path=price_path
    )
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    
    return results, comparison_results


if __name__ == "__main__":
    results, comparison = main()


# ============================================================================
# ADVANCED EXTENSIONS
# ============================================================================

# --------------------------------------------------------------------------
# WAVENET-STYLE ARCHITECTURE
# --------------------------------------------------------------------------

class DilatedCausalConv(nn.Module):
    """Dilated causal convolution with gated activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        
        self.filter_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.gate_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        
        self.skip_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.residual_conv = nn.Conv1d(out_channels, in_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        filter_out = self.filter_conv(x)
        gate_out = self.gate_conv(x)
        
        if self.padding > 0:
            filter_out = filter_out[:, :, :-self.padding]
            gate_out = gate_out[:, :, :-self.padding]
        
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        out = self.norm(out)
        out = self.dropout(out)
        
        skip = self.skip_conv(out)
        residual = self.residual_conv(out) + x
        
        return residual, skip


class WaveNetModel(nn.Module):
    """WaveNet-style model for time series forecasting."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        hidden_channels = config.cnn_channels[0]
        
        # Input projection
        self.input_conv = nn.Conv1d(config.n_features, hidden_channels, 1)
        
        # Dilated causal convolution blocks
        self.blocks = nn.ModuleList()
        num_layers = 8  # Layers per block
        num_stacks = 2  # Number of stacks
        
        for stack in range(num_stacks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                self.blocks.append(
                    DilatedCausalConv(
                        hidden_channels, hidden_channels,
                        kernel_size=2, dilation=dilation,
                        dropout=config.cnn_dropout
                    )
                )
        
        # Output layers
        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, 1)
        )
        
        self.fc = nn.Linear(config.seq_len, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.input_conv(x)
        
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
            
        out = self.output_net(skip_sum)  # (batch, 1, seq_len)
        out = self.fc(out.squeeze(1)).squeeze(-1)
        
        return out


# --------------------------------------------------------------------------
# ENSEMBLE METHODS
# --------------------------------------------------------------------------

class ModelEnsemble(nn.Module):
    """Ensemble of multiple models with different architectures."""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = 'average'
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer('weights', torch.FloatTensor(weights))
        
        # For learned weighting
        if method == 'learned':
            self.weight_net = nn.Sequential(
                nn.Linear(len(models), len(models)),
                nn.Softmax(dim=-1)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            predictions.append(out)
            
        predictions = torch.stack(predictions, dim=-1)  # (batch, n_models)
        
        if self.method == 'average':
            return (predictions * self.weights).sum(dim=-1)
        elif self.method == 'median':
            return predictions.median(dim=-1)[0]
        elif self.method == 'learned':
            weights = self.weight_net(torch.ones(predictions.shape[-1], device=x.device))
            return (predictions * weights).sum(dim=-1)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")


class StackingEnsemble(nn.Module):
    """Stacking ensemble with meta-learner."""
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: Optional[nn.Module] = None,
        freeze_base: bool = True
    ):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.freeze_base = freeze_base
        
        if freeze_base:
            for model in self.base_models:
                for param in model.parameters():
                    param.requires_grad = False
                    
        # Meta-learner
        if meta_learner is None:
            self.meta_learner = nn.Sequential(
                nn.Linear(len(base_models), 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            self.meta_learner = meta_learner
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_predictions = []
        
        with torch.set_grad_enabled(not self.freeze_base):
            for model in self.base_models:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                base_predictions.append(out)
                
        stacked = torch.stack(base_predictions, dim=-1)  # (batch, n_models)
        return self.meta_learner(stacked).squeeze(-1)


# --------------------------------------------------------------------------
# WALK-FORWARD CROSS-VALIDATION
# --------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    initial_train_size: int = 200
    step_size: int = 30
    test_size: int = 30
    min_train_size: int = 100
    expanding: bool = True  # vs sliding window


class WalkForwardValidator:
    """Walk-forward cross-validation for time series."""
    
    def __init__(
        self,
        config: WalkForwardConfig,
        model_builder: Callable,
        trainer_config: TrainingConfig
    ):
        self.config = config
        self.model_builder = model_builder
        self.trainer_config = trainer_config
        self.results = []
        
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """Run walk-forward validation."""
        device = device or get_device()
        n_samples = len(X)
        
        fold_results = []
        fold = 0
        
        train_end = self.config.initial_train_size
        
        while train_end + self.config.test_size <= n_samples:
            fold += 1
            
            # Determine split indices
            if self.config.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - self.config.initial_train_size)
                
            test_start = train_end
            test_end = test_start + self.config.test_size
            
            logger.info(f"\nFold {fold}: Train[{train_start}:{train_end}], Test[{test_start}:{test_end}]")
            
            # Split data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            dates_test = dates[test_start:test_end]
            
            # Build and train model
            model = self.model_builder()
            
            train_dataset = TimeSeriesDataset(X_train, y_train)
            val_size = int(len(X_train) * 0.15)
            train_subset = Subset(train_dataset, range(len(X_train) - val_size))
            val_subset = Subset(train_dataset, range(len(X_train) - val_size, len(X_train)))
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            
            # Simple training loop
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.HuberLoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.trainer_config.epochs):
                # Train
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    out = model(X_batch)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = criterion(out, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                # Validate
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        out = model(X_batch)
                        if isinstance(out, tuple):
                            out = out[0]
                        val_loss += criterion(out, y_batch).item()
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        break
                        
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test).to(device)
                predictions = model(X_test_t)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                predictions = predictions.cpu().numpy()
                
            # Compute metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            fold_results.append({
                'fold': fold,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'predictions': predictions,
                'actuals': y_test,
                'dates': dates_test
            })
            
            logger.info(f"  RMSE: {np.sqrt(mse):.4f}, MAE: {mae:.4f}")
            
            # Move to next window
            train_end += self.config.step_size
            
        # Aggregate results
        all_preds = np.concatenate([r['predictions'] for r in fold_results])
        all_actuals = np.concatenate([r['actuals'] for r in fold_results])
        all_dates = pd.DatetimeIndex(np.concatenate([r['dates'] for r in fold_results]))
        
        summary = {
            'n_folds': len(fold_results),
            'mean_rmse': np.mean([r['rmse'] for r in fold_results]),
            'std_rmse': np.std([r['rmse'] for r in fold_results]),
            'mean_mae': np.mean([r['mae'] for r in fold_results]),
            'std_mae': np.std([r['mae'] for r in fold_results]),
            'all_predictions': all_preds,
            'all_actuals': all_actuals,
            'all_dates': all_dates,
            'fold_results': fold_results
        }
        
        logger.info(f"\nWalk-Forward Summary:")
        logger.info(f"  Folds: {summary['n_folds']}")
        logger.info(f"  Mean RMSE: {summary['mean_rmse']:.4f} (+/- {summary['std_rmse']:.4f})")
        logger.info(f"  Mean MAE: {summary['mean_mae']:.4f} (+/- {summary['std_mae']:.4f})")
        
        return summary


# --------------------------------------------------------------------------
# UNCERTAINTY QUANTIFICATION
# --------------------------------------------------------------------------

class MCDropoutWrapper(nn.Module):
    """Wrapper for Monte Carlo Dropout uncertainty estimation."""
    
    def __init__(self, model: nn.Module, n_samples: int = 50):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional uncertainty estimation.
        
        Returns:
            mean_prediction: Average prediction across samples
            uncertainty: Standard deviation of predictions (if return_uncertainty=True)
        """
        if not return_uncertainty:
            self.model.eval()
            out = self.model(x)
            if isinstance(out, tuple):
                out = out[0]
            return out, None
            
        # Enable dropout for uncertainty estimation
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(x)
                if isinstance(out, tuple):
                    out = out[0]
                predictions.append(out)
                
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch)
        
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


class QuantileRegressionHead(nn.Module):
    """Quantile regression for prediction intervals."""
    
    def __init__(
        self,
        input_dim: int,
        quantiles: List[float] = [0.05, 0.5, 0.95]
    ):
        super().__init__()
        self.quantiles = quantiles
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in quantiles
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions for each quantile."""
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=-1)  # (batch, n_quantiles)
    
    
class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""
    
    def __init__(self, quantiles: List[float] = [0.05, 0.5, 0.95]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, n_quantiles)
            targets: (batch,)
        """
        targets = targets.unsqueeze(-1).expand_as(predictions)
        errors = targets - predictions
        
        losses = []
        for i, q in enumerate(self.quantiles):
            error = errors[:, i]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
            
        return torch.stack(losses, dim=-1).mean()


# --------------------------------------------------------------------------
# CUSTOM LOSS FUNCTIONS
# --------------------------------------------------------------------------

class DirectionalLoss(nn.Module):
    """Loss that penalizes wrong direction predictions more heavily."""
    
    def __init__(self, direction_weight: float = 0.3, mse_weight: float = 0.7):
        super().__init__()
        self.direction_weight = direction_weight
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prev_targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mse_loss = self.mse(predictions, targets)
        
        if prev_targets is None:
            return mse_loss
            
        # Direction accuracy
        actual_direction = torch.sign(targets - prev_targets)
        pred_direction = torch.sign(predictions - prev_targets)
        direction_correct = (actual_direction == pred_direction).float()
        direction_loss = 1 - direction_correct.mean()
        
        return self.mse_weight * mse_loss + self.direction_weight * direction_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for cases where over/under-prediction has different costs."""
    
    def __init__(self, over_weight: float = 1.0, under_weight: float = 1.5):
        super().__init__()
        self.over_weight = over_weight
        self.under_weight = under_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = predictions - targets
        
        over_pred = F.relu(errors) ** 2 * self.over_weight
        under_pred = F.relu(-errors) ** 2 * self.under_weight
        
        return (over_pred + under_pred).mean()


class SharpeRatioLoss(nn.Module):
    """Loss based on Sharpe ratio for financial optimization."""
    
    def __init__(self, risk_free_rate: float = 0.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Predicted returns
            targets: Actual returns
            positions: Trading positions (optional, derived from predictions if None)
        """
        if positions is None:
            positions = torch.sign(predictions)
            
        returns = positions * targets
        
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        return -sharpe  # Negative because we minimize loss


# --------------------------------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------------------------------

class FeatureImportanceAnalyzer:
    """Analyze feature importance using permutation importance."""
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: torch.device = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = device or get_device()
        self.model.to(self.device)
        
    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """Compute permutation importance for each feature."""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Baseline score
        with torch.no_grad():
            baseline_pred = self.model(X_tensor)
            if isinstance(baseline_pred, tuple):
                baseline_pred = baseline_pred[0]
            baseline_mse = F.mse_loss(baseline_pred, y_tensor).item()
            
        importance_scores = {}
        
        for feat_idx, feat_name in enumerate(self.feature_names):
            mse_increases = []
            
            for _ in range(n_repeats):
                # Permute feature across all time steps
                X_permuted = X.copy()
                perm_idx = np.random.permutation(len(X))
                X_permuted[:, :, feat_idx] = X[perm_idx, :, feat_idx]
                
                X_perm_tensor = torch.FloatTensor(X_permuted).to(self.device)
                
                with torch.no_grad():
                    perm_pred = self.model(X_perm_tensor)
                    if isinstance(perm_pred, tuple):
                        perm_pred = perm_pred[0]
                    perm_mse = F.mse_loss(perm_pred, y_tensor).item()
                    
                mse_increases.append(perm_mse - baseline_mse)
                
            importance_scores[feat_name] = np.mean(mse_increases)
            
        # Normalize
        total = sum(abs(v) for v in importance_scores.values())
        if total > 0:
            importance_scores = {k: v / total for k, v in importance_scores.items()}
            
        return dict(sorted(importance_scores.items(), key=lambda x: -abs(x[1])))
    
    def plot_importance(
        self,
        importance_scores: Dict[str, float],
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """Plot feature importance."""
        top_features = list(importance_scores.items())[:top_n]
        features, scores = zip(*top_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['green' if s > 0 else 'red' for s in scores]
        ax.barh(range(len(features)), scores, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance (Permutation)')
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# --------------------------------------------------------------------------
# ATTENTION VISUALIZATION
# --------------------------------------------------------------------------

class AttentionVisualizer:
    """Visualize attention patterns in attention-based models."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or get_device()
        self.attention_weights = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def get_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    # MultiheadAttention returns (output, attention_weights)
                    self.attention_weights[name] = output[1].detach().cpu()
            return hook
            
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(get_attention_hook(name))
                
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention maps for input."""
        self.model.eval()
        self.attention_weights.clear()
        
        x = x.to(self.device)
        with torch.no_grad():
            _ = self.model(x)
            
        return self.attention_weights
    
    def plot_attention_heatmap(
        self,
        attention: torch.Tensor,
        title: str = 'Attention Weights',
        save_path: Optional[str] = None
    ):
        """Plot attention heatmap."""
        if len(attention.shape) == 3:
            # Average over batch
            attention = attention.mean(0)
        if len(attention.shape) == 3:
            # Average over heads
            attention = attention.mean(0)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            attention.numpy(),
            cmap='viridis',
            ax=ax,
            xticklabels=5,
            yticklabels=5
        )
        
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# --------------------------------------------------------------------------
# BACKTESTING UTILITIES
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Results from backtesting."""
    returns: np.ndarray
    positions: np.ndarray
    dates: pd.DatetimeIndex
    metrics: Dict[str, float]


class SimpleBacktester:
    """Simple backtesting framework for model predictions."""
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        position_sizing: str = 'binary'  # 'binary', 'proportional', 'kelly'
    ):
        self.transaction_cost = transaction_cost
        self.position_sizing = position_sizing
        
    def run(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> BacktestResult:
        """Run backtest."""
        n = len(predictions)
        
        # Generate positions
        if self.position_sizing == 'binary':
            positions = np.sign(predictions)
        elif self.position_sizing == 'proportional':
            positions = np.tanh(predictions)  # Scale to [-1, 1]
        else:
            positions = np.sign(predictions)
            
        # Calculate returns
        returns = actuals[1:] / actuals[:-1] - 1  # Simple returns
        
        # Strategy returns (shift positions by 1 for execution delay)
        strategy_returns = positions[:-1] * returns
        
        # Transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], positions[:-1]])))
        costs = position_changes * self.transaction_cost
        strategy_returns -= costs
        
        # Metrics
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        max_drawdown = self._max_drawdown(1 + strategy_returns)
        win_rate = np.mean(strategy_returns > 0)
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': np.sum(position_changes > 0),
            'avg_return_per_trade': np.mean(strategy_returns[position_changes[1:] > 0]) if np.any(position_changes[1:] > 0) else 0
        }
        
        return BacktestResult(
            returns=strategy_returns,
            positions=positions[:-1],
            dates=dates[1:],
            metrics=metrics
        )
        
    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown)
    
    def plot_equity_curve(
        self,
        result: BacktestResult,
        benchmark_returns: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """Plot equity curve."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Equity curve
        ax = axes[0]
        strategy_equity = np.cumprod(1 + result.returns)
        ax.plot(result.dates, strategy_equity, label='Strategy', linewidth=1.5)
        
        if benchmark_returns is not None:
            benchmark_equity = np.cumprod(1 + benchmark_returns)
            ax.plot(result.dates, benchmark_equity, label='Benchmark', 
                   linewidth=1.5, alpha=0.7)
            
        ax.set_ylabel('Equity')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Positions
        ax = axes[1]
        ax.fill_between(result.dates, 0, result.positions, alpha=0.5)
        ax.set_ylabel('Position')
        ax.set_title('Trading Positions')
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[2]
        cumulative = np.cumprod(1 + result.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max * 100
        ax.fill_between(result.dates, 0, -drawdown, color='red', alpha=0.5)
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
