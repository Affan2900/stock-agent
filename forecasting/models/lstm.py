import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple

class QuantileLSTMForecaster(nn.Module):
    """
    LSTM Encoder + Direct Multi-Horizon Quantile Regression Head.
    
    Predicts multi-step horizon targets (H=5) and quantiles Q={0.10, 0.50, 0.90} in a single
    forward pass. Enforces quantile monotonicity (q_lower <= q_median <= q_upper) via Softplus offsets.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 5,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.quantiles = tuple(quantiles)
        self.num_quantiles = len(quantiles)
        
        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc_dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Head for median predictions (H steps)
        self.head_median = nn.Linear(hidden_dim, horizon)
        
        # Head for lower and upper quantile positive offsets
        # If 3 quantiles (0.10, 0.50, 0.90), we need 2 offset heads (lower offset & upper offset)
        self.head_lower_offset = nn.Linear(hidden_dim, horizon)
        self.head_upper_offset = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, T, D_in)
            
        Returns:
            Tensor of shape (B, H, Q) where Q = len(quantiles) ordered [q_lower, q_median, q_upper]
        """
        # lstm_out: (B, T, hidden_dim), (h_n, c_n)
        lstm_out, _ = self.lstm(x)
        
        # Take last time step output h_T
        h_t = lstm_out[:, -1, :] # (B, hidden_dim)
        feat = self.fc_dense(h_t) # (B, hidden_dim)
        
        median = self.head_median(feat) # (B, H)
        lower_offset = F.softplus(self.head_lower_offset(feat)) # (B, H) >= 0
        upper_offset = F.softplus(self.head_upper_offset(feat)) # (B, H) >= 0
        
        q_lower = median - lower_offset # (B, H)
        q_upper = median + upper_offset # (B, H)
        
        # Stack into (B, H, 3) representing [q_0.10, q_0.50, q_0.90]
        out = torch.stack([q_lower, median, q_upper], dim=-1) # (B, H, 3)
        return out
