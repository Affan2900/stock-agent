import torch
import torch.nn as nn
from typing import List, Tuple, Sequence

class PinballLoss(nn.Module):
    """
    Multi-Quantile Pinball (Quantile) Loss Function for PyTorch.
    
    L_q(y, \hat{y}_q) = max(q * (y - \hat{y}_q), (q - 1) * (y - \hat{y}_q))
    
    Computes loss across batch samples B, horizon steps H, and quantiles Q.
    """
    
    def __init__(self, quantiles: Sequence[float] = (0.10, 0.50, 0.90)):
        super().__init__()
        self.quantiles = tuple(quantiles)
        # Register quantiles tensor as buffer
        self.register_buffer("q_tensor", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Tensor of shape (B, H, Q) where Q = len(quantiles)
            y_true: Tensor of shape (B, H)
            
        Returns:
            Scalar loss tensor
        """
        if y_true.ndim == 2:
            # Expand y_true to (B, H, 1) for broadcasting across Q
            y_true = y_true.unsqueeze(-1) # (B, H, 1)
            
        errors = y_true - y_pred # (B, H, Q)
        
        # Calculate pinball loss per element
        # max(q * e, (q - 1) * e)
        q_tensor = self.q_tensor.view(1, 1, -1) # (1, 1, Q)
        loss = torch.max(q_tensor * errors, (q_tensor - 1.0) * errors) # (B, H, Q)
        
        return torch.mean(loss)
