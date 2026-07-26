import copy
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from forecasting.models.losses import PinballLoss
from forecasting.models.lstm import QuantileLSTMForecaster

logger = logging.getLogger(__name__)

def create_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Wrap numpy feature and target arrays into PyTorch DataLoader."""
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_quantile_lstm(
    model: QuantileLSTMForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "cpu"
) -> Tuple[QuantileLSTMForecaster, Dict[str, list]]:
    """
    Standard training loop with Early Stopping and LR Scheduling based on validation Pinball Loss.
    """
    model.to(device)
    criterion = PinballLoss(quantiles=model.quantiles).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    
    best_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x) # (B, H, 3)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            n_train_batches += 1
            
        avg_train_loss = train_loss_sum / max(1, n_train_batches)
        
        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss_sum += loss.item()
                n_val_batches += 1
                
        avg_val_loss = val_loss_sum / max(1, n_val_batches)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss - 1e-6:
            best_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_loss:.6f}")
                break
                
    model.load_state_dict(best_model_weights)
    return model, history

def fine_tune_child_model(
    parent_model: QuantileLSTMForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    freeze_encoder: bool = False,
    epochs: int = 30,
    lr: float = 5e-4,
    device: str = "cpu"
) -> QuantileLSTMForecaster:
    """
    Fine-tune pretrained parent model on child ticker dataset.
    """
    child_model = copy.deepcopy(parent_model)
    if freeze_encoder:
        for param in child_model.lstm.parameters():
            param.requires_grad = False
            
    trained_child, _ = train_quantile_lstm(
        model=child_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device
    )
    return trained_child
