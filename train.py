"""
Phase 1: Train the base credit scoring model M on the full training set D.
Supports FT-Transformer, TabTransformer, and TabDDPM-style diffusion conditioning.
Saves model checkpoint to disk for all subsequent unlearning experiments.
"""

import os
import time
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Tuple

from data.datasets import CreditDataset, make_loader
from models.ft_transformer import FTTransformer
from models.tab_transformer import TabTransformer
from models.tabddpm import TabDDPM
from models.lora import count_parameters


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────

def build_model(arch: str, num_num_features: int, cat_dims, device: torch.device,
                d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                dropout: float = 0.1) -> nn.Module:
    effective_heads = max(1, min(n_heads, d_model))
    while d_model % effective_heads != 0 and effective_heads > 1:
        effective_heads -= 1

    if arch == "ft_transformer":
        model = FTTransformer(
            num_num_features=num_num_features,
            cat_dims=cat_dims,
            d_model=d_model,
            n_heads=effective_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
    elif arch == "tab_transformer":
        tab_d_model = max(d_model // 4, 16)
        tab_heads = min(effective_heads, 4)
        while tab_d_model % tab_heads != 0 and tab_heads > 1:
            tab_heads -= 1
        model = TabTransformer(
            num_num_features=num_num_features,
            cat_dims=cat_dims,
            d_model=tab_d_model,
            n_heads=tab_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
    elif arch == "tabddpm":
        ddpm_d_model = max(d_model, 64)
        ddpm_heads = min(effective_heads, 8)
        while ddpm_d_model % ddpm_heads != 0 and ddpm_heads > 1:
            ddpm_heads -= 1
        model = TabDDPM(
            num_num_features=num_num_features,
            cat_dims=cat_dims,
            d_model=ddpm_d_model,
            n_heads=ddpm_heads,
            n_layers=max(n_layers, 4),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model.to(device)


# ──────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────

def compute_loss(model: nn.Module, batch, device: torch.device, criterion):
    x_num, x_cat, y = batch
    x_num = x_num.to(device) if x_num.numel() > 0 else None
    x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
    y = y.to(device)

    if hasattr(model, "compute_training_loss"):
        return model.compute_training_loss(x_num, x_cat, y, criterion)

    logits = model(x_num, x_cat)
    return criterion(logits, y)


def evaluate(model: nn.Module, loader, device: torch.device):
    """Returns AUC-ROC, accuracy, and average BCE loss."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x_num, x_cat, y = batch
            x_num = x_num.to(device) if x_num.numel() > 0 else None
            x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
            y = y.to(device)
            logits = model(x_num, x_cat)
            total_loss += criterion(logits, y).item()
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
            n_batches += 1

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy().astype(int)

    if labels.sum() == 0 or labels.sum() == len(labels):
        auc = 0.5
    else:
        auc = roc_auc_score(labels, probs)

    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    return {"auc": auc, "acc": acc, "loss": total_loss / max(n_batches, 1)}


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_ds: CreditDataset,
    val_ds: CreditDataset,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    verbose: bool = True,
) -> Tuple[nn.Module, dict]:
    """Train model with early stopping on val AUC. Returns best model and history."""
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_auc": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch, device, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_auc"].append(val_metrics["auc"])
        history["val_acc"].append(val_metrics["acc"])

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train_loss={total_loss/len(train_loader):.4f} "
                  f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    if verbose:
        print(f"  Training done in {elapsed:.1f}s | best val AUC={best_auc:.4f}")

    total_params, trainable_params = count_parameters(model)
    history["elapsed"] = elapsed
    history["best_val_auc"] = best_auc
    history["total_params"] = total_params
    history["trainable_params"] = trainable_params
    return model, history


def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Saved model → {path}")


def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model
