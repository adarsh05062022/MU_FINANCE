"""
Method 8: Random Labels (v4 — entropy-only unlearning).

The correct approach for random-label unlearning:
  1. Maximise entropy on forget set (make model uncertain, not wrong)
  2. Fine-tune on retain-only with low LR to restore utility
  
Key insight: never use train_model on forget-only data because val_AUC-based
early stopping will undo the forgetting by picking the best-memorization checkpoint.
"""

import copy
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.datasets import CreditDataset
from train import train_model


# ── Tensor helpers ────────────────────────────────────────────────────────────

def _repeat_optional_tensor(x: Optional[torch.Tensor], times: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return torch.cat([x] * times, dim=0)


def _concat_optional_tensor(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if a is None:
        return b
    if b is None:
        return a
    return torch.cat([a, b], dim=0)


def _build_mixed_dataset(
    D_retain: CreditDataset,
    D_forget: CreditDataset,
    forget_multiplier: int,
) -> CreditDataset:
    repeated_forget = CreditDataset(
        x_num=_repeat_optional_tensor(D_forget.x_num, forget_multiplier),
        x_cat=_repeat_optional_tensor(D_forget.x_cat, forget_multiplier),
        y=_repeat_optional_tensor(D_forget.y, forget_multiplier),
        cat_dims=list(D_forget.cat_dims),
        num_num_features=D_forget.num_num_features,
        feature_names=list(D_forget.feature_names),
    )
    return CreditDataset(
        x_num=_concat_optional_tensor(D_retain.x_num, repeated_forget.x_num),
        x_cat=_concat_optional_tensor(D_retain.x_cat, repeated_forget.x_cat),
        y=torch.cat([D_retain.y, repeated_forget.y], dim=0),
        cat_dims=list(D_retain.cat_dims),
        num_num_features=D_retain.num_num_features,
        feature_names=list(D_retain.feature_names),
    )


def _forward(model, x_num, x_cat):
    """Unified forward pass handling different model signatures."""
    try:
        if x_num is not None and x_cat is not None:
            return model(x_num, x_cat)
        elif x_num is not None:
            return model(x_num)
        else:
            return model(x_cat)
    except TypeError:
        parts = []
        if x_num is not None:
            parts.append(x_num)
        if x_cat is not None:
            parts.append(x_cat.float())
        return model(torch.cat(parts, dim=-1))


def _get_forget_probs(model, D_forget, device, batch_size=64):
    """Compute sigmoid probabilities on forget set. Used for monitoring."""
    model.eval()
    all_probs = []
    x_num = D_forget.x_num.to(device).float() if D_forget.x_num is not None else None
    x_cat = D_forget.x_cat.to(device).long() if D_forget.x_cat is not None else None
    n = D_forget.y.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            bn = x_num[start:start+batch_size] if x_num is not None else None
            bc = x_cat[start:start+batch_size] if x_cat is not None else None
            logits = _forward(model, bn, bc)
            if logits.shape[-1] == 1 or logits.dim() == 1:
                p = torch.sigmoid(logits.squeeze(-1))
            else:
                p = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(p.cpu())
    probs = torch.cat(all_probs)
    entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
    return probs.mean().item(), entropy.mean().item()


def _entropy_disruption(
    model: torch.nn.Module,
    D_forget: CreditDataset,
    device: torch.device,
    lr: float = 2e-4,
    epochs: int = 10,
    batch_size: int = 32,
    target_entropy: float = 0.65,   # ln(2) ≈ 0.693 is max; stop near there
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Maximise output entropy on forget samples via gradient ascent on H(p).

    Stops early if target entropy is reached to avoid over-disruption.
    Uses small batch size (32) since forget set is only 70 samples.
    No val-AUC early stopping — we control termination by entropy threshold.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_num = D_forget.x_num.to(device).float() if D_forget.x_num is not None else None
    x_cat = D_forget.x_cat.to(device).long() if D_forget.x_cat is not None else None
    n = D_forget.y.shape[0]
    idx_tensor = torch.arange(n)

    for epoch in range(epochs):
        model.train()
        loader = DataLoader(TensorDataset(idx_tensor), batch_size=batch_size, shuffle=True)
        epoch_loss = 0.0

        for (idx_batch,) in loader:
            optimizer.zero_grad()
            bn = x_num[idx_batch] if x_num is not None else None
            bc = x_cat[idx_batch] if x_cat is not None else None

            logits = _forward(model, bn, bc)
            if logits.shape[-1] == 1 or logits.dim() == 1:
                p = torch.sigmoid(logits.squeeze(-1))
            else:
                p = torch.softmax(logits, dim=-1)[:, 1]

            eps = 1e-8
            entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
            loss = -entropy.mean()  # maximise entropy = minimise negative entropy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()

        # Check current entropy on full forget set
        mean_p, mean_h = _get_forget_probs(model, D_forget, device, batch_size)
        if verbose:
            print(f"  [entropy] epoch {epoch+1}/{epochs} | "
                  f"loss={epoch_loss/len(loader):.4f} | "
                  f"forget_p={mean_p:.3f} | forget_H={mean_h:.4f} "
                  f"(target≥{target_entropy:.3f})")

        if mean_h >= target_entropy:
            if verbose:
                print(f"  [entropy] Target entropy reached — stopping early.")
            break

    return model


def unlearn(model, D_forget, D_retain, D_val, config, device):
    """
    v4: Entropy disruption + retain-only recovery.

    config keys:
        entropy_lr       : float (default 2e-4)
        entropy_epochs   : int   (default 10)
        entropy_bs       : int   (default 32)
        target_entropy   : float (default 0.65)  # ln(2)≈0.693 is max
        phase2_lr        : float (default 3e-5)
        phase2_epochs    : int   (default 10)
        phase2_bs        : int   (default 128)
        patience         : int   (default 5)
        verbose          : bool  (default True)
    """
    model = copy.deepcopy(model).to(device)

    entropy_lr     = config.get("entropy_lr",     2e-4)
    entropy_epochs = config.get("entropy_epochs", 10)
    entropy_bs     = config.get("entropy_bs",     32)
    target_entropy = config.get("target_entropy", 0.65)
    phase2_lr      = config.get("phase2_lr",      3e-5)
    phase2_epochs  = config.get("phase2_epochs",  10)
    phase2_bs      = config.get("phase2_bs",      128)
    patience       = config.get("patience",       5)
    verbose        = config.get("verbose",        True)

    # ── Phase 1: Entropy maximisation on forget set ───────────────────────
    # No train_model here — we control stopping by entropy threshold, not val AUC.
    if verbose:
        p0, h0 = _get_forget_probs(model, D_forget, device)
        print(f"[random_labels] Before: forget_p={p0:.3f}, forget_H={h0:.4f}")
        print("[random_labels] Phase 1: Entropy disruption...")

    model = _entropy_disruption(
        model, D_forget, device,
        lr=entropy_lr,
        epochs=entropy_epochs,
        batch_size=entropy_bs,
        target_entropy=target_entropy,
        verbose=verbose,
    )

    if verbose:
        p1, h1 = _get_forget_probs(model, D_forget, device)
        print(f"[random_labels] After Phase 1: forget_p={p1:.3f}, forget_H={h1:.4f}")

    # ── Phase 2: Retain-only recovery ────────────────────────────────────
    # Retain-ONLY (no forget samples) so train_model's val AUC early stopping
    # can't accidentally restore forget memorization.
    if verbose:
        print("[random_labels] Phase 2: Retain-only recovery...")

    model, _ = train_model(
        model, D_retain, D_val, device,
        epochs=phase2_epochs,
        batch_size=phase2_bs,
        lr=phase2_lr,
        patience=patience,
        verbose=verbose,
    )

    if verbose:
        p2, h2 = _get_forget_probs(model, D_forget, device)
        print(f"[random_labels] After Phase 2: forget_p={p2:.3f}, forget_H={h2:.4f}")

    return model


def random_labels_unlearn(model, D_forget, D_retain, D_val, config, device):
    """Convenience wrapper. Returns (model, elapsed_seconds)."""
    start = time.perf_counter()
    result = unlearn(model, D_forget, D_retain, D_val, config, device)
    elapsed = time.perf_counter() - start
    return result, elapsed