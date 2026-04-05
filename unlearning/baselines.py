"""
Baseline unlearning methods to compare against LoRA-Credit-Unlearn.

Baselines implemented:
  1. Full Retrain    — retrain from scratch on D_r only (gold standard, slow)
  2. Gradient Ascent — naïve: gradient ascent on D_f without adapter isolation
  3. Fine-tune D_r   — fine-tune original model only on D_r (catastrophic forgetting risk)
  4. SISA            — simplified: re-partition + retrain affected shards
  5. Influence Fn    — Newton-step parameter update approximation
  6. Random Labels   — relabel D_f and retrain on retain + corrupted forget data

All return a model and a timing dict for fair comparison.
"""

import copy
import time
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.datasets import CreditDataset, make_loader
from train import train_model, evaluate
from unlearning.random_labels import unlearn as random_labels_core_unlearn


# ──────────────────────────────────────────────
# 1. Full Retrain (gold standard)
# ──────────────────────────────────────────────

def baseline_full_retrain(
    model_factory,               # callable() → fresh model
    retain_ds: CreditDataset,
    val_ds: CreditDataset,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = True,
) -> tuple:
    """Retrain a fresh model from scratch on D_r only. Gold standard."""
    if verbose:
        print("  [Baseline] Full Retrain on D_r...")
    t0 = time.time()
    model = model_factory().to(device)
    model, history = train_model(
        model, retain_ds, val_ds, device,
        epochs=epochs, batch_size=batch_size, lr=lr, verbose=False
    )
    elapsed = time.time() - t0
    history["elapsed"] = elapsed
    if verbose:
        print(f"    Elapsed: {elapsed:.1f}s")
    return model, history


# ──────────────────────────────────────────────
# 2. Naïve Gradient Ascent
# ──────────────────────────────────────────────

def baseline_gradient_ascent(
    model: nn.Module,
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    val_ds: CreditDataset,
    device: torch.device,
    max_steps: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    retain_auc_min: float = 0.55,
    verbose: bool = True,
) -> tuple:
    """
    Naïve gradient ascent: update ALL model params (no LoRA isolation).
    Known to damage retain performance if run too long.
    """
    if verbose:
        print("  [Baseline] Naïve Gradient Ascent...")
    model_ga = copy.deepcopy(model).to(device)
    optimizer = Adam(model_ga.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    forget_loader = make_loader(forget_ds, batch_size=batch_size, shuffle=True)
    retain_loader = make_loader(retain_ds, batch_size=256, shuffle=False)

    history = {"step": [], "retain_auc": [], "forget_acc": []}
    forget_iter = iter(forget_loader)
    t0 = time.time()

    for step in range(1, max_steps + 1):
        model_ga.train()
        try:
            batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch = next(forget_iter)

        x_num, x_cat, y = batch
        x_num = x_num.to(device) if x_num.numel() > 0 else None
        x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
        y = y.to(device)

        optimizer.zero_grad()
        logits = model_ga(x_num, x_cat)
        loss = -criterion(logits, y)  # ascent
        loss.backward()
        nn.utils.clip_grad_norm_(model_ga.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == max_steps:
            retain_m = evaluate(model_ga, retain_loader, device)
            forget_m = evaluate(model_ga, make_loader(forget_ds, 256, False), device)
            history["step"].append(step)
            history["retain_auc"].append(retain_m["auc"])
            history["forget_acc"].append(forget_m["acc"])

            if retain_m["auc"] < retain_auc_min:
                if verbose:
                    print(f"    GA: retain AUC collapsed to {retain_m['auc']:.3f} — stopping")
                break

    history["elapsed"] = time.time() - t0
    return model_ga, history


# ──────────────────────────────────────────────
# 3. Fine-tune on D_r
# ──────────────────────────────────────────────

def baseline_finetune_retain(
    model: nn.Module,
    retain_ds: CreditDataset,
    val_ds: CreditDataset,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 5e-5,
    verbose: bool = True,
) -> tuple:
    """Fine-tune original model on D_r only. May erase forget set but hurts utility."""
    if verbose:
        print("  [Baseline] Fine-tune on D_r...")
    model_ft = copy.deepcopy(model).to(device)
    t0 = time.time()
    model_ft, history = train_model(
        model_ft, retain_ds, val_ds, device,
        epochs=epochs, batch_size=batch_size, lr=lr, verbose=False, patience=5
    )
    history["elapsed"] = time.time() - t0
    return model_ft, history


# ──────────────────────────────────────────────
# 4. SISA (Sharded, Isolated, Sliced, Aggregated) — simplified
# ──────────────────────────────────────────────

def baseline_sisa(
    model_factory,
    full_train_ds: CreditDataset,
    forget_indices,               # indices in full_train_ds that must be forgotten
    val_ds: CreditDataset,
    device: torch.device,
    n_shards: int = 4,
    epochs_per_shard: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = True,
) -> tuple:
    """
    Simplified SISA:
    1. Partition D into n_shards
    2. Identify which shard(s) contain D_f
    3. Retrain only those shards
    4. Aggregate predictions by averaging logits

    In practice returns a list of models (one per shard).
    For evaluation, we use majority-vote / average logits.
    """
    if verbose:
        print(f"  [Baseline] SISA with {n_shards} shards...")
    import numpy as np

    n_total = len(full_train_ds)
    all_idx = np.arange(n_total)
    forget_set = set(forget_indices.tolist())

    # Partition into shards
    shard_size = n_total // n_shards
    shards = [all_idx[i * shard_size:(i + 1) * shard_size] for i in range(n_shards)]
    if n_total % n_shards != 0:
        shards[-1] = np.append(shards[-1], all_idx[n_shards * shard_size:])

    # Identify affected shards
    affected = [i for i, s in enumerate(shards) if any(j in forget_set for j in s)]
    if verbose:
        print(f"    Affected shards: {affected} (out of {n_shards})")

    t0 = time.time()
    shard_models = []
    for i, shard_idx in enumerate(shards):
        if i in affected:
            # Remove forget samples from this shard and retrain
            clean_idx = np.array([j for j in shard_idx if j not in forget_set])
            if len(clean_idx) == 0:
                continue
            from data.datasets import subset_dataset
            shard_ds = subset_dataset(full_train_ds, clean_idx)
        else:
            from data.datasets import subset_dataset
            shard_ds = subset_dataset(full_train_ds, shard_idx)

        m = model_factory().to(device)
        m, _ = train_model(m, shard_ds, val_ds, device,
                           epochs=epochs_per_shard, batch_size=batch_size,
                           lr=lr, verbose=False, patience=5)
        shard_models.append(m)

    elapsed = time.time() - t0

    # Create an ensemble wrapper
    ensemble = _SISAEnsemble(shard_models)
    history = {"elapsed": elapsed, "n_shards_retrained": len(affected)}
    if verbose:
        print(f"    SISA elapsed: {elapsed:.1f}s")
    return ensemble, history


class _SISAEnsemble(nn.Module):
    """Wraps a list of shard models. Forward averages their logits."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x_num, x_cat=None):
        logits = torch.stack([m(x_num, x_cat) for m in self.models], dim=0)
        return logits.mean(dim=0)


# ──────────────────────────────────────────────
# 5. Influence Functions (Newton-step approximation)
# ──────────────────────────────────────────────

def baseline_influence_functions(
    model: nn.Module,
    forget_ds: CreditDataset,
    device: torch.device,
    damping: float = 0.01,
    scale: float = 1.0,
    verbose: bool = True,
) -> tuple:
    """
    Approximate influence function unlearning:
    θ* = θ - (1/n) * H^{-1} * ∇L_f(θ)

    H^{-1} approximated by LiSSA (identity + damping for simplicity).
    This is a fast approximation — exact H^{-1} is intractable for large models.
    """
    if verbose:
        print("  [Baseline] Influence Functions (Newton-step approx)...")
    model_if = copy.deepcopy(model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    forget_loader = make_loader(forget_ds, batch_size=len(forget_ds), shuffle=False)

    t0 = time.time()
    model_if.eval()

    # Compute gradient of forget loss w.r.t. all parameters
    for param in model_if.parameters():
        param.requires_grad_(True)

    model_if.zero_grad()
    batch = next(iter(forget_loader))
    x_num, x_cat, y = batch
    x_num = x_num.to(device) if x_num.numel() > 0 else None
    x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
    y = y.to(device)

    logits = model_if(x_num, x_cat)
    loss = criterion(logits, y)
    loss.backward()

    # Newton-step: approximate H^{-1} as (1 / (λ + damping)) * I
    n_f = len(forget_ds)
    with torch.no_grad():
        for param in model_if.parameters():
            if param.grad is not None:
                # Remove influence: subtract H^{-1} ∇L_f / n
                param.data -= scale * param.grad / (n_f * damping)
                param.grad = None

    elapsed = time.time() - t0
    history = {"elapsed": elapsed}
    if verbose:
        print(f"    Influence fn elapsed: {elapsed:.1f}s")
    return model_if, history


def baseline_random_labels(
    model: nn.Module,
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    val_ds: CreditDataset,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 5e-5,
    patience: int = 3,
    forget_multiplier: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Random-label corruption baseline, primarily intended for TabDDPM."""
    if verbose:
        print("  [Baseline] Random Labels on D_f + D_r...")
    t0 = time.time()
    model_rl = random_labels_core_unlearn(
        model,
        forget_ds,
        retain_ds,
        val_ds,
        {
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "forget_multiplier": forget_multiplier,
            "seed": seed,
            "verbose": False,
        },
        device,
    )
    history = {"elapsed": time.time() - t0}
    if verbose:
        print(f"    Random-label elapsed: {history['elapsed']:.1f}s")
    return model_rl, history
