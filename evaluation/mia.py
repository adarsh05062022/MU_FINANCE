"""
Membership Inference Attack (MIA) for evaluating machine unlearning.

Method: Shadow model attack (Shokri et al. 2017).
  1. Train multiple shadow models on subsets of D_r ∪ D_f
  2. For each shadow model, label D_f samples as "member" and out-of-shadow samples as "non-member"
  3. Train a binary classifier (MIA attacker) on shadow model outputs
  4. Apply attacker to M* — if MIA accuracy ≈ 50%, forgetting is certified

MIA Score: accuracy of the attacker on the forget set of M*.
  - ≈ 50% → random (certified forgetting)
  - ≈ 100% → model fully remembers D_f (no forgetting)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Optional

from data.datasets import CreditDataset, make_loader, subset_dataset
from evaluation.metrics import get_predictions


# ──────────────────────────────────────────────
# Shadow model attack
# ──────────────────────────────────────────────

def _get_model_confidence(model: nn.Module, dataset: CreditDataset,
                          device: torch.device) -> np.ndarray:
    """Get confidence features for MIA: [P(y=1), P(y=0), max_conf, entropy]."""
    probs, _ = get_predictions(model, dataset, device)
    probs = np.nan_to_num(np.clip(probs, 1e-6, 1.0 - 1e-6), nan=0.5, posinf=1.0, neginf=0.0)
    conf_pos = probs
    conf_neg = 1 - probs
    max_conf = np.maximum(probs, 1 - probs)
    entropy = -(probs * np.log(probs + 1e-8) + (1 - probs) * np.log(1 - probs + 1e-8))
    return np.nan_to_num(np.stack([conf_pos, conf_neg, max_conf, entropy], axis=1),
                         nan=0.0, posinf=1.0, neginf=0.0)


def run_mia(
    target_model: nn.Module,           # M* — the unlearned model
    forget_ds: CreditDataset,          # D_f — supposed to be forgotten
    retain_ds: CreditDataset,          # D_r — still in the model
    train_ds: CreditDataset,           # Full training set (for shadow models)
    device: torch.device,
    n_shadow: int = 4,                 # number of shadow models
    shadow_frac: float = 0.5,          # fraction of train data per shadow model
    attacker: str = "lr",              # "lr" or "rf"
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Runs shadow-model MIA against target_model.

    Returns:
      mia_score   — attacker accuracy on forget set (goal: ≈ 0.5)
      mia_auc     — attacker AUC (goal: ≈ 0.5)
      attacker    — trained sklearn classifier
    """
    rng = np.random.RandomState(seed)
    n_train = len(train_ds)

    if verbose:
        print(f"  [MIA] Training {n_shadow} shadow models...")

    shadow_features = []
    shadow_labels = []

    for i in range(n_shadow):
        # Sample shadow training set
        shadow_idx = rng.choice(n_train, size=int(n_train * shadow_frac), replace=False)
        out_idx = np.setdiff1d(np.arange(n_train), shadow_idx)

        shadow_train = subset_dataset(train_ds, shadow_idx)
        shadow_out = subset_dataset(train_ds, out_idx[:min(len(out_idx), len(shadow_idx))])

        # Train shadow model (lightweight — just a few epochs)
        shadow_model = copy.deepcopy(target_model).to(device)
        # Reset shadow model weights
        for module in shadow_model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        from train import train_model
        shadow_model, _ = train_model(
            shadow_model, shadow_train, shadow_out, device,
            epochs=20, batch_size=128, lr=1e-3, verbose=False, patience=5
        )

        # Collect features: in-shadow = member (1), out-of-shadow = non-member (0)
        in_feats = _get_model_confidence(shadow_model, shadow_train, device)
        out_feats = _get_model_confidence(shadow_model, shadow_out, device)

        shadow_features.append(np.vstack([in_feats, out_feats]))
        shadow_labels.append(np.concatenate([np.ones(len(in_feats)), np.zeros(len(out_feats))]))

        if verbose:
            print(f"    Shadow model {i+1}/{n_shadow} done")

    X_shadow = np.vstack(shadow_features)
    y_shadow = np.concatenate(shadow_labels)
    X_shadow = np.nan_to_num(X_shadow, nan=0.0, posinf=1.0, neginf=0.0)

    # Train MIA attacker
    if attacker == "lr":
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_shadow, y_shadow)

    # Evaluate on forget set of TARGET model (expected: ≈ 50% if truly forgotten)
    forget_feats = _get_model_confidence(target_model, forget_ds, device)
    forget_feats = np.nan_to_num(forget_feats, nan=0.0, posinf=1.0, neginf=0.0)
    # All samples in D_f were originally members → label = 1
    forget_true = np.ones(len(forget_feats))
    forget_preds = clf.predict(forget_feats)
    forget_probs = clf.predict_proba(forget_feats)[:, 1]

    mia_score = accuracy_score(forget_true, forget_preds)
    if len(np.unique(forget_true)) < 2:
        mia_auc = 0.5
    else:
        try:
            mia_auc = roc_auc_score(forget_true, forget_probs)
        except Exception:
            mia_auc = 0.5

    # Also evaluate on retain set (should be high — model remembers D_r)
    retain_feats = _get_model_confidence(target_model, retain_ds, device)
    retain_feats = np.nan_to_num(retain_feats, nan=0.0, posinf=1.0, neginf=0.0)
    retain_true = np.ones(len(retain_feats))
    retain_mia = accuracy_score(retain_true, clf.predict(retain_feats))

    if verbose:
        print(f"  [MIA] forget_set MIA accuracy={mia_score:.4f} (goal≈0.5) | "
              f"retain_set MIA={retain_mia:.4f} | MIA AUC={mia_auc:.4f}")

    return {
        "mia_score": mia_score,
        "mia_auc": mia_auc,
        "retain_mia": retain_mia,
        "attacker_clf": clf,
    }


# ──────────────────────────────────────────────
# Simpler MIA: loss-based threshold attack
# ──────────────────────────────────────────────

def loss_based_mia(
    target_model: nn.Module,
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """
    Simple threshold-based MIA: samples with lower loss are likely members.
    Apply to D_f after unlearning — if loss ≈ D_r loss → forgetting certified.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    def compute_per_sample_loss(model, dataset):
        loader = make_loader(dataset, batch_size=256, shuffle=False)
        losses = []
        model.eval()
        with torch.no_grad():
            for x_num, x_cat, y in loader:
                x_num = x_num.to(device) if x_num.numel() > 0 else None
                x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
                y = y.to(device)
                logits = torch.nan_to_num(model(x_num, x_cat), nan=0.0, posinf=20.0, neginf=-20.0)
                loss = criterion(logits, y)
                losses.append(np.nan_to_num(loss.cpu().numpy(), nan=0.0, posinf=1e6, neginf=1e6))
        return np.concatenate(losses)

    forget_losses = compute_per_sample_loss(target_model, forget_ds)
    retain_losses = compute_per_sample_loss(target_model, retain_ds)

    # Threshold: retain mean loss
    threshold = retain_losses.mean()

    # Members (D_f) should have low loss → predict "member" if loss < threshold
    # After forgetting, D_f loss should be high → attack fails
    forget_preds = (forget_losses < threshold).astype(int)  # 1 = predicted member
    true_member = np.ones(len(forget_losses), dtype=int)

    attack_acc = accuracy_score(true_member, forget_preds)

    if verbose:
        print(f"  [MIA-Loss] forget_loss_mean={forget_losses.mean():.4f} "
              f"retain_loss_mean={retain_losses.mean():.4f} "
              f"attack_acc={attack_acc:.4f} (goal≈0.5)")

    return {
        "mia_score": attack_acc,
        "forget_loss_mean": float(forget_losses.mean()),
        "retain_loss_mean": float(retain_losses.mean()),
    }
