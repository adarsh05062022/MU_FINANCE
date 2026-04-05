"""
Evaluation metrics for credit scoring unlearning experiments.
"""

import copy
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, roc_auc_score

from data.datasets import CreditDataset, make_loader


def _sanitize_logits_tensor(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)


def _sanitize_probs_array(probs: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.clip(probs, 1e-6, 1.0 - 1e-6), nan=0.5, posinf=1.0, neginf=0.0)


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataset: CreditDataset,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return probabilities and labels as numpy arrays."""
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs, all_labels = [], []
    for x_num, x_cat, y in loader:
        x_num = x_num.to(device) if x_num.numel() > 0 else None
        x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
        logits = _sanitize_logits_tensor(model(x_num, x_cat))
        probs = torch.sigmoid(logits)
        all_probs.append(_sanitize_probs_array(probs.cpu().numpy()))
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def get_logits(
    model: nn.Module,
    dataset: CreditDataset,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Return logits as numpy array."""
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_logits = []
    for x_num, x_cat, _ in loader:
        x_num = x_num.to(device) if x_num.numel() > 0 else None
        x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
        logits = _sanitize_logits_tensor(model(x_num, x_cat))
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


def compute_auc(model: nn.Module, dataset: CreditDataset, device: torch.device) -> float:
    """Compute AUC with safe fallback for degenerate labels."""
    probs, labels = get_predictions(model, dataset, device)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return float(roc_auc_score(labels.astype(int), probs))


def forget_set_accuracy(model: nn.Module, forget_ds: CreditDataset, device: torch.device) -> float:
    probs, labels = get_predictions(model, forget_ds, device)
    preds = (probs > 0.5).astype(int)
    return float(accuracy_score(labels.astype(int), preds))


def forget_set_auc(model: nn.Module, forget_ds: CreditDataset, device: torch.device) -> float:
    return compute_auc(model, forget_ds, device)


def compute_forget_accuracy(
    model: nn.Module,
    D_forget: CreditDataset,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    probs, labels = get_predictions(model, D_forget, device)
    preds = (probs > threshold).astype(int)
    return float((preds == labels.astype(int)).mean())


def compute_forget_confidence(model: nn.Module, D_forget: CreditDataset, device: torch.device) -> float:
    probs, _ = get_predictions(model, D_forget, device)
    return float(probs.mean())


def compute_retain_auc(model: nn.Module, D_retain: CreditDataset, device: torch.device) -> float:
    return compute_auc(model, D_retain, device)


def compute_test_auc(model: nn.Module, D_test: CreditDataset, device: torch.device) -> float:
    return compute_auc(model, D_test, device)


def kl_divergence(
    model_star: nn.Module,
    model_ref: nn.Module,
    test_ds: CreditDataset,
    device: torch.device,
) -> float:
    """KL(M_ref || M_star) on test set."""
    probs_star, _ = get_predictions(model_star, test_ds, device)
    probs_ref, _ = get_predictions(model_ref, test_ds, device)
    eps = 1e-8
    p = np.column_stack([1.0 - probs_ref, probs_ref]) + eps
    q = np.column_stack([1.0 - probs_star, probs_star]) + eps
    return float(np.sum(p * np.log(p / q), axis=1).mean())


def compute_kl_divergence(
    model_unlearn: nn.Module,
    model_retrain: nn.Module,
    D_test: CreditDataset,
    device: torch.device,
) -> float:
    return kl_divergence(model_unlearn, model_retrain, D_test, device)


def compute_js_divergence(
    model_a: nn.Module,
    model_b: nn.Module,
    D_test: CreditDataset,
    device: torch.device,
) -> float:
    probs_a, _ = get_predictions(model_a, D_test, device)
    probs_b, _ = get_predictions(model_b, D_test, device)
    dist_a = np.column_stack([1.0 - probs_a, probs_a])
    dist_b = np.column_stack([1.0 - probs_b, probs_b])
    js = [jensenshannon(dist_a[i], dist_b[i]) ** 2 for i in range(len(dist_a))]
    return float(np.mean(js))


def relearn_time(
    model: nn.Module,
    forget_ds: CreditDataset,
    original_forget_auc: float,
    device: torch.device,
    max_steps: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    tolerance: float = 0.02,
) -> int:
    """Steps needed to recover original forget-set AUC."""
    model_rl = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model_rl.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(forget_ds, batch_size=batch_size, shuffle=True)
    data_iter = iter(loader)

    for step in range(1, max_steps + 1):
        model_rl.train()
        try:
            x_num, x_cat, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x_num, x_cat, y = next(data_iter)

        x_num = x_num.to(device) if x_num.numel() > 0 else None
        x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
        y = y.to(device)

        optimizer.zero_grad()
        logits = _sanitize_logits_tensor(model_rl(x_num, x_cat))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            current_auc = forget_set_auc(model_rl, forget_ds, device)
            if current_auc >= original_forget_auc - tolerance:
                return step

    return max_steps


def compute_relearn_time(
    unlearned_model: nn.Module,
    D_forget: CreditDataset,
    D_val: Optional[CreditDataset],
    config: Dict,
    device: torch.device,
) -> int:
    original_forget_acc = config.get("original_forget_acc")
    if original_forget_acc is None:
        return relearn_time(unlearned_model, D_forget, 0.5, device)

    model = copy.deepcopy(unlearned_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("relearn_lr", 1e-4))
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(D_forget, batch_size=config.get("relearn_batch_size", 64), shuffle=True)
    steps = 0
    max_steps = config.get("relearn_max_steps", 1000)

    while steps < max_steps:
        for x_num, x_cat, y in loader:
            x_num = x_num.to(device) if x_num.numel() > 0 else None
            x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
            y = y.to(device)

            optimizer.zero_grad()
            logits = _sanitize_logits_tensor(model(x_num, x_cat))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            steps += 1

            current_acc = compute_forget_accuracy(model, D_forget, device)
            if current_acc >= original_forget_acc - 0.01:
                return steps
            if steps >= max_steps:
                break

    return steps


def equalized_odds_difference(
    probs: np.ndarray,
    labels: np.ndarray,
    sensitive: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Simple equalized-odds gap for binary sensitive attribute."""
    preds = (probs > threshold).astype(int)
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 0.0

    def _tpr(mask):
        pos = mask & (labels == 1)
        return preds[pos].mean() if pos.sum() else 0.0

    def _fpr(mask):
        neg = mask & (labels == 0)
        return preds[neg].mean() if neg.sum() else 0.0

    mask_a = sensitive == groups[0]
    mask_b = sensitive == groups[1]
    return float(abs(_tpr(mask_a) - _tpr(mask_b)) + abs(_fpr(mask_a) - _fpr(mask_b)))


def compute_ece(
    model: nn.Module,
    D_test: CreditDataset,
    device: torch.device,
    n_bins: int = 15,
) -> float:
    probs, labels = get_predictions(model, D_test, device)
    labels = labels.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)


def time_unlearning(unlearn_fn, *args, **kwargs):
    start = time.perf_counter()
    result = unlearn_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def count_updated_params(
    original_model: nn.Module,
    unlearned_model: nn.Module,
    atol: float = 1e-8,
) -> Tuple[int, int, float]:
    total = 0
    changed = 0
    for (_, p_orig), (_, p_new) in zip(original_model.named_parameters(), unlearned_model.named_parameters()):
        total += p_orig.numel()
        changed += int((~torch.isclose(p_orig.detach().cpu(), p_new.detach().cpu(), atol=atol)).sum().item())
    pct = 100.0 * changed / max(total, 1)
    return changed, total, pct


def full_evaluation(
    model_star: nn.Module,
    model_original: nn.Module,
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    test_ds: CreditDataset,
    device: torch.device,
    original_forget_acc: float = None,
    sensitive_attr: Optional[np.ndarray] = None,
    elapsed_seconds: float = 0.0,
    verbose: bool = True,
) -> Dict:
    """Compact evaluation helper retained for compatibility."""
    results = {
        "forget_acc": compute_forget_accuracy(model_star, forget_ds, device),
        "forget_auc": forget_set_auc(model_star, forget_ds, device),
        "forget_confidence": compute_forget_confidence(model_star, forget_ds, device),
        "retain_auc": compute_retain_auc(model_star, retain_ds, device),
        "test_auc": compute_test_auc(model_star, test_ds, device),
        "kl_div": kl_divergence(model_star, model_original, test_ds, device),
        "js_divergence": compute_js_divergence(model_star, model_original, test_ds, device),
        "ece": compute_ece(model_star, test_ds, device),
        "elapsed_sec": elapsed_seconds,
    }

    target_auc = forget_set_auc(model_original, forget_ds, device)
    results["relearn_steps"] = relearn_time(model_star, forget_ds, target_auc, device)

    if sensitive_attr is not None:
        probs, labels = get_predictions(model_star, forget_ds, device)
        results["delta_eo"] = equalized_odds_difference(probs, labels, sensitive_attr)
    else:
        results["delta_eo"] = None

    if original_forget_acc is not None:
        results["original_forget_acc"] = original_forget_acc

    if verbose:
        print(
            f"    forget_acc={results['forget_acc']:.4f} retain_auc={results['retain_auc']:.4f} "
            f"test_auc={results['test_auc']:.4f} kl={results['kl_div']:.4f} "
            f"js={results['js_divergence']:.4f} time={elapsed_seconds:.1f}s"
        )

    return results
