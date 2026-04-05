"""
Method 5: Influence Functions.

Paper: Koh & Liang, "Understanding Black-Box Predictions via Influence
       Functions" (ICML 2017)

Influence functions analytically compute how removing a training point would
change model parameters, without retraining. This is a first-order approximation:

    theta* = theta - H^{-1} * grad_L(D_forget)

where H is the Hessian of the retain loss.

For large models, uses LiSSA (Linear time Stochastic Second-order Algorithm)
to approximate the inverse Hessian-vector product.

NOTE: Influence functions are SLOW on large models (FT-Transformer).
For German Credit (1000 samples) it is feasible.
For GMSC (150K), use a random subsample of D_retain for Hessian estimation.
"""

import copy
import time
import torch
import torch.nn as nn
from itertools import cycle

from data.datasets import CreditDataset, make_loader


def compute_gradient(model, loader, criterion, device):
    """Compute flat gradient vector of loss over the entire loader."""
    model.zero_grad()
    n_batches = 0
    for x_num, x_cat, y in loader:
        x_num = x_num.to(device) if x_num.numel() > 0 else None
        x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
        y = y.to(device)

        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        loss.backward()
        n_batches += 1

    # Average gradients
    grad_parts = []
    for p in model.parameters():
        if p.grad is not None:
            grad_parts.append((p.grad / n_batches).flatten())
        else:
            grad_parts.append(torch.zeros(p.numel(), device=device))

    return torch.cat(grad_parts)


def hessian_vector_product(model, x_num, x_cat, y, v, criterion, device):
    """Efficient HVP computation via double backprop."""
    model.zero_grad()
    logits = model(x_num, x_cat)
    loss = criterion(logits, y)

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    flat_grads = torch.cat([g.flatten() for g in grads])

    # Grad of (grads dot v) w.r.t. parameters = HVP
    grad_v = (flat_grads * v.to(device)).sum()
    hvp = torch.autograd.grad(grad_v, model.parameters(), retain_graph=False)
    return torch.cat([g.flatten() for g in hvp])


def unlearn(model, D_forget, D_retain, D_val, config, device):
    """
    Approximate unlearning via Newton step:
    theta* = theta - H^{-1} * grad_L(D_forget)

    config keys:
        damping: float       (Hessian damping, default 0.01)
        scale: float         (scaling for H^{-1}v computation, default 25.0)
        recursion_depth: int (LiSSA iterations, default 500)
        batch_size: int      (default 256)
        max_retain_samples: int (subsample retain for Hessian, default 5000)
        max_h_estimate_norm: float (fallback if LiSSA explodes, default 1e4)
        max_update_norm: float (clip final parameter-update norm, default 5.0)
    """
    model = copy.deepcopy(model).to(device)
    criterion = nn.BCEWithLogitsLoss()

    damping = config.get("damping", 0.01)
    scale = config.get("scale", 25.0)
    recursion_depth = config.get("recursion_depth", 500)
    batch_size = config.get("batch_size", 256)
    max_retain = config.get("max_retain_samples", 5000)
    max_h_estimate_norm = config.get("max_h_estimate_norm", 1e4)
    max_update_norm = config.get("max_update_norm", 5.0)
    verbose = config.get("verbose", True)

    # Step 1: Compute gradient of forget loss
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    forget_loader = make_loader(D_forget, batch_size=batch_size, shuffle=False)
    forget_grad = compute_gradient(model, forget_loader, criterion, device)

    if verbose:
        print(f"    Forget gradient norm: {forget_grad.norm().item():.4f}")

    # Step 2: Compute H^{-1} * forget_grad using LiSSA
    # Subsample retain set if too large
    if D_retain is not None and len(D_retain) > max_retain:
        import numpy as np
        from data.datasets import subset_dataset
        sub_idx = np.random.choice(len(D_retain), max_retain, replace=False)
        retain_for_hessian = subset_dataset(D_retain, sub_idx)
    else:
        retain_for_hessian = D_retain

    use_identity_fallback = retain_for_hessian is None

    if retain_for_hessian is not None:
        retain_loader = make_loader(retain_for_hessian, batch_size=batch_size, shuffle=True)
        retain_iter = cycle(iter(retain_loader))

        v = forget_grad.clone()
        h_estimate = v.clone()

        for i in range(recursion_depth):
            x_num, x_cat, y = next(retain_iter)
            x_num = x_num.to(device) if x_num.numel() > 0 else None
            x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
            y = y.to(device)

            try:
                hvp = hessian_vector_product(model, x_num, x_cat, y, h_estimate, criterion, device)
                next_h = v + (1 - damping) * h_estimate - hvp / scale
            except RuntimeError:
                # HVP can fail for certain architectures; fall back to identity approx
                if verbose and i == 0:
                    print("    [Warn] HVP failed, using identity Hessian approximation")
                use_identity_fallback = True
                break

            if not torch.isfinite(next_h).all():
                if verbose:
                    print("    [Warn] LiSSA produced non-finite values; using identity Hessian approximation")
                use_identity_fallback = True
                break

            next_norm = next_h.norm().item()
            if next_norm > max_h_estimate_norm:
                if verbose:
                    print(f"    [Warn] LiSSA norm exploded to {next_norm:.4f}; using identity Hessian approximation")
                use_identity_fallback = True
                break

            h_estimate = next_h

            if verbose and (i + 1) % 100 == 0:
                print(f"    LiSSA iteration {i+1}/{recursion_depth} | h_estimate norm={h_estimate.norm().item():.4f}")

    if use_identity_fallback:
        h_estimate = forget_grad / max(damping, 1e-8)

    # Step 3: Update parameters
    # theta* = theta - (1/|D_retain|) * H^{-1} * grad_L(D_forget)
    n_retain = len(D_retain) if D_retain is not None else 1
    update_vector = h_estimate / max(n_retain, 1)
    if not torch.isfinite(update_vector).all():
        if verbose:
            print("    [Warn] Non-finite update detected; sanitizing before applying")
        update_vector = torch.nan_to_num(update_vector, nan=0.0, posinf=0.0, neginf=0.0)

    update_norm = update_vector.norm().item()
    if update_norm > max_update_norm:
        scale_factor = max_update_norm / max(update_norm, 1e-12)
        update_vector = update_vector * scale_factor
        if verbose:
            print(f"    [Info] Clipped influence-function update norm {update_norm:.4f} -> {max_update_norm:.4f}")

    with torch.no_grad():
        offset = 0
        for param in model.parameters():
            param_size = param.numel()
            update = update_vector[offset:offset + param_size].reshape(param.shape)
            param.data -= update
            offset += param_size

    return model


def influence_fn_unlearn(model, D_forget, D_retain, config, device):
    """Convenience wrapper. Returns (model, elapsed_seconds)."""
    start = time.perf_counter()
    result = unlearn(model, D_forget, D_retain, None, config, device)
    elapsed = time.perf_counter() - start
    return result, elapsed
