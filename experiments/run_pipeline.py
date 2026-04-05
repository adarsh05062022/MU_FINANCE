"""
Full experimental pipeline for LoRA-Credit-Unlearn.

Pipeline steps (from HTML spec):
  1. Data preparation
  2. Train base model M
  3. Run all baselines
  4. Run LoRA method (sweep rank r, forget-set size)
  5. Ablation study
  6. MIA attack experiment
  7. Scalability experiment

Saves all results to results/results_{dataset}_{arch}.json
"""

import os
import sys
import json
import copy
import time
import torch
import numpy as np

# Make sure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import prepare_datasets, make_loader
from models.ft_transformer import FTTransformer
from models.tab_transformer import TabTransformer
from models.lora import count_parameters, merge_lora_into_model
from train import build_model, train_model, save_model, load_model, evaluate
from unlearning.forget_adapter import run_forget_adapter
from unlearning.retain_adapter import run_retain_adapter
from unlearning.baselines import (
    baseline_full_retrain, baseline_gradient_ascent,
    baseline_finetune_retain, baseline_sisa, baseline_influence_functions,
    baseline_random_labels,
)
from evaluation.metrics import (
    full_evaluation, forget_set_accuracy, compute_auc, get_predictions
)
from evaluation.mia import run_mia, loss_based_mia


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DEFAULT_CFG = {
    "dataset": "german",
    "arch": "ft_transformer",
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.1,
    "epochs": 50,
    "batch_size": 128,
    "lr": 1e-3,
    "forget_strategy": "random",
    "forget_frac": 0.10,
    "lora_ranks": [4, 8, 16],
    "forget_fracs": [0.05, 0.10, 0.20],
    "lora_rank_default": 8,
    "fa_steps": 100,
    "fa_lr": 5e-4,
    "lambda_retain": 0.5,
    "retain_auc_min": 0.72,
    "ra_epochs": 25,
    "gamma_forget": 10.0,       # bad-teacher weight in Phase 3 (0 = disabled)
    "max_forget_recovery": 0.15, # Phase 3 ceiling: max allowed forget_auc rise above Phase 2 end
    "n_shadow_mia": 3,
    "run_mia": True,
    "run_baselines": True,
    "run_ablation": True,
    "results_dir": "results",
    "ckpt_dir": "checkpoints",
    "seed": 42,
    "data_dir": "data/raw",
    "verbose": True,
}


# ──────────────────────────────────────────────
# Helper: model factory
# ──────────────────────────────────────────────

def get_model_factory(cfg, num_num_features, cat_dims, device):
    def factory():
        return build_model(
            cfg["arch"], num_num_features, cat_dims, device,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], dropout=cfg["dropout"]
        )
    return factory


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run_pipeline(cfg: dict = None) -> dict:
    if cfg is None:
        cfg = DEFAULT_CFG.copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"LoRA-Credit-Unlearn Pipeline")
    print(f"  dataset={cfg['dataset']} | arch={cfg['arch']} | device={device}")
    print(f"{'='*60}\n")

    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["data_dir"], exist_ok=True)

    all_results = {}

    # ────────────────────────────────────────
    # STEP 1: Data preparation
    # ────────────────────────────────────────
    print("Step 1: Data preparation")
    data = prepare_datasets(
        dataset_name=cfg["dataset"],
        data_dir=cfg["data_dir"],
        forget_strategy=cfg["forget_strategy"],
        forget_frac=cfg["forget_frac"],
        seed=cfg["seed"],
    )
    full_train = data["full_train"]
    val_ds = data["val"]
    test_ds = data["test"]
    forget_ds = data["forget"]
    retain_ds = data["retain"]
    forget_indices = data["forget_indices"]
    cat_dims = data["cat_dims"]
    num_num_features = data["num_num_features"]

    model_factory = get_model_factory(cfg, num_num_features, cat_dims, device)

    # ────────────────────────────────────────
    # STEP 2: Train base model M
    # ────────────────────────────────────────
    print("\nStep 2: Train base model M")
    ckpt_path = os.path.join(cfg["ckpt_dir"],
                             f"base_{cfg['dataset']}_{cfg['arch']}.pt")

    base_model = model_factory()
    if os.path.exists(ckpt_path):
        print(f"  Loading cached checkpoint from {ckpt_path}")
        base_model = load_model(base_model, ckpt_path, device)
    else:
        t0 = time.time()
        base_model, base_hist = train_model(
            base_model, full_train, val_ds, device,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
            verbose=cfg["verbose"],
        )
        base_train_time = time.time() - t0
        save_model(base_model, ckpt_path)
        all_results["base_train_time"] = base_train_time
        all_results["base_history"] = {k: v for k, v in base_hist.items()
                                       if k != "best_state"}

    base_test_auc = compute_auc(base_model, test_ds, device)
    base_forget_acc = forget_set_accuracy(base_model, forget_ds, device)
    from evaluation.metrics import forget_set_auc
    base_forget_auc = forget_set_auc(base_model, forget_ds, device)
    print(f"  Base model | test_auc={base_test_auc:.4f} | forget_auc={base_forget_auc:.4f} | forget_acc={base_forget_acc:.4f}")
    all_results["base_test_auc"] = base_test_auc
    all_results["base_forget_acc"] = base_forget_acc
    all_results["base_forget_auc"] = base_forget_auc
    total_p, _ = count_parameters(base_model)
    all_results["base_total_params"] = total_p

    # Auto-scale fa_steps, fa_lr, and lambda_retain based on forget set size.
    #
    # Small datasets (German, ≤500 forget samples):
    #   - 100 steps, lr=5e-4, lambda=0.5  (conservative — retain utility fragile)
    #
    # Large datasets (GMSC, >500 forget samples):
    #   - 3 full passes over forget set, lr=2e-3 (4x), lambda=0.2
    #   - retain_auc stays at ~0.85 throughout on GMSC, so heavy regularisation
    #     is unnecessary and only slows down forgetting
    steps_per_epoch = max(1, len(forget_ds) // cfg["batch_size"])
    effective_fa_steps = max(cfg["fa_steps"], steps_per_epoch * 3)

    large_dataset = len(forget_ds) > 500
    effective_fa_lr = cfg.get("fa_lr", 5e-4) * (4 if large_dataset else 1)
    effective_lambda = 0.2 if large_dataset else cfg.get("lambda_retain", 0.5)
    # Large datasets need more Phase 3 epochs to converge KL distillation
    effective_ra_epochs = max(cfg["ra_epochs"], 40) if large_dataset else cfg["ra_epochs"]

    changes = []
    if effective_fa_steps != cfg["fa_steps"]:
        changes.append(f"fa_steps {cfg['fa_steps']}→{effective_fa_steps}")
    if large_dataset:
        changes.append(f"fa_lr {cfg.get('fa_lr',5e-4):.0e}→{effective_fa_lr:.0e}")
        changes.append(f"lambda_retain {cfg.get('lambda_retain',0.5)}→{effective_lambda}")
        if effective_ra_epochs != cfg["ra_epochs"]:
            changes.append(f"ra_epochs {cfg['ra_epochs']}→{effective_ra_epochs}")
    if changes:
        print(f"  [Config] Auto-tuned for forget_set={len(forget_ds)}: {', '.join(changes)}")

    cfg = {**cfg, "fa_steps": effective_fa_steps, "fa_lr": effective_fa_lr,
           "lambda_retain": effective_lambda, "ra_epochs": effective_ra_epochs}

    # ────────────────────────────────────────
    # STEP 3: Baselines
    # ────────────────────────────────────────
    baseline_results = {}

    if cfg["run_baselines"]:
        print("\nStep 3: Baselines")

        # 3a. Full Retrain
        print("\n  3a. Full Retrain")
        m_retrain, h_retrain = baseline_full_retrain(
            model_factory, retain_ds, val_ds, device,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=cfg["verbose"]
        )
        r = full_evaluation(m_retrain, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, elapsed_seconds=h_retrain["elapsed"],
                            verbose=True)
        r["elapsed"] = h_retrain["elapsed"]
        baseline_results["full_retrain"] = r

        # 3b. Naïve Gradient Ascent
        print("\n  3b. Naïve Gradient Ascent")
        m_ga, h_ga = baseline_gradient_ascent(
            base_model, forget_ds, retain_ds, val_ds, device,
            max_steps=cfg["fa_steps"] * 2, verbose=cfg["verbose"]
        )
        r = full_evaluation(m_ga, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, elapsed_seconds=h_ga["elapsed"],
                            verbose=True)
        r["elapsed"] = h_ga["elapsed"]
        baseline_results["gradient_ascent"] = r

        # 3c. Fine-tune on D_r
        print("\n  3c. Fine-tune on D_r")
        m_ft, h_ft = baseline_finetune_retain(
            base_model, retain_ds, val_ds, device, verbose=cfg["verbose"]
        )
        r = full_evaluation(m_ft, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, elapsed_seconds=h_ft["elapsed"],
                            verbose=True)
        r["elapsed"] = h_ft["elapsed"]
        baseline_results["finetune_retain"] = r

        # 3d. SISA
        print("\n  3d. SISA")
        m_sisa, h_sisa = baseline_sisa(
            model_factory, full_train, forget_indices, val_ds, device,
            n_shards=4, epochs_per_shard=15, verbose=cfg["verbose"]
        )
        r = full_evaluation(m_sisa, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, elapsed_seconds=h_sisa["elapsed"],
                            verbose=True)
        r["elapsed"] = h_sisa["elapsed"]
        baseline_results["sisa"] = r

        # 3e. Influence Functions
        print("\n  3e. Influence Functions")
        m_if, h_if = baseline_influence_functions(
            base_model, forget_ds, device, verbose=cfg["verbose"]
        )
        r = full_evaluation(m_if, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, elapsed_seconds=h_if["elapsed"],
                            verbose=True)
        r["elapsed"] = h_if["elapsed"]
        baseline_results["influence_fn"] = r

        if cfg["arch"] == "tabddpm":
            print("\n  3f. Random Labels")
            m_rl, h_rl = baseline_random_labels(
                base_model, forget_ds, retain_ds, val_ds, device,
                batch_size=cfg["batch_size"], seed=cfg["seed"], verbose=cfg["verbose"]
            )
            r = full_evaluation(m_rl, base_model, forget_ds, retain_ds, test_ds,
                                device, base_forget_acc, elapsed_seconds=h_rl["elapsed"],
                                verbose=True)
            r["elapsed"] = h_rl["elapsed"]
            baseline_results["random_labels"] = r

    all_results["baselines"] = baseline_results

    # ────────────────────────────────────────
    # STEP 4: LoRA method — sweep rank and forget-set size
    # ────────────────────────────────────────
    print("\nStep 4: LoRA-Credit-Unlearn method")
    lora_results = {}

    for rank in cfg["lora_ranks"]:
        for frac in cfg["forget_fracs"]:
            key = f"r{rank}_f{int(frac*100)}pct"
            print(f"\n  LoRA rank={rank} forget_frac={frac:.0%}")

            # Rebuild forget/retain with this fraction
            data_frac = prepare_datasets(
                dataset_name=cfg["dataset"], data_dir=cfg["data_dir"],
                forget_strategy=cfg["forget_strategy"],
                forget_frac=frac, seed=cfg["seed"],
            )
            f_ds = data_frac["forget"]
            r_ds = data_frac["retain"]

            # Phase 2: Forget adapter
            t_fa_start = time.time()
            model_fa, h_fa = run_forget_adapter(
                base_model, f_ds, r_ds, device,
                lora_r=rank, max_steps=cfg["fa_steps"],
                lr=cfg.get("fa_lr", 5e-4),
                lambda_retain=cfg.get("lambda_retain", 0.5),
                retain_auc_min=cfg.get("retain_auc_min", 0.72),
                batch_size=cfg["batch_size"], verbose=cfg["verbose"]
            )
            fa_time = time.time() - t_fa_start

            # Phase 3: Retain adapter
            t_ra_start = time.time()
            model_star, h_ra = run_retain_adapter(
                model_fa, base_model, r_ds, val_ds, device,
                lora_r=rank, epochs=cfg["ra_epochs"],
                forget_ds=f_ds,
                gamma_forget=cfg.get("gamma_forget", 1.0),
                max_forget_recovery=cfg.get("max_forget_recovery", 0.10),
                batch_size=cfg["batch_size"], verbose=cfg["verbose"]
            )
            ra_time = time.time() - t_ra_start

            total_time = fa_time + ra_time
            r = full_evaluation(
                model_star, base_model, f_ds, r_ds, test_ds,
                device, base_forget_acc, elapsed_seconds=total_time, verbose=True
            )
            r["elapsed"] = total_time
            r["fa_elapsed"] = fa_time
            r["ra_elapsed"] = ra_time
            r["rank"] = rank
            r["forget_frac"] = frac
            lora_results[key] = r

    all_results["lora"] = lora_results

    # ────────────────────────────────────────
    # STEP 5: Ablation study
    # ────────────────────────────────────────
    ablation_results = {}

    if cfg["run_ablation"]:
        print("\nStep 5: Ablation study")
        rank = cfg["lora_rank_default"]

        # 5a. Phase 2 only (no retain adapter)
        print("  5a. Phase 2 only (no retain adapter)")
        m_fa_only, _ = run_forget_adapter(
            base_model, forget_ds, retain_ds, device,
            lora_r=rank, max_steps=cfg["fa_steps"],
            lr=cfg.get("fa_lr", 5e-4),
            lambda_retain=cfg.get("lambda_retain", 0.5),
            retain_auc_min=cfg.get("retain_auc_min", 0.72), verbose=False
        )
        r = full_evaluation(m_fa_only, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, verbose=True)
        ablation_results["phase2_only"] = r

        # 5b. Phase 3 only (no forget adapter — just distill on D_r)
        print("  5b. Phase 3 only (no forget adapter)")
        m_ra_only, _ = run_retain_adapter(
            base_model, base_model, retain_ds, val_ds, device,
            lora_r=rank, epochs=cfg["ra_epochs"], verbose=False
        )
        r = full_evaluation(m_ra_only, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, verbose=True)
        ablation_results["phase3_only"] = r

        # 5c. Full LoRA (both phases, default rank)
        print("  5c. Full LoRA method")
        m_full_fa, _ = run_forget_adapter(
            base_model, forget_ds, retain_ds, device,
            lora_r=rank, max_steps=cfg["fa_steps"],
            lr=cfg.get("fa_lr", 5e-4),
            lambda_retain=cfg.get("lambda_retain", 0.5),
            retain_auc_min=cfg.get("retain_auc_min", 0.72), verbose=False
        )
        m_full, _ = run_retain_adapter(
            m_full_fa, base_model, retain_ds, val_ds, device,
            lora_r=rank, epochs=cfg["ra_epochs"],
            forget_ds=forget_ds,
            gamma_forget=cfg.get("gamma_forget", 1.0),
            max_forget_recovery=cfg.get("max_forget_recovery", 0.10), verbose=False
        )
        r = full_evaluation(m_full, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, verbose=True)
        ablation_results["full_lora"] = r

    all_results["ablation"] = ablation_results

    # ────────────────────────────────────────
    # STEP 6: MIA attack experiment
    # ────────────────────────────────────────
    mia_results = {}

    if cfg["run_mia"]:
        print("\nStep 6: MIA attack experiment")
        rank = cfg["lora_rank_default"]

        # Run LoRA unlearning with default rank
        print("  Running LoRA unlearn for MIA evaluation...")
        m_fa, _ = run_forget_adapter(
            base_model, forget_ds, retain_ds, device,
            lora_r=rank, max_steps=cfg["fa_steps"],
            lr=cfg.get("fa_lr", 5e-4),
            lambda_retain=cfg.get("lambda_retain", 0.5),
            retain_auc_min=cfg.get("retain_auc_min", 0.72), verbose=False
        )
        m_star, _ = run_retain_adapter(
            m_fa, base_model, retain_ds, val_ds, device,
            lora_r=rank, epochs=cfg["ra_epochs"],
            forget_ds=forget_ds,
            gamma_forget=cfg.get("gamma_forget", 1.0),
            max_forget_recovery=cfg.get("max_forget_recovery", 0.10), verbose=False
        )

        # Shadow model MIA
        print("  Shadow model MIA on M* (LoRA unlearn)...")
        mia_lora = run_mia(
            m_star, forget_ds, retain_ds, full_train, device,
            n_shadow=cfg["n_shadow_mia"], verbose=cfg["verbose"]
        )
        mia_results["lora_shadow_mia"] = {
            "mia_score": mia_lora["mia_score"],
            "mia_auc": mia_lora["mia_auc"],
            "retain_mia": mia_lora["retain_mia"],
        }

        # Loss-based MIA (fast)
        print("  Loss-based MIA...")
        mia_loss = loss_based_mia(m_star, forget_ds, retain_ds, device,
                                  verbose=cfg["verbose"])
        mia_results["lora_loss_mia"] = mia_loss

        # MIA on base model (upper bound — should be ~100%)
        print("  MIA on base model (upper bound)...")
        mia_base = loss_based_mia(base_model, forget_ds, retain_ds, device,
                                  verbose=cfg["verbose"])
        mia_results["base_loss_mia"] = mia_base

        # MIA on full retrain (lower bound — should be ~50%)
        if "full_retrain" in baseline_results:
            print("  MIA on full retrain (lower bound)...")
            # full retrain model not stored — skip for now
            pass

    all_results["mia"] = mia_results

    # ────────────────────────────────────────
    # Save results
    # ────────────────────────────────────────
    results_path = os.path.join(
        cfg["results_dir"],
        f"results_{cfg['dataset']}_{cfg['arch']}.json"
    )
    with open(results_path, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Print summary table
    _print_summary(all_results, cfg)

    return all_results


def _jsonify(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)


def _print_summary(results: dict, cfg: dict):
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY — {cfg['dataset']} / {cfg['arch']}")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'ForgetAUC':>10} {'ForgetAcc':>10} {'RetainAUC':>10} {'TestAUC':>10} {'KL':>8} {'Time(s)':>9}")
    print("-" * 80)

    # Base model
    if "base_forget_acc" in results:
        print(f"{'Base Model':<25} {results.get('base_forget_auc', float('nan')):>10.4f} "
              f"{results['base_forget_acc']:>10.4f} "
              f"{'N/A':>10} {results.get('base_test_auc', 0):>10.4f} {'N/A':>8} {'N/A':>9}")

    for method, r in results.get("baselines", {}).items():
        print(f"{method:<25} {r.get('forget_auc', float('nan')):>10.4f} "
              f"{r.get('forget_acc', 0):>10.4f} "
              f"{r.get('retain_auc', 0):>10.4f} {r.get('test_auc', 0):>10.4f} "
              f"{r.get('kl_div', 0):>8.4f} {r.get('elapsed', 0):>9.1f}")

    for key, r in results.get("lora", {}).items():
        label = f"LoRA-{key}"
        print(f"{label:<25} {r.get('forget_auc', float('nan')):>10.4f} "
              f"{r.get('forget_acc', 0):>10.4f} "
              f"{r.get('retain_auc', 0):>10.4f} {r.get('test_auc', 0):>10.4f} "
              f"{r.get('kl_div', 0):>8.4f} {r.get('elapsed', 0):>9.1f}")

    print("=" * 80)

    # MIA summary
    if results.get("mia"):
        print("\nMIA Results (goal: ≈0.5)")
        for key, r in results["mia"].items():
            score = r.get("mia_score", r.get("mia_score", "N/A"))
            print(f"  {key}: MIA score = {score:.4f}")
