"""
Master baseline runner for credit scoring unlearning experiments.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import prepare_all
from evaluation.fairness import build_age_groups, compute_delta_eo
from evaluation.mia import run_mia
from evaluation.metrics import (
    compute_ece,
    compute_forget_accuracy,
    compute_forget_confidence,
    compute_js_divergence,
    compute_kl_divergence,
    compute_relearn_time,
    compute_retain_auc,
    compute_test_auc,
    count_updated_params,
)
from evaluation.reporting import create_run_report_dir, save_baseline_report_bundle
from train import build_model, load_model, save_model, train_model
from unlearning.finetune_retain import unlearn as finetune_retain_unlearn
from unlearning.full_retrain import unlearn as full_retrain_unlearn
from unlearning.gradient_ascent import unlearn as gradient_ascent_unlearn
from unlearning.gradient_diff import unlearn as gradient_diff_unlearn
from unlearning.influence_functions import unlearn as influence_unlearn
from unlearning.random_labels import unlearn as random_labels_unlearn
from unlearning.scrub import unlearn as scrub_unlearn
from unlearning.sisa import unlearn as sisa_unlearn


DEFAULT_CONFIG = {
    "dataset": "german",
    "arch": "ft_transformer",
    "forget_strategy": "random",
    "forget_frac": 0.10,
    "seed": 42,
    "batch_size": 128,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 10,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.1,
    "n_shadow": 4,
    "ga": {
        "ga_lr": 1e-5,
        "max_steps": 100,
        "grad_clip": 1.0,
        "retain_auc_threshold": 0.68,
        "batch_size": 64,
    },
    "graddiff": {
        "alpha_sweep": [0.1, 0.5, 1.0, 2.0],
        "lr": 1e-5,
        "max_steps": 200,
        "grad_clip": 1.0,
        "retain_auc_threshold": 0.70,
        "batch_size": 64,
    },
    "finetune_retain": {
        "lr": 1e-5,
        "epochs": 5,
        "batch_size": 128,
        "patience": 3,
    },
    "sisa": {
        "n_shards": 10,
        "epochs_per_shard": 15,
        "batch_size": 128,
        "lr": 1e-3,
    },
    "influence_fn": {
        "damping": 0.01,
        "scale": 25.0,
        "recursion_depth": 500,
        "batch_size": 128,
        "max_retain_samples": 5000,
    },
    "scrub": {
        "lr": 1e-5,
        "alpha": 1.0,
        "max_steps": 200,
        "batch_size": 64,
        "retain_auc_threshold": 0.68,
    },
    "random_labels": {
        "lr": 5e-5,
        "epochs": 5,
        "batch_size": 128,
        "patience": 3,
        "forget_multiplier": 3,
        "force_mismatch": True,
    },
}


ALL_METHODS = [
    "full_retrain",
    "gradient_diff",
    "gradient_ascent",
    "finetune_retain",
    "influence_fn",
    "scrub",
    "random_labels",
    "sisa",
]


def _load_config(path):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = _deep_update(cfg, user_cfg)
    return cfg


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _normalize_method_selection(methods):
    if methods is None:
        return None
    if isinstance(methods, str):
        raw = methods.split(",")
    else:
        raw = methods
    cleaned = [m.strip() for m in raw if m and m.strip()]
    if not cleaned or "all" in cleaned:
        return None
    unknown = sorted(set(cleaned) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}. Valid methods: {ALL_METHODS}")
    return set(cleaned)


def _model_factory(cfg, num_num_features, cat_dims, device):
    def factory():
        return build_model(
            cfg["arch"],
            num_num_features,
            cat_dims,
            device,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
        )
    return factory


def _train_or_load_base_model(cfg, model_factory, full_train, val_ds, device):
    ckpt_dir = os.path.join("models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = f"{cfg['arch']}_{cfg['dataset']}_{cfg['forget_strategy']}_base.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    model = model_factory()
    if os.path.exists(ckpt_path):
        return load_model(model, ckpt_path, device), ckpt_path

    model, _ = train_model(
        model,
        full_train,
        val_ds,
        device,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        verbose=True,
    )
    save_model(model, ckpt_path)
    return model, ckpt_path


def _save_unlearned_model(model, method_name, cfg):
    ckpt_dir = os.path.join("models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(
        ckpt_dir,
        f"{method_name}_{cfg['dataset']}_{cfg['forget_strategy']}_{cfg['arch']}.pt",
    )
    torch.save(model.state_dict(), path)
    return path


def _evaluate_method(
    method_name,
    unlearned_model,
    base_model,
    retrained_model,
    elapsed,
    retrain_time,
    forget_ds,
    retain_ds,
    test_ds,
    full_train,
    fairness_groups,
    cfg,
    device,
):
    config_for_relearn = {"original_forget_acc": compute_forget_accuracy(base_model, forget_ds, device)}
    changed, total, pct = count_updated_params(base_model, unlearned_model)
    mia = run_mia(
        unlearned_model,
        forget_ds,
        retain_ds,
        full_train,
        device,
        n_shadow=cfg["n_shadow"],
        verbose=False,
    )
    if retrained_model is not None:
        js_divergence = compute_js_divergence(unlearned_model, retrained_model, test_ds, device)
        kl_divergence = compute_kl_divergence(unlearned_model, retrained_model, test_ds, device)
    else:
        js_divergence = np.nan
        kl_divergence = np.nan

    return {
        "method": method_name,
        "mia_score": mia["mia_score"],
        "forget_accuracy": compute_forget_accuracy(unlearned_model, forget_ds, device),
        "forget_confidence": compute_forget_confidence(unlearned_model, forget_ds, device),
        "js_divergence": js_divergence,
        "kl_divergence": kl_divergence,
        "relearn_steps": compute_relearn_time(unlearned_model, forget_ds, None, config_for_relearn, device),
        "retain_auc": compute_retain_auc(unlearned_model, retain_ds, device),
        "test_auc": compute_test_auc(unlearned_model, test_ds, device),
        "wall_clock_seconds": elapsed,
        "speedup_vs_retrain": (retrain_time / max(elapsed, 1e-8)) if retrain_time is not None else np.nan,
        "changed_params": changed,
        "total_params": total,
        "pct_params_changed": pct,
        "delta_eo": compute_delta_eo(unlearned_model, test_ds, fairness_groups, device),
        "ece": compute_ece(unlearned_model, test_ds, device),
    }


def run_all_methods(cfg):
    _set_seed(cfg["seed"])
    device = _get_device()
    selected_methods = _normalize_method_selection(cfg.get("methods"))
    if selected_methods and "random_labels" in selected_methods and cfg["arch"] != "tabddpm":
        raise ValueError("random_labels is only enabled for arch='tabddpm'")
    data = prepare_all(
        dataset_name=cfg["dataset"],
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
    model_factory = _model_factory(cfg, num_num_features, cat_dims, device)
    base_model, _ = _train_or_load_base_model(cfg, model_factory, full_train, val_ds, device)
    fairness_groups = build_age_groups(cfg["dataset"], test_ds)

    results = []

    retrained_model = None
    retrain_elapsed = None
    if selected_methods is None or "full_retrain" in selected_methods:
        retrain_cfg = {
            "model_factory": model_factory,
            "lr": cfg["lr"],
            "wd": cfg["weight_decay"],
            "max_epochs": cfg["epochs"],
            "patience": cfg["patience"],
            "batch_size": cfg["batch_size"],
        }
        start = time.perf_counter()
        retrained_model = full_retrain_unlearn(
            None, forget_ds, retain_ds, val_ds, retrain_cfg, device
        )
        retrain_elapsed = time.perf_counter() - start
        _save_unlearned_model(retrained_model, "full_retrain", cfg)
        results.append(
            _evaluate_method(
                "full_retrain",
                retrained_model,
                base_model,
                retrained_model,
                retrain_elapsed,
                retrain_elapsed,
                forget_ds,
                retain_ds,
                test_ds,
                full_train,
                fairness_groups,
                cfg,
                device,
            )
        )

    method_specs = []
    if selected_methods is None or "gradient_ascent" in selected_methods:
        method_specs.append(("gradient_ascent", gradient_ascent_unlearn, cfg["ga"]))
    if selected_methods is None or "finetune_retain" in selected_methods:
        method_specs.append(("finetune_retain", finetune_retain_unlearn, cfg["finetune_retain"]))
    if selected_methods is None or "influence_fn" in selected_methods:
        method_specs.append(("influence_fn", influence_unlearn, cfg["influence_fn"]))
    if selected_methods is None or "scrub" in selected_methods:
        method_specs.append(("scrub", scrub_unlearn, cfg["scrub"]))
    if cfg["arch"] == "tabddpm":
        rl_cfg = dict(cfg["random_labels"])
        rl_cfg["seed"] = cfg["seed"]
        if selected_methods is None or "random_labels" in selected_methods:
            method_specs.append(("random_labels", random_labels_unlearn, rl_cfg))

    if selected_methods is None or "sisa" in selected_methods:
        sisa_cfg = dict(cfg["sisa"])
        sisa_cfg["model_factory"] = model_factory
        sisa_cfg["full_train_ds"] = full_train
        sisa_cfg["forget_indices"] = forget_indices
        method_specs.append(("sisa", sisa_unlearn, sisa_cfg))

    if selected_methods is None or "gradient_diff" in selected_methods:
        best_graddiff = None
        best_graddiff_auc = -1.0
        for alpha in cfg["graddiff"]["alpha_sweep"]:
            gd_cfg = dict(cfg["graddiff"])
            gd_cfg["alpha"] = alpha
            start = time.perf_counter()
            candidate = gradient_diff_unlearn(base_model, forget_ds, retain_ds, val_ds, gd_cfg, device)
            elapsed = time.perf_counter() - start
            candidate_auc = compute_retain_auc(candidate, retain_ds, device)
            if candidate_auc > best_graddiff_auc:
                best_graddiff_auc = candidate_auc
                best_graddiff = (alpha, candidate, elapsed)

        if best_graddiff is not None:
            alpha, gd_model, gd_elapsed = best_graddiff
            gd_name = f"gradient_diff_alpha_{alpha}"
            _save_unlearned_model(gd_model, gd_name, cfg)
            results.append(
                _evaluate_method(
                    gd_name,
                    gd_model,
                    base_model,
                    retrained_model,
                    gd_elapsed,
                    retrain_elapsed,
                    forget_ds,
                    retain_ds,
                    test_ds,
                    full_train,
                    fairness_groups,
                    cfg,
                    device,
                )
            )

    for method_name, method_fn, method_cfg in method_specs:
        start = time.perf_counter()
        unlearned_model = method_fn(base_model, forget_ds, retain_ds, val_ds, method_cfg, device)
        elapsed = time.perf_counter() - start
        _save_unlearned_model(unlearned_model, method_name, cfg)
        results.append(
            _evaluate_method(
                method_name,
                unlearned_model,
                base_model,
                retrained_model,
                elapsed,
                retrain_elapsed,
                forget_ds,
                retain_ds,
                test_ds,
                full_train,
                fairness_groups,
                cfg,
                device,
            )
        )

    if not results:
        raise RuntimeError("No results were produced. Check the requested methods and architecture.")

    df = pd.DataFrame(results)
    os.makedirs(os.path.join("results", "runs"), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = f"baselines_{cfg['dataset']}_{cfg['forget_strategy']}_{cfg['arch']}_{timestamp}"
    run_dir = create_run_report_dir(
        os.path.join("results", "runs"),
        run_name,
    )
    report_paths = save_baseline_report_bundle(
        df,
        run_dir,
        title=f"Baseline Results: {cfg['dataset']} / {cfg['arch']}",
        subtitle=(
            f"forget_strategy={cfg['forget_strategy']} | "
            f"methods={','.join(sorted(selected_methods)) if selected_methods else 'all'}"
        ),
    )
    print(df)
    print(f"\nSaved run folder to {run_dir}")
    print(f"Saved CSV to {report_paths['csv']}")
    print(f"Saved table image to {report_paths['table_png']}")
    print(f"Saved overview plot to {report_paths['overview_png']}")
    print(f"Saved tradeoff plot to {report_paths['tradeoff_png']}")
    print(f"Saved efficiency plot to {report_paths['efficiency_png']}")
    return df, run_dir


def main():
    parser = argparse.ArgumentParser(description="Run unlearning baselines")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated subset of methods to run: full_retrain,gradient_diff,gradient_ascent,finetune_retain,influence_fn,scrub,random_labels,sisa",
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)
    if args.methods:
        cfg["methods"] = args.methods
    run_all_methods(cfg)


if __name__ == "__main__":
    main()
