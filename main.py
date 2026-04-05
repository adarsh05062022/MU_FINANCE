"""
LoRA-Credit-Unlearn: Parameter-Efficient Machine Unlearning for Transformer-Based Credit Scoring.

Entry point. Run the full pipeline with:
  python main.py
  python main.py --dataset german --arch ft_transformer --mode full
  python main.py --dataset german --arch tabddpm --mode quick
  python main.py --mode quick   # fast smoke test

Modes:
  full     — complete pipeline (train + all baselines + LoRA sweep + MIA + ablation)
  quick    — single run with default config, no MIA, no ablation (fast smoke test)
  ablation — ablation study only (requires base model checkpoint)
  scalability — scalability experiment across datasets
"""

import argparse
import json
import os
import sys
import torch

from experiments.run_pipeline import run_pipeline, DEFAULT_CFG


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA-Credit-Unlearn")
    parser.add_argument("--dataset", default="german",
                        choices=["german", "gmsc"],
                        help="Dataset to use")
    parser.add_argument("--arch", default="ft_transformer",
                        choices=["ft_transformer", "tab_transformer", "tabddpm"],
                        help="Model architecture")
    parser.add_argument("--mode", default="quick",
                        choices=["full", "quick", "ablation", "scalability"],
                        help="Experiment mode")
    parser.add_argument("--forget_strategy", default="random",
                        choices=["random", "demographic"],
                        help="Forget set construction strategy")
    parser.add_argument("--forget_frac", type=float, default=0.10,
                        help="Fraction of training data to forget (for random strategy)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Default LoRA rank")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for base model")
    parser.add_argument("--data_dir", default="data/raw",
                        help="Directory with raw dataset files")
    parser.add_argument("--results_dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip baselines (faster)")
    parser.add_argument("--no_mia", action="store_true",
                        help="Skip MIA (faster)")
    parser.add_argument("--no_ablation", action="store_true",
                        help="Skip ablation (faster)")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = DEFAULT_CFG.copy()
    cfg.update({
        "dataset": args.dataset,
        "arch": args.arch,
        "forget_strategy": args.forget_strategy,
        "forget_frac": args.forget_frac,
        "lora_rank_default": args.lora_rank,
        "epochs": args.epochs,
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "seed": args.seed,
    })

    if args.mode == "quick":
        # Fast smoke test: minimal config
        cfg.update({
            "epochs": 20,
            "fa_steps": 20,
            "ra_epochs": 10,
            "lora_ranks": [8],
            "forget_fracs": [0.10],
            "n_shadow_mia": 2,
            "run_baselines": not args.no_baselines,
            "run_mia": not args.no_mia,
            "run_ablation": not args.no_ablation,
        })
        if args.no_baselines:
            cfg["run_baselines"] = False
        if args.no_mia:
            cfg["run_mia"] = False
        if args.no_ablation:
            cfg["run_ablation"] = False

    elif args.mode == "full":
        cfg.update({
            "run_baselines": True,
            "run_mia": True,
            "run_ablation": True,
            "n_shadow_mia": 4,
        })

    elif args.mode == "scalability":
        from experiments.scalability import run_scalability_experiment
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = run_scalability_experiment(cfg, device)
        out_path = os.path.join(cfg["results_dir"], "scalability_results.json")
        os.makedirs(cfg["results_dir"], exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nScalability results saved to {out_path}")
        return

    # Run main pipeline
    results = run_pipeline(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
