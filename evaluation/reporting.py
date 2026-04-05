"""
Helpers for saving experiment reports as images inside per-run folders.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt(value, digits: int = 3) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.{digits}f}"


def _signal_label(row: pd.Series) -> str:
    mia = _to_float(row.get("mia_score"))
    forget_acc = _to_float(row.get("forget_accuracy"))
    retain_auc = _to_float(row.get("retain_auc"))
    test_auc = _to_float(row.get("test_auc"))

    mia_good = mia is not None and abs(mia - 0.5) <= 0.12
    forget_good = forget_acc is not None and forget_acc <= 0.72
    utility_good = (
        retain_auc is not None and retain_auc >= 0.75 and
        test_auc is not None and test_auc >= 0.72
    )

    if mia_good and forget_good and utility_good:
        return "Strong"
    if (mia_good and utility_good) or (forget_good and utility_good):
        return "Partial"
    return "Weak"


def _signal_color(label: str) -> str:
    return {
        "Strong": "#1f7a45",
        "Partial": "#9a6700",
        "Weak": "#b42318",
    }.get(label, "#6b7280")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def create_run_report_dir(base_dir: str, run_name: str) -> str:
    return _ensure_dir(os.path.join(base_dir, run_name))


def save_dataframe_csv(df: pd.DataFrame, run_dir: str, filename: str = "results.csv") -> str:
    path = os.path.join(run_dir, filename)
    df.to_csv(path, index=False)
    return path


def save_table_image(
    df: pd.DataFrame,
    output_path: str,
    title: str,
    subtitle: str = "",
    columns: Optional[Iterable[str]] = None,
) -> str:
    table_df = df.copy()
    table_df.insert(1, "unlearning_signal", table_df.apply(_signal_label, axis=1))

    if columns is not None:
        keep = [col for col in columns if col in table_df.columns]
        table_df = table_df[keep]

    display_df = table_df.copy()
    for column in display_df.columns:
        if column not in {"method", "unlearning_signal"}:
            display_df[column] = display_df[column].map(lambda x: _fmt(x, digits=3))

    nrows, ncols = display_df.shape
    fig_w = max(12, ncols * 1.6)
    fig_h = max(3.8, 1.1 + nrows * 0.62)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("#f7f2e8")

    title_text = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(title_text, fontsize=18, fontweight="bold", loc="left", pad=18)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e9dcc4")
            cell.set_text_props(weight="bold", color="#1d1b16")
            cell.set_edgecolor("#d3c3a5")
            continue

        cell.set_edgecolor("#e5d8c1")
        if col == 1:
            label = display_df.iloc[row - 1, col]
            cell.set_facecolor(_signal_color(label))
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_facecolor("#fffaf0")
            cell.set_text_props(weight="bold", color="#1d1b16")
        else:
            cell.set_facecolor("#fffdf8")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metric_bars(df: pd.DataFrame, output_path: str, title: str) -> str:
    plot_df = df.copy()
    plot_df["unlearning_signal"] = plot_df.apply(_signal_label, axis=1)
    metrics = ["mia_score", "forget_accuracy", "retain_auc", "test_auc"]
    present = [m for m in metrics if m in plot_df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    fig.patch.set_facecolor("#f7f2e8")

    for ax, metric in zip(axes, present):
        values = pd.to_numeric(plot_df[metric], errors="coerce")
        colors = [_signal_color(x) for x in plot_df["unlearning_signal"]]
        ax.bar(plot_df["method"], values, color=colors, edgecolor="#4a3d2a", linewidth=0.6)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=25)
        if metric in {"mia_score", "forget_accuracy", "retain_auc", "test_auc"}:
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

    for ax in axes[len(present):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_tradeoff_scatter(df: pd.DataFrame, output_path: str, title: str) -> str:
    plot_df = df.copy()
    plot_df["unlearning_signal"] = plot_df.apply(_signal_label, axis=1)
    x = pd.to_numeric(plot_df.get("retain_auc"), errors="coerce")
    y = pd.to_numeric(plot_df.get("mia_score"), errors="coerce")
    sizes = pd.to_numeric(plot_df.get("test_auc"), errors="coerce").fillna(0.5) * 700

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#f7f2e8")
    ax.set_facecolor("#fffdf8")

    colors = [_signal_color(x) for x in plot_df["unlearning_signal"]]
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.82, edgecolors="#4a3d2a", linewidths=0.8)

    for _, row in plot_df.iterrows():
        xv = _to_float(row.get("retain_auc"))
        yv = _to_float(row.get("mia_score"))
        if xv is None or yv is None:
            continue
        ax.annotate(row["method"], (xv, yv), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.axhline(0.5, color="#9a6700", linestyle="--", linewidth=1.2, label="Ideal MIA ≈ 0.5")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Retain AUC")
    ax.set_ylabel("MIA Score")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_efficiency_plot(df: pd.DataFrame, output_path: str, title: str) -> str:
    plot_df = df.copy()
    plot_df["unlearning_signal"] = plot_df.apply(_signal_label, axis=1)
    wall_clock = pd.to_numeric(plot_df.get("wall_clock_seconds"), errors="coerce")
    speedup = pd.to_numeric(plot_df.get("speedup_vs_retrain"), errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#f7f2e8")

    colors = [_signal_color(x) for x in plot_df["unlearning_signal"]]

    axes[0].bar(plot_df["method"], wall_clock, color=colors, edgecolor="#4a3d2a", linewidth=0.6)
    axes[0].set_title("Wall Clock Seconds")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_axisbelow(True)

    axes[1].bar(plot_df["method"], speedup, color=colors, edgecolor="#4a3d2a", linewidth=0.6)
    axes[1].set_title("Speedup Vs Full Retrain")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    axes[1].set_axisbelow(True)

    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_baseline_report_bundle(
    df: pd.DataFrame,
    run_dir: str,
    title: str,
    subtitle: str = "",
) -> dict:
    _ensure_dir(run_dir)
    paths = {}
    paths["csv"] = save_dataframe_csv(df, run_dir, "results.csv")
    paths["table_png"] = save_table_image(
        df,
        os.path.join(run_dir, "01_results_table.png"),
        title=title,
        subtitle=subtitle,
        columns=[
            "method",
            "unlearning_signal",
            "mia_score",
            "forget_accuracy",
            "retain_auc",
            "test_auc",
            "wall_clock_seconds",
            "speedup_vs_retrain",
        ],
    )
    paths["overview_png"] = save_metric_bars(
        df,
        os.path.join(run_dir, "02_metric_overview.png"),
        title=f"{title} - Metric Overview",
    )
    paths["tradeoff_png"] = save_tradeoff_scatter(
        df,
        os.path.join(run_dir, "03_unlearning_tradeoff.png"),
        title=f"{title} - Retain vs MIA",
    )
    paths["efficiency_png"] = save_efficiency_plot(
        df,
        os.path.join(run_dir, "04_efficiency.png"),
        title=f"{title} - Runtime",
    )
    return paths
