#!/usr/bin/env python3
"""Plot ablation curves from WandB for CDC 2026 paper.

Usage:
    python plot_ablation.py --project HGTeam_CDC_Ablation --groups physical_graph no_gnn
    python plot_ablation.py --project HGTeam_CDC_Ablation --groups physical_graph no_gnn --metric eval/reward/EV_mean
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
})

LABELS = {
    "physical_graph": "HGTeam (Physics-Informed GNN)",
    "no_gnn": "HGTeam (No GNN)",
}
COLORS = {
    "physical_graph": "#1f77b4",
    "no_gnn": "#d62728",
}


def fetch_group_runs(project, group):
    """Fetch all runs in a WandB group, return list of (steps, values) arrays."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project, filters={"group": group, "state": "finished"})
    series = []
    for run in runs:
        hist = run.history(samples=5000)
        if hist.empty:
            continue
        series.append(hist)
    return series


def align_and_aggregate(series_list, metric, x_key="_step"):
    """Align runs to common x-axis, return (x, mean, std)."""
    # Find common x range
    all_x = set()
    cleaned = []
    for df in series_list:
        if metric not in df.columns or x_key not in df.columns:
            continue
        sub = df[[x_key, metric]].dropna()
        if sub.empty:
            continue
        cleaned.append(sub)
        all_x.update(sub[x_key].values)

    if not cleaned:
        return None, None, None

    x_common = np.array(sorted(all_x))
    interped = []
    for sub in cleaned:
        vals = np.interp(x_common, sub[x_key].values, sub[metric].values)
        interped.append(vals)

    mat = np.stack(interped)
    return x_common, mat.mean(axis=0), mat.std(axis=0)


def plot_ablation(project, groups, metric="eval/reward/episode_reward_mean",
                  xlabel="Environment Frames", ylabel="Evaluation Return",
                  title="GNN Ablation", output="ablation_gnn.pdf",
                  smooth=5):
    fig, ax = plt.subplots()

    for group in groups:
        print(f"Fetching group '{group}' from project '{project}'...")
        series = fetch_group_runs(project, group)
        if not series:
            print(f"  WARNING: no finished runs found for group '{group}'")
            continue
        print(f"  Found {len(series)} runs")

        x, mean, std = align_and_aggregate(series, metric)
        if x is None:
            print(f"  WARNING: metric '{metric}' not found in runs")
            continue

        # Smooth with rolling average
        if smooth > 1:
            mean = pd.Series(mean).rolling(smooth, min_periods=1).mean().values
            std = pd.Series(std).rolling(smooth, min_periods=1).mean().values

        label = LABELS.get(group, group)
        color = COLORS.get(group, None)
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--metric", default="eval/reward/episode_reward_mean")
    parser.add_argument("--xlabel", default="Environment Frames")
    parser.add_argument("--ylabel", default="Evaluation Return")
    parser.add_argument("--title", default="GNN Ablation")
    parser.add_argument("--output", default="ablation_gnn.pdf")
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()

    plot_ablation(args.project, args.groups, args.metric,
                  args.xlabel, args.ylabel, args.title, args.output, args.smooth)
