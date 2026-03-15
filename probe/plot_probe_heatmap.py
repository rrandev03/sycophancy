"""
Plot R² heatmap per (layer, head) for assertiveness probes. Optionally log to wandb.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.path.insert(0, current_dir)

import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def plot_r2_heatmap(r2_dict, num_layers, num_heads, output_path=None, title_suffix=""):
    """Plot heatmap of R² per (layer, head)."""
    matrix = np.zeros((num_layers, num_heads))
    for key, val in r2_dict.items():
        layer, head = map(int, key.split("_"))
        matrix[layer, head] = val

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=0, vmax=max(matrix.max(), 0.01))    
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"R² per (layer, head) - Assertiveness Probe {title_suffix}".strip())
    plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot R² heatmap for assertiveness probes.")
    parser.add_argument("--model_id", type=str, default="gemma-3")
    parser.add_argument("--probe_dir", type=str, default=None, help="Override probe dir (default: probe/trained_probe_assertiveness/{model_id})")
    parser.add_argument("--metric", type=str, default="r2", choices=["r2", "test_r2"], help="Which R² to plot (r2=train, test_r2=held-out)")
    parser.add_argument("--output", type=str, default=None, help="Output path for heatmap (default: probe_dir/r2_heatmap.png)")
    parser.add_argument("--wandb", action="store_true", help="Log heatmap to wandb")

    args = parser.parse_args()
    model_name = args.model_id.split("/")[-1]
    probe_dir = args.probe_dir or f"probe/trained_probe_assertiveness/{model_name}"
    r2_file = "test_r2_dict.pkl" if args.metric == "test_r2" else "r2_dict.pkl"
    r2_path = os.path.join(probe_dir, r2_file)

    if not os.path.exists(r2_path):
        raise FileNotFoundError(f"Run train_epint first. Missing: {r2_path}")

    with open(r2_path, "rb") as f:
        r2_dict = pickle.load(f)

    keys = list(r2_dict.keys())
    layers = [int(k.split("_")[0]) for k in keys]
    heads = [int(k.split("_")[1]) for k in keys]
    num_layers = max(layers) + 1
    num_heads = max(heads) + 1

    suffix = "test_r2" if args.metric == "test_r2" else "r2"
    output_path = args.output or os.path.join(probe_dir, f"{suffix}_heatmap.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if args.wandb:
        wandb.init(
            project="assertiveness-probes",
            config={"model_id": args.model_id, "probe_dir": probe_dir},
            name=f"heatmap_{model_name}",
        )

    title_suffix = "(Test)" if args.metric == "test_r2" else "(Train)"
    fig = plot_r2_heatmap(r2_dict, num_layers, num_heads, output_path, title_suffix)

    if args.wandb:
        wandb.log({"r2_heatmap": wandb.Image(fig)})
        wandb.finish()

    plt.close(fig)


if __name__ == "__main__":
    main()
