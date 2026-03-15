"""
Steer assertiveness using ridge probe weights.

Loads top-k heads by R², applies h - α * (w/||w||) via pyvene AdditionIntervention.
Uses epistemic-integrity test_data.csv for evaluation.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.path.insert(0, current_dir)

import argparse
import pickle
import pandas as pd
import torch
import pyvene as pv
import wandb
from tqdm.auto import tqdm

from utils import load_model, load_ep_data, generate_and_decode_new_tokens

def get_top_k_keys(r2_dict, k=16):
    """Return top k keys from r2_dict by R² value (descending)."""
    sorted_items = sorted(r2_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_items[: min(k, len(sorted_items))]]


def get_assertiveness_probe_vectors(
    top_k_heads,
    probe_weights,
    num_layers,
    num_heads,
    head_dim,
    scale,
    device="cuda",
):
    """
    Build intervention vectors from ridge probe weights.
    direction = coef / ||coef||; source_representation = scale * direction.
    scale < 0 steers down assertiveness.
    """
    target_heads = {}
    for head_string in top_k_heads:
        layer, head = head_string.split("_")
        layer = int(layer)
        head = int(head)
        if layer in target_heads:
            target_heads[layer].append(head)
        else:
            target_heads[layer] = [head]

    probes = {i: torch.zeros(head_dim * num_heads, dtype=torch.float32, device=device) for i in range(num_layers)}
    for layer in target_heads:
        for head in target_heads[layer]:
            key = f"{layer}_{head}"
            if key not in probe_weights:
                continue
            pw = probe_weights[key]
            coef = torch.tensor(pw["coef"], dtype=torch.float32, device=device)
            direction = coef / (torch.norm(coef, p=2) + 1e-8)
            vec = scale * direction
            probes[layer][head * head_dim : head_dim * (head + 1)] = vec
    return probes


def main():
    parser = argparse.ArgumentParser(description="Steer assertiveness using ridge probe weights.")
    parser.add_argument("--model_id", type=str, default="gemma-3", help="Model ID")
    parser.add_argument("--k_heads", type=int, default=16, help="Number of top heads by R²")
    parser.add_argument("--scale", type=float, default=-5.0, help="Steering strength (negative = steer down assertiveness)")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save outputs (default: predictions_assertiveness/...)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples (default: all)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")

    args = parser.parse_args()
    model_id = args.model_id
    model_name = model_id.split("/")[-1]
    probe_dir = f"probe/trained_probe_assertiveness/{model_name}"

    if args.wandb:
        wandb.init(
            project="assertiveness-steering",
            config={
                "model_id": model_id,
                "k_heads": args.k_heads,
                "scale": args.scale,
                "max_samples": args.max_samples,
            },
            name=f"epint_inference_{model_name}_k{args.k_heads}_scale{args.scale}",
        )

    print(f"Loading model: {model_id}")
    model, processor = load_model(model_id)
    model.eval()

    # Model config
    if "gemma" in str(type(model)).lower():
        cfg = model.config.text_config
    else:
        cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    # Load R² for top-k selection and probe weights
    r2_path = os.path.join(probe_dir, "r2_dict.pkl")
    weights_path = os.path.join(probe_dir, "probe_weights.pkl")
    if not os.path.exists(r2_path):
        raise FileNotFoundError(f"Run train_epint first. Missing: {r2_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Run train_epint first. Missing: {weights_path}")

    with open(r2_path, "rb") as f:
        r2_dict = pickle.load(f)
    with open(weights_path, "rb") as f:
        probe_weights = pickle.load(f)

    print(f"Getting top {args.k_heads} heads by R²")
    top_k_heads = get_top_k_keys(r2_dict, k=args.k_heads)
    print(f"Top heads: {top_k_heads[:5]}...")

    print(f"Creating probe vectors with scale {args.scale}")
    linear_probes = get_assertiveness_probe_vectors(
        top_k_heads,
        probe_weights,
        num_layers,
        num_heads,
        head_dim,
        scale=args.scale,
        device="cuda",
    )

    # Setup pyvene intervention
    print("Setting up intervention components")
    if "gemma" in model_id.lower():
        target_components = [
            {
                "component": f"model.language_model.layers[{i}].self_attn.o_proj.input",
                "intervention": pv.AdditionIntervention(source_representation=linear_probes[i]),
            }
            for i in range(num_layers)
            if torch.count_nonzero(linear_probes[i])
        ]
    else:
        target_components = [
            {
                "component": f"model.layers[{i}].self_attn.o_proj.input",
                "intervention": pv.AdditionIntervention(source_representation=linear_probes[i]),
            }
            for i in range(num_layers)
            if torch.count_nonzero(linear_probes[i])
        ]

    print("Creating intervenable model")
    pv_model = pv.IntervenableModel(target_components, model=model)

    # Load test data
    print("Loading epistemic-integrity test_data.csv")
    ds_test = load_ep_data("test")
    test_texts = list(ds_test["text"])
    test_assertiveness = list(ds_test["assertiveness"])
    if args.max_samples:
        test_texts = test_texts[: args.max_samples]
        test_assertiveness = test_assertiveness[: args.max_samples]
    print(f"Evaluating on {len(test_texts)} samples")

    # Generate with and without steering (initial=base model, steered=intervenable model)
    print("Generating responses...")
    initial_answers = []
    steered_answers = []
    for text in tqdm(test_texts):
        init_res, _ = generate_and_decode_new_tokens(text, model, processor, model_id)
        steer_res, _ = generate_and_decode_new_tokens(text, pv_model, processor, model_id)
        initial_answers.append(init_res)
        steered_answers.append(steer_res)

    # Save outputs
    out_dir = "predictions_assertiveness"
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output_csv or os.path.join(
        out_dir, f"epint_{model_name}_k{args.k_heads}_scale{args.scale}.csv"
    )
    df = pd.DataFrame(
        {
            "text": test_texts,
            "assertiveness": test_assertiveness,
            "initial_answer": initial_answers,
            "steered_answer": steered_answers,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    if args.wandb:
        wandb.log({
            "output_path": out_path,
            "n_samples": len(test_texts),
            "top_k_heads": top_k_heads,
        })
        wandb.save(out_path, base_path=os.path.dirname(out_path))
        wandb.finish()


if __name__ == "__main__":
    main()
