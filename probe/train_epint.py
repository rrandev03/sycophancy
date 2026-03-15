"""
Train ridge regression probes on MHA activations to predict assertiveness.

Uses epistemic-integrity train_data.csv (90/10 train/val) and test_data.csv for evaluation.
Saves probe weights for steering at best layer/heads.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tqdm.auto import tqdm

import wandb

from utils import load_model, load_ep_data
import extract_activation


def _format_and_tokenize(text, processor, model_id, max_length=2048, device="cuda"):
    """Format text as chat and tokenize. Returns input_ids tensor."""
    if "gemma" in model_id.lower():
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]
    inputs_str = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    if "gemma" in model_id.lower():
        encoded = processor.tokenizer(
            text=inputs_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    else:
        encoded = processor(
            inputs_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    return encoded["input_ids"].squeeze(0).to(device)


def extract_activations_batch(model, processor, texts, model_id, max_length=2048, device="cuda"):
    """Extract MHA activations at last token for each text. Returns (n, n_layers, n_heads, head_dim)."""
    model.eval()
    activations_list = []
    for text in tqdm(texts, desc="Extracting activations"):
        input_ids = _format_and_tokenize(text, processor, model_id, max_length, device)
        act = extract_activation.extract_mha_activation(model, processor, input_ids)
        activations_list.append(act.cpu())
    return torch.stack(activations_list)


def train_ridge_probes(activations, assertiveness, num_layers, num_heads, head_dim, alpha=1.0):
    """Train ridge regression per (layer, head). Returns r2_dict, mse_dict, probe_weights."""
    r2_dict = {}
    mse_dict = {}
    probe_weights = {}
    for layer in tqdm(range(num_layers), desc="Training probes"):
        for head in range(num_heads):
            key = f"{layer}_{head}"
            X = activations[:, layer, head, :]
            y = assertiveness
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            ridge = Ridge(alpha=alpha, random_state=3407)
            ridge.fit(X_scaled, y)
            y_pred = ridge.predict(X_scaled)
            r2_dict[key] = float(r2_score(y, y_pred))
            mse_dict[key] = float(mean_squared_error(y, y_pred))
            probe_weights[key] = {
                "coef": ridge.coef_.astype(np.float32),
                "intercept": float(ridge.intercept_),
                "scaler_mean": scaler.mean_.astype(np.float32),
                "scaler_scale": scaler.scale_.astype(np.float32),
            }
    return r2_dict, mse_dict, probe_weights


def evaluate_probes(probe_weights, activations, assertiveness):
    """Evaluate probes on held-out data. Returns r2_dict, mse_dict."""
    r2_dict = {}
    mse_dict = {}
    for key, pw in probe_weights.items():
        layer, head = map(int, key.split("_"))
        X = activations[:, layer, head, :]
        y = assertiveness
        X_scaled = (X - pw["scaler_mean"]) / pw["scaler_scale"]
        y_pred = X_scaled @ pw["coef"] + pw["intercept"]
        r2_dict[key] = float(r2_score(y, y_pred))
        mse_dict[key] = float(mean_squared_error(y, y_pred))
    return r2_dict, mse_dict


def main(args):
    print(f"Loading model: {args.model_id}...")
    output_dir = "probe/trained_probe_assertiveness"
    model, processor = load_model(args.model_id)
    model.eval()
    model.to(args.device)

    

    is_gemma = "gemma" in str(type(model)).lower()
    if is_gemma:
        cfg = model.config.text_config
    else:
        cfg = model.config
    NUM_LAYERS = cfg.num_hidden_layers
    NUM_HEADS = cfg.num_attention_heads
    HEAD_DIM = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"Model Config: Layers={NUM_LAYERS}, Heads={NUM_HEADS}, HeadDim={HEAD_DIM}")

    if args.wandb:
        wandb.init(
            project="assertiveness-probes",
            config={
                "model_id": args.model_id,
                "ridge_alpha": args.ridge_alpha,
                "device": args.device,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "head_dim": HEAD_DIM,
            },
            name=f"epint_{args.model_id.split('/')[-1]}",
        )

    # Step 1: Load train data, 90/10 split
    print("Loading epistemic-integrity train_data.csv...")
    ds_train = load_ep_data("train")
    texts = ds_train["text"]
    assertiveness = np.array(ds_train["assertiveness"], dtype=np.float32)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, assertiveness, test_size=0.1, random_state=3407
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Step 2: Extract activations
    print("Extracting MHA activations (train)...")
    train_activations = extract_activations_batch(
        model, processor, train_texts, args.model_id, max_length=2048, device=args.device
    )
    train_activations_np = train_activations.float().numpy()

    print("Extracting MHA activations (val)...")
    val_activations = extract_activations_batch(
        model, processor, val_texts, args.model_id, max_length=2048, device=args.device
    )
    val_activations_np = val_activations.float().numpy()

    # Step 3: Train ridge probes
    print("Training ridge regression probes...")
    r2_dict, mse_dict, probe_weights = train_ridge_probes(
        train_activations_np,
        train_labels,
        NUM_LAYERS,
        NUM_HEADS,
        HEAD_DIM,
        alpha=args.ridge_alpha,
    )

    # Validation R²
    val_r2, val_mse = evaluate_probes(probe_weights, val_activations_np, val_labels)
    best_train_r2 = max(r2_dict.values())
    best_val_r2 = max(val_r2.values())
    best_train_head = max(r2_dict, key=r2_dict.get)
    best_val_head = max(val_r2, key=val_r2.get)
    print(f"Best train R²: {best_train_r2:.4f} at {best_train_head}")
    print(f"Best val R²: {best_val_r2:.4f} at {best_val_head}")
    if args.wandb:
        wandb.log({
            "train/best_r2": best_train_r2,
            "train/best_head": best_train_head,
            "val/best_r2": best_val_r2,
            "val/best_head": best_val_head,
        })

    # Step 4: Evaluate on test_data.csv
    print("Loading test_data.csv for evaluation...")
    ds_test = load_ep_data("test")
    test_texts = ds_test["text"]
    test_labels = np.array(ds_test["assertiveness"], dtype=np.float32)
    print("Extracting MHA activations (test)...")
    test_activations = extract_activations_batch(
        model, processor, test_texts, args.model_id, max_length=2048, device=args.device
    )
    test_activations_np = test_activations.float().numpy()
    test_r2, test_mse = evaluate_probes(probe_weights, test_activations_np, test_labels)
    best_test_r2 = max(test_r2.values())
    best_test_head = max(test_r2, key=test_r2.get)
    print(f"Best test R²: {best_test_r2:.4f} at {best_test_head}")
    if args.wandb:
        wandb.log({
            "test/best_r2": best_test_r2,
            "test/best_head": best_test_head,
        })
        # Log R² heatmap data as table (layer_head, r2_train, r2_val, r2_test)
        r2_table = wandb.Table(
            columns=["layer_head", "r2_train", "r2_val", "r2_test", "mse_train", "mse_test"],
            data=[
                [k, r2_dict.get(k, 0), val_r2.get(k, 0), test_r2.get(k, 0), mse_dict.get(k, 0), test_mse.get(k, 0)]
                for k in r2_dict
            ],
        )
        wandb.log({"r2_per_head": r2_table})

    # Step 5: Save
    model_name = args.model_id.split("/")[-1]
    save_dir = os.path.join(output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "probe_weights.pkl"), "wb") as f:
        pickle.dump(probe_weights, f)
    with open(os.path.join(save_dir, "r2_dict.pkl"), "wb") as f:
        pickle.dump(r2_dict, f)
    with open(os.path.join(save_dir, "mse_dict.pkl"), "wb") as f:
        pickle.dump(mse_dict, f)
    with open(os.path.join(save_dir, "test_r2_dict.pkl"), "wb") as f:
        pickle.dump(test_r2, f)
    with open(os.path.join(save_dir, "test_mse_dict.pkl"), "wb") as f:
        pickle.dump(test_mse, f)
    # R² as "accuracy" for inference_epint top-k selection
    with open(os.path.join(save_dir, "linear_accuracies_dict_mha.pkl"), "wb") as f:
        pickle.dump(r2_dict, f)

    print(f"Saved to {save_dir}")
    if args.wandb:
        wandb.log({"save_dir": save_dir})
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train assertiveness probes (ridge regression) on MHA activations.")
    parser.add_argument("--model_id", type=str, default="gemma-3", help="Model ID")
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    main(args)
