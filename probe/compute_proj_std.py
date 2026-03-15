import argparse
import sys
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import load_model, load_ep_data

from datasets import load_dataset
from probe_data_utils import construct_data
from tqdm.auto import tqdm
from extract_activation import extract_mha_activation, extract_residual_activation, extract_mlp_activation

import torch


def _format_and_tokenize_ep(text, processor, model_id, max_length=2048, device="cuda"):
    """Format text as chat and tokenize for epistemic-integrity data."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save standard deviation along the learned direction.")
    parser.add_argument("--model_id", type=str, required=True, help="'gemma-3' or 'llama-3.2'")
    parser.add_argument("--activation_type", type=str, required=True, choices=['mha', 'mlp', 'residual'], help="Type of activation to extract and trained on ('mha', 'mlp', or 'residual').")
    parser.add_argument("--concept", type=str, required=True, choices=['sycophancy', 'truthful', 'sycophancy_hypothesis', 'sycophancy_challenged', 'assertiveness'], help="Type of direction to extract")
    parser.add_argument("--probe_type", type=str, default='linear', choices=['linear', 'nonlinear'], help='Type of probe to use {linear or nonlinear}')

    args = parser.parse_args()
    model_id = args.model_id
    model_name = model_id.split("/")[-1]
    activation_type = args.activation_type
    concept = args.concept

    if concept == 'assertiveness' and activation_type != 'mha':
        raise ValueError("For concept 'assertiveness', activation_type must be 'mha'.")

    print("Loading model")
    model, processor = load_model(model_id)

    if concept == 'assertiveness':
        # Load epistemic-integrity train data and extract MHA activations
        print("Loading epistemic-integrity train_data.csv...")
        ds_train = load_ep_data("train")
        texts = ds_train["text"]
        print("Formatting and tokenizing...")
        tokenized_data = [
            _format_and_tokenize_ep(t, processor, model_id)
            for t in tqdm(texts, desc="Tokenizing")
        ]
        extract_fn = extract_mha_activation
    else:
        # Load and Prepare TruthfulQA data
        print("Loading and preparing TruthfulQA data...")
        ds = load_dataset("truthfulqa/truthful_qa", "generation")
        split_ds = ds["validation"].train_test_split(test_size=0.2, seed=3407)
        ds_train_split = split_ds["train"]
        chats, labels = construct_data(ds_train_split, concept=concept)
        print("Applying chat template and tokenizing...")
        chats_templated = processor.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
        tokenized_data = [
            processor(text=chat, return_tensors="pt")["input_ids"].squeeze()
            for chat in tqdm(chats_templated, desc="Tokenizing")
        ]
        if activation_type == 'mha':
            extract_fn = extract_mha_activation
        elif activation_type == 'mlp':
            extract_fn = extract_mlp_activation
        elif activation_type == 'residual':
            extract_fn = extract_residual_activation
        else:
            raise ValueError(f"Unsupported activation_type: {activation_type}.")

    train_activation_list = []
    for datum in tqdm(tokenized_data, total=len(tokenized_data), desc="Extracting Activations"):
        act_tensor = extract_fn(model, processor, datum.to('cuda'))
        train_activation_list.append(act_tensor.cpu())
    tuning_activations = torch.stack(train_activation_list)

    NUM_LAYER = train_activation_list[0].shape[0]
    NUM_HEAD = train_activation_list[0].shape[1]
    print(f"Computing standard deviations: {activation_type}")
    if activation_type == 'mha':
        if concept == 'assertiveness':
            # Load ridge probe weights; direction = coef / ||coef||
            probe_dir = f"probe/trained_probe_assertiveness/{model_name}"
            weights_path = os.path.join(probe_dir, "probe_weights.pkl")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Run train_epint first. Missing: {weights_path}")
            with open(weights_path, "rb") as f:
                probe_weights = pickle.load(f)
            for layer in tqdm(range(NUM_LAYER)):
                for head in range(NUM_HEAD):
                    key = f"{layer}_{head}"
                    if key not in probe_weights:
                        continue
                    coef = torch.tensor(probe_weights[key]["coef"], dtype=torch.float32)
                    direction = coef / (torch.norm(coef, p=2) + 1e-8)
                    activations = torch.tensor(tuning_activations[:, layer, head, :], dtype=torch.float32)
                    proj_vals = activations @ direction
                    proj_val_std = torch.std(proj_vals)
                    save_path = os.path.join(probe_dir, f"linear_std_mha_{layer}_{head}.pt")
                    torch.save(proj_val_std, save_path)
        else:
            probe_dir = f"trained_probe_{concept}/{model_id}"
            for layer in tqdm(range(NUM_LAYER)):
                for head in range(NUM_HEAD):
                    if args.probe_type == 'nonlinear':
                        direction = torch.load(f'{probe_dir}/{args.probe_type}_probe_{layer}_{head}.pth')['net.2.weight'][0, :].cpu().to(torch.bfloat16)
                    else:
                        direction = torch.load(f'{probe_dir}/{args.probe_type}_probe_{layer}_{head}.pth')['linear.weight'][0, :].cpu().to(torch.bfloat16)
                    direction = direction/torch.norm(direction)
                    activations = torch.tensor(tuning_activations[:,layer,head,:], dtype=torch.bfloat16).to("cpu")
                    proj_vals = activations @ direction.T
                    proj_val_std = torch.std(proj_vals)
                    torch.save(proj_val_std, f'{probe_dir}/{args.probe_type}_std_mha_{layer}_{head}.pt')
    elif activation_type == 'residual':
        for layer in tqdm(range(NUM_LAYER)):
            if args.probe_type == 'nonlinear':
                direction = torch.load(f'trained_probe_{concept}/{model_id}/{args.probe_type}_probe_residual_{layer}.pth')['net.2.weight'][0, :].cpu().to(torch.bfloat16)
            else:
                direction = torch.load(f'trained_probe_{concept}/{model_id}/{args.probe_type}_probe_residual_{layer}.pth')['linear.weight'][0, :].cpu().to(torch.bfloat16)
            direction = direction/torch.norm(direction)
            activations = torch.tensor(tuning_activations[:,layer,:], dtype=torch.bfloat16).to("cpu")
            proj_vals = activations @ direction.T
            proj_val_std = torch.std(proj_vals)
            torch.save(proj_val_std, f'trained_probe_{concept}/{model_id}/{args.probe_type}_std_residual_{layer}.pt')
    elif activation_type == 'mlp':
        for layer in tqdm(range(NUM_LAYER)):
            if args.probe_type == 'nonlinear':
                direction = torch.load(f'trained_probe_{concept}/{model_id}/{args.probe_type}_probe_mlp_{layer}.pth')['net.2.weight'][0, :].cpu().to(torch.bfloat16)
            else:
                direction = torch.load(f'trained_probe_{concept}/{model_id}/{args.probe_type}_probe_mlp_{layer}.pth')['linear.weight'][0, :].cpu().to(torch.bfloat16)
            direction = direction/torch.norm(direction)
            activations = torch.tensor(tuning_activations[:,layer,:], dtype=torch.bfloat16).to("cpu")
            proj_vals = activations @ direction.T
            proj_val_std = torch.std(proj_vals)
            torch.save(proj_val_std, f'trained_probe_{concept}/{model_id}/{args.probe_type}_std_mlp_{layer}.pt')
    else:
        raise(f'Activation type not supported: {activation_type}')