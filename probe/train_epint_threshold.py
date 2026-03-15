# train.py
import sys
import os
current_dir = os.path.dirname(os.path.abspath(''))
sys.path.append(current_dir)

import argparse
import pickle
import importlib
from tqdm.auto import tqdm
import numpy as np

import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import load_model, load_ep_data

# Assuming these utils are in the same directory or accessible via PYTHONPATH
import probe_data_utils as probe_data_utils
import extract_activation
from train_epint import extract_activations_batch
from probe import LinearProbe, NonLinearProbe
from probe_data_utils import construct_data

# --- Dataset Class ---
class QADataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Training Function ---
def train_probe(model, processor, train_dataset, val_dataset, batch_size,
                learning_rate, num_epochs, device, target_component,
                activation_type, input_dim, model_id, output_dir,
                hidden_dim, head_dim, probe_type, use_wandb=False, step_offset=0):
    """
    Trains a linear probe on top of LLM representations.
    """
    print(f"----------- Training {activation_type.upper()} Probe for Component: {target_component} --------------")

    def collate_fn(batch):
        padded_inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Determine pad_token_id based on model type
        is_gemma = 'gemma' in str(type(model)).lower()
        if is_gemma:
            pad_val = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.eos_token_id
        else:
            pad_val = processor.pad_token_id if processor.pad_token_id is not None else processor.eos_token_id

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            padded_inputs, batch_first=True, padding_value=pad_val
        )
        labels = torch.tensor(labels)
        return padded_inputs, labels

    # Initialize the linear probe
    if probe_type == 'linear':
        probe = LinearProbe(input_dim).to(device)
    elif probe_type == 'nonlinear':
        probe = NonLinearProbe(input_dim, hidden_dim=head_dim).to(device)
    else:
        raise ValueError("Invalid probe_type specified")

    # Define optimizer and loss function
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        probe.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            target_layer, target_head = [int(x) for x in target_component.split('_')]
            representations = inputs[:, target_layer, target_head, :].to(torch.float32)

            outputs = probe(representations)
            loss = criterion(outputs.squeeze(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        if use_wandb:
            wandb.log({"train_loss": avg_train_loss}, step=step_offset + epoch)

        # Validation
        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                target_layer, target_head = [int(x) for x in target_component.split('_')]
                representations = inputs[:, target_layer, target_head, :].to(torch.float32)

                outputs = probe(representations)
                predicted = torch.sigmoid(outputs).round()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_acc:.2f}%")
        best_val_acc = max(best_val_acc, val_acc) # Keep track of best acc

        if use_wandb:
            wandb.log({"val_acc": val_acc}, step=step_offset + epoch)

    # Save the trained linear probe
    save_dir = os.path.join(output_dir, model_id.split('/')[-1])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{probe_type}_probe_{target_component}.pth")
    torch.save(probe.state_dict(), save_path)
    print(f"Saved probe to {save_path}")

    if use_wandb:
        wandb.log({f"best_val_acc/{target_component}": best_val_acc})

    return best_val_acc # Return best validation accuracy over epochs

# --- Main Function ---
def main(args):
    # Reload utils if needed (useful during interactive development)
    importlib.reload(probe_data_utils)
    importlib.reload(extract_activation)

    # Load Model and Processor
    print(f"Loading model: {args.model_id}...")
    output_dir = "probe/trained_probe_assertiveness_threshold"
    model, processor = load_model(args.model_id)
    model.eval()
    model.to(args.device)

    # Get Model Config
    print("Extracting model configuration...")
    is_gemma = 'gemma' in str(type(model)).lower()
    if is_gemma:
        config = model.config.text_config
        NUM_LAYERS = config.num_hidden_layers
        HIDDEN_DIM = config.hidden_size
        NUM_HEADS = config.num_attention_heads
        HEAD_DIM = config.head_dim
        MLP_DIM = 10240 #hardcoded
    else: # hardcoded for llama 3.2
        config = model.config
        NUM_LAYERS = config.num_hidden_layers
        HIDDEN_DIM = config.hidden_size
        NUM_HEADS = config.num_attention_heads
        HEAD_DIM = config.head_dim
        MLP_DIM = 8192
        if HEAD_DIM is None:
             raise ValueError("Cannot determine HEAD_DIM for MHA on this model architecture.")

    print(f"Model Config: Layers={NUM_LAYERS}, Heads={NUM_HEADS}, HiddenDim={HIDDEN_DIM}, HeadDim={HEAD_DIM}")


    # Load and Prepare Data
    print("Loading and preparing epistemic integrity (assertiveness) data...")
    ds_train = load_ep_data("train")
    texts = ds_train["text"]
    assertiveness_score = np.array(ds_train["assertiveness"], dtype=np.float32)
    assertiveness = np.array(ds_train['assertiveness'] > 5, dtype=np.float32)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, assertiveness, test_size=0.1, random_state=3407
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")


    # if args.concept in ['truthful', 'sycophancy', 'sycophancy_challenged', 'sycophancy_hypothesis']:
    #     chats, labels = construct_data(ds_train, concept=args.concept) # ????
    # else:
    #     raise("Direction/concept not supported")

    # label_encoder = LabelEncoder()
    # labels = label_encoder.fit_transform(labels)
    # numerical_labels = torch.tensor(labels)

    # print("Applying chat template and tokenizing...")
    # chats_templated = processor.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
    # tokenized_data = [
    #     processor(text=chat, return_tensors="pt")["input_ids"].squeeze()
    #     for chat in tqdm(chats_templated, desc="Tokenizing")
    # ]

    # # Split data for probe training
    # print("Splitting data for probe training...")
    # train_tok, val_tok, train_labels, val_labels = train_test_split( # split again since we need validation for probe training
    #     tokenized_data, numerical_labels, test_size=0.2, 
    #     random_state=3407
    # )

    # Extract Activations (includes chat templating and tokenizing)
    print(f"Extracting MHA activations...")
    train_activations = extract_activations_batch(
    model, processor, train_texts, args.model_id, max_length=2048, device=args.device
    )
    train_activations_np = train_activations.float().numpy()
    
    val_activations = extract_activations_batch(
    model, processor, val_texts, args.model_id, max_length=2048, device=args.device
    )
    val_activations_np = val_activations.float().numpy()
    
    # train_activation_list = []
    # for datum in tqdm(train_tok, total=len(train_tok), desc="Extracting Train Activations"):
    #     act_tensor = extract_fn(model, processor, datum)
    #     train_activation_list.append(act_tensor.cpu()) # Move back to CPU for storage

    # val_activation_list = []
    # for datum in tqdm(val_tok, total=len(val_tok), desc="Extracting Val Activations"):
    #     act_tensor = extract_fn(model, processor, datum)
    #     val_activation_list.append(act_tensor.cpu()) # Move back to CPU for storage

    # Create Datasets for Probe Training
    train_dataset = QADataset(train_activations_np, train_labels)
    val_dataset = QADataset(val_activations_np, val_labels)

    # Train Probes
    print("Starting probe training...")
    accuracies = {}
    probe_input_dim = HEAD_DIM

    if args.wandb:
        wandb.init(
            project="sycophancy-probes",
            config=vars(args),
            name=f"{args.model_id}_mha_epint_threshold",
        )

    for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                target_component = f"{layer}_{head}"
                step_offset = (layer * NUM_HEADS + head) * args.epochs
                current_acc = train_probe(model, processor, train_dataset, val_dataset,
                                          args.batch_size, args.lr, args.epochs, args.device,
                                          target_component, args.activation_type, probe_input_dim,
                                          args.model_id, output_dir, HIDDEN_DIM, HEAD_DIM, args.probe_type,
                                          use_wandb=args.wandb, step_offset=step_offset)
                accuracies[target_component] = current_acc

    # Save Accuracies
    output_subdir = os.path.join(output_dir, args.model_id.split('/')[-1])
    os.makedirs(output_subdir, exist_ok=True)
    acc_filename = f"{args.probe_type}_accuracies_dict_mha.pkl"
    acc_path = os.path.join(output_subdir, acc_filename)
    print(f"Saving accuracies to {acc_path}")
    with open(acc_path, 'wb') as f:
        pickle.dump(accuracies, f)

    if args.wandb:
        wandb.log({"accuracies": wandb.Table(columns=["component", "val_acc"], data=[[k, v] for k, v in accuracies.items()])})
        wandb.log({f"acc/{k}": v for k, v in accuracies.items()})
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linear probes on LLM activations.")
    parser.add_argument("--model_id", type=str, required=True, help="'gemma-3' or 'llama-3.2'")
    # parser.add_argument("--activation_type", type=str, required=True, choices=['mha', 'mlp', 'residual'], help="Type of activation to extract and train on ('mha', 'mlp', or 'residual').")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for probe training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for probe training.")
    # parser.add_argument("--concept", type=str, default="sycophancy", choices=["sycophancy", "truthful", "sycophancy_hypothesis", "sycophancy_challenged"], help="Direction/concept to steer")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear", "nonlinear"], help="Type of probe to use ('linear' or 'nonlinear').")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")

    args = parser.parse_args()

    # Add current directory to path for utils etc.
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    sys.path.append(current_dir)

    main(args)