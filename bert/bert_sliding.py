import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig, BertForMaskedLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import os
import json
from collections import defaultdict
from sklearn.metrics import brier_score_loss
from scipy.stats import pearsonr
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_txt(path):
    return np.loadtxt(path, dtype=np.int8)

    return torch.tensor(weights, dtype=torch.float32, device=device)

def sliding_window_chunks(array, window_size=512, stride=256):
    """
    For each row (haplotype sequence), returns chunks of fixed window_size.
    The final chunk ends exactly at the last SNP and includes the last window_size SNPs.
    """
    chunks = []
    for row in array:
        row_len = len(row)
        last_start = row_len - window_size

        for i in range(0, last_start + 1, stride):
            chunk = row[i:i + window_size]
            chunks.append(chunk)

        # Add final chunk if not already included
        if (row_len - window_size) % stride != 0:
            final_chunk = row[-window_size:]
            chunks.append(final_chunk)

    return np.array(chunks, dtype=np.int64)

def get_mask_prob(epoch, total_epochs, mask_prob_start, mask_prob_min=None):
    """
    Linearly anneal mask_prob from mask_prob_start to mask_prob_min across training.
    If mask_prob_min is None, just return mask_prob_start.
    """
    if mask_prob_min is None:
        return mask_prob_start
    # Linear schedule: start → min over epochs
    progress = epoch / max(1, total_epochs - 1)
    return mask_prob_start - (mask_prob_start - mask_prob_min) * progress

def save_minimal_tokenizer(vocab, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Save vocab.json
    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

def mask_input(input_ids, mask_token_id, mask_prob=0.15):
    labels = input_ids.clone()
    masked_input = input_ids.clone()
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_arr = rand < mask_prob
    masked_input[mask_arr] = mask_token_id
    labels[~mask_arr] = -100
    return masked_input, labels

def mask_input_even(input_ids, mask_token_id, mask_prob=0.15):
    labels = input_ids.clone()
    masked_input = input_ids.clone()
    
    B, L = input_ids.shape
    mask_arr = torch.zeros_like(input_ids, dtype=torch.bool)

    # Balance masking for 0s and 1s separately
    for val in [0, 1]:
        val_mask = (input_ids == val)
        rand = torch.rand_like(input_ids, dtype=torch.float)
        val_masking = (rand < mask_prob) & val_mask
        mask_arr |= val_masking  # Combine mask for 0s and 1s

    masked_input[mask_arr] = mask_token_id
    labels[~mask_arr] = -100

    return masked_input, labels

def mask_whole_snps(input_ids, mask_token_id, mask_percent=0.15, seed=None):
    """
    Masks a percentage of SNP positions across the entire batch.
    
    Args:
        input_ids: Tensor of shape [B, L]
        mask_token_id: Token ID used for masking
        mask_percent: Fraction (0-1) of positions in the window to mask
        device: Optional device for torch.Generator
        seed: If provided, makes masking deterministic
    
    Returns:
        masked_input: Masked tensor
        labels: Labels tensor where unmasked positions are -100
    """
    B, L = input_ids.shape
    labels = input_ids.clone()
    masked_input = input_ids.clone()

    # Determine number of sites to mask based on percentage
    num_sites = max(1, round(L * mask_percent))  # ensure at least one position masked

    # Create a local generator for reproducibility
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        pos = torch.randperm(L, generator=g)[:num_sites].to(input_ids.device)
    else:
        pos = torch.randperm(L)[:num_sites].to(input_ids.device)

    # Create mask where every sample has True at these positions
    mask_arr = torch.zeros_like(input_ids, dtype=torch.bool)
    mask_arr[:, pos] = True

    masked_input[mask_arr] = mask_token_id
    labels[~mask_arr] = -100

    return masked_input, labels

# def train_epoch(model, loader, optimizer, scaler, scheduler, device, mask_token_id, mask_prob, epoch):
#     model.train()
#     losses = []
#     for (batch,) in loader:
#         masked, labels = mask_whole_snps(batch, mask_token_id, mask_percent=mask_prob)
#         masked, labels = masked.to(device), labels.to(device)

#         with autocast(device_type=device.type):
#             attention_mask = (masked != -100).long()
#             outputs = model(input_ids=masked, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step()
#         optimizer.zero_grad()
#         losses.append(loss.item())
#     return np.mean(losses)

def train_epoch(model, loader, optimizer, scaler, scheduler, device, mask_token_id, mask_prob, epoch, total_epochs, mask_prob_min=None):
    model.train()
    losses = []
    # compute annealed mask rate for this epoch
    mask_prob_epoch = get_mask_prob(epoch, total_epochs, mask_prob, mask_prob_min)

    for (batch,) in loader:
        masked, labels = mask_whole_snps(batch, mask_token_id, mask_percent=mask_prob_epoch)
        masked, labels = masked.to(device), labels.to(device)

        with autocast(device_type=device.type):
            attention_mask = (masked != -100).long()
            outputs = model(input_ids=masked, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return np.mean(losses), mask_prob_epoch


def eval_epoch(model, loader, device, mask_token_id, mask_prob, max_snps=100, calc_per_snp_r2=True):
    model.eval()
    losses = []
    all_probs = []
    all_labels = []
    per_snp_r2_values = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            seq_len = batch.size(1)
            num_snps = min(seq_len, max_snps)
            snp_indices_to_mask = random.sample(range(seq_len), num_snps)

            # Global masking for overall metrics
            masked, labels = mask_whole_snps(batch, mask_token_id, mask_percent=mask_prob)
            masked, labels = masked.to(device), labels.to(device)
            attention_mask = (masked != -100).long()

            with autocast(device_type=device.type):
                outputs = model(input_ids=masked, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits[..., :2]

            mask = labels != -100
            masked_logits = logits[mask]
            masked_labels = labels[mask]

            probs = torch.softmax(masked_logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(masked_labels.cpu())
            losses.append(loss.item())

            # Per-SNP masking for subset (only if toggle is on)
            if calc_per_snp_r2:
                for snp_idx in snp_indices_to_mask:
                    labels_single = torch.full_like(batch, fill_value=-100)
                    masked_single = batch.clone()
                    masked_single[:, snp_idx] = mask_token_id
                    labels_single[:, snp_idx] = batch[:, snp_idx]

                    attention_mask_single = (masked_single != -100).long()

                    with autocast(device_type=device.type):
                        outputs_single = model(
                            input_ids=masked_single,
                            attention_mask=attention_mask_single,
                            labels=labels_single
                        )
                    logits_single = outputs_single.logits[..., :2]

                    probs_single = torch.softmax(logits_single, dim=-1)[:, snp_idx, 1].cpu().numpy()
                    labels_single_np = batch[:, snp_idx].cpu().numpy()

                    if np.std(probs_single) > 0 and np.std(labels_single_np) > 0:
                        r, _ = pearsonr(probs_single, labels_single_np)
                        per_snp_r2_values.append(r ** 2)
                    else:
                        per_snp_r2_values.append(0)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    if np.std(all_probs) == 0 or np.std(all_labels) == 0:
        r2_global = 0.0
    else:
        r, _ = pearsonr(all_probs, all_labels)
        r2_global = r ** 2

    r2_per_snp_mean = float(np.mean(per_snp_r2_values)) if (calc_per_snp_r2 and per_snp_r2_values) else 0.0
    brier = brier_score_loss(all_labels, all_probs)
    avg_loss = np.mean(losses)

    return avg_loss, brier, r2_global, r2_per_snp_mean

def adaptive_lr(base_lr, base_mask_prob, new_mask_prob):
    return base_lr * ((base_mask_prob / new_mask_prob) ** 0.5)

def main(args):
    set_seed(args.seed)  # Set random seeds

    wandb.init(
        project="snp-imputation",          # Change to your actual project name
        name=f"run-seed{args.seed}",       # Or use wandb.util.generate_id()
        config=vars(args)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = {"0": 0, "1": 1, "[MASK]": 2, "[PAD]": 3}
    vocab_size = len(vocab)
    mask_token_id = vocab["[MASK]"]
    pad_token_id = vocab["[PAD]"]

    print("Loading data...")
    X = load_txt(args.train_data)
    X_chunks = sliding_window_chunks(X, window_size=args.seq_len, stride=args.stride)

    print(X_chunks.shape)
    

    X_input_ids = torch.tensor(X_chunks, dtype=torch.long)
    train_tensor, val_tensor = train_test_split(X_input_ids, test_size=0.1, random_state=args.seed)

    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Building model...")
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_position_embeddings=args.seq_len,
        intermediate_size=4*args.hidden_size,
        type_vocab_size=1,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id
    )
    model = BertForMaskedLM(config)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    lr_new = adaptive_lr(args.lr, 0.15, args.mask_prob)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_new)
    scaler = GradScaler()

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")
    epochs_without_improve = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, mask_prob_epoch = train_epoch(model, train_loader, optimizer, scaler, scheduler, device, mask_token_id, args.mask_prob, epoch, args.epochs, args.mask_prob_min)
        val_loss, val_brier, val_r2, site_r2 = eval_epoch(model, val_loader, device, mask_token_id, args.mask_prob, calc_per_snp_r2=False)
        print(f"Epoch {epoch+1}/{args.epochs} "
          f"- Train Loss: {train_loss:.4f} "
          f"- Val Loss: {val_loss:.4f} "
          f"- Brier: {val_brier:.4f} "
          f"- R²: {val_r2:.4f} "
          f"- Site R²: {site_r2:.4f} "
          f"- Mask Prob: {mask_prob_epoch:.3f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/brier_score": val_brier,
            "val/r2": val_r2,
            "val/r2_site": site_r2,
            "lr": optimizer.param_groups[0]["lr"],
            "mask_prob": mask_prob_epoch,
        })
        
        # Early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            print(f"Validation loss improved. Saving model checkpoint to {args.output_dir}")
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            save_minimal_tokenizer(vocab, args.output_dir)
        else:
            epochs_without_improve += 1
            print(f"No improvement in validation loss for {epochs_without_improve} epochs.")

        if epochs_without_improve >= args.patience:
            print(f"Early stopping triggered after {epochs_without_improve} epochs without improvement.")
            break

    print("Training complete.")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT for SNP Imputation")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data txt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (number of SNPs)")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of BERT model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of BERT layers")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Masking probability")
    parser.add_argument("--mask_prob_min", type=float, default=None, help="Minimum masking probability for annealing schedule. If None, use constant mask_prob.")

    args = parser.parse_args()
    main(args)
