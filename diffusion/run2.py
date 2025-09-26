import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
import logging
from torch.optim.lr_scheduler import LambdaLR
import math

# class DiscreteDiffusionModel:
#     def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.1, device="cpu"):
#         self.device = device
#         self.num_timesteps = num_timesteps
#         self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

#     def q_sample(self, x_0, t):
#         """Bernoulli mixture forward process"""
#         batch_size, snps = x_0.shape
#         beta_t = self.betas[t].view(-1, 1)  # (B, 1)
#         prob_1 = (1 - beta_t) * x_0 + beta_t * (1 - x_0)
#         x_t = torch.bernoulli(prob_1)
#         return x_t

#     def sample_timesteps(self, batch_size):
#         return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

# class ReverseModel(nn.Module):
#     def __init__(self, snps, hidden_dim=2048):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(snps + 1, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, snps),
#         )

#     def forward(self, x_t, t):
#         t_norm = t.float().unsqueeze(1) / 100  # normalize
#         x_in = torch.cat([x_t, t_norm.expand_as(x_t[:, :1])], dim=1)
#         return self.net(x_in)

# def train_masked(model, diffusion, train_dataset, val_dataset, optimizer,
#                  epochs=10, batch_size=128, patience=20,
#                  val_index_file=None, scheduler=None):

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None
#     loss_fn = nn.BCEWithLogitsLoss()

#     # Load validation mask indices once
#     mask_indices = None
#     if val_index_file is not None and os.path.exists(val_index_file):
#         mask_indices = np.loadtxt(val_index_file, dtype=int)
#         logging.info(f"Validation will mask {len(mask_indices)} SNPs: {mask_indices}")

#     best_val_loss = float("inf")
#     epochs_no_improve = 0

#     # --- AMP scaler ---
#     scaler = torch.cuda.amp.GradScaler()

#     for epoch in range(epochs):
#         # ---- Training ----
#         model.train()
#         total_train_loss = 0.0
#         for x_0 in train_loader:
#             x_0 = x_0[0].to(diffusion.device)
#             t = diffusion.sample_timesteps(x_0.shape[0])
#             x_t = diffusion.q_sample(x_0, t)

#             optimizer.zero_grad()

#             # --- Forward + loss with AMP ---
#             with torch.cuda.amp.autocast():
#                 x_0_pred = model(x_t, t)
#                 loss = loss_fn(x_0_pred, x_0)

#             # --- Backward and optimizer step with scaler ---
#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()

#             total_train_loss += loss.item() * x_0.size(0)

#         avg_train_loss = total_train_loss / len(train_dataset)

#         # ---- Validation ----
#         if val_loader is not None and mask_indices is not None:
#             model.eval()
#             total_val_loss = 0.0
#             with torch.no_grad():
#                 for x_0 in val_loader:
#                     x_0 = x_0[0].to(diffusion.device)

#                     x_masked = x_0.clone()
#                     x_masked[:, mask_indices] = 0.0
#                     t_in = torch.zeros(x_0.shape[0], dtype=torch.long, device=diffusion.device)
#                     preds = model(x_masked, t_in)
#                     loss = loss_fn(preds[:, mask_indices], x_0[:, mask_indices])
#                     total_val_loss += loss.item() * x_0.size(0)

#             avg_val_loss = total_val_loss / len(val_dataset)
#             logging.info(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} "
#                          f"| Val Loss (masked): {avg_val_loss:.4f}")

#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 epochs_no_improve = 0
#                 torch.save(model.state_dict(), "best_model.pt")
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= patience:
#                     logging.info(f"Early stopping triggered after {epoch+1} epochs.")
#                     break
#         else:
#             logging.info(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f}")

#         if scheduler is not None:
#             scheduler.step()

#     return epoch + 1

# @torch.no_grad()
# def evaluate_masked_r2_from_index_file(
#     model,
#     diffusion,
#     x_data,
#     index_file,
#     device,
#     output_path=None,
#     batch_size=1024,
# ):
#     """
#     Mask SNP positions specified in index_file, impute them in batches,
#     and compute per-SNP R². Returns average R².
#     """
#     model.eval()
#     x_data = x_data.to(device)
#     samples, features = x_data.shape

#     # Load indices to mask
#     mask_indices = np.loadtxt(index_file, dtype=int)
#     logging.info(f"Masking {len(mask_indices)} SNPs: {mask_indices}")

#     # Prepare masked input
#     x_masked = x_data.clone()
#     x_masked[:, mask_indices] = 0.0

#     preds = torch.zeros_like(x_data, device=device)

#     # --- Batched forward passes ---
#     for start in range(0, samples, batch_size):
#         end = min(start + batch_size, samples)
#         x_batch = x_masked[start:end]
#         t_in = torch.zeros(x_batch.size(0), dtype=torch.long, device=device)

#         preds[start:end] = torch.sigmoid(model(x_batch, t_in))

#     # --- Compute per-SNP R² for masked features ---
#     r2_list = []
#     for feat_idx in mask_indices:
#         true_vals = x_data[:, feat_idx]
#         pred_vals = preds[:, feat_idx]
#         r2 = compute_squared_pearson_correlation(true_vals, pred_vals)
#         r2_list.append(r2.item())

#     avg_r2 = float(np.mean(r2_list))
#     logging.info(f"Average R² across masked SNPs: {avg_r2:.6f}")

#     if output_path:
#         df = pd.DataFrame({"SNP_Index": mask_indices, "R2": r2_list})
#         df.to_csv(output_path, index=False, float_format="%.8f")
#         logging.info(f"Saved SNP-wise R² to {output_path}")

#     return avg_r2

# class ConditionalReverseModel(nn.Module):
#     def __init__(self, snps, hidden_dim=2048):
#         super().__init__()
#         # input: noisy haplotype (snps) + mask (snps) + timestep (1)
#         self.net = nn.Sequential(
#             nn.Linear(2 * snps + 1, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, snps)
#         )

#     def forward(self, x_t, t, mask):
#         # normalize timestep
#         t_norm = t.float().unsqueeze(1) / 100.0
#         x_in = torch.cat([x_t, mask.float(), t_norm.expand(x_t.size(0), 1)], dim=1)
#         return self.net(x_in)

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / optimizer.param_groups[0]["lr"], cosine_decay)
    return LambdaLR(optimizer, lr_lambda)

def load_maf_weights(maf_file, min_weight=0.1, max_weight=20.0, sqrt_scale=True):
    """
    maf_file: tab-separated file with [snp_id, maf]
    min_weight: floor to avoid extremely small weights
    max_weight: ceiling to avoid extremely large weights
    sqrt_scale: if True, use 1/sqrt(maf*(1-maf)); else use 1/(maf*(1-maf))
    
    Returns: torch.FloatTensor of shape (num_snps,)
    """
    df = pd.read_csv(maf_file, sep="\t", header=None, names=["snp_id", "maf"])
    maf = df["maf"].to_numpy(dtype=np.float32)

    if sqrt_scale:
        weights = 1.0 / np.sqrt(maf * (1 - maf) + 1e-8)
    else:
        weights = 1.0 / (maf * (1 - maf) + 1e-8)

    # normalize and clip
    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)

    return torch.tensor(weights, dtype=torch.float32)

def weighted_bce_loss_with_logits(logits, targets, mask, maf_weights):
    """
    logits: (batch, seq_len) raw outputs from model
    targets: (batch, seq_len) true SNP values (0/1)
    mask: (batch, seq_len) 1 = observed, 0 = masked
    maf_weights: (seq_len,) precomputed SNP weights
    """
    # expand maf_weights to match batch
    maf_tensor = maf_weights.to(logits.device).unsqueeze(0).expand_as(targets)

    # only apply loss on masked positions, scaled by MAF
    weights = (1 - mask) * maf_tensor  # (batch, seq_len)

    loss = F.binary_cross_entropy_with_logits(
        logits, targets, weight=weights, reduction="sum"
    )

    # normalize so loss is comparable across batches
    return loss / (weights.sum() + 1e-8)
    
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for timesteps (like in Transformers / Diffusion models)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] tensor of timesteps
        Returns: [B, dim] embedding
        """
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * (-math.log(10000) / half_dim)
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=device)], dim=1)
        return emb


class ResidualConvBlock(nn.Module):
    """1D Residual convolutional block."""
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size//2)*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size//2)*dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class ConditionalReverseConvModel(nn.Module):
    """
    Conditional reverse model for haplotype diffusion using 1D CNNs + residual blocks.
    Input: [batch, snps] noisy haplotype + mask
    """
    def __init__(self, snps, n_channels=256, n_blocks=8, t_emb_dim=128):
        super().__init__()
        self.snps = snps
        self.t_emb = SinusoidalPosEmb(t_emb_dim)

        # initial projection: haplotype + mask
        self.input_proj = nn.Conv1d(2, n_channels, kernel_size=3, padding=1)

        # residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualConvBlock(n_channels, kernel_size=5, dilation=2**i) for i in range(n_blocks)
        ])

        # time embedding projection
        self.t_proj = nn.Linear(t_emb_dim, n_channels)

        # output projection
        self.output_proj = nn.Conv1d(n_channels, 1, kernel_size=1)

    def forward(self, x_t, t, mask):
        """
        x_t: [B, snps]
        mask: [B, snps]
        t: [B]
        """
        B, S = x_t.shape

        # prepare input: [B, 2, snps] (haplotype + mask)
        x = torch.stack([x_t, mask.float()], dim=1)

        # input projection
        x = self.input_proj(x)  # [B, n_channels, snps]

        # sinusoidal timestep embedding
        t_emb = self.t_emb(t)   # [B, t_emb_dim]
        t_emb = self.t_proj(t_emb).unsqueeze(-1)  # [B, n_channels, 1]

        # add timestep embedding to all positions
        x = x + t_emb

        # residual blocks
        for block in self.res_blocks:
            x = block(x)

        # output projection
        x = self.output_proj(x).squeeze(1)  # [B, snps]
        return x

class ConditionalDiscreteDiffusionModel:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.1, device="cpu"):
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

    def q_sample(self, x_0, t, mask):
        """
        Vectorized noise for masked SNPs.
        x_0: [B, snps]
        t: [B]
        mask: [B, snps] (1=observed, 0=masked)
        """
        beta_t = self.betas[t].view(-1, 1)  # [B, 1]
        x_t = x_0.clone()

        # Only apply noise to masked positions
        prob_1 = (1 - beta_t) * x_0 + beta_t * (1 - x_0)
        x_t = mask * x_0 + (1 - mask) * torch.bernoulli(prob_1)
        return x_t

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

def train_masked_conditional(
    model,
    diffusion,
    train_dataset,
    val_dataset=None,
    optimizer=None,
    epochs=100,
    batch_size=256,
    patience=5,
    val_index_file=None,
    scheduler=None,
    device="cpu",
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    snps = train_dataset.tensors[0].shape[1]
    best_val_loss = float("inf")
    patience_counter = 0

    # fixed mask indices for validation
    val_mask_indices = None
    if val_index_file is not None:
        val_mask_indices = np.loadtxt(val_index_file, dtype=int)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for (x_0,) in train_loader:
            x_0 = x_0.to(device)

            # per-sample random mask (vectorized)
            mask = torch.ones_like(x_0, device=device)
            n_mask = int(0.5 * snps)
            for i in range(x_0.size(0)):
                mask_indices = torch.randperm(snps, device=device)[:n_mask]
                mask[i, mask_indices] = 0

            # forward diffusion
            t = diffusion.sample_timesteps(x_0.size(0))
            x_t = diffusion.q_sample(x_0, t, mask)

            # compute BCE loss only on masked positions
            logits = model(x_t, t, mask)
            loss = F.binary_cross_entropy_with_logits(logits, x_0, weight=(1 - mask))
            # loss = weighted_bce_loss_with_logits(logits, x_0, mask, maf_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item() * x_0.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        val_loss = None
        if val_loader and val_mask_indices is not None:
            model.eval()
            total_val_loss = 0.0
            mask = torch.ones(snps, device=device)
            mask[val_mask_indices] = 0

            with torch.no_grad():
                for (x_val,) in val_loader:
                    x_val = x_val.to(device)
                    t = diffusion.sample_timesteps(x_val.size(0))
                    mask_batch = mask.unsqueeze(0).expand(x_val.size(0), -1)

                    x_t = diffusion.q_sample(x_val, t, mask_batch)
                    logits = model(x_t, t, mask_batch)

                    loss = F.binary_cross_entropy_with_logits(logits, x_val, weight=(1 - mask_batch))
                    # loss = weighted_bce_loss_with_logits(logits, x_val, mask_batch, maf_weights)

                    total_val_loss += loss.item() * x_val.size(0)

            val_loss = total_val_loss / len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    return epoch + 1

        logging.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}"
            + (f" | Val Loss: {val_loss:.4f}" if val_loss is not None else "")
        )

    return epochs

def compute_squared_pearson_correlation(x_true, x_pred):
    vx = x_true - x_true.mean()
    vy = x_pred - x_pred.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr ** 2

@torch.no_grad()
def evaluate_masked_r2_reverse_diffusion(
    model,
    diffusion,
    x_data,
    index_file,
    device,
    output_path=None,
    batch_size=1024,
):
    """
    Conditional imputation using multi-step reverse diffusion.
    Mask SNPs specified in index_file, initialize them with random noise,
    then iteratively denoise while keeping observed sites fixed.
    """
    model.eval()
    x_data = x_data.to(device)
    samples, snps = x_data.shape

    # Load fixed indices to mask
    mask_indices = np.loadtxt(index_file, dtype=int)
    logging.info(f"Masking {len(mask_indices)} SNPs: {mask_indices}")

    # Mask vector: 1=observed, 0=masked
    mask_vec = torch.ones(snps, device=device)
    mask_vec[mask_indices] = 0
    mask_full = mask_vec.unsqueeze(0).expand(samples, -1)

    preds = torch.zeros_like(x_data, device=device)

    for start in range(0, samples, batch_size):
        end = min(start + batch_size, samples)
        x_batch = x_data[start:end]
        mask_batch = mask_full[start:end]

        # Initialize masked positions with random noise
        x_t = x_batch.clone()
        rand_noise = torch.bernoulli(0.5 * torch.ones_like(x_t))
        x_t = mask_batch * x_t + (1 - mask_batch) * rand_noise

        # Reverse diffusion: iterate from last timestep to t=0
        for t_inv in reversed(range(diffusion.num_timesteps)):
            t_tensor = torch.full((x_t.size(0),), t_inv, dtype=torch.long, device=device)
            logits = model(x_t, t_tensor, mask_batch)
            x_pred = torch.sigmoid(logits)

            # Update only masked positions
            x_t = mask_batch * x_batch + (1 - mask_batch) * x_pred

        preds[start:end] = x_t

    # Compute per-SNP R² for masked features
    r2_list = []
    for feat_idx in mask_indices:
        true_vals = x_data[:, feat_idx]
        pred_vals = preds[:, feat_idx]
        r2 = compute_squared_pearson_correlation(true_vals, pred_vals)
        r2_list.append(r2.item())

    avg_r2 = float(np.mean(r2_list))
    logging.info(f"Average R² across masked SNPs: {avg_r2:.6f}")

    if output_path:
        df = pd.DataFrame({"SNP_Index": mask_indices, "R2": r2_list})
        df.to_csv(output_path, index=False, float_format="%.8f")
        logging.info(f"Saved SNP-wise R² to {output_path}")

    return avg_r2

if __name__ == "__main__":

    # --- Logging ---
    log_path = "/scratch2/prateek/diffusion/train2.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting training...")

    # --- Setup ---
    torch.manual_seed(0)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # --- Load training data ---
    file_path = "/scratch2/prateek/diffusion/data/1KG/8020_train.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    x_data_np = np.loadtxt(file_path, dtype=np.float32)
    x_data = torch.tensor(x_data_np, dtype=torch.float32)
    samples, snps = x_data.shape
    logging.info(f"Loaded train matrix: {samples} samples × {snps} SNPs")

    maf_weights = load_maf_weights("/scratch2/prateek/genetic_pc_github/aux/10K_legend.maf.txt").to(device)
    print(max(maf_weights))

    # --- Split train/val ---
    x_train, x_val = train_test_split(x_data, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)

    diffusion = ConditionalDiscreteDiffusionModel(device=device)
    reverse_model = ConditionalReverseConvModel(snps=snps).to(device)
    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=1e-3)

    # Scheduler: warmup 10 epochs, cosine decay afterwards
    total_epochs = 1000
    batch_size = 64
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=50, total_epochs=total_epochs)

    # --- Train with early stopping ---
    trained_epochs = train_masked_conditional(
        reverse_model,
        diffusion,
        train_dataset,
        val_dataset,
        optimizer=optimizer,
        epochs=total_epochs,
        batch_size=batch_size,
        patience=5,
        val_index_file="/scratch2/prateek/diffusion/snp_index_file.txt",
        scheduler=scheduler,
        device=device
    )

    # trained_epochs = 226
    logging.info(f"Training stopped after {trained_epochs} epochs (early stopping).")

    # Load best model
    reverse_model.load_state_dict(torch.load("best_model.pt"))

    # --- Fine-tune on full dataset ---
    fine_tune_epochs = max(3, min(int(0.1 * trained_epochs), 20))
    logging.info(f"Fine-tuning on full dataset for {fine_tune_epochs} epochs with cosine LR schedule...")

    full_dataset = TensorDataset(torch.cat([x_train, x_val], dim=0))

    # Fine-tuning: small LR and short warmup
    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=1e-4)
    scheduler_ft = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=2,
        total_epochs=fine_tune_epochs
    )

    train_masked_conditional(
        reverse_model,
        diffusion,
        full_dataset,
        val_dataset=None,
        optimizer=optimizer,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        scheduler=scheduler_ft,
        device=device
    )

    # --- Load test data ---
    x_test = torch.tensor(
        np.loadtxt("/scratch2/prateek/diffusion/data/1KG/8020_test.txt"),
        dtype=torch.float32,
    )
    samples_test, snps_test = x_test.shape
    logging.info(f"Loaded test matrix: {samples_test} samples × {snps_test} SNPs")

    # --- Evaluate using index file ---
    avg_r2 = evaluate_masked_r2_reverse_diffusion(
        model=reverse_model,
        diffusion=diffusion,
        x_data=x_test,
        index_file="/scratch2/prateek/diffusion/snp_index_file.txt",
        device=device,
        output_path="/scratch2/prateek/diffusion/diff_results.csv",
        batch_size=batch_size
    )

    logging.info(f"✅ Final Average R²: {avg_r2:.6f}")