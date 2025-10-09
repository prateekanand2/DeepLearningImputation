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
import random

def load_coarse_map(path):
    bp, chr_, cm = np.loadtxt(path, dtype=float, usecols=(0, 1, 2), unpack=True)
    # ensure sorted by bp
    order = np.argsort(bp)
    return bp[order], cm[order]

# -----------------------------
# Step 2: Load SNP bp positions
# -----------------------------
def load_snp_positions(path):
    bps = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            token = line.split()[0]   # e.g. "15:27379578"
            bp = int(token.split(":")[1])
            bps.append(bp)
    return np.array(bps, dtype=np.int64)

# -----------------------------
# Step 3: Interpolate cM positions for SNPs
# -----------------------------
def interpolate_cM_for_snps(snp_bps, map_bps, map_cms):
    # linear interpolation, extrapolate linearly using edge slopes
    cms_interp = np.interp(snp_bps, map_bps, map_cms)
    left_mask = snp_bps < map_bps[0]
    right_mask = snp_bps > map_bps[-1]
    if left_mask.any():
        slope = (map_cms[1] - map_cms[0]) / (map_bps[1] - map_bps[0])
        cms_interp[left_mask] = map_cms[0] + slope * (snp_bps[left_mask] - map_bps[0])
    if right_mask.any():
        slope = (map_cms[-1] - map_cms[-2]) / (map_bps[-1] - map_bps[-2])
        cms_interp[right_mask] = map_cms[-1] + slope * (snp_bps[right_mask] - map_bps[-1])
    return cms_interp

# -----------------------------
# Step 4: Compute tau (and log tau)
# -----------------------------
def compute_tau(cM_positions, Ne=10000, H=1000):
    cM = np.asarray(cM_positions, dtype=np.float64)
    d_cM = np.diff(cM)             # difference in cM between adjacent SNPs
    d_M = d_cM / 100.0             # convert to Morgans
    d_M[d_M < 0] = 0.0             # guard against rounding issues
    tau = 1.0 - np.exp(-4.0 * Ne * d_M / H)
    tau_full = np.concatenate([[0.0], tau])
    log_tau_full = np.log(tau_full + 1e-12)
    return torch.tensor(tau_full, dtype=torch.float32), torch.tensor(log_tau_full, dtype=torch.float32)

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

def set_seed(seed=0):
    random.seed(seed)                       # Python random
    np.random.seed(seed)                    # NumPy random
    torch.manual_seed(seed)                 # PyTorch CPU
    torch.cuda.manual_seed(seed)            # PyTorch GPU
    torch.cuda.manual_seed_all(seed)        # If multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


# class ConditionalReverseConvModel(nn.Module):
#     """
#     Conditional reverse model for haplotype diffusion using 1D CNNs + residual blocks.
#     Input: [batch, snps] noisy haplotype + mask
#     """
#     def __init__(self, snps, n_channels=256, n_blocks=8, t_emb_dim=128):
#         super().__init__()
#         self.snps = snps
#         self.t_emb = SinusoidalPosEmb(t_emb_dim)

#         # initial projection: haplotype + mask
#         self.input_proj = nn.Conv1d(2, n_channels, kernel_size=3, padding=1)

#         # residual blocks
#         self.res_blocks = nn.ModuleList([
#             ResidualConvBlock(n_channels, kernel_size=5, dilation=2**i) for i in range(n_blocks)
#         ])

#         # time embedding projection
#         self.t_proj = nn.Linear(t_emb_dim, n_channels)

#         # output projection
#         self.output_proj = nn.Conv1d(n_channels, 1, kernel_size=1)

#     def forward(self, x_t, t, mask):
#         """
#         x_t: [B, snps]
#         mask: [B, snps]
#         t: [B]
#         """
#         B, S = x_t.shape

#         # prepare input: [B, 2, snps] (haplotype + mask)
#         x = torch.stack([x_t, mask.float()], dim=1)

#         # input projection
#         x = self.input_proj(x)  # [B, n_channels, snps]

#         # sinusoidal timestep embedding
#         t_emb = self.t_emb(t)   # [B, t_emb_dim]
#         t_emb = self.t_proj(t_emb).unsqueeze(-1)  # [B, n_channels, 1]

#         # add timestep embedding to all positions
#         x = x + t_emb

#         # residual blocks
#         for block in self.res_blocks:
#             x = block(x)

#         # output projection
#         x = self.output_proj(x).squeeze(1)  # [B, snps]
#         return x

class ConditionalReverseConvModel(nn.Module):
    def __init__(self, snps, n_channels=256, n_blocks=8, t_emb_dim=128):
        super().__init__()
        self.snps = snps
        self.t_emb = SinusoidalPosEmb(t_emb_dim)

        # (x_t, mask, tau)
        self.input_proj = nn.Conv1d(3, n_channels, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList([
            ResidualConvBlock(n_channels, kernel_size=5, dilation=2**i) for i in range(n_blocks)
        ])
        self.t_proj = nn.Linear(t_emb_dim, n_channels)
        self.output_proj = nn.Conv1d(n_channels, 1, kernel_size=1)

    def forward(self, x_t, t, mask, tau):
        """
        x_t: [B, snps]
        mask: [B, snps]
        tau: [snps] or [B, snps]  (transition priors)
        """
        B, S = x_t.shape
        if tau.dim() == 1:
            tau = tau.unsqueeze(0).expand(B, -1)

        # [B, 3, snps]
        x = torch.stack([x_t, mask.float(), tau], dim=1)

        x = self.input_proj(x)
        t_emb = self.t_emb(t)
        t_emb = self.t_proj(t_emb).unsqueeze(-1)
        x = x + t_emb

        for block in self.res_blocks:
            x = block(x)

        return self.output_proj(x).squeeze(1)

class ConditionalDiscreteDiffusionModel:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.999, device="cpu"):
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
        # prob_1 = (1 - beta_t) * x_0 + beta_t * (1 - x_0)
        prob_1 = (1 - beta_t) * x_0 + 0.5 * beta_t
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
    patience=10,
    mask_ratio=0.15,
    val_index_file=None,
    scheduler=None,
    device="cpu",
    tau_tensor=None,
    val_seed=1234  # ✅ ensures deterministic validation masking
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    snps = train_dataset.tensors[0].shape[1]
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # --- Load fixed validation mask indices (if provided) ---
    val_mask_indices = None
    if val_index_file is not None and os.path.exists(val_index_file):
        val_mask_indices = np.loadtxt(val_index_file, dtype=int)

    # --- Pre-generate deterministic validation mask if no file provided ---
    fixed_val_mask = None
    if val_loader and val_mask_indices is None:
        logging.info("ℹ️ No val_index_file provided — using deterministic random mask for validation.")
        rng = torch.Generator(device=device)
        rng.manual_seed(val_seed)

        # generate mask for a single validation batch shape (will be reused)
        first_val_sample_count = val_dataset.tensors[0].shape[0]
        fixed_val_mask = torch.ones((first_val_sample_count, snps), device=device)
        n_mask_val = int(mask_ratio * snps)
        for i in range(first_val_sample_count):
            mask_indices = torch.randperm(snps, generator=rng, device=device)[:n_mask_val]
            fixed_val_mask[i, mask_indices] = 0

    for epoch in range(epochs):
        # ====================================================
        # 🟢 TRAINING
        # ====================================================
        model.train()
        total_loss = 0.0

        for (x_0,) in train_loader:
            x_0 = x_0.to(device)
            B = x_0.size(0)

            # --- per-sample random mask ---
            mask = torch.ones_like(x_0, device=device)
            n_mask = int(mask_ratio * snps)
            for i in range(B):
                mask_indices = torch.randperm(snps, device=device)[:n_mask]
                mask[i, mask_indices] = 0

            # --- forward diffusion ---
            t = diffusion.sample_timesteps(B)
            x_t = diffusion.q_sample(x_0, t, mask)
            tau_batch = tau_tensor.to(device)

            logits = model(x_t, t, mask, tau_batch)

            loss = F.binary_cross_entropy_with_logits(logits, x_0, weight=(1 - mask))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B

        avg_loss = total_loss / len(train_loader.dataset)

        # ====================================================
        # 🔵 VALIDATION
        # ====================================================
        val_loss = None
        if val_loader:
            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for batch_idx, (x_val,) in enumerate(val_loader):
                    x_val = x_val.to(device)
                    B_val = x_val.size(0)

                    if val_mask_indices is not None:
                        # --- fixed mask from file ---
                        mask = torch.ones(snps, device=device)
                        mask[val_mask_indices] = 0
                        mask_batch = mask.unsqueeze(0).expand(B_val, -1)
                    else:
                        # --- deterministic random mask ---
                        mask_batch = fixed_val_mask[
                            batch_idx * B_val : batch_idx * B_val + B_val
                        ].to(device)

                    # --- forward diffusion ---
                    t_val = diffusion.sample_timesteps(B_val)
                    x_t_val = diffusion.q_sample(x_val, t_val, mask_batch)
                    tau_batch = tau_tensor.to(device)

                    logits_val = model(x_t_val, t_val, mask_batch, tau_batch)
                    loss_val = F.binary_cross_entropy_with_logits(logits_val, x_val, weight=(1 - mask_batch))
                    total_val_loss += loss_val.item() * B_val

            val_loss = total_val_loss / len(val_loader.dataset)

            # --- Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"⏸ Early stopping at epoch {epoch+1}")
                    return best_epoch

        # ====================================================
        # 🟡 SCHEDULER STEP + LOGGING
        # ====================================================
        if scheduler:
            scheduler.step()

        log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}"
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}"
        logging.info(log_msg)

    return best_epoch

def compute_squared_pearson_correlation(x_true, x_pred):
    vx = x_true - x_true.mean()
    vy = x_pred - x_pred.mean()

    denom = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
    corr = torch.sum(vx * vy) / denom
    corr_sq = corr ** 2

    if torch.isnan(corr_sq):
        return torch.tensor(0.0, device=x_true.device)

    return corr_sq

@torch.no_grad()
def evaluate_masked_r2_reverse_diffusion(
    model,
    diffusion,
    x_data,
    index_file,
    device,
    output_path=None,
    batch_size=1024,
    tau_tensor=None
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

        tau_batch = tau_tensor.to(device)

        for t_inv in reversed(range(diffusion.num_timesteps)):
            t_tensor = torch.full((x_t.size(0),), t_inv, dtype=torch.long, device=device)
            logits = model(x_t, t_tensor, mask_batch, tau_batch)
        # Reverse diffusion: iterate from last timestep to t=0
        # for t_inv in reversed(range(diffusion.num_timesteps)):
        #     t_tensor = torch.full((x_t.size(0),), t_inv, dtype=torch.long, device=device)
        #     logits = model(x_t, t_tensor, mask_batch)
            x_pred = torch.sigmoid(logits)

            # Stochastic update: sample masked positions
            masked_noise = torch.bernoulli(x_pred)
            x_t = mask_batch * x_batch + (1 - mask_batch) * masked_noise

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
    # ============================================================
    # 🧪 HYPERPARAMETERS
    # ============================================================
    # --- Paths ---
    DATA_DIR = "../data/1KG"
    TRAIN_FILE = f"{DATA_DIR}/8020_train.txt"
    TEST_FILE = f"{DATA_DIR}/8020_test.txt"
    INDEX_FILE = "../index_files/snp_index_file.txt"
    MAP_FILE = "/scratch2/prateek/DeepLearningImputation/1kg_15_map.txt"
    LEGEND_FILE = "/scratch2/prateek/genetic_pc_github/aux/10K_legend.maf.txt"

    LOG_PATH = "logs/train_bern.log"
    OUTPUT_RESULTS = "results/diff_results.csv"

    # --- Training ---
    SEED = 42
    LR = 1e-3
    BATCH_SIZE = 32
    TOTAL_EPOCHS = 300
    WARMUP_EPOCHS = 30
    PATIENCE = 10
    VAL_SPLIT = 0.1
    MASK_RATIO = 0.15

    # --- Genetic Map / Tau ---
    Ne = 10000
    H = 4006

    # --- Evaluation ---
    EVAL_BATCH_SIZE = 128
    # ============================================================

    # --- Set seeds for reproducibility ---
    set_seed(SEED)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting training...")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    # --- Load training data ---
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"File not found: {TRAIN_FILE}")

    x_data_np = np.loadtxt(TRAIN_FILE, dtype=np.float32)
    x_data = torch.tensor(x_data_np, dtype=torch.float32)
    samples, snps = x_data.shape
    logging.info(f"Loaded train matrix: {samples} samples × {snps} SNPs")

    # --- Train/val split ---
    x_train, x_val = train_test_split(x_data, test_size=VAL_SPLIT, random_state=SEED)
    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)

    # --- Diffusion + Model ---
    diffusion = ConditionalDiscreteDiffusionModel(device=device)
    reverse_model = ConditionalReverseConvModel(snps=snps).to(device)
    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=LR)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=TOTAL_EPOCHS
    )

    # --- Genetic map and tau ---
    map_bps, map_cms = load_coarse_map(MAP_FILE)
    snp_bps = load_snp_positions(LEGEND_FILE)
    interp_cms = interpolate_cM_for_snps(snp_bps, map_bps, map_cms)
    tau_tensor, log_tau_tensor = compute_tau(interp_cms, Ne=Ne, H=H)

    # --- Initial Training (with early stopping) ---
    trained_epochs = train_masked_conditional(
        reverse_model,
        diffusion,
        train_dataset,
        val_dataset,
        optimizer=optimizer,
        epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        mask_ratio=MASK_RATIO,
        val_index_file=None,
        scheduler=scheduler,
        device=device,
        tau_tensor=log_tau_tensor
    )
    logging.info(f"Training stopped, best model found at {trained_epochs} epochs.")

    # --- Retrain on full dataset ---
    logging.info(f"Retraining on full dataset for {trained_epochs} epochs...")
    full_dataset = TensorDataset(torch.cat([x_train, x_val], dim=0))
    reverse_model = ConditionalReverseConvModel(snps=snps).to(device)  # re-init model
    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=LR)
    scheduler_ft = get_cosine_schedule_with_warmup(
        optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=trained_epochs
    )

    train_masked_conditional(
        reverse_model,
        diffusion,
        full_dataset,
        val_dataset=None,  # no validation
        optimizer=optimizer,
        epochs=trained_epochs,
        batch_size=BATCH_SIZE,
        mask_ratio=MASK_RATIO,
        val_index_file=None,
        scheduler=scheduler_ft,
        device=device,
        tau_tensor=log_tau_tensor
    )

    # --- Load test data ---
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"File not found: {TEST_FILE}")
    x_test = torch.tensor(np.loadtxt(TEST_FILE), dtype=torch.float32)
    samples_test, snps_test = x_test.shape
    logging.info(f"Loaded test matrix: {samples_test} samples × {snps_test} SNPs")

    # --- Evaluate ---
    avg_r2 = evaluate_masked_r2_reverse_diffusion(
        model=reverse_model,
        diffusion=diffusion,
        x_data=x_test,
        index_file=INDEX_FILE,
        device=device,
        output_path=OUTPUT_RESULTS,
        batch_size=EVAL_BATCH_SIZE,
        tau_tensor=log_tau_tensor
    )
    logging.info(f"✅ Final Average R²: {avg_r2:.6f}")