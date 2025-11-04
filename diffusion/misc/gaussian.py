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
    
EPS = 1e-5
def logit(x):
    x = x.clamp(EPS, 1 - EPS)
    return torch.log(x) - torch.log1p(-x)

def sigmoid(x):
    return torch.sigmoid(x)

def set_seed(seed=0):
    random.seed(seed)                       # Python random
    np.random.seed(seed)                    # NumPy random
    torch.manual_seed(seed)                 # PyTorch CPU
    torch.cuda.manual_seed(seed)            # PyTorch GPU
    torch.cuda.manual_seed_all(seed)        # If multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    initial_lr = optimizer.param_groups[0]["lr"]
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / initial_lr, cosine_decay)
    return LambdaLR(optimizer, lr_lambda)
    
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
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, n_channels),
        )

        # output projection
        self.output_proj = nn.Conv1d(n_channels, 1, kernel_size=1)

    def forward(self, z_t, t, mask):
        """
        z_t: [B, snps]
        mask: [B, snps]
        t: [B]
        """
        B, S = z_t.shape

        # prepare input: [B, 2, snps] (haplotype + mask)
        x = torch.stack([z_t, mask.float()], dim=1)

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
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.999, device="cpu"):
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

    def q_sample(self, x_0, t, mask):
        """
        Continuous noise model for SNP dosages in [0,1].
        """
        beta_t = self.betas[t].view(-1, 1)  # [B,1]
        noise = torch.rand_like(x_0)        # Uniform(0,1)
        x_t = (1 - beta_t) * x_0 + beta_t * noise
        # observed sites stay fixed
        x_t = mask * x_0 + (1 - mask) * x_t
        return x_t

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

class GaussianLogitDiffusion:
    def __init__(self, num_timesteps=500, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, mask):
        """
        Forward diffusion: sample z_t given x0 (in [0,1]) and timestep t.
        """
        z0 = logit(x0)  # go to logit space
        noise = torch.randn_like(z0)

        a_bar = self.alpha_bars[t].view(-1, 1)
        zt = torch.sqrt(a_bar) * z0 + torch.sqrt(1 - a_bar) * noise

        # keep observed positions fixed
        # z_obs = logit(x0)
        # zt = mask * z_obs + (1 - mask) * zt

        return zt, noise

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
    val_index_file=None,
    scheduler=None,
    device="cpu",
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    snps = train_dataset.tensors[0].shape[1]
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # fixed validation mask if provided
    val_mask_indices = None
    if val_index_file is not None:
        val_mask_indices = np.loadtxt(val_index_file, dtype=int)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for (x_0,) in train_loader:
            x_0 = x_0.to(device)

            # --- create random per-sample mask (15% missing) ---
            mask = torch.ones_like(x_0, device=device)
            n_mask = int(0.15 * snps)
            rand_idx = torch.argsort(torch.rand(x_0.size(0), snps, device=device), dim=1)
            mask.scatter_(1, rand_idx[:, :n_mask], 0)

            # --- forward diffusion ---
            t = diffusion.sample_timesteps(x_0.size(0)).to(device)
            z_t, noise = diffusion.q_sample(x_0, t, mask)
            z_t = mask * logit(x_0) + (1 - mask) * z_t

            # --- predict noise ---
            eps_pred = model(z_t, t, mask)
            loss = F.mse_loss(eps_pred, noise, reduction="none")
            loss = (loss * (1 - mask)).sum() / (1 - mask).sum()

            # --- update ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_0.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # --- Validation ---
        val_loss = None
        # if val_loader and val_mask_indices is not None:
        #     model.eval()
        #     total_val_loss = 0.0
        #     mask = torch.ones(snps, device=device)
        #     mask[val_mask_indices] = 0

        #     with torch.no_grad():
        #         for (x_val,) in val_loader:
        #             x_val = x_val.to(device)
        #             t = diffusion.sample_timesteps(x_val.size(0)).to(device)
        #             mask_batch = mask.unsqueeze(0).expand(x_val.size(0), -1)

        #             z_t, noise = diffusion.q_sample(x_val, t, mask_batch)
        #             z_t = mask_batch * logit(x_val) + (1 - mask_batch) * z_t
        #             eps_pred = model(z_t, t, mask_batch)
        #             loss = F.mse_loss(eps_pred, noise, reduction="none")
        #             loss = (loss * (1 - mask_batch)).sum() / (1 - mask_batch).sum()

        #             total_val_loss += loss.item() * x_val.size(0)

        #     val_loss = total_val_loss / len(val_loader.dataset)

        if val_loader:
            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for (x_val,) in val_loader:
                    x_val = x_val.to(device)

                    # Use same masking style as training (optional)
                    mask = torch.ones_like(x_val)
                    n_mask = int(0.15 * snps)
                    rand_idx = torch.argsort(torch.rand(x_val.size(0), snps, device=device), dim=1)
                    mask.scatter_(1, rand_idx[:, :n_mask], 0)

                    t = diffusion.sample_timesteps(x_val.size(0)).to(device)
                    z_t, noise = diffusion.q_sample(x_val, t, mask)
                    z_t = mask * logit(x_val) + (1 - mask) * z_t

                    eps_pred = model(z_t, t, mask)
                    loss = F.mse_loss(eps_pred, noise, reduction="none")
                    loss = (loss * (1 - mask)).sum() / (1 - mask).sum()

                    total_val_loss += loss.item() * x_val.size(0)

            val_loss = total_val_loss / len(val_loader.dataset)

            # --- checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), "best_model_gauss.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    return best_epoch

        # --- step scheduler once per epoch ---
        if scheduler is not None:
            scheduler.step()

        logging.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}"
            + (f" | Val Loss: {val_loss:.4f}" if val_loss is not None else "")
        )

    return best_epoch

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

        # initialize masked with Gaussian noise in z-space
        z_obs = logit(x_batch)
        z_t = torch.randn_like(z_obs)
        z_t = mask_batch * z_obs + (1 - mask_batch) * z_t

        # reverse diffusion
        for t_inv in reversed(range(diffusion.num_timesteps)):
            t_tensor = torch.full((z_t.size(0),), t_inv, dtype=torch.long, device=device)

            eps_pred = model(z_t, t_tensor, mask_batch)

            beta_t = diffusion.betas[t_inv]
            alpha_t = diffusion.alphas[t_inv]
            alpha_bar_t = diffusion.alpha_bars[t_inv]

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            z_mean = coef1 * (z_t - coef2 * eps_pred)

            if t_inv > 0:
                z_t = z_mean + torch.sqrt(beta_t) * torch.randn_like(z_t)
            else:
                z_t = z_mean

            # reinsert observed sites
            z_t = mask_batch * z_obs + (1 - mask_batch) * z_t

        x_hat = sigmoid(z_t)
        preds[start:end] = x_hat

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

    # --- Set seeds for reproducibility ---
    set_seed(42)
    
    # --- Logging ---
    log_path = "train_gauss.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting training...")

    # --- Setup ---
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load training data ---
    file_path = "../data/1KG/8020_train.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    x_data_np = np.loadtxt(file_path, dtype=np.float32)
    x_data = torch.tensor(x_data_np, dtype=torch.float32)
    samples, snps = x_data.shape
    logging.info(f"Loaded train matrix: {samples} samples × {snps} SNPs")

    # maf_weights = load_maf_weights("/scratch2/prateek/genetic_pc_github/aux/10K_legend.maf.txt").to(device)
    # print(max(maf_weights))

    # --- Split train/val ---
    x_train, x_val = train_test_split(x_data, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)

    diffusion = GaussianLogitDiffusion(num_timesteps=1000, beta_start=1e-4, beta_end=0.05, device=device)
    reverse_model = ConditionalReverseConvModel(snps=snps, n_channels=128, n_blocks=6).to(device)

    total_epochs = 200
    warmup_epochs = 20
    patience = 10
    lr = 2e-3
    batch_size = 128

    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=warmup_epochs, total_epochs=total_epochs, min_lr=1e-5)

    # --- Train with early stopping ---
    trained_epochs = train_masked_conditional(
        reverse_model,
        diffusion,
        train_dataset,
        val_dataset,
        optimizer=optimizer,
        epochs=total_epochs,
        batch_size=batch_size,
        patience=patience,
        val_index_file="../index_files/snp_index_file.txt",
        scheduler=scheduler,
        device=device
    )

    logging.info(f"Training stopped, best model found at {trained_epochs} epochs.")

    # --- Retrain on full dataset for best number of epochs ---
    full_dataset = TensorDataset(torch.cat([x_train, x_val], dim=0))
    reverse_model = ConditionalReverseConvModel(snps=snps, n_channels=128, n_blocks=6).to(device)  # reinit model

    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=lr)
    scheduler_ft = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=warmup_epochs, total_epochs=trained_epochs, min_lr=1e-5)

    logging.info(f"Retraining on full dataset for {trained_epochs} epochs...")
    train_masked_conditional(
        reverse_model,
        diffusion,
        full_dataset,
        val_dataset=None,   # no validation now
        optimizer=optimizer,
        epochs=trained_epochs,
        batch_size=batch_size,
        scheduler=scheduler_ft,
        device=device
    )

    # --- Load test data ---
    x_test = torch.tensor(
        np.loadtxt("../data/1KG/8020_test.txt"),
        dtype=torch.float32,
    )
    samples_test, snps_test = x_test.shape
    logging.info(f"Loaded test matrix: {samples_test} samples × {snps_test} SNPs")

    # --- Evaluate using index file ---
    avg_r2 = evaluate_masked_r2_reverse_diffusion(
        model=reverse_model,
        diffusion=diffusion,
        x_data=x_test,
        index_file="../index_files/snp_index_file.txt",
        device=device,
        output_path="diff_results_gauss.csv",
        batch_size=batch_size
    )

    logging.info(f"✅ Final Average R²: {avg_r2:.6f}")