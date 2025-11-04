import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import logging
import math
import random
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

# ------------------ Utilities ------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def int2bit(x, scale=1.0):
    return (x * 2.0 - 1.0) * scale  # 0/1 -> -scale/+scale

def bit2int(x):
    return (x + 1.0) / 2.0  # -1/+1 -> 0/1

# def gamma_schedule(t, ns=0.0002, ds=0.00025):
#     return torch.cos(((t + ns) / (1 + ds)) * (math.pi / 2)) ** 2

def gamma_schedule(t):
    return torch.cos((t * math.pi / 2)) ** 4

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / max(1, warmup_epochs)
        progress = float(current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / optimizer.param_groups[0]["lr"], cosine_decay)
    return LambdaLR(optimizer, lr_lambda)

# ------------------ Model ------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * (-math.log(10000) / half_dim))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=t.device)], dim=1)
        return emb

class ResidualConvBlock(nn.Module):
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

# ------------------ Model ------------------
class ConditionalReverseConvModel(nn.Module):
    def __init__(self, snps, n_channels=256, n_blocks=8, t_emb_dim=128):
        super().__init__()
        self.snps = snps
        self.t_emb = SinusoidalPosEmb(t_emb_dim)
        self.input_proj = nn.Conv1d(2, n_channels, kernel_size=3, padding=1)  # x + mask
        self.res_blocks = nn.ModuleList([ResidualConvBlock(n_channels, kernel_size=5, dilation=2**i) for i in range(n_blocks)])
        self.t_proj = nn.Sequential(nn.Linear(t_emb_dim, n_channels), nn.SiLU(), nn.Linear(n_channels, n_channels))
        # Removed Tanh to prevent saturation
        self.output_proj = nn.Conv1d(n_channels, 1, kernel_size=1)

    def forward(self, x, t, mask):
        B = x.shape[0]
        x_in = torch.cat([x.unsqueeze(1), mask.unsqueeze(1)], dim=1)  # [B,2,snps]
        x = self.input_proj(x_in)
        t_emb = self.t_proj(self.t_emb(t)).unsqueeze(-1)
        x = x + t_emb
        for block in self.res_blocks:
            x = block(x)
        out = self.output_proj(x).squeeze(1)  # [-inf, +inf] logits
        return out

# ------------------ Diffusion ------------------
class BitDiffusion:
    def __init__(self, device="cpu"):
        self.device = device
    def q_sample(self, x_bits, t, mask):
        gam = gamma_schedule(t).view(-1,1)  # [B,1]
        gam = gam.to(x_bits.device)
        eps = torch.randn_like(x_bits)
        x_crpt = x_bits * mask + (torch.sqrt(gam) * x_bits + torch.sqrt(1 - gam) * eps) * (1 - mask)
        return x_crpt, eps

# safe/ddim & generate replacement ------------------------------------------------
EPS_GAMMA = 1e-8

@torch.no_grad()
def ddim_step_safe(x_t, x_pred, t_now, t_next):
    """
    Safe DDIM step: handles tiny/zero gammas robustly.
    x_t, x_pred: [B, snps]
    t_now, t_next: [B]
    """
    # compute and clamp gammas
    gamma_now = gamma_schedule(t_now).view(-1, 1).to(x_t.device).clamp(min=EPS_GAMMA, max=1.0)
    gamma_next = gamma_schedule(t_next).view(-1, 1).to(x_t.device).clamp(min=EPS_GAMMA, max=1.0)

    one_minus_gamma_now = torch.clamp(1.0 - gamma_now, min=0.0)
    one_minus_gamma_next = torch.clamp(1.0 - gamma_next, min=0.0)

    # compute eps estimate (safe denominator)
    denom = torch.sqrt(one_minus_gamma_now)
    denom = torch.clamp(denom, min=1e-12)   # avoid division by zero / tiny
    eps = (x_t - torch.sqrt(gamma_now) * x_pred) / denom

    x_next = torch.sqrt(gamma_next) * x_pred + torch.sqrt(one_minus_gamma_next) * eps
    return x_next


@torch.no_grad()
def generate_bit_diffusion(model, diffusion, x_obs, mask, steps=50, eta=0.0, device="cpu", debug=False):
    """
    Stable noise-prediction DDIM sampler for 1-bit diffusion (deterministic by default).

    Args:
        model: trained ConditionalReverseConvModel (predicts noise eps)
        diffusion: BitDiffusion instance (provides q_sample/gamma)
        x_obs: [B, snps] observed ints (0/1)
        mask:  [B, snps] floats (1=observed, 0=masked)
        steps: number of DDIM steps
        eta: stochastic scale (0 -> deterministic). If you want stochastic, use a tiny eta and be careful.
        debug: if True prints/raises when NaNs are detected.

    Returns:
        probs: [B, snps] predicted P(snp=1) in [0,1]
        x0_final: [B, snps] analog values in [-1,1] (reconstructed x0)
    """
    device = device if isinstance(device, torch.device) else torch.device(device)
    x_obs = x_obs.to(device)
    mask = mask.to(device).float()

    B, snps = x_obs.shape
    # convert observed integers to bipolar bits in [-1,1]
    x_obs_bits = int2bit(x_obs).to(device)

    # initialize x_t: observed positions = true bits, masked positions = noise
    x_t = torch.randn_like(x_obs_bits, device=device) * (1.0 - mask) + x_obs_bits * mask

    for step in range(steps):
        # times
        t_now = torch.full((B,), 1.0 - float(step) / float(steps), device=device)
        t_next_val = max(1.0 - float(step + 1) / float(steps), 0.0)
        t_next = torch.full((B,), t_next_val, device=device)

        # clamp gammas used locally to avoid zeros
        gamma_now = gamma_schedule(t_now).view(-1, 1).to(device).clamp(min=EPS_GAMMA, max=1.0)
        gamma_next = gamma_schedule(t_next).view(-1, 1).to(device).clamp(min=EPS_GAMMA, max=1.0)

        # predict noise eps for current x_t
        eps_pred = model(x_t, t_now, mask)  # model trained to predict eps
        # ensure same dtype/device
        eps_pred = eps_pred.to(device)

        # reconstruct x0 from predicted eps: x0 = (x_t - sqrt(1-gamma)*eps) / sqrt(gamma)
        sqrt_gamma_now = torch.sqrt(gamma_now)
        sqrt_one_minus_gamma_now = torch.sqrt(torch.clamp(1.0 - gamma_now, min=0.0))
        x0_pred = (x_t - sqrt_one_minus_gamma_now * eps_pred) / (sqrt_gamma_now + 1e-12)

        # optionally clamp x0_pred to physical range [-1,1] (prevent explosion)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Deterministic DDIM update (stable)
        # x_{t+1} = sqrt(gamma_next) * x0_pred + sqrt(1 - gamma_next) * eps_pred
        sqrt_gamma_next = torch.sqrt(gamma_next)
        sqrt_one_minus_gamma_next = torch.sqrt(torch.clamp(1.0 - gamma_next, min=0.0))
        x_t_next = sqrt_gamma_next * x0_pred + sqrt_one_minus_gamma_next * eps_pred

        # If you want stochastic DDIM (eta > 0), you must compute sigma safely.
        # For now we support deterministic (eta=0) to avoid fragile ratio math; if eta>0 we clamp safely:
        if eta and eta > 0.0:
            # safe fallback: small gaussian noise scaled by eta (do NOT use unstable closed form)
            noise = torch.randn_like(x_t_next) * (eta * 0.01)
            x_t_next = x_t_next + noise

        # set observed positions back to true values (keep context fixed)
        x_t = x_t_next * (1.0 - mask) + x_obs_bits * mask

        # debug checks
        if debug:
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                # print diagnostics
                logging.error(f"NaN/Inf detected at step {step}")
                logging.error(f"gamma_now min/max: {gamma_now.min().item()}, {gamma_now.max().item()}")
                logging.error(f"x_t min/max: {torch.nanmin(x_t).item()}, {torch.nanmax(x_t).item()}")
                raise RuntimeError("NaN/Inf encountered in generate_bit_diffusion")

    # final step: predict eps and reconstruct final x0
    t_zero = torch.zeros((B,), device=device)
    gamma_zero = gamma_schedule(t_zero).view(-1,1).to(device).clamp(min=EPS_GAMMA, max=1.0)
    eps_final = model(x_t, t_zero, mask)
    sqrt_gamma_zero = torch.sqrt(gamma_zero)
    sqrt_one_minus_gamma_zero = torch.sqrt(torch.clamp(1.0 - gamma_zero, min=0.0))
    x0_final = (x_t - sqrt_one_minus_gamma_zero * eps_final) / (sqrt_gamma_zero + 1e-12)
    x0_final = torch.clamp(x0_final, -1.0, 1.0)

    # map bipolar to probability [0,1]
    probs = torch.clamp(bit2int(x0_final), 0.0, 1.0)

    # final debug:
    if debug:
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            logging.error("Final probs contain NaN/Inf")
            raise RuntimeError("NaN/Inf in final probs")

    return probs, x0_final

@torch.no_grad()
def evaluate_r2(model, diffusion, x_data, index_file, device="cpu", batch_size=128, steps=50):
    model.eval()
    mask_indices = np.loadtxt(index_file, dtype=int)
    preds = torch.zeros_like(x_data, device=device)
    N, snps = x_data.shape
    x_data = x_data.to(device)
    for start in range(0, N, batch_size):
        end = min(start+batch_size, N)
        x_batch = x_data[start:end].to(device)
        B = x_batch.size(0)
        mask = torch.ones(B, snps, device=device)
        mask[:, mask_indices] = 0
        ints_pred, _ = generate_bit_diffusion(model, diffusion, x_batch, mask, steps=steps, device=device)
        preds[start:end] = ints_pred

    r2_list = []
    for idx in mask_indices:
        x_true = x_data[:, idx].float()
        x_pred = preds[:, idx]
        vx = x_true - x_true.mean()
        vy = x_pred - x_pred.mean()
        denom = (torch.sum(vx**2) * torch.sum(vy**2)) + 1e-8

        # print(x_true)
        # print(x_pred)
        # Compute r2 and replace NaNs with 0
        r2 = (torch.sum(vx * vy) ** 2) / denom
        if torch.isnan(r2):
            r2 = torch.tensor(0.0, device=r2.device)
        r2_list.append(r2.item())

    avg_r2 = float(np.mean(r2_list))
    return avg_r2

# ------------------ Training ------------------
def train_bit_diffusion_1bit(model, diffusion, train_dataset, val_dataset=None,
                             optimizer=None, epochs=100, batch_size=64,
                             patience=10, val_index_file=None, scheduler=None,
                             device="cuda"):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    snps = train_dataset.tensors[0].shape[1]

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    val_mask_indices = None
    if val_index_file:
        val_mask_indices = torch.tensor(np.loadtxt(val_index_file, dtype=int), device=device)

    for epoch in range(epochs):
        # ---------------- Training ----------------
        model.train()
        total_train_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            B = x.size(0)
            x_bits = int2bit(x)

            # random mask (15%)
            mask = torch.ones(B, snps, device=device)
            n_mask = max(1, int(0.15*snps))
            rand_idx = torch.rand(B, snps, device=device).argsort(dim=1)[:, :n_mask]
            mask[torch.arange(B).unsqueeze(1), rand_idx] = 0

            t = torch.rand(B, device=device)
            x_crpt, eps = diffusion.q_sample(x_bits, t, mask)

            optimizer.zero_grad()
            # Model input: observed positions as context, masked positions are noisy
            x_input = x_crpt * (1 - mask) + x_bits * mask
            logits = model(x_input, t, mask)

            # Predict noise on masked positions
            loss = (((logits - eps)**2) * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * B

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ---------------- Validation ----------------
        avg_val_loss = None
        if val_loader and val_mask_indices is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for (x_val,) in val_loader:
                    x_val = x_val.to(device)
                    Bv = x_val.size(0)
                    x_bits_val = int2bit(x_val)
                    mask_val = torch.ones(Bv, snps, device=device)
                    mask_val[:, val_mask_indices] = 0
                    t_val = torch.rand(Bv, device=device)
                    x_crpt_val, eps_val = diffusion.q_sample(x_bits_val, t_val, mask_val)
                    x_input_val = x_crpt_val * (1 - mask_val) + x_bits_val * mask_val
                    logits_val = model(x_input_val, t_val, mask_val)
                    loss_val = (((logits_val - eps_val)**2) * (1 - mask_val)).sum() / ((1 - mask_val).sum() + 1e-8)
                    total_val_loss += loss_val.item() * Bv
            avg_val_loss = total_val_loss / len(val_loader.dataset)

        # ---------------- Logging ----------------
        log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}"
        if avg_val_loss is not None:
            log_msg += f" | Val Loss: {avg_val_loss:.6f}"
        logging.info(log_msg)

        # ---------------- Early Stopping ----------------
        if avg_val_loss is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), "checkpoints/best_model_1bit.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    return best_epoch

        if scheduler:
            scheduler.step()

    return best_epoch


# ------------------ Main ------------------
if __name__ == "__main__":
    set_seed(42)

    # ---------------- Logging Setup ----------------
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_1bit.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),   # Log to file
            logging.StreamHandler()          # Log to console
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Load Data ----------------
    x_data = torch.tensor(np.loadtxt("../data/1KG/8020_train.txt"), dtype=torch.float32)
    x_train, x_val = train_test_split(x_data, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)

    snps = x_data.shape[1]
    model = ConditionalReverseConvModel(snps=snps).to(device)
    diffusion = BitDiffusion(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---------------- Train ----------------
    # train_bit_diffusion_1bit(model, diffusion, train_dataset, val_dataset=val_dataset,
    #                          optimizer=optimizer, epochs=100, batch_size=64, patience=10,
    #                          device=device, val_index_file="../index_files/snp_index_file.txt")

    model.load_state_dict(torch.load('/scratch2/prateek/DeepLearningImputation/diffusion/checkpoints/best_model_1bit.pt', map_location=device))
    model.eval()

    # ---------------- Evaluate ----------------
    x_test = torch.tensor(np.loadtxt("../data/1KG/8020_test.txt"), dtype=torch.float32)
    avg_r2 = evaluate_r2(model, diffusion, x_test, "../index_files/snp_index_file.txt", device=device)
    logging.info(f"✅ Final Average R²: {avg_r2:.6f}")
