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
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
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

@torch.no_grad()
def ddim_step(x_t, x_pred, t_now, t_next):
    gamma_now = gamma_schedule(t_now).view(-1,1)
    gamma_next = gamma_schedule(t_next).view(-1,1)
    eps = (x_t - torch.sqrt(gamma_now) * x_pred) / torch.sqrt(1 - gamma_now)
    x_next = torch.sqrt(gamma_next) * x_pred + torch.sqrt(1 - gamma_next) * eps
    return x_next

# ------------------ Sampling & Evaluation ------------------
@torch.no_grad()
def generate_bit_diffusion(model, diffusion, x_obs, mask, steps=50, td=0.0, device="cpu"):
    B, snps = x_obs.shape
    x_obs_bits = int2bit(x_obs).to(device)
    x_t = torch.randn_like(x_obs_bits, device=device) * (1 - mask) + x_obs_bits * mask

    for step in range(steps):
        t_now = torch.full((B,), 1.0 - step / steps, device=device)
        t_next_val = max(1.0 - (step + 1 + td) / steps, 0.0)
        t_next = torch.full((B,), t_next_val, device=device)
        x0_pred = model(x_t, t_now, mask)
        x_t = ddim_step(x_t, x0_pred, t_now, t_next)
        x_t = x_t * (1 - mask) + x_obs_bits * mask

    final_logits = model(x_t, torch.zeros(B, device=device), mask)
    probs = torch.clamp(bit2int(final_logits), 0.0, 1.0)  # [0,1] probabilities
    return probs, final_logits

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

        # Compute r2 and replace NaNs with 0
        r2 = (torch.sum(vx * vy) ** 2) / denom
        if torch.isnan(r2):
            r2 = torch.tensor(0.0, device=r2.device)
        r2_list.append(r2.item())

    avg_r2 = float(np.mean(r2_list))
    return avg_r2

# ------------------ Training ------------------
def train_bit_diffusion_1bit(
    model,
    diffusion,
    train_dataset,
    val_dataset=None,
    optimizer=None,
    epochs=100,
    batch_size=64,
    patience=10,
    mask_ratio=0.15,
    val_index_file=None,
    scheduler=None,
    device="cuda",
    val_seed=1234,   # ✅ for deterministic val masking
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    snps = train_dataset.tensors[0].shape[1]

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # --- Load fixed validation mask indices from file if provided ---
    val_mask_indices = None
    if val_index_file:
        val_mask_indices = torch.tensor(np.loadtxt(val_index_file, dtype=int), device=device)

    # --- Pre-generate deterministic random validation mask if no file ---
    fixed_val_mask = None
    if val_loader and val_mask_indices is None:
        logging.info("ℹ️ No val_index_file provided — using deterministic random mask for validation.")
        rng = torch.Generator(device=device)
        rng.manual_seed(val_seed)

        val_samples = val_dataset.tensors[0].shape[0]
        fixed_val_mask = torch.ones((val_samples, snps), device=device)
        n_mask_val = max(1, int(mask_ratio * snps))
        for i in range(val_samples):
            mask_indices = torch.randperm(snps, generator=rng, device=device)[:n_mask_val]
            fixed_val_mask[i, mask_indices] = 0

    for epoch in range(epochs):
        # ======================================================
        # 🟢 TRAINING
        # ======================================================
        model.train()
        total_train_loss = 0.0

        for (x,) in train_loader:
            x = x.to(device)
            B = x.size(0)
            x_bits = int2bit(x)

            # random training mask
            mask = torch.ones(B, snps, device=device)
            n_mask = max(1, int(mask_ratio * snps))
            rand_idx = torch.rand(B, snps, device=device).argsort(dim=1)[:, :n_mask]
            mask[torch.arange(B).unsqueeze(1), rand_idx] = 0

            t = torch.rand(B, device=device)
            x_crpt, _ = diffusion.q_sample(x_bits, t, mask)

            optimizer.zero_grad()
            x_input = x_crpt * (1 - mask) + x_bits * mask
            logits = model(x_input, t, mask)

            loss = (((logits - x_bits) ** 2) * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * B

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ======================================================
        # 🔵 VALIDATION
        # ======================================================
        avg_val_loss = None
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (x_val,) in enumerate(val_loader):
                    x_val = x_val.to(device)
                    Bv = x_val.size(0)
                    x_bits_val = int2bit(x_val)

                    if val_mask_indices is not None:
                        # ✅ Fixed mask indices from file
                        mask_val = torch.ones(Bv, snps, device=device)
                        mask_val[:, val_mask_indices] = 0
                    else:
                        # ✅ Deterministic mask slice from precomputed
                        start = batch_idx * Bv
                        end = start + Bv
                        mask_val = fixed_val_mask[start:end]

                    t_val = torch.rand(Bv, device=device)
                    x_crpt_val, _ = diffusion.q_sample(x_bits_val, t_val, mask_val)

                    x_input_val = x_crpt_val * (1 - mask_val) + x_bits_val * mask_val
                    logits_val = model(x_input_val, t_val, mask_val)

                    loss_val = (((logits_val - x_bits_val) ** 2) * (1 - mask_val)).sum() / (
                        (1 - mask_val).sum() + 1e-8
                    )
                    total_val_loss += loss_val.item() * Bv

            avg_val_loss = total_val_loss / len(val_loader.dataset)

        # ======================================================
        # 📝 LOGGING
        # ======================================================
        log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}"
        if avg_val_loss is not None:
            log_msg += f" | Val Loss: {avg_val_loss:.6f}"
        logging.info(log_msg)

        # ======================================================
        # 🛑 EARLY STOPPING
        # ======================================================
        if avg_val_loss is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), "checkpoints/best_model_1bit_predlog.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"⏸ Early stopping at epoch {epoch+1}")
                    return best_epoch

        # ======================================================
        # ⏳ SCHEDULER STEP
        # ======================================================
        if scheduler:
            scheduler.step()

    return best_epoch

# ------------------ Main ------------------
if __name__ == "__main__":
    # ============================================================
    # 🧪 HYPERPARAMETERS
    # ============================================================
    DATA_DIR = "../data/1KG"
    INDEX_FILE = "../index_files/snp_index_file.txt"

    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"
    INITIAL_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model_1bit_predlog.pt"
    FINAL_MODEL_PATH = f"{CHECKPOINT_DIR}/final_model_1bit_predlog.pt"

    TRAIN_FILE = f"{DATA_DIR}/8020_train.txt"
    TEST_FILE = f"{DATA_DIR}/8020_test.txt"

    TEST_SIZE = 0.1            # fraction of train set held out for val
    MASK_RATIO = 0.5
    LR = 1e-3
    BATCH_SIZE = 64
    MAX_EPOCHS = 200
    PATIENCE = 10
    MIN_LR = 1e-7
    R2_STEPS = 50
    SEED = 42
    # ============================================================

    # ---------------- Logging Setup ----------------
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, "train_1bit_predlog.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    # ---------------- Load Data ----------------
    x_data = torch.tensor(np.loadtxt(TRAIN_FILE), dtype=torch.float32)
    x_train, x_val = train_test_split(x_data, test_size=TEST_SIZE, random_state=SEED)
    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)

    snps = x_data.shape[1]
    model = ConditionalReverseConvModel(snps=snps).to(device)
    diffusion = BitDiffusion(device=device)
    
    # compute warmup as 10% of total
    warmup_epochs_initial = max(1, int(0.1 * MAX_EPOCHS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=warmup_epochs_initial,
        total_epochs=MAX_EPOCHS,
        min_lr=MIN_LR
    )

    # ---------------- First Training ----------------
    logging.info(f"🚀 Starting initial training (warmup={warmup_epochs_initial}, total={MAX_EPOCHS})...")
    best_epoch = train_bit_diffusion_1bit(
        model,
        diffusion,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        mask_ratio=MASK_RATIO,
        val_index_file=None,
        scheduler=scheduler
    )
    logging.info(f"✅ Best epoch from initial training: {best_epoch}")

    # ---------------- Retrain on full train+val ----------------
    logging.info("🔁 Retraining on full train+val set...")
    x_full = torch.cat([x_train, x_val], dim=0)
    full_dataset = TensorDataset(x_full)

    model = ConditionalReverseConvModel(snps=snps).to(device)
    diffusion = BitDiffusion(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # recompute warmup as 10% of best_epoch
    warmup_epochs_retrain = max(1, int(0.1 * best_epoch))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=warmup_epochs_retrain,
        total_epochs=best_epoch,
        min_lr=MIN_LR
    )

    logging.info(f"🔁 Retraining for {best_epoch} epochs (warmup={warmup_epochs_retrain})")
    train_bit_diffusion_1bit(
        model,
        diffusion,
        full_dataset,
        val_dataset=None,
        optimizer=optimizer,
        epochs=best_epoch,
        batch_size=BATCH_SIZE,
        patience=0,  # no early stopping
        device=device,
        mask_ratio=MASK_RATIO,
        val_index_file=None,
        scheduler=scheduler
    )

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    logging.info(f"💾 Saved final retrained model to {FINAL_MODEL_PATH}")

    # ---------------- Evaluate ----------------
    x_test = torch.tensor(np.loadtxt(TEST_FILE), dtype=torch.float32)
    avg_r2 = evaluate_r2(
        model,
        diffusion,
        x_test,
        INDEX_FILE,
        batch_size=128,
        device=device,
        steps=R2_STEPS
    )
    logging.info(f"🏁 Final Average R² after retraining: {avg_r2:.6f}")
