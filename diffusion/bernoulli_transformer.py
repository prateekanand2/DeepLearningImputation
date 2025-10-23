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
from tqdm import tqdm

def chunk_sequence_with_stride(x, chunk_len=1024, stride=512):
    """
    x: [B, L]
    returns: list of [B, chunk_len]
    """
    B, L = x.shape
    chunks = []
    positions = []
    for start in range(0, L, stride):
        end = start + chunk_len
        if end > L:
            # pad at the end if needed
            pad_len = end - L
            chunk = torch.cat([x[:, start:], torch.zeros(B, pad_len, device=x.device)], dim=1)
        else:
            chunk = x[:, start:end]
            pad_len = 0
        chunks.append(chunk)
        positions.append((start, end, pad_len))
        if end >= L:
            break
    return chunks, positions

def reconstruct_from_chunks(pred_chunks, positions, seq_len):
    """
    pred_chunks: list of [B, chunk_len]
    positions: (start, end, pad_len)
    seq_len: original length
    returns: [B, seq_len]
    """
    B = pred_chunks[0].shape[0]
    preds_full = torch.zeros(B, seq_len, device=pred_chunks[0].device)
    counts = torch.zeros(B, seq_len, device=pred_chunks[0].device)

    for pred, (start, end, pad_len) in zip(pred_chunks, positions):
        valid_len = pred.shape[1] - pad_len
        preds_full[:, start:start+valid_len] += pred[:, :valid_len]
        counts[:, start:start+valid_len] += 1

    preds_full /= (counts + 1e-8)
    return preds_full
    
def log_hyperparameters(params: dict):
    """Log hyperparameters in a clean, aligned format."""
    logging.info("📋 HYPERPARAMETERS")
    logging.info("=" * 50)
    for key, value in params.items():
        logging.info(f"{key:<20} : {value}")
    logging.info("=" * 50)


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

class ConditionalReverseTransformerModel(nn.Module):
    def __init__(self, snps, max_seq_len=1024, d_model=256, nhead=4, num_layers=4, t_emb_dim=128):
        super().__init__()
        self.snps = snps
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # --- Embed inputs: (x_t, mask, tau, x_self_cond) ---
        self.input_proj = nn.Linear(4, d_model)

        # --- Time embedding ---
        self.time_emb = SinusoidalPosEmb(t_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # --- Positional embedding for sequence ---
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # --- Transformer backbone ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output projection ---
        self.output_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t, t, mask, tau, x_self_cond=None):
        """
        x_t: [B, L]
        t: [B] (scalar timestep per sample)
        mask, tau: [B, L]
        """
        device = x_t.device

        B, L = x_t.shape

        mask = mask.to(device)
        tau = tau.to(device)
        
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x_t, device=device)
        else:
            x_self_cond = x_self_cond.to(device)

        # stack inputs and embed
        x_stack = torch.stack([x_t, mask.float(), tau, x_self_cond], dim=-1)
        h = self.input_proj(x_stack) + self.pos_emb[:, :L, :]

        # --- time embedding ---
        t_emb = self.time_emb(t)               # [B, t_emb_dim]
        t_emb = self.time_mlp(t_emb)           # [B, d_model]
        t_emb = t_emb[:, None, :].expand(-1, L, -1)  # broadcast over sequence length
        h = h + t_emb                          # additive conditioning

        # transformer
        h = self.transformer(h)

        # output
        logits = self.output_proj(h).squeeze(-1)
        return logits

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
    val_seed=1234,
    chunk_len=1024,
    stride=512
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    snps = train_dataset.tensors[0].shape[1]
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # --- Load fixed validation mask indices ---
    val_mask_indices = None
    if val_index_file is not None and os.path.exists(val_index_file):
        val_mask_indices = np.loadtxt(val_index_file, dtype=int)

    # --- Deterministic val mask ---
    fixed_val_mask = None
    if val_loader and val_mask_indices is None:
        logging.info("ℹ️ No val_index_file provided — using deterministic random mask for validation.")
        rng = torch.Generator(device=device)
        rng.manual_seed(val_seed)
        n_val = val_dataset.tensors[0].shape[0]
        fixed_val_mask = torch.ones((n_val, snps), device=device)
        n_mask_val = int(mask_ratio * snps)
        for i in range(n_val):
            mask_indices = torch.randperm(snps, generator=rng, device=device)[:n_mask_val]
            fixed_val_mask[i, mask_indices] = 0

    # ====================================================
    # EPOCH LOOP
    # ====================================================
    for epoch in range(epochs):
        # ====================================================
        # 🟢 TRAINING
        # ====================================================
        model.train()
        total_loss = 0.0

        for (x_0,) in train_loader:
            x_0 = x_0.to(device)
            B = x_0.size(0)

            # --- random mask per sample ---
            mask = torch.ones_like(x_0)
            n_mask = int(mask_ratio * snps)
            for i in range(B):
                mask_indices = torch.randperm(snps, device=device)[:n_mask]
                mask[i, mask_indices] = 0

            # --- chunk inputs ---
            x_chunks, _ = chunk_sequence_with_stride(x_0, chunk_len, stride)
            mask_chunks, _ = chunk_sequence_with_stride(mask, chunk_len, stride)
            tau_chunks, _ = chunk_sequence_with_stride(tau_tensor.unsqueeze(0).expand(B, -1), chunk_len, stride)

            loss_total = 0
            for x_chunk, mask_chunk, tau_chunk in zip(x_chunks, mask_chunks, tau_chunks):
                t = diffusion.sample_timesteps(B)
                x_t = diffusion.q_sample(x_chunk, t, mask_chunk)

                # --- self-conditioning ---
                use_self_cond = torch.rand(1).item() < 0.5
                if use_self_cond:
                    with torch.no_grad():
                        prev_logits = model(x_t, t, mask_chunk, tau_chunk)
                    x_self_cond = torch.sigmoid(prev_logits)
                else:
                    x_self_cond = None

                logits = model(x_t, t, mask_chunk, tau_chunk, x_self_cond=x_self_cond)
                loss = F.binary_cross_entropy_with_logits(logits, x_chunk, weight=(1 - mask_chunk))
                loss_total += loss

            loss_total /= len(x_chunks)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            total_loss += loss_total.item() * B

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

                    # --- fixed mask ---
                    if val_mask_indices is not None:
                        mask = torch.ones(snps, device=device)
                        mask[val_mask_indices] = 0
                        mask_batch = mask.unsqueeze(0).expand(B_val, -1)
                    else:
                        mask_batch = fixed_val_mask[
                            batch_idx * B_val : batch_idx * B_val + B_val
                        ].to(device)

                    # --- chunk val ---
                    x_chunks_val, _ = chunk_sequence_with_stride(x_val, chunk_len, stride)
                    mask_chunks_val, _ = chunk_sequence_with_stride(mask_batch, chunk_len, stride)
                    tau_chunks_val, _ = chunk_sequence_with_stride(tau_tensor.unsqueeze(0).expand(B_val, -1), chunk_len, stride)

                    loss_val_total = 0
                    for x_chunk_val, mask_chunk_val, tau_chunk_val in zip(x_chunks_val, mask_chunks_val, tau_chunks_val):
                        t_val = diffusion.sample_timesteps(B_val)
                        x_t_val = diffusion.q_sample(x_chunk_val, t_val, mask_chunk_val)

                        use_self_cond_val = torch.rand(1).item() < 0.5
                        if use_self_cond_val:
                            prev_logits_val = model(x_t_val, t_val, mask_chunk_val, tau_chunk_val)
                            x_self_cond_val = torch.sigmoid(prev_logits_val)
                        else:
                            x_self_cond_val = None

                        logits_val = model(x_t_val, t_val, mask_chunk_val, tau_chunk_val, x_self_cond=x_self_cond_val)
                        loss_val_chunk = F.binary_cross_entropy_with_logits(
                            logits_val, x_chunk_val, weight=(1 - mask_chunk_val)
                        )
                        loss_val_total += loss_val_chunk

                    loss_val_total /= len(x_chunks_val)
                    total_val_loss += loss_val_total.item() * B_val

            val_loss = total_val_loss / len(val_loader.dataset)

            # --- Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), "checkpoints/best_model_bern_transformer.pt")
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

        msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        logging.info(msg)

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
    tau_tensor=None,
    n_reverse_runs=1,
    num_sampling_steps=1000,
    window_size=1024,
    stride=512  # overlap if stride < window_size
):
    """
    Evaluate masked R² by running reverse diffusion with self-conditioning
    on chunked overlapping windows. If a masked SNP appears in more than one
    window, its predicted values are averaged across windows.
    """

    model.eval()
    x_data = x_data.to(device)
    samples, snps = x_data.shape

    # --- Load fixed mask indices ---
    mask_indices = np.loadtxt(index_file, dtype=int)
    mask_vec = torch.ones(snps, device=device)
    mask_vec[mask_indices] = 0
    full_mask = mask_vec.unsqueeze(0).expand(samples, -1)

    # --- prepare accumulation buffers ---
    pred_sum = torch.zeros_like(x_data)
    pred_count = torch.zeros_like(x_data)

    # --- sampling schedule ---
    t_schedule = np.linspace(
        diffusion.num_timesteps - 1,
        0,
        num_sampling_steps,
        dtype=int
    )

    # --- define chunk start positions ---
    chunk_starts = list(range(0, snps - window_size + 1, stride))
    if chunk_starts[-1] + window_size < snps:
        chunk_starts.append(snps - window_size)  # last tail chunk

    num_chunks = len(chunk_starts)
    num_batches = (samples + batch_size - 1) // batch_size

    # Move tau_tensor to device once
    tau_tensor = tau_tensor.to(device)

    for start in tqdm(range(0, samples, batch_size), total=num_batches, desc="Batches"):
        end = min(start + batch_size, samples)
        x_batch = x_data[start:end]
        mask_batch_full = full_mask[start:end]

        for chunk_i, chunk_start in enumerate(tqdm(chunk_starts, leave=False, desc="Chunks")):
            chunk_end = chunk_start + window_size

            x_chunk = x_batch[:, chunk_start:chunk_end]
            mask_chunk = mask_batch_full[:, chunk_start:chunk_end]

            # ✅ slice tau correctly for this chunk
            tau_chunk = tau_tensor[chunk_start:chunk_end]
            tau_chunk = tau_chunk.unsqueeze(0).expand(x_chunk.size(0), -1)

            pred_chunk_accum = torch.zeros_like(x_chunk)

            for run_idx in range(n_reverse_runs):
                # initialize noise at masked positions
                rand_noise = torch.bernoulli(0.5 * torch.ones_like(x_chunk))
                x_t = mask_chunk * x_chunk + (1 - mask_chunk) * rand_noise

                # initialize self-conditioning as zeros
                x_self_cond = torch.zeros_like(x_t)

                # reverse diffusion timesteps
                for t_inv in t_schedule:
                    t_tensor = torch.full((x_t.size(0),), t_inv, dtype=torch.long, device=device)

                    logits = model(x_t, t_tensor, mask_chunk, tau_chunk, x_self_cond=x_self_cond)
                    x_pred = torch.sigmoid(logits)

                    # deterministic update: observed + predicted masked
                    x_t = mask_chunk * x_chunk + (1 - mask_chunk) * x_pred

                    # update self-conditioning
                    x_self_cond = x_pred.detach()

                pred_chunk_accum += x_t

            # average across runs
            pred_chunk = pred_chunk_accum / n_reverse_runs

            # --- accumulate results into global prediction tensor ---
            pred_sum[start:end, chunk_start:chunk_end] += pred_chunk
            pred_count[start:end, chunk_start:chunk_end] += 1

    # --- finalize averaged predictions ---
    pred_count[pred_count == 0] = 1
    preds = pred_sum / pred_count

    # --- compute R² on masked SNPs ---
    r2_list = []
    for feat_idx in tqdm(mask_indices, desc="Computing R²"):
        true_vals = x_data[:, feat_idx]
        pred_vals = preds[:, feat_idx]
        r2 = compute_squared_pearson_correlation(true_vals, pred_vals)
        r2_list.append(r2.item())

    avg_r2 = float(np.mean(r2_list))
    logging.info(
        f"Average R² across masked SNPs (n={n_reverse_runs} runs, "
        f"{num_sampling_steps} steps, self-cond=ON, chunked inference): {avg_r2:.6f}"
    )

    # --- optionally save per-SNP R² ---
    if output_path:
        df = pd.DataFrame({"SNP_Index": mask_indices, "R2": r2_list})
        df.to_csv(output_path, index=False, float_format="%.8f")

    return avg_r2


if __name__ == "__main__":
    # ============================================================
    # 🧪 HYPERPARAMETERS
    # ============================================================
    # --- Paths ---
    DATA_DIR = "/scratch2/prateek/genetic_pc_github/results/b38/8020/data"
    TRAIN_FILE = f"{DATA_DIR}/8020_train.txt"
    TEST_FILE = f"{DATA_DIR}/8020_test.txt"
    INDEX_FILE = "/scratch2/prateek/genetic_pc_github/results/b38/missing_indices.txt"
    MAP_FILE = "/scratch2/prateek/DeepLearningImputation/1kg_15_map_b38.txt"
    LEGEND_FILE = "/scratch2/prateek/genetic_pc_github/aux/b38_legend.maf.txt"

    LOG_PATH = "logs/train_bern_transformer_b38.log"
    OUTPUT_RESULTS = "results/diff_results_bern_transformer_b38.csv"

    # --- Model checkpoints ---
    FINAL_MODEL_PATH = "checkpoints/best_model_bern_transformer_b38.pt"

    # --- Training ---
    SEED = 42
    LR = 1e-3
    MIN_LR = 1e-7
    BATCH_SIZE = 8
    TOTAL_EPOCHS = 200
    PATIENCE = 20
    VAL_SPLIT = 0.1
    MASK_RATIO = 0.5
    SKIP_TRAINING = False   # Set True to skip training and load model from disk
    WINDOW_SIZE = 1024
    STRIDE = 512

    # --- Genetic Map / Tau ---
    Ne = 10000
    H = 4006

    # --- Evaluation ---
    EVAL_BATCH_SIZE = 512
    # ============================================================

    # --- Setup ---
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting conditional diffusion pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load training data ---
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"File not found: {TRAIN_FILE}")

    x_data_np = np.loadtxt(TRAIN_FILE, dtype=np.float32)
    x_data = torch.tensor(x_data_np, dtype=torch.float32)
    samples, snps = x_data.shape
    logging.info(f"Loaded train matrix: {samples} samples × {snps} SNPs")

    # --- Genetic map and tau ---
    map_bps, map_cms = load_coarse_map(MAP_FILE)
    snp_bps = load_snp_positions(LEGEND_FILE)
    interp_cms = interpolate_cM_for_snps(snp_bps, map_bps, map_cms)
    tau_tensor, log_tau_tensor = compute_tau(interp_cms, Ne=Ne, H=H)

    # --- Model & diffusion setup ---
    diffusion = ConditionalDiscreteDiffusionModel(device=device)
    reverse_model = ConditionalReverseTransformerModel(snps=snps).to(device)

    # ============================================================
    # TRAINING (optional)
    # ============================================================
    if not SKIP_TRAINING:
        logging.info("Training enabled.")

        # Split into train/val
        x_train, x_val = train_test_split(x_data, test_size=VAL_SPLIT, random_state=SEED)
        train_dataset = TensorDataset(x_train)
        val_dataset = TensorDataset(x_val)

        optimizer = torch.optim.Adam(reverse_model.parameters(), lr=LR)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=int(0.1 * TOTAL_EPOCHS),
            total_epochs=TOTAL_EPOCHS,
            min_lr=MIN_LR
        )

        # --- Initial training with early stopping ---
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
            tau_tensor=log_tau_tensor,
            chunk_len=WINDOW_SIZE,
            stride=STRIDE
        )
        logging.info(f"Best validation model found at {trained_epochs} epochs.")

        # --- Retrain on full dataset ---
        logging.info(f"Retraining on full dataset for {trained_epochs} epochs...")
        full_dataset = TensorDataset(torch.cat([x_train, x_val], dim=0))
        reverse_model = ConditionalReverseTransformerModel(snps=snps).to(device)
        optimizer = torch.optim.Adam(reverse_model.parameters(), lr=LR)
        scheduler_ft = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=int(0.1 * trained_epochs),
            total_epochs=trained_epochs,
            min_lr=MIN_LR
        )

        train_masked_conditional(
            reverse_model,
            diffusion,
            full_dataset,
            val_dataset=None,
            optimizer=optimizer,
            epochs=trained_epochs,
            batch_size=BATCH_SIZE,
            mask_ratio=MASK_RATIO,
            val_index_file=None,
            scheduler=scheduler_ft,
            device=device,
            tau_tensor=log_tau_tensor,
            chunk_len=WINDOW_SIZE,
            stride=STRIDE
        )

        # --- Save retrained model ---
        torch.save(reverse_model.state_dict(), FINAL_MODEL_PATH)
        logging.info(f"Saved final full-dataset model to {FINAL_MODEL_PATH}")

    else:
        # ============================================================
        # SKIP TRAINING — LOAD MODEL
        # ============================================================
        logging.info("⚡ Skipping training — loading pre-trained model.")
        if os.path.exists(FINAL_MODEL_PATH):
            reverse_model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
            logging.info(f"✅ Loaded final model from {FINAL_MODEL_PATH}")
        else:
            raise FileNotFoundError("No saved model found. Please run training first.")

    # ============================================================
    # EVALUATION
    # ============================================================
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"File not found: {TEST_FILE}")

    x_test = torch.tensor(np.loadtxt(TEST_FILE), dtype=torch.float32)
    samples_test, snps_test = x_test.shape
    logging.info(f"Loaded test matrix: {samples_test} samples × {snps_test} SNPs")

    avg_r2 = evaluate_masked_r2_reverse_diffusion(
        model=reverse_model,
        diffusion=diffusion,
        x_data=x_test,
        index_file=INDEX_FILE,
        device=device,
        output_path=OUTPUT_RESULTS,
        batch_size=EVAL_BATCH_SIZE,
        tau_tensor=log_tau_tensor,
        window_size=WINDOW_SIZE,
        stride=STRIDE
    )
    logging.info(f"Final Average R²: {avg_r2:.6f}")
