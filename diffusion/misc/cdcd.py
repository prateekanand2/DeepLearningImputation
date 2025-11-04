# binary_cdcd.py
import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import List, Tuple

# ----------------------------
# Utilities
# ----------------------------

def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Toy dataset (replace with real data)
# ----------------------------

class RandomBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len=128, size=10000):
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tokens = torch.randint(0, 2, (self.seq_len,), dtype=torch.long)
        return tokens

# ----------------------------
# RoPE helpers
# ----------------------------

def rotate_half(x):
    # x: ... D where D is even
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rope(q, k, rope_cache):
    """
    q,k: [B, H, L, Dh]
    rope_cache: (cos, sin) each [L, Dh]
    """
    cos, sin = rope_cache
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,L,Dh]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

def build_rope_cache(seq_len, head_dim, base=10000.0, device="cpu"):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [L, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [L, head_dim]
    return torch.cos(emb), torch.sin(emb)  # each [L, head_dim]


# ----------------------------
# Random Fourier time embedding (optional)
# ----------------------------

class RandomFourierTimeEmbedding(nn.Module):
    def __init__(self, out_dim, scale=10.0):
        super().__init__()
        assert out_dim % 2 == 0
        half = out_dim // 2
        B = torch.randn(half) * scale
        self.register_buffer("B", B, persistent=True)
        self.out_dim = out_dim

    def forward(self, t: torch.Tensor):
        # t: [B] or [B,1]
        if t.ndim == 1:
            t = t[:, None].float()
        arg = 2.0 * math.pi * t * self.B[None, :]  # [B, half]
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return emb  # [B, out_dim]


# ----------------------------
# Pre-LN Transformer block (RoPE inside attention)
# ----------------------------

class PreLNTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_cache):
        # x: [B, L, D]
        B, L, D = x.shape
        h = self.norm1(x)  # pre-LN

        qkv = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, L, Dh]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        # Apply RoPE to q,k
        q, k = apply_rope(q, k, rope_cache)  # both [B,H,L,Dh]

        # Scaled dot-product
        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,L,L]
        attn = F.softmax(attn_logits, dim=-1)
        out = attn @ v  # [B,H,L,Dh]

        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B,L,D]
        out = self.proj(out)
        x = x + out

        # FFN
        h2 = self.norm2(x)
        x = x + self.ff(h2)
        return x


# ----------------------------
# CDCD Transformer (8 blocks, 1024 dim)
# ----------------------------

class CDCDTransformerModel(nn.Module):
    def __init__(self, vocab_size=2, dim=1024, n_heads=8, n_layers=8, max_len=512, dropout=0.1, use_time_emb=True):
        super().__init__()
        assert dim % n_heads == 0
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        # token embedding (note: token embedding dim == model dim)
        self.token_emb = nn.Embedding(vocab_size, dim)

        # projections for input streams (x, c, m, p) -> keep same dim and sum per specification
        self.proj_x = nn.Linear(dim, dim)
        self.proj_c = nn.Linear(dim, dim)
        self.proj_m = nn.Linear(1, dim)
        self.proj_p = nn.Linear(dim, dim)

        # optional time embedding
        self.use_time_emb = use_time_emb
        if use_time_emb:
            self.time_emb = nn.Sequential(
                RandomFourierTimeEmbedding(dim // 2, scale=10.0),
                nn.Linear(dim // 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.GELU(),
            )

        # transformer backbone
        self.blocks = nn.ModuleList([PreLNTransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

        # RoPE cache buffers (head_dim for RoPE)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _ensure_rope(self, L, device):
        head_dim = self.dim // self.n_heads
        if (self.rope_cos is None) or (self.rope_cos.shape[0] < L):
            cos, sin = build_rope_cache(L, head_dim, device=device)
            self.rope_cos, self.rope_sin = cos, sin

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, x: torch.Tensor, c: torch.Tensor, p: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        tokens: [B, L] int (not used directly in computation except for maybe diagnostics)
        mask:   [B, L] binary (1 = to generate/noisy, 0 = conditioning/clean)
        x:      [B, L, D] noisy embeddings (zeros at conditioning positions)
        c:      [B, L, D] conditioning embeddings (zeros at generated positions)
        p:      [B, L, D] self-conditioning embeddings (zeros at conditioning positions)
        t:      [B] optional timestep (int or float) for time embedding
        """
        B, L = tokens.shape
        device = tokens.device
        self._ensure_rope(L, device)
        rope_cache = (self.rope_cos[:L], self.rope_sin[:L])

        m_proj = self.proj_m(mask.unsqueeze(-1).float())  # [B,L,D]
        h = self.proj_x(x) + self.proj_c(c) + m_proj + self.proj_p(p)  # [B,L,D]

        if self.use_time_emb and (t is not None):
            t_emb = self.time_emb(t)  # [B, D]
            h = h + t_emb.unsqueeze(1)  # broadcast to [B,L,D]

        # Transformer stack
        for block in self.blocks:
            h = block(h, rope_cache)

        h = self.final_ln(h)
        logits = self.head(h)  # [B,L,V]
        return logits

# ----------------------------
# Embedding helpers
# ----------------------------

def add_noise_to_embeddings(token_emb: torch.Tensor, mask: torch.Tensor, sigma: torch.Tensor):
    """
    Add Gaussian noise to embeddings according to CDCD schedule.
    token_emb: [B,L,D] already L2-normalized and scaled
    mask: [B,L] 1 = noisy/generate, 0 = conditioning
    sigma: [B] per-sample noise level (σ = t)
    """
    sigma = sigma.view(-1, 1, 1)  # [B,1,1] for broadcasting
    noise = torch.randn_like(token_emb) * sigma
    x_noisy = (token_emb + noise) / torch.sqrt(1.0 + sigma**2)  # CDCD-style scaling
    return x_noisy * mask.unsqueeze(-1).float()


def make_conditioning_embeddings(token_emb: torch.Tensor, mask: torch.Tensor):
    """
    token_emb: [B,L,D]
    mask: [B,L] 1 = noisy/generate, 0 = conditioning
    returns c: [B,L,D] embeddings for conditioning positions only
    """
    return token_emb * (1.0 - mask.unsqueeze(-1).float())


def build_inputs_from_tokens(token_ids: torch.Tensor, model, mask: torch.Tensor, sigma: torch.Tensor):
    """
    token_ids: [B,L]
    mask: [B,L] 1 = generate, 0 = conditioning
    sigma: [B] per-sample noise level
    returns x, c
    """
    token_embs = model.token_emb(token_ids)  # [B,L,D]

    # L2-normalize and scale
    D = token_embs.size(-1)
    token_embs = F.normalize(token_embs, dim=-1) * (D ** 0.5)

    # Conditioning embeddings
    c = make_conditioning_embeddings(token_embs, mask)

    # Noisy embeddings
    x = add_noise_to_embeddings(token_embs, mask, sigma)

    return x, c

# ----------------------------
# Self-conditioning
# ----------------------------

def maybe_self_condition(model, tokens, mask, x, c, t=None):
    B = tokens.size(0)
    device = tokens.device
    use_self_cond = torch.rand(B, device=device) < 0.5
    p = torch.zeros_like(x)

    if use_self_cond.any():
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(tokens, mask, x, c, p=torch.zeros_like(x), t=t)
            probs = F.softmax(logits, dim=-1)
            pred_p_emb = probs @ model.token_emb.weight
            D = pred_p_emb.size(-1)
            pred_p_emb = F.normalize(pred_p_emb, dim=-1) * (D ** 0.5)
            pred_p_emb = pred_p_emb * mask.unsqueeze(-1).float()
        if was_training:
            model.train()

        p[use_self_cond] = pred_p_emb[use_self_cond]

    return p


def train_model(
    model: CDCDTransformerModel,
    dataloader,
    n_steps: int = 2000,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    tmin: float = 1.0,
    tmax: float = 300.0,
    warmup_steps: int = 200,
    val_frac: float = 0.1,
    patience: int = 10
):
    """
    Train the CDCD model with:
    - linear warmup + cosine decay scheduler
    - validation loss evaluation
    - early stopping with patience
    """
    device = device or default_device()
    model.to(device)

    # Split dataloader into train/val
    dataset_size = len(dataloader.dataset)
    val_size = int(dataset_size * val_frac)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, n_steps - warmup_steps), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    while step < n_steps:
        model.train()
        for batch in train_loader:
            batch = batch[0].to(device)  # [B,L]
            B, L = batch.shape

            # Sample per-instance mask probability and derive mask
            p_rand = torch.rand(B, device=device) * 0.6 + 0.1
            mask = torch.bernoulli(p_rand.unsqueeze(-1).expand(-1, L)).to(device)

            t = tmin + (tmax - tmin) * torch.rand(B, device=device)

            # Build inputs
            x, c = build_inputs_from_tokens(batch, model, mask, sigma=t)

            # Self-conditioning
            p = maybe_self_condition(model, batch, mask, x, c, t)

            # Forward pass
            logits = model(batch, mask, x, c, p, t)  # [B,L,V]

            # Compute CE only on generated positions
            logits_flat = logits.view(-1, model.vocab_size)
            targets_flat = batch.view(-1)
            mask_flat = mask.view(-1).bool()
            if mask_flat.sum() == 0:
                continue

            loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()  # update learning rate

            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"step {step:6d} train_loss {loss.item():.4f} sigma {t[0].item():.2f} lr {current_lr:.2e}")

            step += 1

            # Validation + early stopping
            if step % 100 == 0 or step == n_steps:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch[0].to(device)
                        Bv, Lv = val_batch.shape

                        p_rand_val = torch.rand(Bv, device=device) * 0.6 + 0.1
                        mask_val = torch.bernoulli(p_rand_val.unsqueeze(-1).expand(-1, Lv)).to(device)
                        t_val = tmin + (tmax - tmin) * torch.rand(Bv, device=device)

                        x_val, c_val = build_inputs_from_tokens(val_batch, model, mask_val, sigma=t_val)
                        p_val = maybe_self_condition(model, val_batch, mask_val, x_val, c_val, t_val)
                        logits_val = model(val_batch, mask_val, x_val, c_val, p_val, t=t_val)

                        logits_flat_val = logits_val.view(-1, model.vocab_size)
                        targets_flat_val = val_batch.view(-1)
                        mask_flat_val = mask_val.view(-1).bool()
                        if mask_flat_val.sum() > 0:
                            val_loss = F.cross_entropy(logits_flat_val[mask_flat_val], targets_flat_val[mask_flat_val])
                            val_losses.append(val_loss.item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"step {step:6d} validation_loss {avg_val_loss:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), "best_model_cdcd.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at step {step}")
                        model.load_state_dict(torch.load("best_model_cdcd.pt"))
                        return model

            if step >= n_steps:
                break

    # Load best model at the end
    model.load_state_dict(torch.load("best_model_cdcd.pt"))
    return model

# ----------------------------
# Sampling (iterative denoise with self-conditioning)
# ----------------------------

def compute_squared_pearson_correlation(x_true: torch.Tensor, x_pred: torch.Tensor):
    """
    Compute squared Pearson correlation between two tensors of shape [N]
    """
    vx = x_true - x_true.mean()
    vy = x_pred - x_pred.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr ** 2

def evaluate_imputation_iterative_noise_init_chunked(
    model: CDCDTransformerModel,
    test_data_full: torch.Tensor,        # [N, L_full]
    test_chunks: torch.Tensor,           # [num_chunks, seq_len]
    chunk_indices: List[Tuple[int,int,int]],  # (sample_idx, start, end)
    drop_frac: float = 0.15,
    n_steps: int = 200,
    tmin: float = 1.0,
    tmax: float = 300.0,
    device=None,
):
    """
    Evaluate iterative diffusion imputation chunk-by-chunk.
    - test_data_full: unchunked data [N, L_full]
    - test_chunks: chunked sequences [num_chunks, seq_len]
    - chunk_indices: list of (sample_idx, start, end)
    - 15% of features dropped consistently across all chunks
    - overlapping chunks merged by averaging predicted probabilities
    """
    device = device or default_device()
    model.to(device)
    model.eval()

    test_data_full = test_data_full.to(device)
    N, L_full = test_data_full.shape
    seq_len = test_chunks.size(1)
    D = model.dim

    # 1. Create consistent mask (same dropped features across all samples)
    num_drop = max(1, int(L_full * drop_frac))
    drop_indices = torch.randperm(L_full, device=device)[:num_drop]
    global_mask = torch.zeros(L_full, device=device)
    global_mask[drop_indices] = 1

    # 2. Storage for accumulated predictions
    prob_sum = torch.zeros_like(test_data_full, dtype=torch.float32)
    prob_count = torch.zeros_like(test_data_full, dtype=torch.float32)

    # 3. Loop over chunks
    t_schedule = torch.linspace(tmax, tmin, n_steps, device=device)
    with torch.no_grad():
        for (chunk_idx, (sample_idx, start, end)) in enumerate(chunk_indices):
            chunk = test_chunks[chunk_idx:chunk_idx+1].to(device)  # [1, seq_len]
            mask_chunk = global_mask[start:end].unsqueeze(0).expand(1, -1)  # [1, seq_len]

            # token embeddings
            token_embs = model.token_emb(chunk)
            token_embs = F.normalize(token_embs, dim=-1) * (D ** 0.5)

            c = make_conditioning_embeddings(token_embs, mask_chunk)

            # initialize x: masked positions random noise, others real emb
            x = torch.randn_like(token_embs)
            x = x * mask_chunk.unsqueeze(-1).float() + token_embs * (1 - mask_chunk).unsqueeze(-1).float()
            prev_p = torch.zeros_like(x)

            # iterative denoising
            for t_step in t_schedule:
                t_vec = t_step * torch.ones(1, device=device)
                logits = model(chunk, mask_chunk, x, c, prev_p, t_vec)
                probs = F.softmax(logits, dim=-1)
                p_emb = probs @ model.token_emb.weight
                p_emb = F.normalize(p_emb, dim=-1) * (D ** 0.5)
                p_emb = p_emb * mask_chunk.unsqueeze(-1).float()
                prev_p = p_emb
                noise = torch.randn_like(x) * t_step * 0.05
                x = p_emb + noise * mask_chunk.unsqueeze(-1).float()
                x = x * mask_chunk.unsqueeze(-1).float() + token_embs * (1 - mask_chunk).unsqueeze(-1).float()

            # final probs
            final_logits = model(chunk, mask_chunk, x, c, prev_p, t_vec)
            final_probs = F.softmax(final_logits, dim=-1)
            prob_1 = final_probs[..., 1] * mask_chunk  # [1, seq_len]

            # aggregate into full-sample matrix
            prob_sum[sample_idx, start:end] += prob_1.squeeze(0)
            prob_count[sample_idx, start:end] += mask_chunk.squeeze(0)

    # 4. Average overlapping predictions
    avg_prob = torch.where(prob_count > 0, prob_sum / prob_count, torch.zeros_like(prob_sum))

    # 5. Compute per-feature R^2 (only masked features)
    r2_per_feature = []
    for j in drop_indices:
        y_true = test_data_full[:, j].float()
        y_pred = avg_prob[:, j].float()
        corr2 = compute_squared_pearson_correlation(y_true, y_pred)
        r2_per_feature.append(corr2.item())

    avg_r2 = sum(r2_per_feature) / len(r2_per_feature)
    print(f"Average per-feature R^2 at masked features: {avg_r2:.4f}")

    return avg_prob.cpu(), global_mask.cpu(), r2_per_feature

# ----------------------------
# Example run
# ----------------------------

def chunk_sequences(data: np.ndarray, seq_len: int = 1024, overlap: int = 512):
    """
    Split each row of `data` (shape [N, L]) into overlapping chunks of length `seq_len`.
    Overlap controls how much consecutive chunks overlap.
    Returns a new array of shape [num_chunks, seq_len].
    """
    N, L = data.shape
    step = seq_len - overlap
    chunks = []
    for i in range(N):
        for start in range(0, L - seq_len + 1, step):
            end = start + seq_len
            chunks.append(data[i, start:end])
    return np.stack(chunks)

def build_chunk_indices(data, seq_len=1024, overlap=512):
        step = seq_len - overlap
        indices = []
        for i in range(data.shape[0]):
            for start in range(0, data.shape[1] - seq_len + 1, step):
                end = start + seq_len
                indices.append((i, start, end))
        return indices

def example_run(train_path="train.txt", test_path="test.txt"):
    set_seed(1)
    device = default_device()

    # ----------------------------
    # Load and chunk datasets
    # ----------------------------
    print(f"Loading training data from {train_path} ...")
    train_data = np.loadtxt(train_path, dtype=np.int64)
    test_data = np.loadtxt(test_path, dtype=np.int64)

    # Chunk each row into overlapping windows
    seq_len = 1024
    overlap = 512
    print(f"Chunking data into sequences of {seq_len} with {overlap} overlap...")

    train_chunks = chunk_sequences(train_data, seq_len=seq_len, overlap=overlap)
    test_chunks = chunk_sequences(test_data, seq_len=seq_len, overlap=overlap)

    print(f"Train chunks: {train_chunks.shape}, Test chunks: {test_chunks.shape}")

    # Convert to tensors
    train_tensor = torch.tensor(train_chunks, dtype=torch.long)
    test_tensor = torch.tensor(test_chunks, dtype=torch.long)

    # Wrap into dataset and dataloader
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # ----------------------------
    # Create model
    # ----------------------------
    print("Creating model (8 blocks, 1024 dim, 8 heads)...")
    model = CDCDTransformerModel(
        vocab_size=2,
        dim=512,
        n_heads=8,
        n_layers=8,
        max_len=seq_len,
        use_time_emb=True,
        dropout=0.1
    )

    # ----------------------------
    # Train model
    # ----------------------------
    print("Training model...")
    model = train_model(
        model,
        train_loader,
        n_steps=5000,
        lr=1e-4,
        tmin=0.5,
        tmax=50.0,
        warmup_steps=500,
        val_frac=0.1,
        patience=2,
        device=device
    )

    # ----------------------------
    # Evaluation
    # ----------------------------
    # Build mapping info for chunks
    chunk_indices = build_chunk_indices(test_data, seq_len=seq_len, overlap=overlap)

    print(f"Evaluating imputation on {len(test_tensor)} test chunks...")
    prob_1, mask, r2_features = evaluate_imputation_iterative_noise_init_chunked(
        model,
        torch.tensor(test_data, dtype=torch.long),
        test_tensor,
        chunk_indices,
        drop_frac=0.15,
        n_steps=50,
        tmin=0.5,
        tmax=50.0,
        device=device,
    )

    print("\nExample for first test sequence:")
    masked_positions = mask[0].nonzero().squeeze().tolist()
    print("Masked positions:", masked_positions)
    print("Predicted probability of 1 at masked positions:", prob_1[0][mask[0] == 1].tolist())
    print(f"Average per-feature R^2 at masked positions: {sum(r2_features)/len(r2_features):.4f}")


if __name__ == "__main__":
    example_run("/scratch2/prateek/DeepLearningImputation/data/1KG/8020_train.txt", "/scratch2/prateek/DeepLearningImputation/data/1KG/8020_test.txt")
