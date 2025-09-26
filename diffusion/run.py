import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
import logging

class DiscreteDiffusionModel:
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.1, device='cpu'):
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)  # Flip probabilities

    # def q_sample(self, x_0, t):
    #     """
    #     Bernoulli forward process: q(x_t | x_0) flips bits with probability β_t.
    #     """
    #     # x_0: (batch, snps), binary
    #     batch_size, snps = x_0.shape
    #     beta_t = self.betas[t].view(-1, 1)  # (batch, 1)
        
    #     # Flip mask: flip each bit with probability beta_t
    #     flip_mask = torch.bernoulli(beta_t.expand(-1, snps)).to(x_0.device)
    #     x_t = (1 - flip_mask) * x_0 + flip_mask * (1 - x_0)  # flip bits where mask==1
    #     return x_t
    
    def q_sample(self, x_0, t):
        """
        Bernoulli mixture forward process:
        q(x_t=1 | x_0) = (1 - beta_t) * x_0 + beta_t * (1 - x_0)
        """
        batch_size, snps = x_0.shape
        beta_t = self.betas[t].view(-1, 1)  # (B, 1)

        prob_1 = (1 - beta_t) * x_0 + beta_t * (1 - x_0)
        x_t = torch.bernoulli(prob_1)  # Sample from Bernoulli
        return x_t

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

class ReverseModel(nn.Module):
    def __init__(self, snps, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(snps + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, snps),
            # nn.Sigmoid()
        )

    def forward(self, x_t, t):
        # Concatenate time step t as a feature
        t_norm = t.float().unsqueeze(1) / 100  # normalize
        x_in = torch.cat([x_t, t_norm.expand_as(x_t[:, :1])], dim=1)
        return self.net(x_in)

def train(model, diffusion, train_dataset, val_dataset, optimizer, epochs=10, batch_size=128, patience=20):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for x_0 in train_loader:
            x_0 = x_0[0].to(diffusion.device)  # x_0 is a tuple (data,)
            t = diffusion.sample_timesteps(x_0.shape[0])
            x_t = diffusion.q_sample(x_0, t)
            x_0_pred = model(x_t, t)
            loss = loss_fn(x_0_pred, x_0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item() * x_0.size(0)

        avg_train_loss = total_train_loss / len(train_dataset)

        # Evaluate on validation set
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x_0 in val_loader:
                    x_0 = x_0[0].to(diffusion.device)
                    t = diffusion.sample_timesteps(x_0.shape[0])
                    x_t = diffusion.q_sample(x_0, t)
                    x_0_pred = model(x_t, t)
                    val_loss = loss_fn(x_0_pred, x_0)
                    total_val_loss += val_loss.item() * x_0.size(0)

            avg_val_loss = total_val_loss / len(val_dataset)

            logging.info(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Optionally save best model here
                torch.save(model.state_dict(), "best_model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        else:
            # No validation dataset, so just logging.info training loss
            logging.info(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f}")

    return epoch + 1

def sample(model, diffusion, shape):
    """
    Generate samples starting from noise and reversing diffusion.
    """
    x_t = torch.bernoulli(0.5 * torch.ones(shape)).to(model.net[0].weight.device)
    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=model.net[0].weight.device)
        x_t = model(x_t, t_tensor).bernoulli()  # Sample from predicted probability
    return x_t

def generate_and_save_samples(reverse_model, diffusion, total_samples, snps, batch_size=500, output_file="samples.txt"):
    all_samples = []

    for i in range(0, total_samples, batch_size):
        current_batch_size = min(batch_size, total_samples - i)
        logging.info(f"Generating batch {i // batch_size + 1} of size {current_batch_size}...")
        with torch.no_grad():
            batch_samples = sample(reverse_model, diffusion, shape=(current_batch_size, snps))
        all_samples.append(batch_samples.cpu())

    final_tensor = torch.cat(all_samples, dim=0)

    # Save to file
    np.savetxt(output_file, final_tensor.numpy(), fmt='%d')
    logging.info(f"✅ Saved {total_samples} samples to {output_file}")

def compute_squared_pearson_correlation(x_true, x_pred):
    """
    Computes squared Pearson correlation (R^2) between two 1D tensors.
    """
    vx = x_true - x_true.mean()
    vy = x_pred - x_pred.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr ** 2

@torch.no_grad()
def evaluate_featurewise_r2_batch_and_save(
    model,
    diffusion,
    x_data,
    legend_path,
    output_path,
    batch_size_features=128,
):
    """
    Batch imputations of multiple features in parallel. Computes R² per feature,
    joins with legend file, and saves to CSV.

    Args:
      model: trained ReverseModel
      diffusion: DiscreteDiffusionModel
      x_data: torch.Tensor, shape (samples, features)
      legend_path: path to legend file (tab-delimited, no header)
      output_path: path to save output CSV
      batch_size_features: int, number of features masked simultaneously
    """
    model.eval()
    device = diffusion.device
    x_data = x_data.to(device)
    samples, features = x_data.shape

    r2_list = []

    for start in range(0, features, batch_size_features):
        end = min(start + batch_size_features, features)
        batch_feats = end - start

        x_batch = x_data.unsqueeze(0).repeat(batch_feats, 1, 1)
        x_batch = x_batch.permute(1, 0, 2).contiguous()

        for i, feat_idx in enumerate(range(start, end)):
            x_batch[:, i, feat_idx] = 0.0  # Mask the feature

        x_in = x_batch.reshape(-1, features)
        t_in = torch.zeros(x_in.shape[0], dtype=torch.long, device=device)

        preds = torch.sigmoid(model(x_in, t_in))
        preds = preds.reshape(samples, batch_feats, features)

        for i, feat_idx in enumerate(range(start, end)):
            true_vals = x_data[:, feat_idx]
            pred_vals = preds[:, i, feat_idx]
            r2 = compute_squared_pearson_correlation(true_vals, pred_vals)
            r2_list.append(r2.item())

    # Convert to NumPy array
    r2_array = np.array(r2_list)

    # Load legend file
    legend_df = pd.read_csv(legend_path, sep="\t", header=None, names=["SNP_Set", "MAF"])

    assert len(legend_df) == len(r2_array), \
        f"Legend file rows ({len(legend_df)}) and R2 array length ({len(r2_array)}) mismatch."

    output_df = pd.DataFrame({
        "SNP Set": legend_df["SNP_Set"],
        "R2": r2_array,
        "MAF": legend_df["MAF"]
    })

    output_df.to_csv(output_path, index=False, float_format="%.8f")
    logging.info(f"Saved R2 results to {output_path} (shape: {output_df.shape})")

def evaluate_partitioned_masked_r2_and_save(
    model,
    diffusion,
    x_data,
    legend_path,
    output_path,
    num_features_to_mask=128,
    seed=42,
):
    """
    Divide features into non-overlapping random batches and mask one batch at a time.
    Each feature is masked exactly once. All R² values are computed and saved.

    Args:
      model: trained ReverseModel
      diffusion: DiscreteDiffusionModel
      x_data: torch.Tensor, shape (samples, features)
      legend_path: path to legend file (tab-delimited, no header)
      output_path: path to save output CSV
      num_features_to_mask: int, number of features masked in each round
      seed: int, random seed for reproducibility
    """
    model.eval()
    device = diffusion.device
    x_data = x_data.to(device)
    samples, features = x_data.shape

    assert features % num_features_to_mask == 0, \
        f"num_features_to_mask={num_features_to_mask} must divide total features={features}"

    # Set reproducible seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Shuffle and partition feature indices
    all_indices = np.arange(features)
    np.random.shuffle(all_indices)
    partitions = np.array_split(all_indices, features // num_features_to_mask)

    # Initialize R² array
    r2_array = np.zeros(features)

    for batch_id, feature_indices in enumerate(partitions):
        feature_indices = np.sort(feature_indices)
        logging.info(f"Evaluating batch {batch_id + 1}/{len(partitions)} on features {feature_indices}")

        # Create masked input
        x_masked = x_data.clone()
        x_masked[:, feature_indices] = 0.0

        # Predict all features (we'll just extract predictions for the masked ones)
        t_in = torch.zeros(samples, dtype=torch.long, device=device)
        preds = torch.sigmoid(model(x_masked, t_in))

        # Compute R² only for the masked features
        for feat_idx in feature_indices:
            true_vals = x_data[:, feat_idx]
            pred_vals = preds[:, feat_idx]
            r2 = compute_squared_pearson_correlation(true_vals, pred_vals)
            r2_array[feat_idx] = r2.item()

    # Load legend file
    legend_df = pd.read_csv(legend_path, sep="\t", header=None, names=["SNP_Set", "MAF"])

    assert len(legend_df) == features, \
        f"Legend file rows ({len(legend_df)}) do not match number of features ({features})"

    # Save full R² table
    output_df = pd.DataFrame({
        "SNP Set": legend_df["SNP_Set"],
        "R2": r2_array,
        "MAF": legend_df["MAF"]
    })

    output_df.to_csv(output_path, index=False, float_format="%.8f")
    logging.info(f"Saved full R2 results to {output_path} (shape: {output_df.shape})")

if __name__ == "__main__":
    
    # Set up logging
    log_path = "/scratch2/prateek/diffusion/train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting training...")

    torch.manual_seed(0)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    file_path = "/scratch2/prateek/genetic_pc/reproduce_final/1KG/8020/data/8020_train.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    x_data_np = np.loadtxt(file_path, dtype=np.float32)
    x_data = torch.tensor(x_data_np, dtype=torch.float32)
    samples, snps = x_data.shape
    logging.info(f"Loaded train matrix: {samples} samples × {snps} SNPs")

    # --- Split into train/val ---
    x_train, x_val = train_test_split(x_data, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)

    diffusion = DiscreteDiffusionModel(device=device)
    reverse_model = ReverseModel(snps=snps).to(device)
    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=1e-3)  # Initial LR

    # --- Train with early stopping on train/val split ---
    trained_epochs = train(
        reverse_model,
        diffusion,
        train_dataset,
        val_dataset,
        optimizer=optimizer,
        epochs=1000,
        batch_size=2048,
        patience=20
    )
    logging.info(f"Training stopped after {trained_epochs} epochs (early stopping).")

    # Save best model
    torch.save(reverse_model.state_dict(), "best_model.pt")

    # Load best model weights before fine-tuning
    reverse_model.load_state_dict(torch.load("best_model.pt"))

    # --- Train on full train+val dataset using trained_epochs ---
    fine_tune_epochs = max(3, min(int(0.1 * trained_epochs), 20))
    logging.info(f"Fine-tuning on full dataset for {fine_tune_epochs} epochs with reduced LR...")

    # Reduce learning rate by factor of 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-4

    full_dataset = TensorDataset(torch.cat([x_train, x_val], dim=0))
    train(
        reverse_model,
        diffusion,
        full_dataset,
        val_dataset=None,
        optimizer=optimizer,
        epochs=fine_tune_epochs,
        batch_size=2048,
        patience=11  # Not used since no val set, but harmless
    )

    # Load test data
    x_test = torch.tensor(np.loadtxt("/scratch2/prateek/genetic_pc/reproduce_final/1KG/8020/data/8020_test.txt"), dtype=torch.float32)
    samples_test, snps_test = x_test.shape
    logging.info(f"Loaded test matrix: {samples_test} samples × {snps_test} SNPs")

    # Evaluate R2
    evaluate_featurewise_r2_batch_and_save(
        model=reverse_model,
        diffusion=diffusion,
        x_data=x_test,
        legend_path="/scratch2/prateek/genetic_pc/reproduce_final/10K_legend.maf.txt",
        output_path="/scratch2/prateek/genetic_pc/results_final/r2s/bootstrap/diff_r2_8020.csv",
        batch_size_features=32
    )

    evaluate_partitioned_masked_r2_and_save(
        model=reverse_model,
        diffusion=diffusion,
        x_data=x_test,
        legend_path="/scratch2/prateek/genetic_pc/reproduce_final/10K_legend.maf.txt",
        output_path="/scratch2/prateek/genetic_pc/results_final/r2s/bootstrap/diff_r2_partitioned_8020.csv",
        num_features_to_mask=10,
        seed=42
    )
    # generate_and_save_samples(reverse_model, diffusion, total_samples=5008, snps=snps, batch_size=5008)