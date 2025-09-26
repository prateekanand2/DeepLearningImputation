import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM
from tqdm import tqdm
import argparse
from scipy.stats import pearsonr
import json
import os
import csv
from collections import Counter

class SimpleTokenizer:
    def __init__(self, tokenizer_dir):
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.mask_token = "[MASK]"
        self.mask_token_id = self.vocab[self.mask_token]

    def encode(self, tokens):
        return [self.vocab.get(t, -1) for t in tokens]

    def decode(self, token_ids):
        return [self.id_to_token[i] for i in token_ids]

def compute_r2(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return corr ** 2

class SNPDataset(Dataset):
    def __init__(self, hap_array):
        self.hap_array = hap_array

    def __len__(self):
        return len(self.hap_array)

    def __getitem__(self, idx):
        return self.hap_array[idx]

def load_maf(maf_file):
    mafs = []
    with open(maf_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mafs.append(float(parts[1]))
    return np.array(mafs)

def sliding_window_indices(seq_len, window_size=512, stride=256):
    starts = list(range(0, max(1, seq_len - window_size + 1), stride))
    if starts[-1] != seq_len - window_size:
        starts.append(seq_len - window_size)
    return starts

def infer_and_compute_r2(model_path, test_file, out_file, window_size, stride, batch_size=128, device='cuda', maf_file=None, snp_filter="all", snp_index_file=None, ):
    model = BertForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()
    tokenizer = SimpleTokenizer(model_path)
    hap_array = np.loadtxt(test_file, dtype=int)
    num_indiv, seq_len = hap_array.shape
    print(f"Loaded test data with shape {hap_array.shape}")

    # SNP filtering
    snp_indices = np.arange(seq_len)
    if snp_filter == "common" and maf_file:
        mafs = load_maf(maf_file)
        assert len(mafs) == seq_len
        snp_indices = np.where(mafs > 1e-1)[0]
    elif snp_filter == "rare" and maf_file:
        mafs = load_maf(maf_file)
        assert len(mafs) == seq_len
        snp_indices = np.where(mafs < 1e-3)[0]
    elif snp_filter == "file" and snp_index_file:
        with open(snp_index_file, "r") as f:
            snp_indices = [int(line.strip()) for line in f if line.strip()]
        snp_indices = np.array(snp_indices, dtype=int)

    print(f"{len(snp_indices)} SNPs selected for filter: '{snp_filter}'")

    half_window = window_size // 2
    starts = sliding_window_indices(seq_len, window_size, stride=stride)

    # Assign each SNP to its best window
    snp_best_window = {}
    for snp_idx in snp_indices:
        best_start = min(starts, key=lambda s: abs((s + half_window) - snp_idx))
        snp_best_window[snp_idx] = best_start

    # Group SNPs by window start
    window_to_snps = {}
    for snp_idx, w_start in snp_best_window.items():
        window_to_snps.setdefault(w_start, []).append(snp_idx)

    snp_to_r2 = {}

    # Process each window once
    for w_start, snps_in_window in tqdm(window_to_snps.items(), desc="Processing windows"):
        window_data = hap_array[:, w_start:w_start + window_size]

        # Mask all SNPs in this window at once
        masked_copy = window_data.copy()
        positions_in_window = [snp_idx - w_start for snp_idx in snps_in_window]
        for pos in positions_in_window:
            masked_copy[:, pos] = tokenizer.mask_token_id

        dataset = SNPDataset(torch.tensor(masked_copy, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Collect predictions for all individuals, all positions
        all_probs = []
        for batch in dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                attention_mask = torch.ones_like(batch, dtype=torch.long)
                outputs = model(input_ids=batch, attention_mask=attention_mask)
                logits = outputs.logits[..., :2]
                probs = torch.softmax(logits, dim=-1)  # shape [B, window_size, 2]

            all_probs.append(probs[:, :, 1].cpu().numpy())  # take allele=1 probs

        all_probs = np.vstack(all_probs)  # shape (num_indiv, window_size)

        # Compute R² for each masked SNP
        for snp_idx, pos_in_window in zip(snps_in_window, positions_in_window):
            snp_preds = all_probs[:, pos_in_window]        # model probs
            snp_labels = window_data[:, pos_in_window]     # true alleles
            r2 = compute_r2(snp_labels, snp_preds)
            snp_to_r2[snp_idx] = r2

    # Return R² in the order of snp_indices
    ordered_r2s = [snp_to_r2[i] for i in snp_indices]
    print(f"Mean R² over selected SNPs: {np.mean(ordered_r2s):.4f}")

    # Only do this if a MAF file is provided
    if maf_file:
        # Reload SNP names and MAFs from file
        snp_names = []
        mafs = []
        with open(maf_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    snp_names.append(parts[0])
                    mafs.append(float(parts[1]))

        # Filter SNPs if needed (same logic as before)
        if snp_filter == "common":
            filtered_indices = np.where(np.array(mafs) > 1e-1)[0]
        elif snp_filter == "rare":
            filtered_indices = np.where(np.array(mafs) < 1e-3)[0]
        elif snp_filter == "file":
            assert snp_index_file is not None, "snp_index_file must be provided when snp_filter='file'"
            with open(snp_index_file) as f:
                filtered_indices = [int(line.strip()) for line in f if line.strip()]
            filtered_indices = np.array(filtered_indices)
        else:
            filtered_indices = np.arange(len(mafs))

        with open(out_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["SNP Set", "R2", "MAF"])
            i = 0
            for idx in filtered_indices:
                writer.writerow([snp_names[idx], round(ordered_r2s[i], 6), mafs[idx]])
                i += 1

        print(f"R² results written to {out_file}")
    return ordered_r2s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained BERT model directory")
    parser.add_argument("--test_file", required=True, help="Path to test data (haplotype matrix)")
    parser.add_argument("--out", required=True, help="Path to output file")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--maf_file", type=str, help="Path to legend file with MAFs")
    parser.add_argument("--snp_filter", choices=["all", "common", "rare", "file"], default="all", help="Filter SNPs by MAF")
    parser.add_argument("--snp_index_file", help="Filter SNPs by indices")
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    infer_and_compute_r2(
        model_path=args.model_path,
        test_file=args.test_file,
        out_file=args.out,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device,
        maf_file=args.maf_file,
        snp_filter=args.snp_filter,
        snp_index_file=args.snp_index_file
    )
