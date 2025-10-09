import numpy as np
import torch

# -----------------------------
# Step 1: Load coarse genetic map
# -----------------------------
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

# -----------------------------
# Step 5: Run it
# -----------------------------
map_bps, map_cms = load_coarse_map("/scratch2/prateek/DeepLearningImputation/1kg_15_map.txt")
snp_bps = load_snp_positions("/scratch2/prateek/genetic_pc_github/aux/10K_legend.maf.txt")

interp_cms = interpolate_cM_for_snps(snp_bps, map_bps, map_cms)
tau_tensor, log_tau_tensor = compute_tau(interp_cms, Ne=10000, H=4006)

print("Interpolated cM positions:\n", interp_cms)
print("Tau values:\n", tau_tensor)
print("log Tau values:\n", log_tau_tensor)