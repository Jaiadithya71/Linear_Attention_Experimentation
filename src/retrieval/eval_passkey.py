"""
src/retrieval/eval_passkey.py
=============================
Rigorous Key-Value Associative Retrieval Benchmark & Non-Inferiority Suite.
Person 3: Retrieval Lead - Team 3.

Empirical evaluation of associative retrieval under controlled layer setup:
- N in [64, 256, 512, 1024, 2048, 4096]
- 20 distinct tasks (seeds 1000 to 1019) per sequence length
- C=16 balanced orthonormal value classes (DV=64, chance level = 6.25%)
- Queries with noise=0.3 seeking target key, all other N-1 keys are distractors
- Evaluates both 'global target' and 'local target (|dist|<=64)'
- Methods: Softmax (naive), Flash/SDPA (exact), Sparse (local w=64), Linear (ReLU+1)
- 2,000 bootstrap iterations for 95% Confidence Intervals of paired differences (Delta = acc_m - acc_softmax)
- Saves authentic, empirical data to data/retrieval_raw.csv and data/noninferiority.csv
"""

import os
import sys
import math
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

# Configuration
CONFIG = {
    "RET_LENGTHS": [64, 256, 512, 1024, 2048, 4096],
    "RET_TASKS": 20,
    "N_CLASSES": 16,
    "DK": 64,
    "DV": 64,
    "W": 64,
    "KEY_SCALE": 8.0,
    "QUERY_NOISE": 0.3,
    "DELTA": 0.05,
    "N_BOOTSTRAP": 2000,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# -----------------------------------------------------------------------------
# Attention Mechanism Implementations
# -----------------------------------------------------------------------------
def phi(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) + 1.0


def softmax_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ V


def flash_sdpa_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(Q, K, V)


def linear_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    fq = phi(Q).float()
    fk = phi(K).float()
    v_acc = V.float()
    
    kv = fk.transpose(-1, -2) @ v_acc                      # (B, H, r, dv)
    z = fk.sum(dim=-2, keepdim=True)                       # (B, H, 1, r)
    num = fq @ kv                                          # (B, H, N, dv)
    den = (fq * z).sum(dim=-1, keepdim=True).clamp_min(1e-6) # (B, H, N, 1)
    return (num / den).to(V.dtype)


def sparse_local_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, w: int = CONFIG["W"]) -> torch.Tensor:
    Bq, Hq, N, d = Q.shape
    nb = math.ceil(N / w)
    Np = nb * w
    pad = Np - N

    def blocks(x: torch.Tensor) -> torch.Tensor:
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        return x.reshape(Bq, Hq, nb, w, x.shape[-1])

    def with_neighbours(xb: torch.Tensor) -> torch.Tensor:
        xp = F.pad(xb, (0, 0, 0, 0, 1, 1))
        return torch.cat([xp[:, :, :-2], xp[:, :, 1:-1], xp[:, :, 2:]], dim=3)

    Qb = blocks(Q)
    Kn = with_neighbours(blocks(K))
    Vn = with_neighbours(blocks(V))

    valid = torch.zeros(nb + 2, w, dtype=torch.bool, device=Q.device)
    valid[1:-1] = (torch.arange(Np, device=Q.device) < N).reshape(nb, w)
    mask = torch.cat([valid[:-2], valid[1:-1], valid[2:]], dim=1)

    scale = 1.0 / math.sqrt(d)
    s = (Qb @ Kn.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask[None, None, :, None, :], float("-inf"))
    out = torch.softmax(s, dim=-1) @ Vn
    return out.reshape(Bq, Hq, Np, -1)[:, :, :N]


METHODS = {
    "softmax (naive)": softmax_attention,
    "flash/SDPA (exact)": flash_sdpa_attention,
    "sparse (local w=64)": sparse_local_attention,
    "linear (ReLU+1)": linear_attention,
}

# -----------------------------------------------------------------------------
# Benchmark Task Generator & Accuracy Evaluator
# -----------------------------------------------------------------------------
def make_retrieval_task(
    N: int,
    seed: int,
    max_dist: Optional[int] = None,
    key_scale: float = CONFIG["KEY_SCALE"],
    noise: float = CONFIG["QUERY_NOISE"],
    c: int = CONFIG["N_CLASSES"],
    dk: int = CONFIG["DK"],
    dv: int = CONFIG["DV"],
    device: torch.device = CONFIG["DEVICE"]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs an authentic key-value associative retrieval task.
    N keys are normalized unit vectors scaled by key_scale.
    Each key is mapped to one of C orthonormal value classes.
    Query is a noisy copy of the target key.
    """
    gen = torch.Generator().manual_seed(seed)
    # Generate orthonormal class codebook via QR decomposition
    codes = torch.linalg.qr(torch.randn(dv, c, generator=gen))[0].T.contiguous().to(device) # (C, DV)
    keys = F.normalize(torch.randn(N, dk, generator=gen), dim=-1).to(device)
    cls = (torch.arange(N) % c)[torch.randperm(N, generator=gen)].to(device)

    if max_dist is None:
        # Global needle target (needle can be anywhere in [0, N-1])
        tgt = torch.randperm(N, generator=gen).to(device)
    else:
        # Local needle target (|distance| <= max_dist)
        offsets = torch.randint(-max_dist, max_dist + 1, (N,), generator=gen).to(device)
        tgt = (torch.arange(N, device=device) + offsets).clamp(0, N - 1)

    q_noise = F.normalize(torch.randn(N, dk, generator=gen, device=device), dim=-1)
    q = F.normalize(keys[tgt] + noise * q_noise, dim=-1)

    # Reshape for multi-head attention (B=1, H=1, N, D)
    T = lambda x: x[None, None].to(device, dtype=torch.float32)
    return (
        T(key_scale * q),
        T(key_scale * keys),
        T(codes[cls]),
        codes,
        cls[tgt]
    )


def compute_retrieval_accuracy(fn, task: Tuple) -> float:
    """Computes exact top-1 retrieval classification accuracy."""
    Q, K, V, codes, labels = task
    out = fn(Q, K, V)[0, 0] # (N, DV)
    # Nearest neighbor cosine matching to class codebook
    pred_classes = (F.normalize(out.float(), dim=-1) @ codes.T).argmax(dim=-1)
    return pred_classes.eq(labels).float().mean().item()


def bootstrap_ci(diffs: np.ndarray, n: int = CONFIG["N_BOOTSTRAP"], seed: int = 42) -> Tuple[float, float]:
    """Computes 95% bootstrap confidence interval [ci_lo, ci_hi] over independent tasks."""
    diffs = np.asarray(diffs)
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


# -----------------------------------------------------------------------------
# Main Experiment Execution Suite
# -----------------------------------------------------------------------------
def run_retrieval_experiment(output_dir: str = os.path.join(ROOT_DIR, "data")):
    os.makedirs(output_dir, exist_ok=True)
    lengths = CONFIG["RET_LENGTHS"]
    n_tasks = CONFIG["RET_TASKS"]
    device = CONFIG["DEVICE"]
    w = CONFIG["W"]
    delta_threshold = CONFIG["DELTA"]

    print("=" * 75)
    print("RUNNING AUTHENTIC P3 KEY-VALUE RETRIEVAL EXPERIMENTS")
    print(f"Device: {device} | Sequence Lengths: {lengths} | Tasks per length: {n_tasks}")
    print(f"Classes: {CONFIG['N_CLASSES']} (Chance = {1.0/CONFIG['N_CLASSES']:.4f}) | Non-inferiority margin: {delta_threshold}")
    print("=" * 75)

    modes = [
        ("global target", None),
        (f"local target (|dist|<={w})", w)
    ]

    raw_records = []
    t_start = time.time()

    for mode_name, max_dist in modes:
        print(f"\n[MODE] Starting evaluation: {mode_name}")
        for N in lengths:
            t0 = time.time()
            for sd in range(n_tasks):
                seed = 1000 + sd
                task = make_retrieval_task(N=N, seed=seed, max_dist=max_dist, device=device)

                for method_name, method_fn in METHODS.items():
                    acc = compute_retrieval_accuracy(method_fn, task)
                    raw_records.append({
                        "mode": mode_name,
                        "N": N,
                        "task": sd,
                        "method": method_name,
                        "acc": round(acc, 4)
                    })
            dt = time.time() - t0
            print(f"  N = {N:5d} completed ({n_tasks} tasks x {len(METHODS)} methods) in {dt:.2f}s")

    ret_df = pd.DataFrame(raw_records)
    raw_path = os.path.join(output_dir, "retrieval_raw.csv")
    ret_df.to_csv(raw_path, index=False)
    print(f"\n[SAVED] Raw retrieval results saved to: {raw_path} ({len(ret_df)} rows)")

    # Print Summary Tables
    print("\n" + "=" * 75)
    print("MEASURED EMPIRICAL RETRIEVAL ACCURACY (MEAN ACROSS 20 TASKS)")
    print("=" * 75)
    for mode_name, _ in modes:
        print(f"\nMode: {mode_name}")
        pivot = ret_df[ret_df["mode"] == mode_name].pivot_table(
            index="N", columns="method", values="acc"
        )[list(METHODS.keys())]
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    # Paired Differences & Non-Inferiority Testing
    print("\n" + "=" * 75)
    print(f"NON-INFERIORITY HYPOTHESIS TEST (Margin delta = -{delta_threshold})")
    print("=" * 75)

    ni_records = []
    for mode_name, _ in modes:
        for N in lengths:
            sub = ret_df[(ret_df["mode"] == mode_name) & (ret_df["N"] == N)]
            ref_accs = sub[sub["method"] == "softmax (naive)"].sort_values("task")["acc"].values

            for method_name in ["linear (ReLU+1)", "sparse (local w=64)", "flash/SDPA (exact)"]:
                m_accs = sub[sub["method"] == method_name].sort_values("task")["acc"].values
                diffs = m_accs - ref_accs
                delta_mean = float(np.mean(diffs))
                ci_lo, ci_hi = bootstrap_ci(diffs, n=CONFIG["N_BOOTSTRAP"])
                is_non_inferior = ci_lo > -delta_threshold

                ni_records.append({
                    "mode": mode_name,
                    "N": N,
                    "method": method_name,
                    "delta": round(delta_mean, 4),
                    "ci_lo": round(ci_lo, 4),
                    "ci_hi": round(ci_hi, 4),
                    "non_inferior": is_non_inferior
                })

    ni_df = pd.DataFrame(ni_records)
    ni_path = os.path.join(output_dir, "noninferiority.csv")
    ni_df.to_csv(ni_path, index=False)
    print(f"[SAVED] Non-inferiority decisions saved to: {ni_path}")
    print("\nSummary Table:")
    print(ni_df.to_string(index=False))

    # Also mirror into Linear_Attention_Experimentation/data
    alt_dir = os.path.join(ROOT_DIR, "Linear_Attention_Experimentation", "data")
    if os.path.exists(alt_dir):
        ret_df.to_csv(os.path.join(alt_dir, "retrieval_raw.csv"), index=False)
        ni_df.to_csv(os.path.join(alt_dir, "noninferiority.csv"), index=False)
        print(f"[MIRROR] Mirrored authentic CSVs to: {alt_dir}")

    total_time = time.time() - t_start
    print(f"\n[DONE] P3 Retrieval suite completed in {total_time:.1f}s.")
    return ret_df, ni_df


if __name__ == "__main__":
    run_retrieval_experiment()
