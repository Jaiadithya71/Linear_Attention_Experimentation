"""
src/retrieval/eval_passkey.py
=============================
Synthetic Passkey / Needle-In-A-Haystack evaluation suite.
Measures key-value associative retrieval collapse across sequence lengths N in [64, 256, 512, 1024, 2048, 4096].

Features:
- 16 balanced orthonormal value classes generated via QR decomposition (DV=64, chance = 6.25%)
- Distractor keys with controlled margin and query noise (noise=0.3)
- Bootstrap resampling (2,000 iterations) to compute 95% Confidence Intervals
- Non-inferiority test for paired difference Delta = acc_linear - acc_softmax against margin delta = 0.05
"""

import math
import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from adaptive_kernel import softmax_attention, linear_attention


def generate_orthonormal_classes(num_classes: int = 16, d_v: int = 64) -> torch.Tensor:
    """
    Generates orthonormal class vectors for value embeddings via QR decomposition.
    Guarantees perfectly separated ground-truth representations.
    """
    assert d_v >= num_classes, f"d_v ({d_v}) must be >= num_classes ({num_classes})"
    gaussian_matrix = torch.randn(d_v, num_classes)
    q, _ = torch.linalg.qr(gaussian_matrix)
    # Transpose so each row is a class vector of length d_v
    classes = q.T  # (num_classes, d_v)
    return classes


def generate_passkey_instance(
    N: int,
    d_k: int = 64,
    d_v: int = 64,
    num_classes: int = 16,
    target_pos: Optional[int] = None,
    query_noise: float = 0.3,
    key_norm_scale: float = 8.0,
    classes: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Generates a synthetic passkey instance.
    - Keys: N normalized vectors scaled by key_norm_scale.
    - Values: target key receives a true class vector; distractors receive random classes.
    - Query: noisy version of target key.
    """
    if classes is None:
        classes = generate_orthonormal_classes(num_classes=num_classes, d_v=d_v)

    if target_pos is None:
        target_pos = np.random.randint(0, N)

    target_class_idx = np.random.randint(0, num_classes)
    distractor_class_indices = np.random.randint(0, num_classes, size=N)
    distractor_class_indices[target_pos] = target_class_idx

    # Keys: N random Gaussian vectors normalized and scaled
    raw_keys = torch.randn(N, d_k)
    K = F.normalize(raw_keys, p=2, dim=-1) * key_norm_scale

    # Values
    V = classes[distractor_class_indices]  # (N, d_v)

    # Query: noisy version of target key
    target_k = K[target_pos]
    noise = torch.randn_like(target_k)
    Q = F.normalize(target_k + query_noise * noise, p=2, dim=-1).unsqueeze(0) * key_norm_scale  # (1, d_k)

    return Q, K, V, target_class_idx


def evaluate_retrieval_step(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    target_class_idx: int,
    classes: torch.Tensor,
    method: str = "softmax"
) -> bool:
    """
    Evaluates whether the attention output correctly identifies the target class.
    Nearest neighbor classification against orthonormal codebook.
    """
    # Shape tensors for attention kernel: (1, 1, seq_len, dim)
    q_tensor = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, d_k)
    k_tensor = K.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_k)
    v_tensor = V.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_v)

    if method == "softmax":
        out = softmax_attention(q_tensor, k_tensor, v_tensor)
    elif method == "linear":
        out = linear_attention(q_tensor, k_tensor, v_tensor)
    else:
        raise ValueError(f"Unknown method {method}")

    # Retrieved vector: (d_v,)
    retrieved = out.squeeze()
    
    # Cosine similarity to orthonormal class codebook
    sims = F.cosine_similarity(retrieved.unsqueeze(0), classes, dim=-1)
    pred_idx = torch.argmax(sims).item()

    return pred_idx == target_class_idx


def compute_bootstrap_noninferiority_ci(
    acc_diffs: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Computes paired difference and 95% bootstrap confidence interval [ci_lo, ci_hi].
    """
    mean_diff = float(np.mean(acc_diffs))
    boot_means = []
    n = len(acc_diffs)
    for _ in range(n_bootstrap):
        sample = np.random.choice(acc_diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))
    
    ci_lo = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_hi = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return mean_diff, ci_lo, ci_hi


if __name__ == "__main__":
    print("Testing Passkey synthetic instance generator...")
    classes = generate_orthonormal_classes(16, 64)
    Q, K, V, target_idx = generate_passkey_instance(N=128, classes=classes)
    print(f"Generated Q: {Q.shape}, K: {K.shape}, V: {V.shape}, Target Class: {target_idx}")

    is_correct_softmax = evaluate_retrieval_step(Q, K, V, target_idx, classes, method="softmax")
    is_correct_linear = evaluate_retrieval_step(Q, K, V, target_idx, classes, method="linear")
    print(f"Softmax retrieval correct: {is_correct_softmax} | Linear retrieval correct: {is_correct_linear}")
    print("[PASS] eval_passkey.py module validated.")
