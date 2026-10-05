"""
src/theory/adaptive_kernel.py
=============================
Production implementation of Error-Controlled Adaptive-Rank Linear Attention.

Contract:
    def adaptive_rank_attention(Q, K, V, delta=0.05, r_candidates=(64, 256, 1024), is_causal=False, fallback_to_exact=True)

Guardrails enforced:
- FP32 accumulators for all summations across sequence length N
- Denominator clamping with + 1e-6 to avoid division-by-zero
- Strict kernel substitution clarity: (phi(Q) phi(K)^T) V = phi(Q) (phi(K)^T V)
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn.functional as F

from src.theory.difficulty_metric import estimate_required_rank, RankDecision


def make_positive_random_features(
    d_k: int,
    r: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    orthogonal: bool = True
):
    """
    Constructs FAVOR+ / FAVOR# positive random feature mapping phi: R^{d_k} -> R^r.
    """
    gen = torch.Generator().manual_seed(seed)
    
    if orthogonal and r >= d_k:
        # Structured orthogonal random features (FAVOR#)
        num_blocks = math.ceil(r / d_k)
        blocks = []
        for _ in range(num_blocks):
            G = torch.randn(d_k, d_k, generator=gen, dtype=torch.float32)
            q, _ = torch.linalg.qr(G)
            # Row multiplier from Chi distribution
            d = torch.randn(d_k, generator=gen).norm(p=2)
            blocks.append(q * d)
        W = torch.cat(blocks, dim=0)[:r].to(device=device, dtype=dtype)
    else:
        # Standard Gaussian random projections (FAVOR+)
        W = torch.randn(r, d_k, generator=gen, device=device, dtype=dtype)

    def phi(x: torch.Tensor) -> torch.Tensor:
        # Positive random feature map for softmax approximation:
        # phi(x) = exp(x @ W^T - ||x||^2 / 2) / sqrt(r)
        # Scale x by d_k^(-0.25) so that (x @ W^T) has variance scaling matching 1/sqrt(d_k)
        x_scaled = x.float() * (d_k ** -0.25)
        proj = torch.matmul(x_scaled, W.t())
        norm_sq = torch.sum(x_scaled ** 2, dim=-1, keepdim=True)
        return (torch.exp(proj - norm_sq * 0.5) / math.sqrt(r)).to(x.dtype)

    return phi


def linear_attention_reordered(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    feat_fn,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Reordered non-causal linear attention with O(N r d_v) complexity.
    Never materializes the N x N attention matrix.
    Enforces FP32 accumulation to prevent half-precision overflow.
    """
    orig_dtype = V.dtype
    acc_dtype = torch.float32

    fq = feat_fn(Q).to(acc_dtype)  # (..., N, r)
    fk = feat_fn(K).to(acc_dtype)  # (..., N, r)
    v_acc = V.to(acc_dtype)        # (..., N, d_v)

    # Fixed-size state: S = phi(K)^T V of shape (..., r, d_v)
    kv_state = torch.matmul(fk.transpose(-1, -2), v_acc)

    # Denominator normalizer: Z = phi(K)^T 1 of shape (..., r, 1)
    k_sum = torch.sum(fk, dim=-2, keepdim=True).transpose(-1, -2)

    # Associative contraction: phi(Q) (phi(K)^T V)
    num = torch.matmul(fq, kv_state)           # (..., N, d_v)
    den = torch.matmul(fq, k_sum) + eps        # (..., N, 1)

    out = num / den
    return out.to(orig_dtype)


def causal_linear_attention_reordered(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    feat_fn,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Causal reordered linear attention via running prefix-sums.
    Enforces FP32 accumulation.
    """
    orig_dtype = V.dtype
    acc_dtype = torch.float32

    fq = feat_fn(Q).to(acc_dtype)  # (..., N, r)
    fk = feat_fn(K).to(acc_dtype)  # (..., N, r)
    v_acc = V.to(acc_dtype)        # (..., N, d_v)

    # Outer product per token: (..., N, r, d_v)
    # Cumulative sum over sequence dimension (dim=-3)
    kv_outer = torch.einsum("...nr,...nd->...nrd", fk, v_acc)
    kv_prefix = torch.cumsum(kv_outer, dim=-3)

    # Running sum of keys: (..., N, r)
    k_prefix = torch.cumsum(fk, dim=-2)

    # Numerator per query:
    num = torch.einsum("...nr,...nrd->...nd", fq, kv_prefix)
    # Denominator per query:
    den = torch.sum(fq * k_prefix, dim=-1, keepdim=True) + eps

    out = num / den
    return out.to(orig_dtype)


def adaptive_rank_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    delta: float = 0.05,
    r_candidates: Tuple[int, ...] = (64, 256, 1024),
    is_causal: bool = False,
    fallback_to_exact: bool = True,
    feature_family: str = "favor_sharp"
) -> Tuple[torch.Tensor, RankDecision]:
    """
    Main entrypoint for Error-Controlled Adaptive-Rank Linear Attention.
    
    1. Evaluates input difficulty metric and error certificate condition:
       varepsilon_r <= delta * min_i(Z_i) / [N * (2*V_max + delta)]
    2. Dynamically selects optimal feature rank r*(X, delta) in r_candidates.
    3. If certificate passes, computes linear attention with rank r*.
       If certificate fails and fallback_to_exact is True, routes query to SDPA.
       
    Args:
        Q: Query tensor of shape (..., N, d_k)
        K: Key tensor of shape (..., N, d_k)
        V: Value tensor of shape (..., N, d_v)
        delta: Error tolerance (default 0.05)
        r_candidates: Candidates (64, 256, 1024)
        is_causal: Whether to apply causal masking
        fallback_to_exact: If True, uses SDPA when certificate cannot be satisfied
        feature_family: "favor_sharp" or "relu"
        
    Returns:
        (output, rank_decision): Output tensor and RankDecision diagnostic object
    """
    d_k = Q.shape[-1]
    device = Q.device
    v_max = torch.norm(V.float(), p=2, dim=-1).max().item()
    v_max = max(v_max, 1.0)

    # 1. Estimate required rank using theoretical difficulty metric
    decision = estimate_required_rank(
        Q=Q,
        K=K,
        delta=delta,
        r_candidates=r_candidates,
        V_max=v_max,
        return_info=False
    )

    # 2. If certificate failed and fallback is requested, execute exact SDPA
    if not decision.certified and fallback_to_exact:
        scale = 1.0 / math.sqrt(d_k)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            scale=scale,
            is_causal=is_causal
        )
        return out, decision

    # 3. Otherwise, execute linear attention with dynamically selected rank r*
    r_star = decision.rank

    if feature_family == "relu":
        def phi_feat(x):
            return F.relu(x) + 1.0
    else:
        phi_feat = make_positive_random_features(
            d_k=d_k,
            r=r_star,
            device=device,
            dtype=Q.dtype,
            orthogonal=True
        )

    if is_causal:
        out = causal_linear_attention_reordered(Q, K, V, feat_fn=phi_feat)
    else:
        out = linear_attention_reordered(Q, K, V, feat_fn=phi_feat)

    return out, decision
