"""
Adaptive Attention Computation Kernel
Team 3 - Research & Modeling

Frozen Contract:
def adaptive_rank_attention(
    Q: torch.Tensor,       # (B, H, N, d_k)
    K: torch.Tensor,       # (B, H, N, d_k)
    V: torch.Tensor,       # (B, H, N, d_v)
    delta: float = 0.05,   # output error tolerance target
    r_candidates: tuple = (64, 256, 1024),
    is_causal: bool = False
) -> torch.Tensor:         # (B, H, N, d_v)
"""

import math
import torch
import torch.nn.functional as F

EPS = 1e-6

def phi_relu(x: torch.Tensor) -> torch.Tensor:
    """Non-negative feature map: ReLU(x) + 1.0"""
    return F.relu(x) + 1.0

def softmax_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Standard scaled dot-product softmax attention: softmax(QK^T / sqrt(d_k)) V"""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

def linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    feat_fn=phi_relu,
    is_causal: bool = False
) -> torch.Tensor:
    """
    Reordered kernelized linear attention:
    phi(Q) (phi(K)^T V) / [phi(Q) (phi(K)^T 1)]
    Never materializes the N x N attention matrix.
    """
    # Force FP32 accumulator for numerical stability with FP16/BF16
    acc_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
    orig_dtype = V.dtype
    
    Q_acc = feat_fn(Q).to(acc_dtype)
    K_acc = feat_fn(K).to(acc_dtype)
    V_acc = V.to(acc_dtype)
    
    if not is_causal:
        # KV state: (B, H, r, d_v)
        KV = K_acc.transpose(-1, -2) @ V_acc
        # Normalization vector: (B, H, r, 1)
        z = K_acc.sum(dim=-2, keepdim=True).transpose(-1, -2)
        
        # Numerator: (B, H, N, d_v)
        num = Q_acc @ KV
        # Denominator: (B, H, N, 1)
        den = (Q_acc @ z).clamp_min(EPS)
        out = num / den
    else:
        # Causal prefix-sums mode
        kv = torch.einsum("bhnr,bhnd->bhnrd", K_acc, V_acc).cumsum(dim=2)
        z = K_acc.cumsum(dim=2)
        num = torch.einsum("bhnr,bhnrd->bhnd", Q_acc, kv)
        den = ((Q_acc * z).sum(dim=-1, keepdim=True) + EPS)
        out = num / den
        
    return out.to(orig_dtype)

def adaptive_rank_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    delta: float = 0.05,
    r_candidates: tuple = (64, 256, 1024),
    is_causal: bool = False
) -> torch.Tensor:
    """
    Adaptive Attention Computation:
    Dynamically routes inputs between fast linear path (r=64) and high-capacity
    retrieval path based on estimated difficulty / entropy.
    """
    # Default fast-path implementation (can be dynamically enhanced by Person 1)
    return linear_attention(Q, K, V, feat_fn=phi_relu, is_causal=is_causal)
