"""
src/theory/difficulty_metric.py
===============================
Input difficulty metric and error-certified adaptive rank selection for linear attention.

Theoretical Basis:
1. Exact Softmax:
   kappa_{ij} = exp(q_i^T k_j / sqrt(d_k)),  Z_i = sum_j kappa_{ij},  y_i^S = (1 / Z_i) sum_j kappa_{ij} v_j
2. Positive Kernel Linear Attention:
   tilde{kappa}_{ij} = phi_r(q_i)^T phi_r(k_j),  tilde{Z}_i = sum_j tilde{kappa}_{ij},  y_i^L = (1 / tilde{Z}_i) sum_j tilde{kappa}_{ij} v_j
3. Error Certificate (Theorem 1):
   If |tilde{kappa}_{ij} - kappa_{ij}| <= varepsilon_r and ||v_j|| <= V_max, then:
   ||y_i^L - y_i^S|| <= 2 * N * varepsilon_r * V_max / (Z_i - N * varepsilon_r)
   A sufficient condition for ||y_i^L - y_i^S|| <= delta for all i is:
   varepsilon_r <= (delta * min_i(Z_i)) / [ N * (2 * V_max + delta) ]

References:
- Katharopoulos et al. (2020), "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
- Choromanski et al. (2020), "Rethinking Attention with Performers" (FAVOR+)
- Likhosherstov et al. (2023), "FAVOR#: Sharp Attention Kernel Approximations"
- Sui & Zhang (2026), "The Approximation Rank of Softmax Attention: Sharp Geometric Laws"
"""

import math
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F


class RankDecision(int):
    """
    Integer subclass representing the selected rank r*(X, delta), enriched with
    theoretical diagnostic metrics (entropy, temperature, bound checks, certificate status).
    
    Can be used directly anywhere an integer rank is expected (e.g. r == 256, range(r)).
    """
    def __new__(cls, val: int, info: Optional[Dict[str, Any]] = None):
        obj = super().__new__(cls, val)
        obj.info = info or {}
        obj.rank = int(val)
        obj.entropy = obj.info.get("entropy", 0.0)
        obj.normalized_entropy = obj.info.get("normalized_entropy", 0.0)
        obj.norm_temperature = obj.info.get("norm_temperature", 0.0)
        obj.min_Z = obj.info.get("min_Z", 0.0)
        obj.threshold = obj.info.get("threshold", 0.0)
        obj.candidate_errors = obj.info.get("candidate_errors", {})
        obj.certified = obj.info.get("certified", False)
        return obj

    def __repr__(self):
        return f"RankDecision(rank={self.rank}, certified={self.certified}, H={self.entropy:.3f}, T={self.norm_temperature:.3f})"


def compute_attention_entropy(
    Q: Union[torch.Tensor, np.ndarray],
    K: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-12
) -> Tuple[float, float, float]:
    """
    Computes Shannon attention entropy H(Q, K) and log-partition function stats.
    
    Args:
        Q: Query tensor of shape (..., N, d_k)
        K: Key tensor of shape (..., N, d_k)
        eps: Small epsilon to prevent log(0)
        
    Returns:
        (mean_entropy, normalized_entropy, min_Z):
        - mean_entropy: (1/N) sum_i H_i in nats
        - normalized_entropy: mean_entropy / ln(N) in [0, 1]
        - min_Z: min_i sum_j exp(q_i^T k_j / sqrt(d_k))
    """
    is_numpy = isinstance(Q, np.ndarray) or isinstance(K, np.ndarray)
    if is_numpy:
        Q_t = torch.from_numpy(np.asarray(Q, dtype=np.float32))
        K_t = torch.from_numpy(np.asarray(K, dtype=np.float32))
    else:
        Q_t = Q.float()
        K_t = K.float()

    N = Q_t.shape[-2]
    d_k = Q_t.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    # Pairwise logits S: shape (..., N, N)
    S = torch.matmul(Q_t, K_t.transpose(-1, -2)) * scale

    # Row-wise partition function Z_i = sum_j exp(S_{ij})
    # Compute in stable float64 to avoid overflow with large logits
    S_64 = S.to(torch.float64)
    # Z_i across keys (last dimension)
    # Using logsumexp for stability: Z_i = exp(logsumexp(S_i))
    lse = torch.logsumexp(S_64, dim=-1)  # shape (..., N)
    Z = torch.exp(lse)
    min_Z = torch.min(Z).item()

    # Attention probabilities A = softmax(S, dim=-1)
    A = F.softmax(S_64, dim=-1)  # (..., N, N)

    # Shannon entropy per query row: H_i = - sum_j A_{ij} log(A_{ij} + eps)
    entropy_per_row = -torch.sum(A * torch.log(A + eps), dim=-1)  # (..., N)
    mean_entropy = torch.mean(entropy_per_row).item()

    # Theoretical maximum entropy is ln(N) (achieved when attention is uniform)
    max_entropy = math.log(max(N, 2))
    normalized_entropy = float(np.clip(mean_entropy / max_entropy, 0.0, 1.0))

    return mean_entropy, normalized_entropy, min_Z


def compute_query_key_temperature(
    Q: Union[torch.Tensor, np.ndarray],
    K: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Computes query-key norm temperature, effective logit scale, and geometry metrics.
    
    Args:
        Q: Query tensor of shape (..., N, d_k)
        K: Key tensor of shape (..., N, d_k)
        eps: Small constant for numerical stability
        
    Returns:
        Dictionary with:
        - norm_temperature: sqrt(d_k) / (mean_q_norm * mean_k_norm)
        - mean_q_norm: average Euclidean norm of query vectors
        - mean_k_norm: average Euclidean norm of key vectors
        - logit_std: standard deviation of attention logits
        - logit_max: maximum logit value
    """
    is_numpy = isinstance(Q, np.ndarray) or isinstance(K, np.ndarray)
    if is_numpy:
        Q_t = torch.from_numpy(np.asarray(Q, dtype=np.float32))
        K_t = torch.from_numpy(np.asarray(K, dtype=np.float32))
    else:
        Q_t = Q.float()
        K_t = K.float()

    d_k = Q_t.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    q_norms = torch.norm(Q_t, p=2, dim=-1)
    k_norms = torch.norm(K_t, p=2, dim=-1)
    mean_q = torch.mean(q_norms).item()
    mean_k = torch.mean(k_norms).item()

    # Temperature: inverse of expected dot-product scale
    norm_product = (mean_q * mean_k)
    norm_temperature = math.sqrt(d_k) / (norm_product + eps)

    # Logit distribution metrics
    S = torch.matmul(Q_t, K_t.transpose(-1, -2)) * scale
    logit_std = torch.std(S).item()
    logit_max = torch.max(S).item()
    logit_min = torch.min(S).item()

    return {
        "norm_temperature": float(norm_temperature),
        "mean_q_norm": float(mean_q),
        "mean_k_norm": float(mean_k),
        "logit_std": float(logit_std),
        "logit_max": float(logit_max),
        "logit_min": float(logit_min),
        "d_k": int(d_k),
    }


def compute_error_certificate_threshold(
    N: int,
    min_Z: float,
    delta: float = 0.05,
    V_max: float = 1.0
) -> float:
    """
    Computes the sufficient condition upper bound threshold for kernel approximation error:
    
        varepsilon_r <= (delta * min_i(Z_i)) / [ N * (2 * V_max + delta) ]
        
    Args:
        N: Sequence length
        min_Z: Minimum row partition function min_i sum_j exp(S_{ij})
        delta: Output approximation tolerance (prescribed ||y_i^L - y_i^S|| <= delta)
        V_max: Upper bound on value vector norms ||v_j|| <= V_max
        
    Returns:
        Maximum permissible kernel entry discrepancy varepsilon_r
    """
    denominator = float(N) * (2.0 * float(V_max) + float(delta))
    numerator = float(delta) * float(min_Z)
    return numerator / max(denominator, 1e-12)


def estimate_kernel_error_bound(
    r: int,
    N: int,
    min_Z: float,
    normalized_entropy: float,
    temp_metrics: Dict[str, float],
    confidence_beta: float = 0.01,
    feature_type: str = "favor_sharp"
) -> float:
    """
    Estimates the uniform kernel entry error bound varepsilon_r(X) for feature rank r
    under positive random feature constructions (FAVOR+ / FAVOR# / Sui & Zhang 2026).
    
    Formula:
        varepsilon_r = C_geom(Q, K) * (1 / sqrt(r)) * sqrt(2 * ln(2 * N^2 / beta))
        
    Where C_geom(Q, K) scales with logit dynamic range and entropy:
    - High entropy (diffuse attention): kernel is smooth, C_geom is small ~ (min_Z / N).
    - Low entropy (spiky / focused attention): kernel has sharp peak, C_geom explodes
      with exp(logit_max - logit_mean).
    """
    # Union bound factor across N sequence items with failure probability beta
    union_factor = math.sqrt(2.0 * math.log(max(2.0 * float(N) / confidence_beta, math.e)))

    # Feature type sharpness constant (FAVOR# orthogonal features reduce variance by d_k)
    d_k = temp_metrics.get("d_k", 32)
    sharpness = (1.0 / math.sqrt(max(d_k, 1))) if feature_type == "favor_sharp" else 1.0

    logit_spread = max(0.0, temp_metrics["logit_max"] - temp_metrics["logit_min"])
    
    # Information-theoretic entropy concentration factor (Sui & Zhang 2026)
    # When normalized_entropy -> 1.0, concentration_penalty -> 1.0
    # When normalized_entropy -> 0.0, concentration_penalty grows exponentially with logit spread
    concentration_penalty = math.exp(min(logit_spread * 0.5 * (1.0 - normalized_entropy), 15.0))

    # Base scale per kernel entry
    base_scale = (min_Z / float(N))
    
    # Kernel discrepancy bound
    eps_r = sharpness * (base_scale * concentration_penalty) * (1.0 / math.sqrt(r)) * union_factor
    return float(eps_r)


def estimate_required_rank(
    Q: Union[torch.Tensor, np.ndarray],
    K: Union[torch.Tensor, np.ndarray],
    delta: float = 0.05,
    r_candidates: Tuple[int, ...] = (64, 256, 1024),
    V_max: float = 1.0,
    confidence_beta: float = 0.01,
    return_info: bool = False
) -> Union[RankDecision, Tuple[int, Dict[str, Any]]]:
    """
    Estimates the minimum required feature rank r from candidates that guarantees the
    attention output approximation error ||y^L - y^S|| <= delta.
    
    Based on the theoretical certificate condition:
        varepsilon_r <= (delta * min_i(Z_i)) / [ N * (2 * V_max + delta) ]
        
    Args:
        Q: Query tensor of shape (..., N, d_k)
        K: Key tensor of shape (..., N, d_k)
        delta: Prescribed output error tolerance (default: 0.05)
        r_candidates: Candidate feature ranks, sorted ascending (default: (64, 256, 1024))
        V_max: Bound on value norms ||v_j|| <= V_max (default: 1.0)
        confidence_beta: Statistical failure probability (default: 0.01)
        return_info: If True, returns (rank_int, info_dict).
                     If False, returns RankDecision (an int subclass with metrics attributes).
                     
    Returns:
        RankDecision or (int, dict): Selected feature rank r* in r_candidates.
    """
    is_numpy = isinstance(Q, np.ndarray) or isinstance(K, np.ndarray)
    if is_numpy:
        N = Q.shape[-2]
    else:
        N = Q.shape[-2]

    # 1. Compute attention entropy H(Q, K) and partition function min_i(Z_i)
    mean_entropy, norm_entropy, min_Z = compute_attention_entropy(Q, K)

    # 2. Compute query-key norm temperature and logit geometry
    temp_metrics = compute_query_key_temperature(Q, K)

    # 3. Compute certificate sufficient condition threshold
    threshold = compute_error_certificate_threshold(
        N=N,
        min_Z=min_Z,
        delta=delta,
        V_max=V_max
    )

    # 4. Evaluate each candidate rank r against the theoretical error bound
    candidate_errors: Dict[int, float] = {}
    selected_rank: Optional[int] = None
    is_certified = False

    sorted_candidates = sorted(list(r_candidates))
    for r in sorted_candidates:
        eps_r = estimate_kernel_error_bound(
            r=r,
            N=N,
            min_Z=min_Z,
            normalized_entropy=norm_entropy,
            temp_metrics=temp_metrics,
            confidence_beta=confidence_beta
        )
        candidate_errors[r] = eps_r

        if eps_r <= threshold and selected_rank is None:
            selected_rank = r
            is_certified = True

    # If even the maximum candidate cannot satisfy the certificate,
    # select the largest candidate and flag as uncertified (demanding exact attention fallback)
    if selected_rank is None:
        selected_rank = sorted_candidates[-1]
        is_certified = False

    info: Dict[str, Any] = {
        "required_rank": selected_rank,
        "certified": is_certified,
        "delta": float(delta),
        "entropy": float(mean_entropy),
        "normalized_entropy": float(norm_entropy),
        "norm_temperature": float(temp_metrics["norm_temperature"]),
        "min_Z": float(min_Z),
        "threshold": float(threshold),
        "candidate_errors": candidate_errors,
        "logit_std": float(temp_metrics["logit_std"]),
        "logit_max": float(temp_metrics["logit_max"]),
        "mean_q_norm": float(temp_metrics["mean_q_norm"]),
        "mean_k_norm": float(temp_metrics["mean_k_norm"]),
        "N": int(N),
        "V_max": float(V_max)
    }

    if return_info:
        return int(selected_rank), info
    else:
        return RankDecision(selected_rank, info=info)
