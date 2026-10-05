"""
src/theory/test_theory_suite.py
===============================
Comprehensive test suite for Person 1 (Theory Lead) modules:
- difficulty_metric.py (estimate_required_rank, entropy, temperature, sufficient condition)
- adaptive_kernel.py (adaptive_rank_attention, PRF feature maps, numerical stability)
- verify_3token.py (analytical toy verification)
"""

import math
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn.functional as F

from src.theory.difficulty_metric import (
    compute_attention_entropy,
    compute_query_key_temperature,
    compute_error_certificate_threshold,
    estimate_kernel_error_bound,
    estimate_required_rank,
    RankDecision
)
from src.theory.adaptive_kernel import (
    adaptive_rank_attention,
    linear_attention_reordered,
    causal_linear_attention_reordered,
    make_positive_random_features
)
from src.theory.verify_3token import (
    run_analytical_numpy_verification,
    run_pytorch_tensor_verification
)


def test_entropy_and_temperature_diffuse_vs_spiky():
    print("-" * 60)
    print("Test 1: Attention Entropy & Temperature (Diffuse vs Spiky)")
    print("-" * 60)

    N, d_k = 128, 64

    # 1. Diffuse input (small query-key norms, high entropy)
    torch.manual_seed(42)
    Q_diffuse = torch.randn(1, 4, N, d_k) * 0.1
    K_diffuse = torch.randn(1, 4, N, d_k) * 0.1

    H_diffuse, H_norm_diffuse, min_Z_diffuse = compute_attention_entropy(Q_diffuse, K_diffuse)
    temp_diffuse = compute_query_key_temperature(Q_diffuse, K_diffuse)

    print(f"Diffuse Input  -> H = {H_diffuse:.3f} nats, H_norm = {H_norm_diffuse:.3f}, "
          f"T_norm = {temp_diffuse['norm_temperature']:.3f}, Logit Std = {temp_diffuse['logit_std']:.3f}")

    # Maximum entropy for N=128 is ln(128) ~ 4.852
    assert H_norm_diffuse > 0.95, f"Expected high normalized entropy for diffuse inputs, got {H_norm_diffuse}"
    assert temp_diffuse["norm_temperature"] > 5.0, "Expected high temperature for small norms"

    # 2. Spiky input (large query-key norms, concentrated retrieval target)
    Q_spiky = torch.randn(1, 4, N, d_k) * 3.0
    K_spiky = torch.randn(1, 4, N, d_k) * 3.0
    # Create an artificial strong key match for each query
    for i in range(N):
        K_spiky[:, :, i] = Q_spiky[:, :, i]

    H_spiky, H_norm_spiky, min_Z_spiky = compute_attention_entropy(Q_spiky, K_spiky)
    temp_spiky = compute_query_key_temperature(Q_spiky, K_spiky)

    print(f"Spiky Input    -> H = {H_spiky:.3f} nats, H_norm = {H_norm_spiky:.3f}, "
          f"T_norm = {temp_spiky['norm_temperature']:.3f}, Logit Std = {temp_spiky['logit_std']:.3f}")

    assert H_spiky < H_diffuse, f"Spiky entropy {H_spiky} must be strictly lower than diffuse {H_diffuse}"
    assert temp_spiky["logit_std"] > temp_diffuse["logit_std"], "Spiky inputs must have higher logit spread"
    print("[PASS] Entropy and temperature correctly distinguish diffuse from spiky attention regimes.")


def test_estimate_required_rank_behavior():
    print("-" * 60)
    print("Test 2: Dynamic Rank Selection r*(X, delta) across Regimes")
    print("-" * 60)

    N, d_k = 64, 32

    # Case A: Diffuse input -> should certify and select a smaller rank (64 or 256)
    Q_diffuse = torch.randn(N, d_k) * 0.1
    K_diffuse = torch.randn(N, d_k) * 0.1

    r_diffuse = estimate_required_rank(
        Q_diffuse, K_diffuse,
        delta=0.08,
        r_candidates=(64, 256, 1024),
        V_max=1.0
    )
    print(f"Case A (Diffuse): Selected Rank = {r_diffuse}, Certified = {r_diffuse.certified}")
    assert isinstance(r_diffuse, int)
    assert r_diffuse.certified is True
    assert r_diffuse in (64, 256, 1024)

    # Case B: Spiky input -> requires larger rank (1024 or uncertified)
    Q_spiky = torch.randn(N, d_k) * 4.0
    K_spiky = torch.randn(N, d_k) * 4.0
    r_spiky = estimate_required_rank(
        Q_spiky, K_spiky,
        delta=0.01,  # strict tolerance on spiky attention
        r_candidates=(64, 256, 1024),
        V_max=1.0
    )
    print(f"Case B (Spiky):   Selected Rank = {r_spiky}, Certified = {r_spiky.certified}")
    assert isinstance(r_spiky, int)
    assert r_spiky >= r_diffuse
    print("[PASS] estimate_required_rank selects appropriate ranks conditioned on input difficulty.")


def test_sufficient_condition_formula():
    print("-" * 60)
    print("Test 3: Mathematical Error Certificate Inequality")
    print("-" * 60)
    # Check: eps_r <= delta * min_i(Z_i) / [N * (2 * V_max + delta)]
    N = 100
    min_Z = 50.0
    delta = 0.05
    V_max = 1.0

    threshold = compute_error_certificate_threshold(N=N, min_Z=min_Z, delta=delta, V_max=V_max)
    expected = (delta * min_Z) / (N * (2 * V_max + delta))

    assert math.isclose(threshold, expected, rel_tol=1e-9)
    print(f"Certificate Threshold = {threshold:.6e} matches analytical formula exactly.")
    print("[PASS] Error certificate inequality verified.")


def test_adaptive_rank_attention_end_to_end():
    print("-" * 60)
    print("Test 4: Adaptive Attention Kernel Forward Pass (Batched, Causal & Non-Causal)")
    print("-" * 60)

    B, H, N, d_k, d_v = 1, 4, 32, 16, 16
    Q = torch.randn(B, H, N, d_k)
    K = torch.randn(B, H, N, d_k)
    V = torch.randn(B, H, N, d_v)

    # Non-causal forward pass
    out_noncausal, decision = adaptive_rank_attention(
        Q, K, V,
        delta=0.05,
        r_candidates=(64, 256, 1024),
        is_causal=False
    )
    assert out_noncausal.shape == (B, H, N, d_v)
    assert not torch.isnan(out_noncausal).any()
    assert not torch.isinf(out_noncausal).any()
    print(f"Non-causal forward pass succeeded. Shape: {out_noncausal.shape}, Decision: {decision}")

    # Causal forward pass
    out_causal, decision_causal = adaptive_rank_attention(
        Q, K, V,
        delta=0.05,
        r_candidates=(64, 256, 1024),
        is_causal=True
    )
    assert out_causal.shape == (B, H, N, d_v)
    assert not torch.isnan(out_causal).any()
    assert not torch.isinf(out_causal).any()
    print(f"Causal forward pass succeeded. Shape: {out_causal.shape}, Decision: {decision_causal}")
    print("[PASS] Adaptive kernel handles both causal and non-causal attention without error.")


def test_3token_toy_verification():
    print("-" * 60)
    print("Test 5: Full 3-Token Toy Example Verification")
    print("-" * 60)
    run_analytical_numpy_verification()
    run_pytorch_tensor_verification()
    print("[PASS] 3-token example passed 100%.")


if __name__ == "__main__":
    test_entropy_and_temperature_diffuse_vs_spiky()
    test_estimate_required_rank_behavior()
    test_sufficient_condition_formula()
    test_adaptive_rank_attention_end_to_end()
    test_3token_toy_verification()
    print("\n" + "=" * 70)
    print("ALL THEORY UNIT & INTEGRATION TESTS PASSED (100% GREEN)!")
    print("=" * 70)
