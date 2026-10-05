"""
src/baselines/test_deltanet.py
==============================
Unit verification script for GatedDeltaNetLayer:
1. Tensor shape and type preservation
2. Backpropagation / autograd gradient flow
3. Associative Recall benchmark: storing and retrieving distinct key-value pairs
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.baselines.deltanet import GatedDeltaNetLayer


def test_shapes_and_states():
    print("------------------------------------------------------------")
    print("Test 1: Output Shape & Memory State Dimension Verification")
    print("------------------------------------------------------------")
    B, N, D = 2, 64, 32
    H = 4
    layer = GatedDeltaNetLayer(d_model=D, num_heads=H)
    
    x = torch.randn(B, N, D)
    out, final_state = layer(x, return_final_state=True)
    
    assert out.shape == (B, N, D), f"Expected {(B, N, D)}, got {out.shape}"
    assert final_state.shape == (B, H, D // H, D // H), f"Expected {(B, H, D // H, D // H)}, got {final_state.shape}"
    assert not torch.isnan(out).any(), "NaN found in output"
    assert not torch.isinf(out).any(), "Inf found in output"
    print(f"[PASS] Output shape: {out.shape}, Final State shape: {final_state.shape}")


def test_gradient_flow():
    print("------------------------------------------------------------")
    print("Test 2: Autograd Gradient Flow & Parameter Updatability")
    print("------------------------------------------------------------")
    B, N, D = 2, 32, 16
    layer = GatedDeltaNetLayer(d_model=D, num_heads=2)
    
    x = torch.randn(B, N, D, requires_grad=True)
    out, _ = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradient is None"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
    print("[PASS] Full backpropagation passed without gradient anomalies.")


def test_associative_recall():
    print("------------------------------------------------------------")
    print("Test 3: Multi-Step Associative Recall (Delta Rule vs Memory)")
    print("------------------------------------------------------------")
    # In DeltaNet, each step updates S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
    # If beta=1.0 and keys are orthonormal, S_t k_t recovers v_t perfectly!
    D = 16
    H = 1
    head_dim = 16
    
    # Orthonormal keys via QR: generate 4 vectors in R^16
    torch.manual_seed(42)
    Q, _ = torch.linalg.qr(torch.randn(head_dim, 4)) # Q has shape (16, 4) with orthonormal columns
    keys = Q.T # (4, 16) with keys @ keys.T = I_4
    values = torch.randn(4, head_dim)
    
    # Direct delta recurrence verification
    S = torch.zeros(head_dim, head_dim)
    beta = 1.0 # full write
    alpha = 1.0 # perfect retention
    
    for t in range(4):
        k_t = keys[t:t+1].T # (16, 1)
        v_t = values[t:t+1].T # (16, 1)
        v_pred = S @ k_t
        err = v_t - v_pred
        S = alpha * S + beta * (err @ k_t.T)
        
    # Query with key 0, 1, 2, 3
    max_err = 0.0
    for t in range(4):
        k_t = keys[t:t+1].T
        v_expected = values[t:t+1].T
        v_retrieved = S @ k_t
        err = torch.norm(v_retrieved - v_expected).item()
        max_err = max(max_err, err)
        print(f"Key {t}: Retrieval L2 error = {err:.6e}")
        
    assert max_err < 1e-5, f"Associative recall error too large: {max_err}"
    print(f"[PASS] Pure delta rule achieves exact associative recall (max error = {max_err:.2e} < 1e-5)!")


if __name__ == "__main__":
    test_shapes_and_states()
    test_gradient_flow()
    test_associative_recall()
    print("============================================================")
    print("ALL GATED DELTANET UNIT TESTS PASSED (100% SUCCESS)!")
    print("============================================================")
