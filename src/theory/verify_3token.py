"""
src/theory/verify_3token.py
===========================
Unit verification script for the analytical 3-token toy example from Team 3 Proposal.

Validates:
1. Exact score matrix QK^T = [[1, 2, 3], [2, 1, 3], [1, 1, 2]]
2. Scaled softmax attention row 1 = [4.280169, 5.719831]
3. Reordered linear attention (ReLU+1) row 1 = [4.8, 5.2]
4. Explicit kernel attention (K) == Reordered linear attention (L) to machine precision
5. Exact relative Frobenius discrepancy between linear and scaled softmax ~ 8.43% (~8.4%)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def run_analytical_numpy_verification():
    print("=" * 70)
    print("STEP 1: Analytical Verification (NumPy float64 / Exact Precision)")
    print("=" * 70)

    # 1. Inputs defined in the proposal (Section 8.3)
    # d_k = d_v = 2, N = 3
    Q = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 1.0]
    ], dtype=np.float64)

    K = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ], dtype=np.float64)

    V = np.array([
        [10.0, 0.0],
        [0.0, 10.0],
        [5.0, 5.0]
    ], dtype=np.float64)

    d_k = 2.0
    scale = 1.0 / math.sqrt(d_k)

    # 2. Check QK^T
    QK_T = Q @ K.T
    expected_QK_T = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 3.0],
        [1.0, 1.0, 2.0]
    ], dtype=np.float64)

    assert np.allclose(QK_T, expected_QK_T, atol=1e-12), f"QK^T mismatch: {QK_T}"
    print("[PASS] QK^T matches [[1,2,3],[2,1,3],[1,1,2]] exactly.")

    # 3. Scaled Softmax Attention (Section 8.5)
    # S = QK^T / sqrt(2)
    S = QK_T * scale
    exp_S = np.exp(S)
    Z = exp_S.sum(axis=1, keepdims=True)
    A_S = exp_S / Z
    Y_S = A_S @ V

    # Expected values for row 1
    # Scores: (1/sqrt(2), 2/sqrt(2), 3/sqrt(2)) = (0.707107, 1.414214, 2.121320)
    # exp(scores): (2.028115, 4.113250, 8.342140), sum = 14.483505
    # weights: (0.140029, 0.283995, 0.575975)
    # Y_S[0] = 0.140029*(10,0) + 0.283995*(0,10) + 0.575975*(5,5) = (4.280169, 5.719831)
    expected_Y_S_row0 = np.array([4.28016918, 5.71983082])
    assert np.allclose(Y_S[0], expected_Y_S_row0, atol=1e-6), f"Y_S[0] mismatch: {Y_S[0]}"
    print(f"[PASS] Scaled Softmax Output Row 1 = [{Y_S[0, 0]:.6f}, {Y_S[0, 1]:.6f}] "
          f"(Matches expected [4.280169, 5.719831])")
    print(f"       Full Y_S:\n{Y_S}")

    # 4. Feature map phi(x) = ReLU(x) + 1 (Section 8.6)
    phi_Q = np.maximum(0.0, Q) + 1.0
    phi_K = np.maximum(0.0, K) + 1.0

    expected_phi_Q = np.array([
        [2.0, 3.0],
        [3.0, 2.0],
        [2.0, 2.0]
    ])
    expected_phi_K = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 2.0]
    ])
    assert np.allclose(phi_Q, expected_phi_Q), f"phi(Q) mismatch: {phi_Q}"
    assert np.allclose(phi_K, expected_phi_K), f"phi(K) mismatch: {phi_K}"

    mapped_scores = phi_Q @ phi_K.T
    expected_mapped_scores = np.array([
        [7.0, 8.0, 10.0],
        [8.0, 7.0, 10.0],
        [6.0, 6.0, 8.0]
    ])
    assert np.allclose(mapped_scores, expected_mapped_scores), f"phi(Q)phi(K)^T mismatch: {mapped_scores}"
    print("[PASS] Mapped scores phi(Q)phi(K)^T match [[7,8,10],[8,7,10],[6,6,8]].")

    # 5. Method K: Explicit Kernel Attention
    A_K = mapped_scores / mapped_scores.sum(axis=1, keepdims=True)
    Y_K = A_K @ V

    # 6. Method L: Reordered Linear Attention (Associativity, Section 8.7)
    # KV = phi(K)^T @ V, K1 = phi(K)^T @ 1
    KV = phi_K.T @ V
    K1 = phi_K.T @ np.ones((3, 1))

    expected_KV = np.array([
        [30.0, 20.0],
        [20.0, 30.0]
    ])
    expected_K1 = np.array([
        [5.0],
        [5.0]
    ])
    assert np.allclose(KV, expected_KV), f"KV mismatch: {KV}"
    assert np.allclose(K1, expected_K1), f"K1 mismatch: {K1}"

    numerators = phi_Q @ KV
    denominators = phi_Q @ K1

    expected_numerators = np.array([
        [120.0, 130.0],
        [130.0, 120.0],
        [100.0, 100.0]
    ])
    expected_denominators = np.array([
        [25.0],
        [25.0],
        [20.0]
    ])
    assert np.allclose(numerators, expected_numerators), f"Numerators mismatch: {numerators}"
    assert np.allclose(denominators, expected_denominators), f"Denominators mismatch: {denominators}"

    Y_L = numerators / denominators
    expected_Y_L = np.array([
        [4.8, 5.2],
        [5.2, 4.8],
        [5.0, 5.0]
    ])
    assert np.allclose(Y_L, expected_Y_L, atol=1e-12), f"Y_L mismatch: {Y_L}"
    print(f"[PASS] Reordered Linear Output Row 1 = [{Y_L[0, 0]:.1f}, {Y_L[0, 1]:.1f}] "
          f"(Matches expected [4.8, 5.2])")
    print(f"       Full Y_L:\n{Y_L}")

    # 7. Associativity Check (K vs L agreement)
    diff_KL = np.max(np.abs(Y_K - Y_L))
    assert diff_KL < 1e-14, f"Explicit K and Reordered L disagree: max diff = {diff_KL}"
    print(f"[PASS] Associativity check: max|Y_K - Y_L| = {diff_KL:.2e} < 1e-14 (Machine Precision)")

    # 8. Relative Frobenius Error (Section 8.8)
    # norm(Y_L - Y_S, 'fro') / norm(Y_S, 'fro')
    frob_diff = np.linalg.norm(Y_L - Y_S, 'fro')
    frob_S = np.linalg.norm(Y_S, 'fro')
    rel_error = frob_diff / frob_S

    print(f"[PASS] ||Y_L - Y_S||_F = {frob_diff:.6f}")
    print(f"[PASS] ||Y_S||_F       = {frob_S:.6f}")
    print(f"[PASS] Relative Frobenius Error = {rel_error * 100.0:.3f}% (~8.4%)")
    assert 0.084 < rel_error < 0.085, f"Unexpected relative error: {rel_error}"


def run_pytorch_tensor_verification():
    print("\n" + "=" * 70)
    print("STEP 2: PyTorch Tensor Verification (batched B=1, H=1, N=3, d=2)")
    print("=" * 70)

    Q = torch.tensor([[[[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)
    K = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)
    V = torch.tensor([[[[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]]], dtype=torch.float32)

    # Softmax
    d_k = 2.0
    S = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
    A_S = F.softmax(S, dim=-1)
    Y_S = A_S @ V

    # Linear
    def phi_torch(x):
        return F.relu(x) + 1.0

    fq = phi_torch(Q)
    fk = phi_torch(K)

    # Reordered
    kv = fk.transpose(-1, -2) @ V
    k1 = fk.sum(dim=-2, keepdim=True).transpose(-1, -2)
    num = fq @ kv
    den = fq @ k1
    Y_L = num / den

    # Explicit
    A_K = fq @ fk.transpose(-1, -2)
    A_K = A_K / A_K.sum(dim=-1, keepdim=True)
    Y_K = A_K @ V

    assert torch.allclose(Y_K, Y_L, atol=1e-6), "PyTorch K and L disagree"
    rel_err = (torch.norm(Y_L - Y_S, p='fro') / torch.norm(Y_S, p='fro')).item()

    row0_S = Y_S[0, 0, 0].tolist()
    row0_L = Y_L[0, 0, 0].tolist()

    print(f"[PASS] PyTorch Softmax Row 1 = [{row0_S[0]:.6f}, {row0_S[1]:.6f}]")
    print(f"[PASS] PyTorch Linear Row 1  = [{row0_L[0]:.4f}, {row0_L[1]:.4f}]")
    print(f"[PASS] PyTorch Rel Frobenius = {rel_err * 100.0:.3f}%")
    assert abs(rel_err - 0.08431) < 1e-4, f"PyTorch rel error out of range: {rel_err}"


if __name__ == "__main__":
    run_analytical_numpy_verification()
    run_pytorch_tensor_verification()
    print("\n" + "=" * 70)
    print("ALL 3-TOKEN ANALYTICAL UNIT TESTS PASSED (100% SUCCESS)!")
    print("=" * 70)
