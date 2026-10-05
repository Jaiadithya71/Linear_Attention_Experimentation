"""
src/integration/test_qwen_forward.py
===================================
Unit verification for Qwen2Attention monkey-patching.
Validates forward pass, shape preservation, and numerical stability.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2Config,
    Qwen2RotaryEmbedding
)
from src.integration.patch_qwen import patch_qwen_attention
from adaptive_kernel import linear_attention, adaptive_rank_attention


def test_qwen_patch_forward():
    print("=" * 70)
    print("Test: Qwen2Attention Monkey-Patch Forward Pass Verification")
    print("=" * 70)

    # 1. Initialize lightweight Qwen2Config matching Qwen2.5 architectural proportions
    config = Qwen2Config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,   # GQA with 4 groups
        max_position_embeddings=2048,
        attention_dropout=0.0
    )
    
    # 2. Instantiate attention layer and RoPE embeddings
    layer_idx = 0
    attn = Qwen2Attention(config, layer_idx=layer_idx)
    rotary_emb = Qwen2RotaryEmbedding(config)

    # 3. Create synthetic input tensors
    B, N = 1, 128
    hidden_states = torch.randn(B, N, config.hidden_size)
    position_ids = torch.arange(N).unsqueeze(0)
    cos, sin = rotary_emb(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    # Baseline forward pass (SDPA / eager)
    with torch.no_grad():
        out_base, _ = attn(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
    print(f"[BASE] Native Qwen2Attention output shape: {out_base.shape}")
    assert out_base.shape == (B, N, config.hidden_size)

    # 4. Monkey-patch with Linear Attention
    count = patch_qwen_attention(attn, linear_attention)
    print(f"[PATCH] Patched {count} attention layer(s) with Linear Attention.")
    assert count == 1, "Failed to patch layer"

    with torch.no_grad():
        out_linear, _ = attn(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
    
    print(f"[LINEAR] Patched output shape: {out_linear.shape}")
    assert out_linear.shape == (B, N, config.hidden_size), "Shape mismatch after patching"
    assert not torch.isnan(out_linear).any(), "NaN detected in patched output"
    assert not torch.isinf(out_linear).any(), "Inf detected in patched output"
    print("[PASS] Linear Attention monkey-patch verified without errors.")

    # 5. Monkey-patch with Adaptive Rank Attention
    count = patch_qwen_attention(attn, adaptive_rank_attention)
    with torch.no_grad():
        out_adapt, _ = attn(hidden_states, position_embeddings=position_embeddings, attention_mask=None)

    print(f"[ADAPTIVE] Patched output shape: {out_adapt.shape}")
    assert out_adapt.shape == (B, N, config.hidden_size)
    assert not torch.isnan(out_adapt).any()
    assert not torch.isinf(out_adapt).any()
    print("[PASS] Adaptive Rank Attention monkey-patch verified without errors.")
    
    print("=" * 70)
    print("ALL QWEN MONKEY-PATCH TESTS PASSED (100% SUCCESS)!")
    print("=" * 70)


if __name__ == "__main__":
    test_qwen_patch_forward()
