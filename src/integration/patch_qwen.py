"""
src/integration/patch_qwen.py
=============================
Monkey-patching utility for Qwen2Attention modules in Hugging Face Transformers.
Dynamically swaps standard quadratic attention with our linear / adaptive attention kernels.

Preserves:
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA) key-value expansion
- Linear projections (q_proj, k_proj, v_proj, o_proj)
- Autoregressive causal masking
"""

import math
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        repeat_kv,
        apply_rotary_pos_emb
    )
    HAVE_QWEN = True
except ImportError:
    HAVE_QWEN = False


def make_patched_qwen_forward(attention_kernel: Callable, is_causal: bool = False):
    """
    Creates a replacement forward method for Qwen2Attention using the provided kernel.
    """
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
        cache_position = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        B, N = input_shape
        
        # 1. Projections
        q = self.q_proj(hidden_states).view(B, N, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 2. Rotary Positional Embeddings (RoPE)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Grouped-Query Attention (GQA) repeat for KV heads
        if hasattr(self, "num_key_value_groups") and self.num_key_value_groups > 1:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # 4. Dispatch to linear / adaptive attention kernel
        # Kernel contract: (B, H, N, d_k) -> (B, H, N, d_v)
        causal_flag = getattr(self, "is_causal", is_causal)
        attn_output = attention_kernel(q, k, v, is_causal=causal_flag)

        # 5. Output reshape and projection
        attn_output = attn_output.transpose(1, 2).reshape(B, N, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    return patched_forward


def patch_qwen_attention(
    model: nn.Module,
    attention_kernel: Callable,
    is_causal: bool = False
) -> int:
    """
    Recursively patches all Qwen2Attention modules in a model with the custom attention kernel.
    
    Args:
        model: Hugging Face Qwen2ForCausalLM, Qwen2Model, or individual module
        attention_kernel: Function with signature (Q, K, V, is_causal) -> Output
        is_causal: Whether causal masking is enforced
        
    Returns:
        count: Number of attention modules successfully patched
    """
    patched_count = 0
    replacement_fn = make_patched_qwen_forward(attention_kernel, is_causal=is_causal)

    if HAVE_QWEN and isinstance(model, Qwen2Attention):
        model.forward = replacement_fn.__get__(model, Qwen2Attention)
        return 1

    for name, module in model.named_modules():
        if HAVE_QWEN and isinstance(module, Qwen2Attention):
            module.forward = replacement_fn.__get__(module, Qwen2Attention)
            patched_count += 1
        elif module.__class__.__name__ == "Qwen2Attention":
            module.forward = replacement_fn.__get__(module, module.__class__)
            patched_count += 1

    return patched_count
