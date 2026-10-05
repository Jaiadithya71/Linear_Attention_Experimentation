"""
src/baselines/deltanet.py
========================
Pure-PyTorch implementation of the Gated DeltaNet layer.
Reference: "DeltaNet: Addressing the Associative Recall Bottleneck in Linear Attention"
           (Schlag et al., 2021; Yang et al., 2024; SOTA Linear RNNs)

Core Recurrent Update:
    k_t = normalize(W_k x_t)
    v_t = W_v x_t
    q_t = W_q x_t
    beta_t = sigmoid(W_beta x_t)   # Data-dependent write / learning rate gate in [0, 1]
    alpha_t = sigmoid(W_alpha x_t) # Data-dependent state retention / decay gate in [0, 1]

    # Delta Rule Error-Correction:
    # Predict prior value: v_pred = S_{t-1} k_t
    # Error vector: e_t = v_t - v_pred
    # State update: S_t = alpha_t * S_{t-1} + beta_t * (e_t) k_t^T
    
    # Readout:
    y_t = S_t q_t
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedDeltaNetLayer(nn.Module):
    """
    Pure-PyTorch Standalone Gated DeltaNet Layer.
    Requires no custom CUDA/C++ extensions; runs efficiently on CPU and CUDA.
    """
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        use_decay: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else (d_model // num_heads)
        self.use_decay = use_decay
        self.total_dim = self.num_heads * self.head_dim

        # Input projections
        self.q_proj = nn.Linear(d_model, self.total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.total_dim, bias=False)
        
        # Data-dependent gating projections
        self.beta_proj = nn.Linear(d_model, self.num_heads, bias=True)
        if self.use_decay:
            self.alpha_proj = nn.Linear(d_model, self.num_heads, bias=True)
        else:
            self.alpha_proj = None

        # Output projection
        self.out_proj = nn.Linear(self.total_dim, d_model, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier uniform for projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Initialize beta gate with mild bias towards active writing (sigmoid(0) = 0.5)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 0.0)
        
        if self.alpha_proj is not None:
            nn.init.zeros_(self.alpha_proj.weight)
            # Initialize decay gate close to 1.0 (long memory retention)
            nn.init.constant_(self.alpha_proj.bias, 2.0)  # sigmoid(2.0) ~ 0.88

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_final_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, d_model)
            initial_state: Optional prior memory state of shape (B, H, head_dim, head_dim)
            return_final_state: If True, returns (out, final_state)
            
        Returns:
            out: (B, N, d_model)
            final_state: (B, H, head_dim, head_dim) if return_final_state is True else None
        """
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # Project Q, K, V: shape (B, N, H, D) -> transpose to (B, H, N, D)
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        # L2 normalize keys along head_dim so ||k_t||_2 = 1 (crucial for delta rule stability)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        # Project gates: shape (B, N, H) -> transpose to (B, H, N, 1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(-1)
        if self.alpha_proj is not None:
            alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2).unsqueeze(-1)
        else:
            alpha = torch.ones_like(beta)

        # Initialize recurrent memory state S: (B, H, D, D)
        if initial_state is not None:
            S = initial_state.clone()
        else:
            S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

        # Sequential scan over sequence length N
        # (For inference or exact associative recall verification)
        outputs = []
        for t in range(N):
            q_t = q[:, :, t, :]       # (B, H, D)
            k_t = k[:, :, t, :]       # (B, H, D)
            v_t = v[:, :, t, :]       # (B, H, D)
            b_t = beta[:, :, t, :]    # (B, H, 1)
            a_t = alpha[:, :, t, :]   # (B, H, 1)

            # 1. Delta rule prediction from prior state:
            # v_pred = S k_t: (B, H, D, D) @ (B, H, D, 1) -> (B, H, D)
            v_pred = torch.einsum("bhde,bhe->bhd", S, k_t)
            
            # 2. Prediction error:
            error = v_t - v_pred  # (B, H, D)
            
            # 3. Recurrent state update:
            # S_t = alpha_t * S_{t-1} + beta_t * (error @ k_t^T)
            delta_S = b_t.unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", error, k_t)
            S = a_t.unsqueeze(-1) * S + delta_S

            # 4. Readout with current query:
            # y_t = S_t q_t: (B, H, D)
            y_t = torch.einsum("bhde,bhe->bhd", S, q_t)
            outputs.append(y_t)

        # Stack outputs: (B, N, H, D) -> (B, N, total_dim)
        y = torch.stack(outputs, dim=2).transpose(1, 2).reshape(B, N, self.total_dim)
        out = self.out_proj(y)

        if return_final_state:
            return out, S
        return out, None
