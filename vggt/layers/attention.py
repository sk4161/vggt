# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False

# Try to import xformers functions
try:
    from xformers.ops import memory_efficient_attention, unbind
    XFORMERS_AVAILABLE = True
except ImportError:
    memory_efficient_attention = None
    unbind = None


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attention_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Create attention mask for scaled_dot_product_attention
        attn_mask = None
        if attention_mask is not None:
            # Convert boolean mask to additive mask for attention
            # attention_mask: [B, N] where True=MASK, False=ATTEND
            # Create proper 4D mask for scaled_dot_product_attention
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            attn_mask = attn_mask.expand(-1, -1, N, -1)  # [B, 1, N, N]
            # True=MASK -> -inf, False=ATTEND -> 0.0
            attn_mask = torch.where(attn_mask, float('-inf'), 0.0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            
            # Apply attention mask if provided
            if attn_mask is not None:
                attn = attn + attn_mask
                
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, attention_mask=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos=pos, attention_mask=attention_mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if unbind is not None:
            q, k, v = unbind(qkv, 2)
        else:
            q, k, v = qkv.unbind(2)

        # Convert attention_mask to attn_bias format if provided
        if attention_mask is not None and attn_bias is None:
            # Convert boolean mask to additive bias
            # attention_mask: [B, N] where True=MASK, False=ATTEND
            attn_bias = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            # True=MASK -> -1e9, False=ATTEND -> 0.0
            attn_bias = torch.where(attn_bias, -1e9, 0.0)

        if memory_efficient_attention is not None:
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        else:
            # Fallback to standard attention if xformers not available
            return super().forward(x, pos=pos, attention_mask=attention_mask)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
