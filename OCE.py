from typing import Optional, Any, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from serialization import get_morton_code

def get_lcp_buckets(morton_a, morton_b, num_buckets, max_depth=14):
    """
    morton_a, morton_b: [B, N] (dtype: torch.int64)
    """
    xor_res = morton_a.unsqueeze(2) ^ morton_b.unsqueeze(1)
    mask = (xor_res == 0)
    diff_bit_pos = torch.log2(xor_res.float() + 1e-9).floor()
    lcp_depth = max_depth - (diff_bit_pos / 3.0).floor()
    lcp_depth = torch.where(mask, torch.tensor(float(max_depth)).to(morton_a.device), lcp_depth)
    scale = (num_buckets - 1) / max_depth
    buckets = (lcp_depth * scale).long()
    return torch.clamp(buckets, 0, num_buckets - 1)

class TopologyAwareAttention(nn.Module):
    def __init__(self, dim, num_heads, num_buckets):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.scale = (dim // num_heads) ** -0.5 
        self.qkv = nn.Linear(dim, dim * 3)
        self.qkv_conv = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.proj = nn.Linear(dim, dim)
        self.topo_bias_table = nn.Parameter(torch.zeros(num_heads, num_buckets))


    def forward(self, x, morton_codes):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 3, 1).contiguous() # 3, B, C, N
        qkv = self.qkv_conv(qkv.reshape(3*B, C, N)).reshape(3, B, C, N).permute(0, 1, 3, 2).contiguous() # 3, B, N, C
        # 3, B, n_h, N, C//n_h
        qkv = qkv.reshape(3, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2] # B, n_h, N, C//n_h
        attn = (q @ k.transpose(-2, -1)) * self.scale
        buckets = get_lcp_buckets(morton_codes, morton_codes, self.num_buckets) 
        bias = self.topo_bias_table[:, buckets].permute(1, 0, 2, 3)
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
class TMABlock(nn.Module):
    def __init__(self, dim, num_heads, num_buckets):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.topo_attn = TopologyAwareAttention(dim, num_heads, num_buckets)
        self.modulation_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )
        self.fn = nn.Linear(dim, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=5, padding=2)

    def forward(self, x, morton_codes, depth):
        B, N, C = x.shape
        
        mod_params = self.modulation_mlp(depth) # [B, C*6]
        a1, b1, g1, a2, b2, g2 = torch.split(mod_params.unsqueeze(1), C, dim=-1) # B, 1, C

        res = x
        x_norm = self.ln1(x)
        x_mod = a1 * x_norm + b1 # alpha * x + beta
        x_attn = self.topo_attn(x_mod, morton_codes)
        x = res + g1 * x_attn # gamma * attn

        res = x
        x_norm = self.ln2(x)
        x_mod = a2 * x_norm + b2
        x_conv = self.conv1d(self.fn(x_mod).transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        x = res + g2 * x_conv
        
        return x


class OCE(nn.Module):
    def __init__(self, input_dim=6, embed_dim=128, num_layers=4, num_heads=8, num_buckets=32):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.local_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)

        self.blocks = nn.ModuleList([
            TMABlock(embed_dim, num_heads, num_buckets) for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 256) 
        )

    def forward(self, features:torch.Tensor):
        # features: [B, N, 6]
        center_position = features[:, :, 0:3]
        morton_codes = get_morton_code(center_position)
        depth = features[:, 0, 3:4] # B, 1

        x = self.embedding(features) # B,N,C
        x = x.transpose(1, 2).contiguous()
        x = self.local_conv(x).transpose(1, 2).contiguous()

        for block in self.blocks:
            x = block(x, morton_codes, depth)    
        logits = self.mlp_head(x)
        return logits



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = OCE(input_dim=6, embed_dim=128, num_layers=4).to(device)

    batch_size = 2
    num_points = 512
    dummy_features = torch.randn(batch_size, num_points, 6).to(device)

    print(f"Input features shape: {dummy_features.shape}")

    output_logits = model(dummy_features)
    print(f"Forward pass SUCCESS! Output shape: {output_logits.shape}")
    print("Expected output shape: torch.Size([2, 512, 256])")
    output_logits.sum().backward()

    
    
        
