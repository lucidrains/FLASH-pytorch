import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# class

class OffsetScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.gamma + self.beta

class GAU(nn.Module):
    def __init__(
        self,
        *,
        dim,
        query_key_dim = 128,
        hidden_dim = None,
        add_residual = True
    ):
        super().__init__()
        self.scale = query_key_dim ** -0.5

        hidden_dim = default(hidden_dim, dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.to_hidden = nn.Linear(dim, hidden_dim * 2)

        self.to_qk = nn.Linear(dim, query_key_dim, bias = False)

        self.q_offsetscale = OffsetScale(query_key_dim)
        self.k_offsetscale = OffsetScale(query_key_dim)

        self.to_out = nn.Linear(hidden_dim, dim)
        self.add_residual = add_residual

    def forward(
        self,
        x,
        rel_pos_bias = None
    ):
        v, gate = self.to_hidden(x).chunk(2, dim = -1)

        qk = self.to_qk(x)
        q, k = self.q_offsetscale(qk), self.k_offsetscale(qk)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = F.relu(sim) ** 2

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate

        out = self.to_out(out)

        if self.add_residual:
            out = out + x

        return out
