import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# scalenorm

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# class

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

# gated attention unit

class GAU(nn.Module):
    def __init__(
        self,
        *,
        dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        causal = False,
        dropout = 0.,
        norm_klass = nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads = 2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        seq_len, device = x.shape[-2], x.device

        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)

        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)

        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 j')
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate

        out = self.to_out(out)

        if self.add_residual:
            out = out + x

        return out

# FLASH

class FLASH(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        dropout = 0.,
        rotary_pos_emb = None,
        norm_klass = nn.LayerNorm,
        shift_tokens = False
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb
        self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal = causal)

        # norm

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # projections

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # prenorm

        normed_x = self.norm(x)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)
        qk = self.to_qk(normed_x)

        # offset and scale

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        # mask out linear attention keys

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # padding for groups

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

        # calculate quadratic attention output

        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        sim = sim + self.rel_pos_bias(sim)

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

        # calculate linear attention output

        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g

            # exclusive cumulative sum along group dimension

            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)

            lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out))

        # gate

        out = gate * (quad_attn_out + lin_attn_out)

        # projection out and residual

        return self.to_out(out) + x

# FLASH Transformer

class FLASHTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        attn_dropout = 0.,
        norm_type = 'scalenorm',
        shift_tokens = True
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.abs_pos_emb = ScaledSinuEmbedding(dim)
        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J

        self.layers = nn.ModuleList([FLASH(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens) for _ in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        x = self.token_emb(x)
        x = self.abs_pos_emb(x) + x

        for flash in self.layers:
            x = flash(x, mask = mask)

        return self.to_logits(x)
