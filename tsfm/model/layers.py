import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class ResidualBlock(nn.Module):
    """Two-layer perceptron with a linear skip connection (Chronos-2 / t0 patch embedding and decoder)."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.hidden = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden, d_out)
        self.skip = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x))) + self.skip(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        n = x.shape[-2]
        angles = torch.arange(n, device=x.device).float()[:, None] * self.inv_freq[None]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, rotary=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim) if rotary else None

    def forward(self, x, mask):
        n_seq, n, d = x.shape
        q, k, v = self.qkv(x).view(n_seq, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(n_seq, n, d))


class TransformerLayer(nn.Module):
    """Pre-norm attention + SwiGLU block, applied either along time or along variates."""

    def __init__(self, d_model, n_heads, d_ff, axis):
        super().__init__()
        self.axis = axis
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, rotary=axis == "time")
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, h, time_mask, variate_mask):
        b, v, n, d = h.shape
        if self.axis == "time":
            x = h.reshape(b * v, n, d)
            x = x + self.attn(self.attn_norm(x), time_mask)
            x = x.view(b, v, n, d)
        else:
            x = h.transpose(1, 2).reshape(b * n, v, d)
            x = x + self.attn(self.attn_norm(x), variate_mask)
            x = x.view(b, n, v, d).transpose(1, 2)
        return x + self.ff(self.ff_norm(x))
