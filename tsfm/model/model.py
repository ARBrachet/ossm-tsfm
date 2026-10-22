import torch
import torch.nn as nn

from tsfm.model.heads import FUTURE, HEADS, PAD, PAST, TARGET
from tsfm.model.layers import ResidualBlock, RMSNorm, TransformerLayer


def causal_stats(x, weight):
    """Running mean / std of x over the weighted steps seen so far (inclusive), shape [..., T]."""
    w = weight.double()
    xd = torch.where(weight, x.double(), 0.0)
    count = w.cumsum(-1)
    mean = xd.cumsum(-1) / count.clamp_min(1.0)
    var = (xd.square().cumsum(-1) / count.clamp_min(1.0) - mean.square()).clamp_min(0.0)
    std = torch.maximum(var.sqrt(), 1e-3 * mean.abs())
    std = torch.where(std > 0, std, 1.0)
    return mean.float(), std.float(), count


def full_stats(x, weight):
    mean, std, count = causal_stats(x, weight)
    return mean[..., -1:].expand_as(x), std[..., -1:].expand_as(x), count[..., -1:].expand_as(x)


class TSFM(nn.Module):
    """t0-style patch transformer (alternating time / variate attention) with a pluggable prediction head."""

    def __init__(self, patch_size=16, d_model=256, n_heads=4, d_ff=1024, n_blocks=2, head="ossm",
                 quantiles=(0.1, 0.5, 0.9), mask_correction=True, expert_loss_weight=0.5, **head_kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.mask_correction = mask_correction
        self.expert_loss_weight = expert_loss_weight
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32), persistent=False)
        self.embed = ResidualBlock(3 * patch_size, d_ff, d_model)
        self.role_embed = nn.Embedding(4, d_model)
        self.layers = nn.ModuleList(
            TransformerLayer(d_model, n_heads, d_ff, axis) for _ in range(n_blocks) for axis in ("time", "time", "variate")
        )
        self.norm = RMSNorm(d_model)
        self.head = HEADS[head](d_model, d_ff, patch_size, len(quantiles), n_heads=n_heads, **head_kwargs)

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"]["model"])
        model.load_state_dict(ckpt["model"])
        return model.eval()

    def constrain(self):
        if hasattr(self.head, "constrain"):
            self.head.constrain()

    def scale(self, values, observed, visible, role):
        is_future = (role == FUTURE)[..., None]
        c_mean, c_std, c_count = causal_stats(values, visible)
        f_mean, f_std, f_count = full_stats(values, observed)
        mean = torch.where(is_future, f_mean, c_mean)
        std = torch.where(is_future, f_std, c_std)
        count = torch.where(is_future, f_count, c_count)
        return mean, std, count

    def forward(self, values, observed, visible, role):
        """values / observed / visible: [B, V, T] with T a multiple of the patch size, role: [B, V].

        `visible` is what the model may read: observed and not masked (context masking, forecast horizon).
        Known-future rows are always visible where observed. Output [B, V, N, P, Q]: index p is the
        forecast of patch p+1 in the normalized space of the statistics at the end of patch p.
        """
        return self.predict_patches(values, observed, visible, role)[0]

    def predict_patches(self, values, observed, visible, role):
        """Same as forward, also returning the expert forecasts of a mixture head (empty list otherwise)."""
        visible = torch.where((role == FUTURE)[..., None], observed, visible)
        mean, std, _ = self.scale(values, observed, visible, role)
        z = torch.where(visible, torch.asinh((values - mean) / std), 0.0)

        b, v, t = values.shape
        n, p = t // self.patch_size, self.patch_size
        offsets = (torch.arange(p, device=values.device) / p).expand(b, v, n, p)
        patches = torch.cat([z.view(b, v, n, p), offsets, visible.view(b, v, n, p).float()], dim=-1)
        h = self.embed(patches) + self.role_embed(role)[:, :, None]

        patch_visible = visible.view(b, v, n, p).any(-1)
        after_start = patch_visible.cumsum(-1) > 0
        time_mask, variate_mask = self.attention_masks(role, after_start)
        for layer in self.layers:
            h = layer(h, time_mask, variate_mask)
        e = self.norm(h)

        correction = patch_visible if self.mask_correction else after_start
        out = self.head(e, correction, role, patch_visible)
        return out if isinstance(out, tuple) else (out, [])

    @staticmethod
    def attention_masks(role, after_start):
        """Time: causal (bidirectional for known-future rows), never reading the left padding before a row starts.
        Variates: no padding rows, known-future rows only read known-future rows."""
        b, v, n = after_start.shape
        causal = torch.ones(n, n, dtype=torch.bool, device=role.device).tril()
        bidirectional = (role == FUTURE).reshape(b * v, 1, 1, 1)
        started = after_start.reshape(b * v, 1, 1, n)
        time_mask = (causal | bidirectional) & (started | torch.eye(n, dtype=torch.bool, device=role.device))

        query, key = role[:, :, None], role[:, None, :]
        allowed = (key != PAD) & ((query != FUTURE) | (key == FUTURE))
        allowed = allowed | torch.eye(v, dtype=torch.bool, device=role.device)
        variate_mask = allowed[:, None, None].expand(b, n, 1, v, v).reshape(b * n, 1, v, v)
        return time_mask, variate_mask

    def loss(self, values, observed, visible, role):
        pred, experts = self.predict_patches(values, observed, visible, role)
        b, v, t = values.shape
        n, p = t // self.patch_size, self.patch_size

        visible = torch.where((role == FUTURE)[..., None], observed, visible)
        mean, std, count = self.scale(values, observed, visible, role)
        end = slice(p - 1, t - p, p)
        mean, std, count = mean[..., end, None], std[..., end, None], count[..., end, None]
        target = values.view(b, v, n, p)[:, :, 1:]
        z = torch.asinh((target - mean) / std)

        weight = observed.view(b, v, n, p)[:, :, 1:] & (count > 0) & (role == TARGET)[:, :, None, None]
        loss = self.quantile_loss(pred[:, :, :-1], z, weight)
        if experts:
            expert_losses = [self.quantile_loss(x[:, :, :-1], z, weight) for x in experts]
            loss = loss + self.expert_loss_weight * sum(expert_losses) / len(experts)
        return loss

    def quantile_loss(self, pred, z, weight):
        """Pinball loss averaged over quantiles, then over the scored steps of each row, then over rows (t0 Eq. 4)."""
        q = self.quantiles
        diff = z.unsqueeze(-1) - pred
        pinball = torch.maximum(q * diff, (q - 1) * diff).mean(-1)
        pinball = torch.where(weight, pinball, 0.0)
        per_row = pinball.flatten(2).sum(-1) / weight.flatten(2).sum(-1).clamp_min(1)
        has_loss = weight.flatten(2).any(-1)
        return (per_row * has_loss).sum() / has_loss.sum().clamp_min(1)

    @torch.no_grad()
    def forecast(self, values, observed, role, context_patches, horizon_patches):
        """Single-pass forecast: the horizon is appended as blank patches. Returns [B, V, H*P, Q] in data space."""
        b, v, t = values.shape
        p = self.patch_size
        is_horizon = torch.zeros_like(observed)
        is_horizon[..., context_patches * p :] = True
        visible = observed & ~is_horizon
        pred = self(values, observed, visible, role)
        first = context_patches - 1
        pred = pred[:, :, first : first + horizon_patches]

        visible = torch.where((role == FUTURE)[..., None], observed, visible)
        mean, std, _ = self.scale(values, observed, visible, role)
        t_end = context_patches * p - 1
        mean, std = mean[..., t_end, None, None, None], std[..., t_end, None, None, None]
        out = mean + std * torch.sinh(pred.float())
        return out.flatten(2, 3)
