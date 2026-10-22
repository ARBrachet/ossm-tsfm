import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsfm.model.layers import ResidualBlock, RMSNorm

PAD, TARGET, PAST, FUTURE = 0, 1, 2, 3


class QuantileDecoder(nn.Module):
    """Maps a d_model vector to the quantiles of one patch, monotone by construction around the median.
    linear=True: a single matrix D (the OSSM readout of Eq. 4), the only nonlinearity being the monotone ordering."""

    def __init__(self, d_model, d_ff, patch_size, n_quantiles, linear=False):
        super().__init__()
        self.patch_size, self.n_quantiles = patch_size, n_quantiles
        self.mid = n_quantiles // 2
        out = patch_size * n_quantiles
        self.block = nn.Linear(d_model, out) if linear else ResidualBlock(d_model, d_ff, out)

    def forward(self, x):
        r = self.block(x).unflatten(-1, (self.patch_size, self.n_quantiles))
        median = r[..., self.mid : self.mid + 1]
        up = median + F.softplus(r[..., self.mid + 1 :]).cumsum(-1)
        down = median - F.softplus(r[..., : self.mid]).flip(-1).cumsum(-1).flip(-1)
        return torch.cat([down, median, up], dim=-1)


class MLPHead(nn.Module):
    """t0 / Chronos-2 decoder: the representation of patch p is decoded into the quantiles of patch p+1."""

    def __init__(self, d_model, d_ff, patch_size, n_quantiles, **_):
        super().__init__()
        self.decoder = QuantileDecoder(d_model, d_ff, patch_size, n_quantiles)

    def forward(self, e, observed_patch, role, patch_visible):
        return self.decoder(e)


def _set_spectral_radius(m, rho):
    return m * rho / torch.linalg.eigvals(m).abs().max()


SEASONAL_PERIODS = (7, 12, 24, 52, 96, 144, 168, 288, 336, 365.25, 672, 1008, 1440, 2016, 8766)


def _initial_angles(n, patch_size, min_patches=2.2, max_patches=256.0):
    """Per-patch rotation angles in (0, pi]: common seasonal periods (aliased to the patch grid), then log-spaced."""
    angles = []
    for period in SEASONAL_PERIODS:
        theta = (2 * math.pi * patch_size / period) % (2 * math.pi)
        theta = min(theta, 2 * math.pi - theta)
        if theta > 1e-3 and all(abs(theta - a) > 0.02 for a in angles):
            angles.append(theta)
    angles = angles[:n]
    fill = [2 * math.pi / period for period in torch.logspace(math.log10(min_patches), math.log10(max_patches), 8 * n).tolist()]
    fill = [theta for theta in fill if all(abs(theta - a) > 0.02 for a in angles)]
    picks = torch.linspace(0, len(fill) - 1, n - len(angles)).round().long().tolist() if n > len(angles) else []
    return torch.tensor(angles + [fill[i] for i in picks], dtype=torch.float32)


class ModalTransition(nn.Module):
    """Controlled-mode transition: A is block-diagonal with 2x2 blocks whose eigenvalues have modulus 0 < r_k < 1 by
    construction (stable free run). Oscillating blocks are rotation-scaling blocks r_k [cos, -sin; sin, cos] with
    learned frequencies theta_k; level / trend blocks are Jordan-type blocks [r_k, kappa_k; 0, r_k], which can ramp."""

    def __init__(self, d_state, patch_size, rho=0.99, real_radii=(0.999, 0.99, 0.95, 0.8), kappa_init=0.1):
        super().__init__()
        n_blocks = d_state // 2
        self.n_real = min(len(real_radii), n_blocks)
        radius = torch.full((n_blocks,), rho)
        radius[: self.n_real] = torch.tensor(real_radii[: self.n_real])
        self.radius_logit = nn.Parameter(torch.logit(radius))
        self.theta = nn.Parameter(_initial_angles(n_blocks - self.n_real, patch_size))
        self.kappa = nn.Parameter(torch.full((self.n_real,), kappa_init))

    @property
    def n_modulation(self):
        return len(self.radius_logit) + len(self.theta) + len(self.kappa)

    def coefficients(self, delta=None):
        """Entries (a11, a12, a21, a22) of every 2x2 block, optionally shifted by delta [..., n_modulation]
        (offsets of the radius logits, angles and trend couplings)."""
        logit, theta, kappa = self.radius_logit, self.theta, self.kappa
        if delta is not None:
            d_logit, d_theta, d_kappa = delta.split([len(logit), len(theta), len(kappa)], dim=-1)
            logit, theta, kappa = logit + d_logit, theta + d_theta, kappa + d_kappa
        r = torch.sigmoid(logit)
        theta = torch.cat([theta.new_zeros(theta.shape[:-1] + (self.n_real,)), theta], dim=-1)
        cos, sin = r * torch.cos(theta), r * torch.sin(theta)
        upper = torch.cat([kappa, -sin[..., self.n_real :]], dim=-1)
        return cos, upper, sin, cos

    def forward(self):
        a11, a12, a21, a22 = self.coefficients()
        blocks = torch.stack([torch.stack([a11, a12], -1), torch.stack([a21, a22], -1)], -2)
        return torch.block_diag(*blocks)


class CovariateAttention(nn.Module):
    """Each row attends, at every patch, over the allowed covariate rows of its series plus a learned null slot."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.null_key = nn.Parameter(torch.zeros(n_heads, 1, self.head_dim))
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, keys, allowed):
        """query, keys: [B, V, N, d], allowed: [B, V, N] (key row usable at that patch) -> [B, V, N, d]."""
        b, v, n, d = query.shape
        heads = lambda x: x.transpose(1, 2).reshape(b * n, v, self.n_heads, self.head_dim).transpose(1, 2)
        k, val = self.kv(keys).chunk(2, dim=-1)
        q, k, val = self.q_norm(heads(self.q(query))), self.k_norm(heads(k)), heads(val)
        k = torch.cat([self.null_key.expand(b * n, -1, -1, -1), k], dim=2)
        val = torch.cat([torch.zeros_like(val[:, :, :1]), val], dim=2)
        mask = torch.cat([allowed.new_ones(b, n, 1), allowed.transpose(1, 2)], dim=-1).reshape(b * n, 1, 1, v + 1)
        out = F.scaled_dot_product_attention(q, k, val, attn_mask=mask)
        return self.out(out.transpose(1, 2).reshape(b, n, v, d).transpose(1, 2))


class CovariateMean(nn.Module):
    """Ablation: plain mean of the allowed covariate rows, shared by every row of the series."""

    def __init__(self, d_model, n_heads):
        super().__init__()

    def forward(self, query, keys, allowed):
        w = allowed.unsqueeze(-1).to(keys.dtype)
        return ((keys * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1.0)).expand_as(query)


POOLING = {"attention": CovariateAttention, "mean": CovariateMean}


class OSSMHead(nn.Module):
    """Observer State-Space Model read-out (Brachet et al., Eq. 4) over the sequence of encoded patches.

    x_{p+1} = A x_p + m_p L (e_p - C x_p) + B_pk u_pk_p + B_fk u_fk_{p+1} [+ G mean_{other targets} x_p]
    patch p+1 ~ D(C x_{p+1})
    With d_hidden, the measurement is h_p = ReLU(W e_p) in a wider space and replaces e_p in the innovation.
    transition: dense A | modal A (stable 2x2 blocks) | selective A(e_p): modal blocks whose radii, angles and trend
    couplings are shifted by a linear map of the row encoding e_p, so the dynamics depend on the series (causally).

    u_pk_p: attention of row e_p over the visible past covariates at patch p.
    u_fk_{p+1}: attention of row e_p over the known-future covariates at patch p+1.
    Queries come from patch p only, and the weights do not depend on the state: the recurrence stays linear.
    """

    def __init__(self, d_model, d_ff, patch_size, n_quantiles, d_state=64, rho_a=0.9, rho_observer=0.91,
                 couple_targets=False, covariate_pooling="attention", n_heads=4, d_hidden=None, transition="dense",
                 a_cap=None, observer="free", selective_bc=False, readout_activation=None, decoder="mlp", **_):
        super().__init__()
        self.lift = nn.Sequential(nn.Linear(d_model, d_hidden), nn.ReLU()) if d_hidden else nn.Identity()
        d_hidden = d_hidden or d_model
        self.a_cap = a_cap
        self.modulation = None
        if transition in ("modal", "selective"):
            self.modal = ModalTransition(d_state, patch_size, rho_a)
            a = self.modal().detach()
            if transition == "selective":
                self.modulation = nn.Linear(d_model, self.modal.n_modulation)
                nn.init.zeros_(self.modulation.weight)
                nn.init.zeros_(self.modulation.bias)
        else:
            a = _set_spectral_radius(torch.randn(d_state, d_state), rho_a)
            self.A = nn.Parameter(a)
        c = torch.randn(d_hidden, d_state) / d_state**0.5
        self.C = nn.Parameter(c)
        self.error_modes = None
        if observer == "controlled":
            self.error_modes = ModalTransition(d_state, patch_size, rho_observer, (rho_observer,) * 4, kappa_init=0.0)
        else:
            observer_error = _set_spectral_radius(torch.randn(d_state, d_state), rho_observer)
            self.L = nn.Parameter((a - observer_error) @ torch.linalg.pinv(c))
        self.pool_pk = POOLING[covariate_pooling](d_model, n_heads)
        self.pool_fk = POOLING[covariate_pooling](d_model, n_heads)
        self.B_pk = nn.Linear(d_model, d_state, bias=False)
        self.B_fk = nn.Linear(d_model, d_state, bias=False)
        self.G = nn.Parameter(torch.zeros(d_state, d_state)) if couple_targets else None
        self.readout_activation = readout_activation
        self.gain_b = self.gain_c = None
        if selective_bc:
            self.gain_b, self.gain_c = nn.Linear(d_model, d_state), nn.Linear(d_model, d_state)
            for layer in (self.gain_b, self.gain_c):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.decoder = QuantileDecoder(d_hidden, d_ff, patch_size, n_quantiles, linear=decoder == "linear")

    def transition_matrix(self):
        return self.modal() if hasattr(self, "modal") else self.A

    @torch.no_grad()
    def constrain(self):
        """Projection after an optimizer step: rescale a dense A whose spectral radius exceeds a_cap."""
        if self.a_cap is None or hasattr(self, "modal"):
            return
        rho = torch.linalg.eigvals(self.A.float().cpu()).abs().max().item()
        if rho > self.a_cap:
            self.A.mul_(self.a_cap / rho)

    def forward(self, e, observed_patch, role, patch_visible):
        measurement = self.lift(e)
        e_next = F.pad(e[:, :, 1:], (0, 0, 0, 1))
        future_next = F.pad(((role == FUTURE)[:, :, None] & patch_visible)[:, :, 1:], (0, 1))
        u_pk = self.pool_pk(e, e, (role == PAST)[:, :, None] & patch_visible)
        u_fk = self.pool_fk(e, e_next, future_next)
        started = (patch_visible.cumsum(-1) > 0).unsqueeze(-1)
        with torch.autocast(device_type=e.device.type, enabled=False):
            drive = (self.B_pk(u_pk.float()) + self.B_fk(u_fk.float())) * started
            blocks = self.modal.coefficients(self.modulation(e.float())) if self.modulation is not None else None
            readout_gain = None
            if self.gain_c is not None:
                drive = drive * torch.exp(torch.tanh(self.gain_b(e.float())))
                readout_gain = torch.exp(torch.tanh(self.gain_c(e.float())))
            states = self.observe(measurement.float(), drive, observed_patch, role, blocks, readout_gain)
            readout = (states if readout_gain is None else states * readout_gain) @ self.C.T
        if self.readout_activation == "relu":
            readout = F.relu(readout)
        return self.decoder(readout)

    @staticmethod
    def _transition(x, a, blocks, p):
        """A_p x for a fixed A, or for the per-row 2x2 blocks of a selective transition at patch p."""
        if blocks is None:
            return x @ a.T
        a11, a12, a21, a22 = (c[:, :, p] for c in blocks)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([a11 * x1 + a12 * x2, a21 * x1 + a22 * x2], dim=-1).flatten(-2)

    def observe(self, e, drive, observed_patch, role, blocks=None, readout_gain=None):
        """blocks: per row and patch 2x2 block entries of a selective transition A(e_p), each [B, V, N, d_state / 2].
        readout_gain: per row and patch g_p > 0 of a selective readout C_p = C diag(g_p), [B, V, N, d_state].

        Free observer: learned gain L. Controlled observer: L_p = (A_p - M) C_p^+, so that A_p - L_p C_p = M for every
        input; with z_p = C_p^+ h_p the correction is (A_p - M)(z_p - x_p) and the estimation error obeys e' = M e."""
        b, v, n, _ = e.shape
        is_target = role == TARGET
        a = self.transition_matrix()
        m = observed_patch.float().unsqueeze(-1)
        if self.error_modes is None:
            innovation_in, lc = e @ self.L.T, self.L @ self.C
        else:
            z, error_dynamics = e @ torch.linalg.pinv(self.C).T, self.error_modes()
            if readout_gain is not None:
                z = z / readout_gain

        x = e.new_zeros(b, v, a.shape[0])
        states = []
        for p in range(n):
            ax = self._transition(x, a, blocks, p)
            if self.error_modes is None:
                read = x if readout_gain is None else x * readout_gain[:, :, p]
                correction = innovation_in[:, :, p] - read @ lc.T
            else:
                gap = z[:, :, p] - x
                correction = self._transition(gap, a, blocks, p) - gap @ error_dynamics.T
            x_next = ax + m[:, :, p] * correction + drive[:, :, p]
            if self.G is not None:
                x_next = x_next + self._mean_other_targets(x, is_target) @ self.G.T
            x = x_next
            states.append(x)
        return torch.stack(states, dim=2)

    @staticmethod
    def _mean_other_targets(x, is_target):
        w = is_target.unsqueeze(-1).float()
        total, count = (x * w).sum(1, keepdim=True), w.sum(1, keepdim=True)
        return (total - x * w) / (count - w).clamp_min(1.0) * w


class MixtureHead(nn.Module):
    """Mixture of experts over the MLP and OSSM heads. A gate reading e_p gives, per row and patch, the weight g of
    the OSSM forecast: quantiles = g q_ossm + (1 - g) q_mlp (a quantile average, so still monotone)."""

    def __init__(self, d_model, d_ff, patch_size, n_quantiles, **ossm_kwargs):
        super().__init__()
        self.mlp = MLPHead(d_model, d_ff, patch_size, n_quantiles)
        self.ossm = OSSMHead(d_model, d_ff, patch_size, n_quantiles, **ossm_kwargs)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.last_gate = None

    def forward(self, e, observed_patch, role, patch_visible):
        experts = [self.mlp(e, observed_patch, role, patch_visible), self.ossm(e, observed_patch, role, patch_visible)]
        g = torch.sigmoid(self.gate(e).float()).unsqueeze(-1)
        self.last_gate = g.detach()[..., 0, 0]
        return g * experts[1] + (1 - g) * experts[0], experts

    def constrain(self):
        self.ossm.constrain()


HEADS = {"mlp": MLPHead, "ossm": OSSMHead, "mixture": MixtureHead}
