import torch

from tsfm.model.heads import FUTURE, PAD, PAST, TARGET, OSSMHead
from tsfm.model.model import TSFM

P = 8


def make_model(head, **kw):
    torch.manual_seed(0)
    return TSFM(patch_size=P, d_model=32, n_heads=2, d_ff=64, n_blocks=1, head=head,
                quantiles=[0.1, 0.5, 0.9], d_state=8, **kw).eval()


def make_batch(t=8 * P):
    torch.manual_seed(1)
    values = torch.randn(2, 4, t).cumsum(-1)
    observed = torch.ones(2, 4, t, dtype=torch.bool)
    observed[1, :, : 2 * P] = False
    role = torch.tensor([[TARGET, PAST, FUTURE, PAD], [TARGET, TARGET, FUTURE, PAD]])
    observed[:, 3] = False
    return values, observed, role


def test_ossm_initialisation():
    rho = lambda m: torch.linalg.eigvals(m).abs().max().item()
    for d_hidden, rho_a, rho_obs in [(None, 0.9, 0.91), (64, 0.99, 0.9)]:
        head = OSSMHead(32, 64, P, 3, d_state=8, rho_a=rho_a, rho_observer=rho_obs, d_hidden=d_hidden)
        assert abs(rho(head.A) - rho_a) < 1e-4
        assert abs(rho(head.A - head.L @ head.C) - rho_obs) < 1e-3


def test_no_future_leak():
    for head, kw in [("mlp", {}), ("ossm", {}), ("ossm", {"couple_targets": True}), ("ossm", {"mask_correction": False}),
                     ("ossm", {"covariate_pooling": "mean"}), ("ossm", {"d_hidden": 64}), ("mixture", {}),
                     ("ossm", {"transition": "selective", "selective_bc": True, "readout_activation": "relu"}),
                     ("ossm", {"transition": "selective", "selective_bc": True, "decoder": "linear"})]:
        model = make_model(head, **kw)
        values, observed, role = make_batch()
        out = model(values, observed, observed.clone(), role)
        cut = 4 * P
        changed = values.clone()
        changed[:, :2, cut:] += 100.0
        out2 = model(changed, observed, observed.clone(), role)
        p_cut = cut // P
        assert torch.allclose(out[:, :, : p_cut], out2[:, :, : p_cut], atol=1e-4), head
        assert not torch.allclose(out[:, :2, p_cut:], out2[:, :2, p_cut:]), head


def test_known_future_is_used():
    model = make_model("ossm")
    values, observed, role = make_batch()
    out = model(values, observed, observed.clone(), role)
    changed = values.clone()
    changed[:, 2, 6 * P :] += 5.0
    out2 = model(changed, observed, observed.clone(), role)
    assert not torch.allclose(out[0, 0, 4], out2[0, 0, 4])


def test_loss_and_forecast_shapes():
    model = make_model("ossm")
    values, observed, role = make_batch()
    loss = model.loss(values, observed, observed.clone(), role)
    assert torch.isfinite(loss)
    loss.backward()
    fc = model.forecast(values, observed, role, context_patches=6, horizon_patches=2)
    assert fc.shape == (2, 4, 2 * P, 3)
    assert (fc[..., 1:] >= fc[..., :-1]).all()


def test_past_covariate_future_does_not_leak():
    model = make_model("ossm")
    values, observed, role = make_batch()
    out = model(values, observed, observed.clone(), role)
    changed = values.clone()
    changed[0, 1, 5 * P :] += 50.0
    out2 = model(changed, observed, observed.clone(), role)
    assert torch.allclose(out[0, :, :5], out2[0, :, :5], atol=1e-4)
    assert not torch.allclose(out[0, 0, 5:], out2[0, 0, 5:])


def test_univariate_without_covariates_is_finite():
    for pooling in ("attention", "mean"):
        model = make_model("ossm", covariate_pooling=pooling)
        values = torch.randn(3, 1, 6 * P).cumsum(-1)
        observed = torch.ones_like(values, dtype=torch.bool)
        role = torch.full((3, 1), TARGET)
        out = model(values, observed, observed.clone(), role)
        assert torch.isfinite(out).all()


def test_forecast_does_not_depend_on_left_padding():
    for head in ("mlp", "ossm", "mixture"):
        model = make_model(head)
        values, observed, role = make_batch()
        base = model.forecast(values, observed, role, context_patches=6, horizon_patches=2)
        pad = 5 * P
        padded = torch.cat([torch.zeros(2, 4, pad), values], dim=-1)
        padded_obs = torch.cat([torch.zeros(2, 4, pad, dtype=torch.bool), observed], dim=-1)
        out = model.forecast(padded, padded_obs, role, context_patches=11, horizon_patches=2)
        assert torch.allclose(base[:, :3], out[:, :3], atol=1e-3), head


def test_all_zero_context_stays_bounded():
    for head in ("mlp", "ossm", "mixture"):
        model = make_model(head)
        values = torch.zeros(1, 1, 8 * P)
        observed = torch.ones_like(values, dtype=torch.bool)
        out = model.forecast(values, observed, torch.full((1, 1), TARGET), context_patches=6, horizon_patches=2)
        assert out.abs().max() < 1e3, head


def test_mixture_starts_as_the_average_of_its_experts():
    model = make_model("mixture")
    values, observed, role = make_batch()
    mixed, (q_mlp, q_ossm) = model.predict_patches(values, observed, observed.clone(), role)
    assert torch.allclose(mixed, 0.5 * (q_mlp + q_ossm), atol=1e-5)
    assert (mixed[..., 1:] >= mixed[..., :-1]).all()
    loss = model.loss(values, observed, observed.clone(), role)
    loss.backward()
    assert model.head.gate[-1].weight.grad is not None


def test_modal_transition_is_stable_by_construction():
    head = OSSMHead(32, 64, P, 3, d_state=8, rho_a=0.99, rho_observer=0.9, transition="modal")
    rho = lambda m: torch.linalg.eigvals(m).abs().max().item()
    assert abs(rho(head.transition_matrix() - head.L @ head.C) - 0.9) < 1e-3
    with torch.no_grad():
        head.modal.radius_logit.add_(10.0 * torch.randn_like(head.modal.radius_logit))
        head.modal.theta.add_(torch.randn_like(head.modal.theta))
    assert rho(head.transition_matrix()) < 1.0


def test_cap_rescales_an_unstable_dense_transition():
    head = OSSMHead(32, 64, P, 3, d_state=8, a_cap=1.0)
    with torch.no_grad():
        head.A.mul_(1.5 / torch.linalg.eigvals(head.A).abs().max())
    head.constrain()
    assert torch.linalg.eigvals(head.A).abs().max().item() <= 1.0 + 1e-5


def test_modal_head_has_no_leak_and_ignores_padding():
    model = make_model("ossm", transition="modal", rho_a=0.99)
    values, observed, role = make_batch()
    out = model(values, observed, observed.clone(), role)
    changed = values.clone()
    changed[:, :2, 4 * P :] += 100.0
    assert torch.allclose(out[:, :, :4], model(changed, observed, observed.clone(), role)[:, :, :4], atol=1e-4)
    base = model.forecast(values, observed, role, context_patches=6, horizon_patches=2)
    padded = torch.cat([torch.zeros(2, 4, 3 * P), values], dim=-1)
    padded_obs = torch.cat([torch.zeros(2, 4, 3 * P, dtype=torch.bool), observed], dim=-1)
    assert torch.allclose(base[:, :3], model.forecast(padded, padded_obs, role, 9, 2)[:, :3], atol=1e-3)


def test_modal_trend_block_ramps_while_stable():
    from tsfm.model.heads import ModalTransition
    modal = ModalTransition(8, P, rho=0.99, real_radii=(0.999,))
    a = modal().detach()
    assert torch.linalg.eigvals(a).abs().max() < 1.0
    x = torch.zeros(8)
    x[1] = 1.0
    path = [x]
    for _ in range(20):
        path.append(a @ path[-1])
    level = torch.stack(path)[:, 0]
    assert (level[1:] > level[:-1]).all()


def test_selective_transition_starts_as_modal_and_has_no_leak():
    torch.manual_seed(0)
    modal = make_model("ossm", transition="modal", rho_a=0.99)
    torch.manual_seed(0)
    selective = make_model("ossm", transition="selective", rho_a=0.99)
    selective.load_state_dict(modal.state_dict(), strict=False)
    values, observed, role = make_batch()
    out = selective(values, observed, observed.clone(), role)
    assert torch.allclose(out, modal(values, observed, observed.clone(), role), atol=1e-4)

    with torch.no_grad():
        selective.head.modulation.weight.normal_(0, 0.05)
    out = selective(values, observed, observed.clone(), role)
    changed = values.clone()
    changed[:, :2, 4 * P :] += 100.0
    assert torch.allclose(out[:, :, :4], selective(changed, observed, observed.clone(), role)[:, :, :4], atol=1e-4)
    base = selective.forecast(values, observed, role, context_patches=6, horizon_patches=2)
    padded = torch.cat([torch.zeros(2, 4, 3 * P), values], dim=-1)
    padded_obs = torch.cat([torch.zeros(2, 4, 3 * P, dtype=torch.bool), observed], dim=-1)
    assert torch.allclose(base[:, :3], selective.forecast(padded, padded_obs, role, 9, 2)[:, :3], atol=1e-3)


def test_selective_blocks_match_the_modal_matrix():
    from tsfm.model.heads import ModalTransition
    modal = ModalTransition(8, P, rho=0.99)
    x = torch.randn(3, 8)
    a11, a12, a21, a22 = modal.coefficients(torch.zeros(3, modal.n_modulation))
    x1, x2 = x[..., 0::2], x[..., 1::2]
    blockwise = torch.stack([a11 * x1 + a12 * x2, a21 * x1 + a22 * x2], dim=-1).flatten(-2)
    assert torch.allclose(blockwise, x @ modal().T, atol=1e-6)


def test_controlled_observer_fixes_the_error_dynamics_for_every_input():
    torch.manual_seed(0)
    model = make_model("ossm", transition="selective", observer="controlled", rho_a=0.99, rho_observer=0.7)
    head = model.head
    with torch.no_grad():
        head.modulation.weight.normal_(0, 0.1)
    c, c_pinv, m = head.C, torch.linalg.pinv(head.C), head.error_modes()
    assert abs(torch.linalg.eigvals(m).abs().max().item() - 0.7) < 1e-4
    a11, a12, a21, a22 = head.modal.coefficients(head.modulation(torch.randn(5, 32)))
    for k in range(5):
        blocks = [torch.stack([torch.stack([a11[k, j], a12[k, j]]), torch.stack([a21[k, j], a22[k, j]])])
                  for j in range(a11.shape[1])]
        a_p = torch.block_diag(*blocks)
        l_p = (a_p - m) @ c_pinv
        assert torch.allclose(a_p - l_p @ c, m, atol=1e-4)

    values, observed, role = make_batch()
    out = model(values, observed, observed.clone(), role)
    changed = values.clone()
    changed[:, :2, 4 * P :] += 100.0
    assert torch.allclose(out[:, :, :4], model(changed, observed, observed.clone(), role)[:, :, :4], atol=1e-4)
    base = model.forecast(values, observed, role, context_patches=6, horizon_patches=2)
    padded = torch.cat([torch.zeros(2, 4, 3 * P), values], dim=-1)
    padded_obs = torch.cat([torch.zeros(2, 4, 3 * P, dtype=torch.bool), observed], dim=-1)
    assert torch.allclose(base[:, :3], model.forecast(padded, padded_obs, role, 9, 2)[:, :3], atol=1e-3)
    model.loss(values, observed, observed.clone(), role).backward()


def test_selective_control_and_readout():
    for observer in ("free", "controlled"):
        torch.manual_seed(0)
        base = make_model("ossm", transition="selective", observer=observer, rho_a=0.99)
        torch.manual_seed(0)
        model = make_model("ossm", transition="selective", observer=observer, rho_a=0.99, selective_bc=True)
        model.load_state_dict(base.state_dict(), strict=False)
        values, observed, role = make_batch()
        out = model(values, observed, observed.clone(), role)
        assert torch.allclose(out, base(values, observed, observed.clone(), role), atol=1e-4), observer

        with torch.no_grad():
            for layer in (model.head.modulation, model.head.gain_b, model.head.gain_c):
                layer.weight.normal_(0, 0.1)
        out = model(values, observed, observed.clone(), role)
        changed = values.clone()
        changed[:, :2, 4 * P :] += 100.0
        assert torch.allclose(out[:, :, :4], model(changed, observed, observed.clone(), role)[:, :, :4], atol=1e-4)
        base_fc = model.forecast(values, observed, role, context_patches=6, horizon_patches=2)
        padded = torch.cat([torch.zeros(2, 4, 3 * P), values], dim=-1)
        padded_obs = torch.cat([torch.zeros(2, 4, 3 * P, dtype=torch.bool), observed], dim=-1)
        assert torch.allclose(base_fc[:, :3], model.forecast(padded, padded_obs, role, 9, 2)[:, :3], atol=1e-3)
        model.loss(values, observed, observed.clone(), role).backward()
        assert model.head.gain_c.weight.grad.abs().sum() > 0


def test_controlled_observer_with_selective_readout_keeps_error_dynamics():
    torch.manual_seed(0)
    c = torch.randn(32, 8)
    g = torch.exp(torch.tanh(torch.randn(8)))
    c_p = c @ torch.diag(g)
    c_p_pinv = torch.diag(1 / g) @ torch.linalg.pinv(c)
    a_p, m = torch.randn(8, 8), torch.randn(8, 8)
    assert torch.allclose(a_p - (a_p - m) @ c_p_pinv @ c_p, m, atol=1e-4)



def test_linear_decoder_is_a_matrix():
    model = make_model("ossm", transition="selective", selective_bc=True, decoder="linear")
    assert isinstance(model.head.decoder.block, torch.nn.Linear)
    values, observed, role = make_batch()
    fc = model.forecast(values, observed, role, context_patches=6, horizon_patches=2)
    assert (fc[..., 1:] >= fc[..., :-1]).all()
