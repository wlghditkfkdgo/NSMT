"""Pre-registered numerical gates for Population f-LIF v3-A.

Gate list: NSMT/docs/Population_fLIF_v3_prereg_KO.md section 3 and 3A (rev.1).
Every gate here runs without training and without a dataset.

    python check_model.py --phase A
    python check_model.py --phase C
    python check_model.py --phase all
"""
import argparse
import math

import torch

import layers

TOL = 1e-12


class Report:
    def __init__(self):
        self.rows = []

    def add(self, gate, name, passed, detail):
        self.rows.append((gate, name, passed, detail))
        mark = {True: 'PASS', False: 'FAIL', None: 'NOT RUN'}[passed]
        print(f'  [{mark:>7}] {gate:<4} {name}\n            {detail}')

    def failures(self):
        return [r for r in self.rows if r[2] is False]


def build(seed=0, embed_dim=6, double=True, **kwargs):
    torch.manual_seed(seed)
    model = layers.PopulationNeuron(embed_dim=embed_dim, **kwargs)
    return model.double() if double else model


def neutral(model):
    """Force eta = 0 exactly (sigmoid saturates, so the branch is the bare kernel)."""
    with torch.no_grad():
        model.selector.eta_hat.fill_(-1e30)
    return model


# --------------------------------------------------------------------------- Phase A
def phase_a(report, steps=24, dim=4):
    # G1a: the coefficient table telescopes to the exact fractional integral gain.
    worst = 0.
    for alpha in (.3, .5, .7, 1.):
        b = layers.fractional_coefficients(alpha, steps, dtype=torch.float64)
        exact = torch.arange(1, steps + 1, dtype=torch.float64).pow(alpha) / math.exp(math.lgamma(alpha + 1.))
        worst = max(worst, (b.cumsum(0) - exact).abs().max().item())
    report.add('G1a', 'sum_d b_d == (n+1)^alpha / Gamma(alpha+1)', worst < TOL,
               f'max |err| = {worst:.2e} over alpha in 0.3/0.5/0.7/1.0, n<={steps}')

    # G1b: constant forcing on the model. A leak-free branch must trace U = c t^alpha/Gamma.
    big = 1e7
    model = neutral(build(tau=(big,) * 4, embed_dim=dim, alpha=.7))
    current = torch.ones(steps, 1, dim, dtype=torch.float64)
    state = model(current, mode='full', return_aux=True)[1]['state'][:, 0, 0, 0]
    exact = torch.arange(1, steps + 1, dtype=torch.float64).pow(.7) / (big * math.exp(math.lgamma(1.7)))
    rel = ((state - exact).abs() / exact).max().item()
    report.add('G1b', 'constant forcing integrator U = U0 + c t^alpha/Gamma(alpha+1)', rel < 1e-6,
               f'max relative err = {rel:.2e} (tau = {big:.0e}, leak suppressed)')

    # G2: alpha = 1 with eta = 0 collapses onto the Euler leaky integrator.
    model = neutral(build(alpha=1., embed_dim=dim))
    torch.manual_seed(1)
    current = torch.randn(steps, 3, dim, dtype=torch.float64)
    state = model(current, mode='full', return_aux=True)[1]['state']
    euler = torch.zeros(3, dim, 4, dtype=torch.float64)
    reference = []
    for n in range(steps):
        euler = euler + (current[n].unsqueeze(-1) - euler) / model.tau
        reference.append(euler.clone())
    err = (state - torch.stack(reference)).abs().max().item()
    report.add('G2', 'alpha=1, eta=0, g=0  ==  Euler leaky integrator', err < TOL,
               f'max |err| = {err:.2e}  (the g=0 condition is what rev.0 omitted)')

    # G3: the loop equals the closed form, and the naive B @ I / tau does not.
    model = neutral(build(embed_dim=dim))
    state = model(current, mode='full', return_aux=True)[1]['state']
    fast = model.fast_path(current)
    b = model.b[:steps]
    rows = torch.arange(steps).unsqueeze(1) - torch.arange(steps).unsqueeze(0)
    kernel = torch.where(rows >= 0, b[rows.clamp_min(0)], torch.zeros(1, dtype=b.dtype))
    naive = torch.einsum('nm,mdc->ndc', kernel, current).unsqueeze(-1) / model.tau
    err, err_naive = (state - fast).abs().max().item(), (state - naive).abs().max().item()
    report.add('G3', 'loop == (Id + B J/tau)^-1 B/tau   (and != B/tau)', err < 1e-10,
               f'|loop - H_eff| = {err:.2e}   |loop - naive B@I/tau| = {err_naive:.3f}')

    # G4: parity against a pinned spikeDE trajectory. Needs a CPU torch 2.x environment.
    report.add('G4', 'golden trajectory parity against pinned f-SNN code', None,
               'blocked: spikeDE needs torch>=2.0 (snn_recall is 1.12, snn_jelly has no CUDA). '
               'Grade stays "mathematical validation" until reference/make_golden.py can run.')

    # G15: the last patch must reach the last spike (D-D soma alignment).
    model = build(embed_dim=dim, double=False)
    torch.manual_seed(2)
    current = torch.randn(steps, 2, dim)
    with torch.no_grad():
        model.soma.weight.fill_(1. / model.num_population)
    base = model(current, mode='sparse', return_aux=True)[1]['voltage']
    bumped = current.clone(); bumped[-1] += 50.
    moved = model(bumped, mode='sparse', return_aux=True)[1]['voltage']
    delta_last = (base[-1] - moved[-1]).abs().max().item()
    delta_prev = (base[:-1] - moved[:-1]).abs().max().item()
    report.add('G15', 'perturbing I_{T-1} moves v_{T-1} and nothing earlier', delta_last > 1e-3 and delta_prev == 0.,
               f'|d v_last| = {delta_last:.4f}, |d v_earlier| = {delta_prev:.2e}')


# --------------------------------------------------------------------------- Phase B/C
def phase_c(report, steps=24, dim=4):
    torch.manual_seed(3)
    current = torch.randn(steps, 3, dim, dtype=torch.float64)

    # G5a: K=1 reduces to a scalar fractional branch driving the soma.
    single = neutral(build(num_population=1, tau=(8.,), embed_dim=dim))
    state = single(current, mode='full', return_aux=True)[1]['state'][..., 0]
    scalar, trace = torch.zeros(3, dim, dtype=torch.float64), []
    increments = []
    for n in range(steps):
        increments.append((current[n] - scalar) / 8.)
        scalar = sum(single.b[n - j] * increments[j] for j in range(n + 1))
        trace.append(scalar.clone())
    err = (state - torch.stack(trace)).abs().max().item()
    report.add('G5a', 'K=1 == scalar fractional branch (not the paper neuron: no branch reset)',
               err < TOL, f'max |err| = {err:.2e}')

    # G5b: a homogeneous population must hold identical constituents.
    homo = neutral(build(embed_dim=dim, heterogeneous=False))
    state = homo(current, mode='full', return_aux=True)[1]['state']
    spread = (state - state[..., :1]).abs().max().item()
    report.add('G5b', 'homogeneous tau => identical constituents (tau_hom = K/sum(1/tau_k))',
               spread < TOL, f'max spread = {spread:.2e}, tau_hom = {homo.tau[0].item():.3f}')

    # G6: causality -- a future patch may not touch any earlier quantity.
    model = build(embed_dim=dim, double=False)
    cut = steps // 2
    base = model(current.float(), mode='sparse', return_aux=True)[1]
    bumped = current.float().clone(); bumped[cut:] += 7.
    moved = model(bumped, mode='sparse', return_aux=True)[1]
    err = max((base[k][:cut] - moved[k][:cut]).abs().max().item() for k in ('state', 'voltage', 'spikes'))
    report.add('G6', 'perturbing patches >= t leaves every state/score/spike before t intact',
               err == 0., f'max |err| over state/voltage/spikes = {err:.2e}')

    # G7: eta = 0 is bitwise the full-history branch.
    model = neutral(build(embed_dim=dim))
    a = model(current, mode='sparse', return_aux=True)[1]['state']
    c = model(current, mode='full', return_aux=True)[1]['state']
    report.add('G7', 'neutral limit: eta=0 sparse == full, bitwise', torch.equal(a, c),
               f'allclose(atol=0, rtol=0) = {torch.equal(a, c)}')

    # G7b: a uniform p is mathematically the same condition, so `uniform` is not a control.
    model = neutral(build(embed_dim=dim))
    with torch.no_grad():
        model.selector.eta_hat.fill_(1e30)                       # eta = 1: rho = rho_tilde
    uniform = [None] + [torch.full((3, dim, n), 1. / n, dtype=torch.float64) for n in range(1, steps)]
    a = model(current, mode='oracle', oracle_p=uniform, return_aux=True)[1]['state']
    c = model(current, mode='full', return_aux=True)[1]['state']
    err = (a - c).abs().max().item()
    report.add('G7b', 'uniform p == full  (why `uniform` was dropped as a control)', err < 1e-12,
               f'max |err| = {err:.2e} at eta=1')

    # G8/G12: mass and cap behaviour.
    model = neutral(build(embed_dim=dim))
    aux = model(current, mode='sparse', return_aux=True)[1]
    kappa_err = (aux['kappa'][1:] - 1.).abs().max().item()
    cap = aux['cap_rate'].max().item()
    report.add('G12', 'cap is inactive at eta=0 (b_d <= b_0 for d>=1)', cap == 0.,
               f'max cap_rate = {cap:.1e}')
    report.add('G8a', 'kappa == 1.000 while the cap is inactive', kappa_err < 1e-12,
               f'max |kappa - 1| = {kappa_err:.2e}')

    model = build(embed_dim=dim, double=False)
    with torch.no_grad():
        model.selector.eta_hat.fill_(1e30)
    aux = model(current.float(), mode='sparse', return_aux=True)[1]
    zeros = [(c == 0.).any().item() for c in aux['coeff'][1:] if c is not None]
    kappa = aux['kappa'][1:]
    report.add('G8b', 'eta=1 sparsemax produces exact zeros in the final coefficients', any(zeros),
               f'{sum(zeros)}/{len(zeros)} steps contain an exact zero; '
               f'kappa in [{kappa.min():.3f}, {kappa.max():.3f}] (<1 where the cap binds)')

    # G13: support has to be reported per layer, because they differ.
    model = build(embed_dim=dim, double=False)
    aux = model(current.float(), mode='sparse', return_aux=True)[1]
    eta = model.selector.eta.item()
    n = steps - 1
    b_hist = model.b[1:n + 1].flip(0)
    with torch.no_grad():
        xi_now = torch.cat([aux['state'][n - 1], current.float()[n].unsqueeze(-1)], dim=-1)
        hist = torch.stack([torch.cat([aux['state'][j - 1] if j else torch.zeros_like(aux['state'][0]),
                                       current.float()[j].unsqueeze(-1)], dim=-1) for j in range(n)], dim=-2)
        coeff, sel = model.selector.coefficients(xi_now, hist, b_hist, model.b[0].item(), 'sparse')
    sizes = {'p': (sel['p'] > 0).double().mean().item(), 'rho': (sel['rho'] > 0).double().mean().item(),
             'b*rho': (b_hist * sel['rho'] > 0).double().mean().item(),
             'c': (coeff > 0).double().mean().item()}
    report.add('G13', 'support(p) / support(rho) / support(b*rho) / support(c) reported separately',
               sizes['p'] <= sizes['rho'],
               f'eta={eta:.4f}  ' + '  '.join(f'{k}={v:.3f}' for k, v in sizes.items())
               + '  -- dense residual (1-eta)*b_d survives wherever p = 0')

    # G10: every selector parameter must receive a finite, nonzero gradient.
    model = build(embed_dim=dim, double=False)
    out = model(current.float(), mode='sparse')
    out.sum().backward()
    grads = {'W_Q': model.selector.query.weight.grad, 'W_K': model.selector.key.weight.grad,
             'eta_hat': model.selector.eta_hat.grad, 'w': model.soma.weight.grad}
    finite = all(g is not None and torch.isfinite(g).all() for g in grads.values())
    nonzero = {k: (g is not None and g.abs().sum().item() > 0) for k, g in grads.items()}
    report.add('G10', 'gradients finite and nonzero for W_Q, W_K, eta_hat, w', finite and all(nonzero.values()),
               'finite=' + str(finite) + '  nonzero=' + str(nonzero))

    # G9: windows must not talk to each other through the batch axis.
    model = build(embed_dim=dim, double=False)
    batched = model(current.float(), mode='sparse', return_aux=True)[1]['state']
    single = torch.cat([model(current.float()[:, i:i + 1], mode='sparse', return_aux=True)[1]['state']
                        for i in range(current.shape[1])], dim=1)
    err = (batched - single).abs().max().item()
    report.add('G9', 'batched == per-window evaluation', err < 1e-6, f'max |err| = {err:.2e}')

    # G11: bounded state under the adversarial-worst policy the sweep identified.
    worst = {}
    for length in (42, 84):
        model = build(embed_dim=dim, max_length=96, double=False)
        with torch.no_grad():
            model.selector.eta_hat.fill_(1e30)                   # eta = 1, the worst case
        torch.manual_seed(5)
        drive = torch.randn(length, 2, dim) * 2.
        aux = model(drive, mode='recent', return_aux=True)[1]     # recent-only == the loop policy
        worst[length] = aux['state'].abs().max().item()
    bound = 10. * 2. * 3.                                          # 10 * max|I|, roughly
    report.add('G11', 'state stays bounded at eta=1 under the recent-only policy, tau=[4,8,16,32]',
               max(worst.values()) < bound,
               f'max|u| = {worst[42]:.3f} (T=42), {worst[84]:.3f} (T=84); declared bound {bound:.1f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all', choices=['A', 'C', 'all'])
    args = parser.parse_args()
    report = Report()
    if args.phase in ('A', 'all'):
        print('Phase A -- scalar definition and reference parity')
        phase_a(report)
        print()
    if args.phase in ('C', 'all'):
        print('Phase C -- population reduction, causality, selection')
        phase_c(report)
        print()
    failed = report.failures()
    skipped = [r for r in report.rows if r[2] is None]
    print(f'{len(report.rows) - len(failed) - len(skipped)} passed, {len(failed)} failed, {len(skipped)} not run')
    for gate, name, _, _ in failed:
        print(f'  FAILED {gate}: {name}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
