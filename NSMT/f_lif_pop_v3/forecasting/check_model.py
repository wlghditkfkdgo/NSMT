"""Pre-registered numerical gates for Population f-LIF v3-A.

Runs every gate that needs no training and no dataset, so that a wrong definition
is caught before any GPU time is spent. The gate list itself is pre-registered in
``NSMT/docs/Population_fLIF_v3_prereg_KO.md`` sections 3 and 3A (rev.1); this file
is only the executor and must not invent or relax a gate.

Phases
------
A : scalar definition and reference parity  (G1, G2, G3, G4, G15)
C : population reduction, causality, selection  (G5 - G13)

Phase B (firing-rate calibration) needs data and lives in ``calibrate.py``.

Usage
-----
    python check_model.py --phase A
    python check_model.py --phase C
    python check_model.py --phase all
"""

import csv
import math
import hashlib
import argparse
from pathlib import Path

import torch

import layers

GOLDEN = Path(__file__).resolve().parents[1] / 'reference' / 'golden' / 'scalar_trajectories.csv'
GOLDEN_SHA256 = '2fda1abbb03743e1b9074d2fb7b7b84feeec8e322201b42b0b66bc0fc657a7b7'

TOL = 1e-12                    # float64 게이트의 공통 허용오차


class GateReport:
    """Collects gate outcomes and prints them in registration order.

    ``passed`` is tri-state: True / False / None, where None means the gate could
    not be run at all (missing reference environment, say). A not-run gate is never
    silently dropped -- the prereg requires it be reported as "not run".
    """
    def __init__(self, tag='gate'):
        self.tag = tag
        self.rows = []

    def add(self, gate, name, passed, detail):
        self.rows.append((gate, name, passed, detail))
        mark = {True: 'PASS', False: 'FAIL', None: 'NOT RUN'}[passed]
        print(f"[{self.tag}] {mark:>7} | {gate:<4} {name}")
        print(f"[{self.tag}]         | {detail}")

    def failures(self):
        return [row for row in self.rows if row[2] is False]

    def skipped(self):
        return [row for row in self.rows if row[2] is None]


def build_neuron(seed=0, embed_dim=6, double=True, **neuron_args):
    """Fresh neuron with a fixed seed. float64 by default so 1e-12 gates are meaningful."""
    torch.manual_seed(seed)
    dtype = torch.float64 if double else torch.float32
    model = layers.PopulationNeuron(embed_dim=embed_dim, dtype=dtype, **neuron_args)

    return model.to(dtype)


def force_eta(model, value):
    """Saturate the sigmoid so eta is exactly 0 or exactly 1, not merely close."""
    with torch.no_grad():
        model.selector.eta_hat.fill_(-1e30 if value == 0 else 1e30)

    return model


@torch.no_grad()
def phase_a(report, T=24, D=4):
    """Scalar definition: does the branch integrate what the paper says it integrates?

    Args
    ----
    report : GateReport
    T : int, default 24
        Sequence length. Long enough that the kernel tail matters, short enough to be fast.
    D : int, default 4
        Embedding dim.
    """
    # G1a: 계수표가 정확히 분수적분 이득으로 telescoping 되는가 (해석해 대조)
    worst = 0.
    for alpha in (.3, .5, .7, 1.):
        b = layers.fractional_coefficients(alpha, T, dtype=torch.float64)
        exact = torch.arange(1, T + 1, dtype=torch.float64).pow(alpha) / math.exp(math.lgamma(alpha + 1.))
        worst = max(worst, (b.cumsum(0) - exact).abs().max().item())
    report.add('G1a', 'sum_d b_d == (n+1)^alpha / Gamma(alpha+1)', worst < TOL,
               f"max |err| = {worst:.2e} over alpha in 0.3/0.5/0.7/1.0, n <= {T}")

    # G1b: 모델 수준 상수 forcing. tau를 크게 두어 누설을 억제하면 U = c t^alpha / Gamma 여야 한다
    big = 1e7
    model = force_eta(build_neuron(tau=(big,) * 4, embed_dim=D, alpha=.7), 0)
    x = torch.ones(T, 1, D, dtype=torch.float64)
    u = model(x, mode='full', return_aux=True)[1]['state'][:, 0, 0, 0]
    exact = torch.arange(1, T + 1, dtype=torch.float64).pow(.7) / (big * math.exp(math.lgamma(1.7)))
    rel = ((u - exact).abs() / exact).max().item()
    report.add('G1b', 'constant forcing integrator U = U0 + c t^alpha / Gamma(alpha+1)', rel < 1e-6,
               f"max relative err = {rel:.2e} (tau = {big:.0e}, leak suppressed; residual IS the leak)")

    # G2: alpha=1, eta=0, g=0 이면 보통 오일러 leaky integrator와 같아야 한다
    model = force_eta(build_neuron(alpha=1., embed_dim=D), 0)
    torch.manual_seed(1)
    x = torch.randn(T, 3, D, dtype=torch.float64)
    u = model(x, mode='full', return_aux=True)[1]['state']
    euler, ref = torch.zeros(3, D, 4, dtype=torch.float64), []
    for t in range(T):
        euler = euler + (x[t].unsqueeze(-1) - euler) / model.tau
        ref.append(euler.clone())
    err = (u - torch.stack(ref)).abs().max().item()
    report.add('G2', 'alpha=1, eta=0, g=0  ==  Euler leaky integrator', err < TOL,
               f"max |err| = {err:.2e}  (the g=0 condition is exactly what rev.0 omitted)")

    # G3: 루프와 폐형식이 같은가. 그리고 되먹임을 빠뜨린 단순 행렬곱은 틀려야 한다
    model = force_eta(build_neuron(embed_dim=D), 0)
    u = model(x, mode='full', return_aux=True)[1]['state']
    fast = model.fast_path(x)
    b = model.b[:T]
    lag = torch.arange(T).unsqueeze(1) - torch.arange(T).unsqueeze(0)
    kernel = torch.where(lag >= 0, b[lag.clamp_min(0)], torch.zeros(1, dtype=b.dtype))
    naive = torch.einsum('nm,mbd->nbd', kernel, x).unsqueeze(-1) / model.tau
    err, err_naive = (u - fast).abs().max().item(), (u - naive).abs().max().item()
    report.add('G3', 'loop == (Id + B J/tau)^-1 B/tau   (and != B/tau)', err < 1e-10,
               f"|loop - H_eff| = {err:.2e}   |loop - naive B@I/tau| = {err_naive:.3f}")

    # G4: 고정 커밋 spikeDE 궤적과의 대조. 가지가 리셋하지 않으므로 첫 스파이크 전까지만 같다.
    report.add(*golden_parity())

    # G15: D-D 정렬. 마지막 patch가 마지막 스파이크에 도달해야 한다
    model = build_neuron(embed_dim=D, double=False)
    torch.manual_seed(2)
    x = torch.randn(T, 2, D) * 6.                            # 임계값을 실제로 넘는 구동
    base_s, base = model(x, mode='sparse', return_aux=True)
    bumped = x.clone()
    bumped[-1] += 50.
    moved_s, moved = model(bumped, mode='sparse', return_aux=True)
    dv = (base['voltage'][-1] - moved['voltage'][-1]).abs().max().item()
    ds = (base_s[-1] - moved_s[-1]).abs().sum().item()       # 마지막 스파이크가 실제로 바뀌는가
    quiet_v = (base['voltage'][:-1] - moved['voltage'][:-1]).abs().max().item()
    quiet_s = (base_s[:-1] - moved_s[:-1]).abs().sum().item()
    report.add('G15', 'perturbing I_{T-1} flips s_{T-1} and leaves every earlier spike alone',
               ds > 0 and quiet_s == 0. and quiet_v == 0.,
               f"changed spikes at T-1 = {ds:.0f} (rate {base_s.mean():.3f}), |d v_last| = {dv:.3f}; "
               f"earlier: d spikes = {quiet_s:.0f}, |d v| = {quiet_v:.2e}")


def golden_parity(atol=1e-10):
    """Compare the fractional integrator against the pinned upstream trajectories.

    The comparison is deliberately partial. v3-A's branches never reset (prereg D1), while
    the upstream neuron subtracts at threshold, so the two are the same recurrence only up to
    the step before the first spike. 7 of the 24 recorded conditions never spike, and the
    rest are compared on their pre-spike prefix. This is parity for the integrator core, not
    for the full neuron, and the grade is reported as such.
    """
    if not GOLDEN.exists():
        return ('G4', 'golden trajectory parity against pinned f-SNN code', None,
                f'golden file missing at {GOLDEN}; run reference/make_golden.py')
    digest = hashlib.sha256(GOLDEN.read_bytes()).hexdigest()
    if digest != GOLDEN_SHA256:
        return ('G4', 'golden trajectory parity against pinned f-SNN code', False,
                f'golden file changed: sha256 {digest[:16]} != pinned {GOLDEN_SHA256[:16]}')

    rows = list(csv.DictReader(GOLDEN.open()))
    cases, worst, compared, clean = {}, 0., 0, 0
    for row in rows:
        cases.setdefault((row['input'], float(row['alpha']), float(row['tau'])), []).append(row)
    for (name, alpha, tau), steps in cases.items():
        steps.sort(key=lambda r: int(r['n']))
        spike = [i for i, r in enumerate(steps) if float(r['upstream_spike']) > 0]
        limit = spike[0] if spike else len(steps)             # 첫 스파이크 이후는 규약이 갈린다
        if limit == 0:
            continue
        clean += not spike
        model = build_neuron(num_population=1, tau=(tau,), embed_dim=1, alpha=alpha,
                             max_length=len(steps) + 1)
        force_eta(model, 0)
        current = torch.tensor([float(r['current']) for r in steps],
                               dtype=torch.float64).reshape(-1, 1, 1)
        state = model(current, mode='full', return_aux=True)[1]['state'][:, 0, 0, 0]
        target = torch.tensor([float(r['upstream_U_next']) for r in steps], dtype=torch.float64)
        worst = max(worst, (state[:limit] - target[:limit]).abs().max().item())
        compared += limit

    passed = worst < atol
    return ('G4', 'integrator parity against the pinned spikeDE commit fcd743b', passed,
            f'max |err| = {worst:.2e} over {compared} steps in {len(cases)} conditions '
            f'({clean} of them spike-free, the rest compared before their first spike); '
            f'golden sha256 {GOLDEN_SHA256[:16]}. Grade: source parity for the fractional '
            f'integrator. The full v3-A neuron differs by design (D1: branches never reset).')


def phase_c(report, T=24, D=4):
    """Population behaviour: reductions, causality, and what the selection actually does.

    Args
    ----
    report : GateReport
    T, D : int
        Sequence length and embedding dim, as in :func:`phase_a`.
    """
    torch.manual_seed(3)
    x64 = torch.randn(T, 3, D, dtype=torch.float64)
    x32 = x64.float()

    # G5a: K=1은 scalar fractional branch + 소마다. 원 논문 뉴런과 같지 않다 (D-L)
    with torch.no_grad():
        model = force_eta(build_neuron(num_population=1, tau=(8.,), embed_dim=D), 0)
        u = model(x64, mode='full', return_aux=True)[1]['state'][..., 0]
        scalar, incs, ref = torch.zeros(3, D, dtype=torch.float64), [], []
        for t in range(T):
            incs.append((x64[t] - scalar) / 8.)
            scalar = sum(model.b[t - j] * incs[j] for j in range(t + 1))
            ref.append(scalar.clone())
        err = (u - torch.stack(ref)).abs().max().item()
    report.add('G5a', 'K=1 == scalar fractional branch (no branch reset, so not the paper neuron)',
               err < TOL, f"max |err| = {err:.2e}")

    # G5b: 동질 population이면 구성원이 모두 같아야 한다
    with torch.no_grad():
        model = force_eta(build_neuron(embed_dim=D, heterogeneous=False), 0)
        u = model(x64, mode='full', return_aux=True)[1]['state']
        spread = (u - u[..., :1]).abs().max().item()
    report.add('G5b', 'homogeneous tau => identical constituents', spread < TOL,
               f"max spread = {spread:.2e}, tau_hom = K/sum(1/tau_k) = {model.tau[0].item():.3f}")

    # G6: 인과성. 미래 patch를 건드려도 과거 어느 양도 변하면 안 된다
    with torch.no_grad():
        model = build_neuron(embed_dim=D, double=False)
        cut = T // 2
        base = model(x32, mode='sparse', return_aux=True)[1]
        bumped = x32.clone()
        bumped[cut:] += 7.
        moved = model(bumped, mode='sparse', return_aux=True)[1]
        err = max((base[k][:cut] - moved[k][:cut]).abs().max().item()
                  for k in ('state', 'voltage', 'spikes'))
    report.add('G6', 'perturbing patches >= t leaves every state/voltage/spike before t intact',
               err == 0., f"max |err| over state/voltage/spikes = {err:.2e}")

    # G7: 중립극한. eta=0이면 sparse 경로가 full과 비트 단위로 같아야 한다
    with torch.no_grad():
        model = force_eta(build_neuron(embed_dim=D), 0)
        a = model(x64, mode='sparse', return_aux=True)[1]['state']
        c = model(x64, mode='full', return_aux=True)[1]['state']
    report.add('G7', 'neutral limit: eta=0 sparse == full, bitwise', torch.equal(a, c),
               f"torch.equal (atol=0, rtol=0) = {torch.equal(a, c)}")

    # G7b: uniform p도 같은 조건이다. `uniform`을 대조군에서 뺀 이유의 직접 확인 (D-H)
    with torch.no_grad():
        model = force_eta(build_neuron(embed_dim=D), 1)
        flat = [None] + [torch.full((3, D, t), 1. / t, dtype=torch.float64) for t in range(1, T)]
        a = model(x64, mode='oracle', oracle_p=flat, return_aux=True)[1]['state']
        c = model(x64, mode='full', return_aux=True)[1]['state']
        err = (a - c).abs().max().item()
    bitwise = torch.equal(a, c)
    report.add('G7b', 'uniform p == full  (why `uniform` is a gate, not a control)', err < 1e-12,
               f"max |err| = {err:.2e} at eta=1; bitwise = {bitwise}. The registered wording said "
               f"bitwise, but the uniform-p path divides by sum(b*p) while `full` does not, so the "
               f"two differ in the last ulp by construction. Declared tolerance 1e-12 "
               f"(prereg amendment 2026-09-21).")

    # G12/G8a: eta=0에서 상한은 구조적으로 작동할 수 없고, 따라서 질량이 보존된다
    with torch.no_grad():
        model = force_eta(build_neuron(embed_dim=D), 0)
        aux = model(x64, mode='sparse', return_aux=True)[1]
        cap_rate = aux['cap_rate'].max().item()
        kappa_err = (aux['kappa'][1:] - 1.).abs().max().item()
    report.add('G12', 'cap is inactive at eta=0 (b_d <= b_0 for d >= 1)', cap_rate == 0.,
               f"max cap_rate = {cap_rate:.1e}")
    report.add('G8a', 'kappa == 1.000 while the cap is inactive', kappa_err < 1e-12,
               f"max |kappa - 1| = {kappa_err:.2e}")

    # G8b: eta=1이면 sparsemax의 0이 최종 계수까지 살아남고, 상한이 실제로 물린다
    with torch.no_grad():
        model = force_eta(build_neuron(embed_dim=D, double=False), 1)
        aux = model(x32, mode='sparse', return_aux=True)[1]
        zeros = [(c == 0.).any().item() for c in aux['coeff'][1:] if c is not None]
        kappa = aux['kappa'][1:]
    report.add('G8b', 'eta=1 sparsemax leaves exact zeros in the final coefficients', any(zeros),
               f"{sum(zeros)}/{len(zeros)} steps contain an exact zero; "
               f"kappa in [{kappa.min():.3f}, {kappa.max():.3f}] -- below 1 wherever the cap binds")

    # G13: support는 층위마다 다르다. 실제 실행 전체의 aux에서 모은다 (D-K, audit A08)
    with torch.no_grad():
        model = build_neuron(embed_dim=D, double=False)
        aux = model(x32, mode='sparse', return_aux=True)[1]
        live = aux['has_history']                            # t=0은 history가 없어 평균에서 뺀다
        support = {k.replace('support_', ''): aux[k][live].mean().item()
                   for k in ('support_p', 'support_rho', 'support_braw', 'support_c')}
    report.add('G13', 'support(p) / support(rho) / support(b*rho) / support(c) over the whole run',
               support['p'] <= support['rho'],
               f"eta = {model.selector.eta.item():.4f}   "
               + "   ".join(f"{k} = {v:.3f}" for k, v in support.items())
               + f"   over {int(live.sum())} steps with history"
               + "   -- the dense residual (1-eta)*b_d survives wherever p = 0")

    # G10: 선택자 파라미터 전부에 유한하고 0이 아닌 gradient가 흘러야 한다
    model = build_neuron(embed_dim=D, double=False)
    model(x32, mode='sparse').sum().backward()
    grads = {'W_Q': model.selector.query.weight.grad, 'W_K': model.selector.key.weight.grad,
             'eta_hat': model.selector.eta_hat.grad, 'w': model.soma.weight.grad}
    finite = all(g is not None and torch.isfinite(g).all() for g in grads.values())
    nonzero = {k: bool(g is not None and g.abs().sum().item() > 0) for k, g in grads.items()}
    report.add('G10', 'gradients finite and nonzero for W_Q, W_K, eta_hat, w',
               finite and all(nonzero.values()), f"finite = {finite}   nonzero = {nonzero}")

    # G9: 창끼리 배치 축을 통해 정보가 새면 안 된다
    with torch.no_grad():
        model = build_neuron(embed_dim=D, double=False)
        batched = model(x32, mode='sparse', return_aux=True)[1]['state']
        single = torch.cat([model(x32[:, i:i + 1], mode='sparse', return_aux=True)[1]['state']
                            for i in range(x32.shape[1])], dim=1)
        err = (batched - single).abs().max().item()
    report.add('G9', 'batched == per-window evaluation', err < 1e-6, f"max |err| = {err:.2e}")

    # G11: F1(tau 이동)이 실제 모델에서도 상태를 유계로 유지하는가. eta=1이 최악 조건이다
    with torch.no_grad():
        worst = {}
        for length in (42, 84):
            model = force_eta(build_neuron(embed_dim=D, max_length=96, double=False), 1)
            torch.manual_seed(5)
            drive = torch.randn(length, 2, D) * 2.
            aux = model(drive, mode='recent', return_aux=True)[1]
            worst[length] = aux['state'].abs().max().item()
        bound = 10. * drive.abs().max().item()                       # 사전 선언 상한 10 * max|I|
    report.add('G11', 'state stays bounded at eta=1, recent-only policy, tau=[4,8,16,32]',
               max(worst.values()) < bound,
               f"max|u| = {worst[42]:.3f} (T=42), {worst[84]:.3f} (T=84); declared bound {bound:.1f}")


def phase_c_audit(report, T=42, D=2):
    """Gates added after the 2026-09-21 audit. Each one is an issue's pass condition."""
    torch.manual_seed(11)
    b = layers.fractional_coefficients(.7, T, dtype=torch.float64)
    b_hist, mass, b0 = b[1:T].flip(0), b[1:T].sum(), b[0].item()
    xi = torch.randn(1, D, 5, dtype=torch.float64)
    hist = torch.randn(1, D, T - 1, 5, dtype=torch.float64) * 3.
    hist[0, :, 0] = xi[0]                                    # 가장 오래된 slot이 정답이 되도록

    # G16 (audit A01): mass_matched는 cap 뒤의 질량을 맞춰야 한다
    with torch.no_grad():
        sel = layers.Selector(4, None, 1., 0.).double()       # eta = 0.5
        c_sel, a_sel = sel(xi, hist, b_hist, b0, 'sparse')
        c_mm, a_mm = sel(xi, hist, b_hist, b0, 'mass_matched')
        c_full, _ = sel(xi, hist, b_hist, b0, 'full')
        same_mass = (c_mm.sum(-1) - c_sel.sum(-1)).abs().max().item()
        differs = (c_mm - c_full).abs().max().item()
        under_cap = c_mm.max().item() <= b0 + 1e-12
        sel0 = layers.Selector(4, None, 1., 0., eta_fixed=0.).double()
        neutral = torch.equal(sel0(xi, hist, b_hist, b0, 'mass_matched')[0],
                              sel0(xi, hist, b_hist, b0, 'full')[0])
    report.add('G16', 'mass_matched matches the POST-cap mass and does not collapse onto full',
               same_mass < 1e-12 and differs > 1e-6 and under_cap and neutral,
               f"kappa sel {a_sel['kappa'].mean():.12f} vs control {a_mm['kappa'].mean():.12f}; "
               f"|sum c_control - sum c_sel| = {same_mass:.2e}; |control - full| = {differs:.4f}; "
               f"max c_control <= b_0: {under_cap}; cap-inactive => full bitwise: {neutral}")

    # G17 (audit A06): --eta_fixed / --no-cap이 실제 모델을 바꾸는가
    with torch.no_grad():
        exact = {}
        for value in (0., 1.):
            fixed = layers.Selector(4, None, 1., -4., eta_fixed=value).double()
            exact[value] = (fixed.eta.item(), fixed.eta_hat.requires_grad)
        free = layers.Selector(4, None, 1., -4.).double()
        uncapped = layers.Selector(4, None, 1., 0., cap=False).double()
        c_un, _ = uncapped(xi, hist, b_hist, b0, 'sparse')
    report.add('G17', '--eta_fixed is an exact override and --no-cap actually removes the cap',
               exact[0.][0] == 0. and exact[1.][0] == 1. and not exact[0.][1]
               and free.eta_hat.requires_grad and c_un.max().item() > b0,
               f"eta_fixed 0 -> {exact[0.][0]!r}, 1 -> {exact[1.][0]!r} (exact, not a sigmoid "
               f"approximation); eta_hat trainable: fixed {exact[0.][1]}, free {free.eta_hat.requires_grad}; "
               f"no-cap max c = {c_un.max().item():.3f} > b_0 = {b0:.3f}")

    # G18 (prereg 2J): mode='hard' -- q >= 1 is bitwise full; q < 1 keeps exactly round(q J) slots
    # with the untouched kernel value, matches the rule in analysis/hard_mask_screen.py, and
    # actually changes the coefficients.
    with torch.no_grad():
        x = torch.randn(T, 3, D, dtype=torch.float64)
        neutral = build_neuron(double=True, embed_dim=D, hard_q=1.)
        bitwise = torch.equal(neutral(x, mode='hard'), neutral(x, mode='full'))
        st_h = neutral(x, mode='hard', return_aux=True)[1]['state']
        st_f = neutral(x, mode='full', return_aux=True)[1]['state']
        bitwise = bitwise and torch.equal(st_h, st_f)
        q = .25
        sel_h = layers.Selector(4, None, 1., -4., hard_axis='shared', hard_stat='pearson', hard_q=q).double()
        c_h, a_h = sel_h(xi, hist, b_hist, b0, 'hard')
        J = hist.shape[-2]
        want = max(1, int(round(q * J)))
        m = (c_h > 0).to(c_h.dtype)
        exact_kernel = torch.equal(c_h, b_hist * m)              # 남긴 칸은 b 그대로, 나머지는 0
        count_ok = bool(((m.sum(-1) == want)).all())
        under = c_h.max().item() <= b0 + 1e-12
        # the same rule written independently: flatten units, centre, cosine, top-k
        qv = xi.reshape(1, -1) - xi.reshape(1, -1).mean(-1, keepdim=True)
        kv = hist.permute(0, 2, 1, 3).reshape(1, J, -1)
        kv = kv - kv.mean(-1, keepdim=True)
        sim = (qv.unsqueeze(1) * kv).sum(-1) / (qv.norm(dim=-1, keepdim=True) * kv.norm(dim=-1))
        ref = torch.zeros(1, J, dtype=torch.float64).scatter_(-1, sim.topk(want, -1).indices, 1.)
        same_rule = torch.equal(m[0, 0], ref[0]) and torch.equal(m[0, 1], ref[0])   # shared: 단위 간 동일
        changes = not torch.equal(c_h, sel_h(xi, hist, b_hist, b0, 'full')[0])
    report.add('G18', "mode='hard': q>=1 is bitwise full; q<1 keeps round(qJ) slots at exact b, "
                      "matches the screening rule, and differs from full",
               bitwise and exact_kernel and count_ok and under and same_rule and changes,
               f"q=1 bitwise full (spikes and states): {bitwise}; q={q}: kept {int(m.sum(-1)[0, 0])}/{J} "
               f"(want {want}), c == m*b: {exact_kernel}, max c <= b_0: {under}, rule matches "
               f"independent top-k: {same_rule}, differs from full: {changes}")

    # G19 (prereg 2K): the same-budget controls keep exactly k slots, look at no content, and
    # share the parameter initialisation with the pearson model at the same seed.
    with torch.no_grad():
        q = .5
        J = hist.shape[-2]
        want = max(1, int(round(q * J)))
        rec = layers.Selector(4, None, 1., -4., hard_stat='recent', hard_q=q, hard_seed=3).double()
        c_rec, _ = rec(xi, hist, b_hist, b0, 'hard')
        m_rec = (c_rec > 0).to(c_rec.dtype)
        last_k = torch.zeros(J, dtype=torch.float64)
        last_k[J - want:] = 1.                                   # 가장 최근 k칸
        recent_ok = torch.equal(m_rec[0, 0], last_k) and bool((m_rec.sum(-1) == want).all())
        c_rec2, _ = rec(xi * 3. + 1., hist.flip(-1), b_hist, b0, 'hard')
        recent_blind = torch.equal(c_rec, c_rec2)                # 내용을 바꿔도 같은 mask
        rnd = layers.Selector(4, None, 1., -4., hard_stat='random', hard_q=q, hard_seed=3).double()
        c_r1, _ = rnd(xi, hist, b_hist, b0, 'hard')
        c_r2, _ = rnd(xi, hist, b_hist, b0, 'hard')
        rnd.reseed_hard()
        c_r3, _ = rnd(xi, hist, b_hist, b0, 'hard')
        m_r1 = (c_r1 > 0).to(c_r1.dtype)
        random_ok = bool((m_r1.sum(-1) == want).all()) and torch.equal(c_r1, b_hist * m_r1)
        random_fresh = not torch.equal(c_r1, c_r2)               # forward마다 새 추첨
        random_reseed = torch.equal(c_r1, c_r3)                  # 재설정하면 같은 추첨
        # same initialisation across stats at the same seed
        inits = []
        for stat in ('pearson', 'recent', 'random'):
            torch.manual_seed(5)
            n = layers.PopulationNeuron(embed_dim=D, hard_stat=stat, hard_q=q, hard_seed=5, dtype=torch.float64)
            inits.append(torch.cat([p.detach().flatten() for p in n.parameters()]))
        same_init = all(torch.equal(inits[0], v) for v in inits[1:])
    report.add('G19', "2K controls: 'recent' keeps the last k slots and ignores content; 'random' keeps k slots, "
                      "fresh per forward, reproducible after reseed; all stats share the init at a seed",
               recent_ok and recent_blind and random_ok and random_fresh and random_reseed and same_init,
               f"recent == last {want}/{J}: {recent_ok}, content-blind: {recent_blind}; random keeps k at exact b: "
               f"{random_ok}, fresh per forward: {random_fresh}, reseed reproduces: {random_reseed}; "
               f"same init across pearson/recent/random: {same_init}")


def main():
    parser = argparse.ArgumentParser(description='pre-registered numerical gates for v3-A')
    parser.add_argument('--phase', dest='phase', nargs='?', default='all', choices=['A', 'C', 'all'],
                        help='Which gate phase to run \n\t default: %(default)s')
    args = parser.parse_args()

    report = GateReport()
    if args.phase in ('A', 'all'):
        print("[gate] Phase A -- scalar definition and reference parity")
        phase_a(report)
    if args.phase in ('C', 'all'):
        print("[gate] Phase C -- population reduction, causality, selection")
        phase_c(report)
        print("[gate] Phase C (audit) -- pass conditions for the 2026-09-21 audit issues")
        phase_c_audit(report)

    failed, skipped = report.failures(), report.skipped()
    passed = len(report.rows) - len(failed) - len(skipped)
    print(f"[gate] {passed} passed, {len(failed)} failed, {len(skipped)} not run")
    for gate, name, _, _ in failed:
        print(f"[gate] FAILED {gate}: {name}")

    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
