#!/usr/bin/env python3
"""Gates for the spikeDE-based neuron family in neurons.py (torch 1.12; CPU except W7).

    python check_neurons.py [--spikede <spikeDE root at fcd743b>] [legacy run directory ...]

  S1   spikede.py is the verbatim spikeDE copy (spikede_copy.py check; needs --spikede)
  N1   FLIFNeuron(alpha=1, pred) == LIFNeuron (Euler)             float64, rel 1e-12, spikes equal
  N2   FLIFNeuron(alpha=1, gl)   == LIFNeuron
  N3   q = 1 removes the selection exactly: Selective* == its class without Selective, bitwise
  N4   K = 1 removes the population exactly: Population* (one tau) == its single-member class, bitwise
  N5   PopulationSelectiveFLIF(K=1, q=.5) == SelectiveFLIF, bitwise
  N6   FLIFNeuron == spikede.snn_solve(spikede.LIFNeuron, AdamsBashforthSNN) called directly, bitwise
  N7   Soma == the v3 soma recursion with the reset theta/tau_s (spikes bitwise, V to 1e-12);
       with v3's reset theta it differs (the decided change)
  N8   PopulationFLIFSoma: member states == layers.PopulationNeuron mode 'full' to 1e-12 (heterogeneous and
       homogeneous); its spikes == the soma reference on its own states, bitwise
  N9   PopulationSelectiveFLIFSoma(pearson, shared, q .5): states and coefficients == legacy mode 'hard' to
       1e-12, the same slots kept
  N10  recent / random controls == legacy mode 'hard' with the same seed (states to 1e-12, same slots)
  N11  part='input': q = 1 is bitwise the class without selection; q < 1 differs from part='all'; states
       == the 2Q (A) leak-kept recursion (analysis/selection_stability.run) to 1e-12
  N12  memory >= T is bitwise the full history (pred, gl); memory = 5 is not
  N13  gradients reach the input current (and the soma weights), finite and nonzero, all nine classes
  N14  shapes, out_features, refused configurations
  N15  select_mask keeps exactly max(1, round(qJ)) slots; 'recent' keeps the newest
  N16  composition: every class inherits FLIFNeuron (LIF: LIFNeuron) and owns Population / Selective / Soma
       INSTANCES; members and soma are spikede.LIFNeuron; Selective is a spikede.AdamsBashforthSNN
  W1   every (neuron, soma) builds through config -> model.load_mymodel -> ours.myModel and runs
  W2   myModel pop_flif (soma) vs legacy --mode full with shared initial values: member states to float32
       rounding (rel 1e-5); spikes == the soma reference on its own states
  W3   three AdamW steps with clipping: finite, parameters move (pop_flif, sel_pop_flif)
  W4   --soma none: the heads read D*K spike trains
  W5   config.check_neuron_args accepts consistent settings and refuses ignored ones; names and calibration
       tags distinct (legacy: none)
  W6   the legacy neuron_kwargs is the dict it was before the neurons.py wiring
  W7   (run directories) a saved legacy checkpoint reproduces its recorded best validation MSE
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import layers                                                    # noqa: E402
import neurons as N                                              # noqa: E402
import spikede                                                   # noqa: E402
from config import parse_defaults, neuron_kwargs, check_neuron_args, neuron_variant, calibration_tag  # noqa: E402
from model import load_mymodel                                   # noqa: E402

LEGACY_KEYS = ['num_population', 'alpha', 'tau', 'heterogeneous', 'max_length', 'theta', 'eta_init',
               'eta_fixed', 'cap', 'key_norm', 'qk_norm', 'qk_eps', 'tau_s', 'threshold',
               'surrogate_scale', 'hard_axis', 'hard_stat', 'hard_q', 'hard_seed']   # 연결 이전 (cf21ed68b)
T, B, D = 42, 3, 6
RESULTS = []


def gate(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"[neurons] {'PASS' if ok else 'FAIL'} | {name:<5} {detail}")


def current(dtype=torch.float64, seed=1, scale=3., shape=(T, B, D)):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(*shape, generator=g, dtype=dtype)


def build(name, soma=None, dtype=torch.float64, **kw):
    kw.setdefault('max_length', T)
    return N.build_neuron(name, D, soma=soma, dtype=dtype, **kw)


def legacy(dtype=torch.float64, **kw):
    kw.setdefault('max_length', T)
    return layers.PopulationNeuron(D, dtype=dtype, **kw)


def run(neuron, x, **kw):
    out, aux = neuron(x, return_aux=True, **kw)
    return out, aux['state'], aux


def same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def rel(a, b):
    return float((a - b).abs().max() / max(1., float(b.abs().max())))


def soma_reference(u, weight, tau_s=2., threshold=1., reset=None):
    """The v3 soma recursion (layers.Soma) with a chosen reset; reset = theta/tau_s is spikeDE's."""
    reset = threshold / tau_s if reset is None else reset
    a = (weight * u).sum(-1)
    v, spikes, volts = torch.zeros_like(a[0]), [], []
    for k in range(a.shape[0]):
        charge = v + (a[k] - v) / tau_s
        s = (charge - threshold >= 0.).to(a.dtype)
        v = charge - reset * s
        spikes.append(s)
        volts.append(v)
    return torch.stack(spikes), torch.stack(volts)


def slots(coeffs):
    return [None if c is None else (c > 0) for c in coeffs]


@torch.no_grad()
def classes(spikede_root):
    x = current()

    # S1 -------------------------------------------------------------------------------------
    if spikede_root:
        import subprocess
        r = subprocess.run([sys.executable, str(Path(__file__).with_name('spikede_copy.py')), 'check', spikede_root],
                           capture_output=True, text=True)
        gate('S1', r.returncode == 0, r.stdout.strip().splitlines()[0].split('| ', 1)[-1] if r.stdout else r.stderr)
    else:
        print('[neurons] S1    not run (no --spikede root given)')

    # N1, N2 ---------------------------------------------------------------------------------
    for gate_name, method in (('N1', 'pred'), ('N2', 'gl')):
        lif = run(build('lif', tau=2.), x)
        frac = run(build('flif', tau=2., alpha=1., method=method), x)
        gate(gate_name, rel(frac[1], lif[1]) < 1e-12 and torch.equal(frac[0], lif[0]),
             f"f-LIF(alpha=1, {method}) vs LIF (Euler): rel state err {rel(frac[1], lif[1]):.1e}, "
             f"spikes equal {torch.equal(frac[0], lif[0])} ({int(lif[0].sum())} spikes)")

    # N3, N4, N5 -----------------------------------------------------------------------------
    ok3 = {}
    for dtype in (torch.float32, torch.float64):
        xd = current(dtype)
        for sel, plain, kw in (('sel_flif', 'flif', dict(tau=2.)), ('sel_pop_flif', 'pop_flif', dict())):
            for soma in ('none', 'lif'):
                ok3[f'{sel}/{soma}/{str(dtype)[6:]}'] = same(run(build(sel, soma, dtype, q=1., **kw), xd),
                                                            run(build(plain, soma, dtype, **kw), xd))
    gate('N3', all(ok3.values()), f"q=1 == the class without Selective, bitwise: {all(ok3.values())} ({len(ok3)} cases)")
    ok4 = {soma: same(run(build('pop_flif', soma, tau=(2.,)), x), run(build('flif', soma, tau=2.), x))
           for soma in ('none', 'lif')}
    gate('N4', all(ok4.values()), f"PopulationFLIF[Soma](K=1) == FLIFNeuron / FLIFSoma bitwise: {ok4}")
    a = run(build('sel_pop_flif', 'none', tau=(2.,), q=.5), x)
    b = run(build('sel_flif', 'none', tau=2., q=.5), x)
    gate('N5', same(a, b), f"PopulationSelectiveFLIF(K=1, q=.5) == SelectiveFLIF bitwise: {same(a, b)}")

    # N6 -------------------------------------------------------------------------------------
    n = build('flif', tau=2., alpha=.7)
    cell = spikede.LIFNeuron(tau=2., threshold=1., surrogate_grad_scale=5.)
    seen = []

    def rhs(t, y):
        k = int(round(float(t)))
        seen.append(y[0])
        return cell(y[0], x[k] if k < T else torch.zeros_like(x[0]))
    hist = spikede.snn_solve(rhs, (torch.zeros_like(x[0]), torch.zeros_like(x[0])), [.7],
                             torch.arange(T + 2, dtype=x.dtype), spikede.AdamsBashforthSNN())
    direct_state, direct_spikes = torch.stack(seen[1:T + 1]), torch.stack(hist[1][:T]).to(x.dtype)
    mine = run(n, x)
    ok6 = torch.equal(mine[1][..., 0], direct_state) and torch.equal(mine[0], direct_spikes)
    gate('N6', ok6, f"FLIFNeuron == spikede.snn_solve(LIFNeuron, AdamsBashforthSNN) called directly, bitwise: {ok6}")

    # N7 -------------------------------------------------------------------------------------
    u = current(shape=(T, B, D, 4), scale=2.) + .8
    soma = N.Soma(D, 4, dtype=torch.float64)
    soma.weight.copy_(torch.rand(D, 4, generator=torch.Generator().manual_seed(3), dtype=torch.float64))
    s_new, v_new, _ = soma(u)
    s_ref, v_ref = soma_reference(u, soma.weight)
    s_v3, _ = soma_reference(u, soma.weight, reset=1.)
    ok7 = torch.equal(s_new, s_ref) and rel(v_new, v_ref) < 1e-12 and not torch.equal(s_new, s_v3)
    gate('N7', ok7, f"Soma == v3 soma with reset theta/tau_s: spikes bitwise {torch.equal(s_new, s_ref)}, V rel err "
                    f"{rel(v_new, v_ref):.1e}; {int(s_new.sum())} spikes vs {int(s_v3.sum())} with v3's reset theta")

    # N8, N9, N10 ----------------------------------------------------------------------------
    for gate_name, cases in (('N8', [('pop_flif', dict(), dict(), 'full'),
                                      ('pop_flif', dict(heterogeneous=False), dict(heterogeneous=False), 'full')]),
                             ('N9', [('sel_pop_flif', dict(q=.5), dict(hard_q=.5), 'hard')]),
                             ('N10', [('sel_pop_flif', dict(q=.5, stat=s, seed=11), dict(hard_q=.5, hard_stat=s, hard_seed=11),
                                       'hard') for s in ('recent', 'random')])):
        ok, worst = [], 0.
        for name, kw_new, kw_old, mode in cases:
            new = run(build(name, 'lif', **kw_new), x)
            old = run(legacy(**kw_old), x, mode=mode)
            err = rel(new[1], old[1])
            worst = max(worst, err) if err == err else float('nan')
            ref_spikes, _ = soma_reference(new[1], new_soma_weight(name, kw_new))
            coeff_ok = all((cn is None) == (co is None) and (cn is None or rel(cn, co) < 1e-12)
                           for cn, co in zip(new[2]['coeff'], old[2]['coeff']))
            slot_ok = all((sn is None and so is None) or torch.equal(sn, so)
                          for sn, so in zip(slots(new[2]['coeff']), slots(old[2]['coeff'])))
            ok.append(err < 1e-12 and coeff_ok and slot_ok and torch.equal(new[0], ref_spikes))
        gate(gate_name, all(ok), f"{[c[0] + str(c[1]) for c in cases]} vs layers.PopulationNeuron(mode='{cases[0][3]}'): "
                                 f"member states rel err <= {worst:.1e}, coefficients and kept slots equal, spikes == "
                                 f"soma reference on own states: {ok}")

    # N11 ------------------------------------------------------------------------------------
    import selection_stability as S
    one = same(run(build('sel_flif', tau=2., q=1., part='input'), x), run(build('flif', tau=2.), x))
    full_sel = run(build('sel_pop_flif', 'lif', q=.5, part='all'), x)
    in_sel = run(build('sel_pop_flif', 'lif', q=.5, part='input'), x)
    differs = not torch.equal(full_sel[1], in_sel[1])
    nn_ = build('sel_pop_flif', 'lif', q=.5, part='input')
    diag, _, _ = S.run(x, layers.fractional_coefficients(.7, T, torch.float64), nn_.tau.double(), 'pearson', .5,
                       leak_keep=True)
    err = rel(in_sel[1], diag)
    gate('N11', one and differs and err < 1e-12,
         f"part='input': q=1 bitwise without selection {one}; differs from part='all' {differs}; "
         f"states vs the 2Q (A) leak-kept recursion rel err {err:.1e}")

    # N12 ------------------------------------------------------------------------------------
    ok12 = {}
    for method in ('pred', 'gl'):
        full = run(build('flif', tau=2., method=method), x)
        long = run(build('flif', tau=2., method=method, memory=T), x)
        short = run(build('flif', tau=2., method=method, memory=5), x)
        ok12[method] = same(full, long) and not torch.equal(full[1], short[1])
    gate('N12', all(ok12.values()), f"memory >= T is bitwise the full history and memory=5 is not: {ok12}")

    # N14, N15 -------------------------------------------------------------------------------
    shapes = {}
    for name in N.NEURONS:
        for soma in ('none', 'lif'):
            if name == 'lif' and soma == 'lif':
                continue
            kw = dict(tau=2.) if name in ('lif', 'flif', 'sel_flif') else dict()
            n = build(name, soma, **kw, alpha=1. if name == 'lif' else .7, q=.5, unrelated_key=1)
            shapes[f'{name}/{soma}'] = (n(x).shape[-1], n.out_features)
    shape_ok = all(o == f for o, f in shapes.values()) and shapes['pop_flif/none'][0] == 4 * D \
        and shapes['pop_flif/lif'][0] == D and shapes['flif/lif'][0] == D and shapes['flif/none'][0] == D
    refused = []
    for bad in (lambda: build('flif', tau=(2., 4.)), lambda: build('sel_flif', tau=2., method='gl'),
                lambda: build('lif', tau=0.), lambda: build('flif', tau=2.)(torch.zeros(T + 1, B, D)),
                lambda: N.neuron_class('lif', 'lif'), lambda: N.neuron_class('pop_flif', 'both'),
                lambda: build('flif', tau=2.)(x, analog='analog')):
        try:
            bad()
            refused.append(False)
        except ValueError:
            refused.append(True)
    gate('N14', shape_ok and all(refused), f"out_features {shapes}; refused configurations {refused}")
    J, ok15 = 41, []
    xi, hist_ = torch.randn(B, D, 5, dtype=torch.float64), torch.randn(B, D, J, 5, dtype=torch.float64)
    for q in (.1, .25, .5, .75):
        for stat in N.STATS:
            m, _ = N.select_mask(xi, hist_, q, stat, 'shared', torch.Generator().manual_seed(0))
            ok15.append(bool((m.sum(-1) == max(1, int(round(q * J)))).all()))
    m, _ = N.select_mask(xi, hist_, .25, 'recent')
    newest = bool((m[..., -10:] == 1).all() and (m[..., :-10] == 0).all())
    gate('N15', all(ok15) and newest, f"kept exactly round(qJ) in 16 cases: {all(ok15)}; recent keeps the newest 10/41: {newest}")

    # N16 ------------------------------------------------------------------------------------
    ok16 = {}
    for name in N.NEURONS:
        for soma in ('none', 'lif'):
            if name == 'lif' and soma == 'lif':
                continue
            cls = N.neuron_class(name, soma)
            n = build(name, soma, **(dict(tau=2.) if name in ('lif', 'flif', 'sel_flif') else {}))
            base = N.LIFNeuron if name == 'lif' else N.FLIFNeuron
            pure = cls is base or cls.__mro__[1] is base             # 바로 위 부모가 FLIFNeuron (구성 요소를 상속하지 않음)
            pop = (n.population is None) == (name not in ('pop_flif', 'sel_pop_flif')) and \
                (n.population is None or isinstance(n.population, N.Population))
            sel = (n.selective is None) == (name not in ('sel_flif', 'sel_pop_flif')) and \
                (n.selective is None or isinstance(n.selective, spikede.AdamsBashforthSNN))
            som = (n.soma is None) == (soma == 'none') and (n.soma is None or isinstance(n.soma.neuron, spikede.LIFNeuron))
            cells = all(isinstance(c, spikede.LIFNeuron) for c in n.cells)
            silent = all(c.threshold == (N.SILENT if soma == 'lif' else 1.) for c in n.cells)
            ok16[f'{name}/{soma}'] = pure and pop and sel and som and cells and silent
    gate('N16', all(ok16.values()), f"each class = FLIFNeuron + Population/Selective/Soma instances, spikeDE cells, "
                                    f"silent members under a soma: {ok16}")


def new_soma_weight(name, kw):
    """The initial soma weight of a freshly built neuron (deterministic: 1/K)."""
    return build(name, 'lif', **kw).soma.weight


def gradients():
    ok13 = {}
    for name in N.NEURONS:
        for soma in ('none', 'lif'):
            if name == 'lif' and soma == 'lif':
                continue
            kw = dict(tau=2.) if name in ('lif', 'flif', 'sel_flif') else dict()
            kw.update(dict(q=.5) if name.startswith('sel') else {})
            n = build(name, soma, **kw)
            xg = current().requires_grad_(True)
            out = n(xg)
            (out * torch.linspace(.5, 1.5, out.shape[-1], dtype=torch.float64)).sum().backward()
            grads = [xg.grad] + ([n.soma.weight.grad] if n.has_soma else [])
            ok13[f'{name}/{soma}'] = all(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
                                         for g in grads)
    gate('N13', all(ok13.values()), f"finite, nonzero gradients to the input (and soma weights): {ok13}")


# ==== pipeline =====================================================================================
def cli(**kw):
    """Parser defaults with some settings changed, as Config would hold them (no run directory)."""
    ns = parse_defaults()
    for key, value in kw.items():
        setattr(ns, key, value)
    ns.task, ns.revin, ns.input_scale, ns.cpu = kw.get('task', 'ett'), kw.get('revin', True), 6., True
    ns.num_patches = ns.seq_len // ns.patch_size
    ns.device = torch.device('cpu')

    return ns


def ett_batch(seed=3, B_=4, C=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.cumsum(torch.randn(B_, 336, C, generator=g), dim=1)
    y = x[:, -1:, :] + torch.randn(B_, 96, C, generator=g)

    return x, y


def wiring():
    x, y = ett_batch()
    one = dict(tau=[8.], num_population=1)
    combos = [('lif', 'none', dict(alpha=1., mode='full', **one)), ('flif', 'none', dict(mode='full', **one)),
              ('flif', 'lif', dict(mode='full', **one)), ('sel_flif', 'none', dict(mode='hard', hard_q=.5, **one)),
              ('sel_flif', 'lif', dict(mode='hard', hard_q=.5, **one)), ('pop_flif', 'none', dict(mode='full')),
              ('pop_flif', 'lif', dict(mode='full')), ('sel_pop_flif', 'none', dict(mode='hard', hard_q=.5)),
              ('sel_pop_flif', 'lif', dict(mode='hard', hard_q=.5))]

    # W1 -------------------------------------------------------------------------------------
    ok1 = {}
    for name, soma, kw in combos:
        for head in ('flatten', 'linear'):
            ns = cli(neuron=name, soma=soma, head_mode=head, **kw)
            check_neuron_args(ns)
            model = load_mymodel(ns)
            with torch.no_grad():
                out = model(x, mode=ns.mode)
            ok1[f'{name}/{soma}/{head}'] = tuple(out.shape) == (x.shape[0], 96, x.shape[2]) and bool(torch.isfinite(out).all())
    gate('W1', all(ok1.values()), f"every (neuron, soma) through config -> load_mymodel -> myModel runs, finite: "
                                  f"{sum(ok1.values())}/{len(ok1)}")

    # W2 -------------------------------------------------------------------------------------
    old = load_mymodel(cli(mode='full'))
    new = load_mymodel(cli(neuron='pop_flif', soma='lif', mode='full'))
    state, mine = old.state_dict(), new.state_dict()
    missing = [k for k in mine if k not in state]
    b_equal = torch.equal(state['embedding.neuron.b'], mine['embedding.neuron.b'])
    new.load_state_dict({k: state[k] for k in mine}, strict=True)
    with torch.no_grad():
        _, aux_old = old(x, mode='full', return_aux=True)
        _, aux_new = new(x, mode='full', return_aux=True)
    err = rel(aux_new['state'], aux_old['state'])
    ref_spikes, _ = soma_reference(aux_new['state'], new.embedding.neuron.soma.weight)
    ok2 = not missing and b_equal and err < 1e-5 and torch.equal(aux_new['spikes'], ref_spikes)
    gate('W2', ok2, f"myModel pop_flif(soma) vs legacy --mode full (shared initial values): member states rel err {err:.1e} "
                    f"(float32), spikes == soma reference on own states {torch.equal(aux_new['spikes'], ref_spikes)}, "
                    f"keys without legacy counterpart {missing}, kernel table equal {b_equal}")

    # W3 -------------------------------------------------------------------------------------
    ok3 = {}
    with torch.enable_grad():
        for name, kw in (('pop_flif', dict(mode='full')), ('sel_pop_flif', dict(mode='hard', hard_q=.5))):
            model = load_mymodel(cli(neuron=name, soma='lif', **kw))
            before = {k: p.detach().clone() for k, p in model.named_parameters()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=.01)
            finite = True
            for _ in range(3):
                loss = torch.nn.functional.mse_loss(model(x, mode=kw['mode']), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                finite = finite and bool(torch.isfinite(loss))
            moved = [k for k, p in model.named_parameters() if not torch.equal(p, before[k])]
            ok3[name] = (finite and 'embedding.neuron.soma.weight' in moved, f'{len(moved)}/{len(before)} moved')
    gate('W3', all(v[0] for v in ok3.values()), f"3 AdamW steps (lr 1e-2, clip 1): finite, soma weights and others move: {ok3}")

    # W4 -------------------------------------------------------------------------------------
    ok4 = {}
    for head in ('flatten', 'linear'):
        model = load_mymodel(cli(neuron='pop_flif', mode='full', soma='none', head_mode=head))
        K, Dm = 4, model.embedding.emb_linear.out_features
        with torch.no_grad():
            out = model(x)
        widths = (model.recall_head.in_features, getattr(model.head_compress, 'in_features', None), model.head.in_features)
        want = (Dm * K, Dm * K if head == 'flatten' else None,
                model.head.in_features if head == 'flatten' else Dm * K * model.num_patches)
        ok4[head] = widths == want and tuple(out.shape) == (x.shape[0], 96, x.shape[2]) and bool(torch.isfinite(out).all())
    gate('W4', all(ok4.values()), f"--soma none: heads read {Dm}*{K} spike trains, forward finite: {ok4}")

    # W5 -------------------------------------------------------------------------------------
    accepted = [dict(neuron=n, soma=s, **kw) for n, s, kw in combos] + [
        dict(neuron='flif', mode='full', fde_method='gl', fde_memory=10, **one),
        dict(neuron='pop_flif', mode='full', **one),
        dict(neuron='pop_flif', soma='lif', mode='full', readout='analog'),
        dict(neuron='flif', soma='lif', mode='full', readout='drive', **one),
        dict(neuron='sel_pop_flif', mode='hard', hard_q=.5, select_part='input'),
        dict(neuron='sel_pop_flif', mode='hard', hard_q=.5, hard_stat='random')]
    refused = [dict(neuron='lif', mode='full', **one),                                   # alpha .7
               dict(neuron='lif', soma='lif', alpha=1., mode='full', **one),             # LIF+soma 없음
               dict(neuron='flif', mode='full'),                                         # tau 4개
               dict(neuron='pop_flif', mode='hard'),
               dict(neuron='sel_pop_flif', mode='full', hard_q=.5),
               dict(neuron='sel_pop_flif', mode='hard', hard_q=.5, hard_stat='oracle'),
               dict(neuron='sel_flif', mode='hard', fde_method='gl', **one),
               dict(neuron='flif', mode='full', readout='analog', **one),               # soma 없음
               dict(neuron='pop_flif', mode='full', soma='none', readout='drive'),
               dict(neuron='pop_flif', mode='full', cap=False),
               dict(neuron='pop_flif', mode='full', hard_q=.5),
               dict(neuron='flif', mode='full', model='GRU', **one),
               dict(neuron='flif', mode='full', fde_step=0., **one),
               dict(neuron='flif', mode='full', heterogeneous=False, **one)]
    take, names, tags = [], set(), set()
    for kw in accepted:
        try:
            ns = cli(**kw)
            check_neuron_args(ns)
            take.append(True)
            names.add(f"{kw.get('readout', 'spike')}_{neuron_variant(ns)}")
            if kw.get('readout', 'spike') == 'spike':
                tags.add(calibration_tag(ns))
        except ValueError:
            take.append(False)
    refuse = []
    for kw in refused:
        try:
            check_neuron_args(cli(**kw))
            refuse.append(False)
        except ValueError:
            refuse.append(True)
    n_spike = len([kw for kw in accepted if kw.get('readout', 'spike') == 'spike'])
    own = len(tags) == n_spike and calibration_tag(cli()) == ''
    gate('W5', all(take) and all(refuse) and len(names) == len(accepted) and own,
         f"accepted {sum(take)}/{len(take)}, refused {sum(refuse)}/{len(refuse)}, distinct names {len(names)}/{len(accepted)}, "
         f"one calibration tag per configuration and none for legacy {own} "
         f"(e.g. {calibration_tag(cli(neuron='flif', soma='lif', mode='full', **one))!r})")

    # W6 -------------------------------------------------------------------------------------
    keys = list(neuron_kwargs(cli()).keys())
    gate('W6', keys == LEGACY_KEYS, f"legacy neuron_kwargs keys unchanged: {keys == LEGACY_KEYS}")


def reproduce(run_dir):
    """W7: rebuild a legacy run from its config.pt and evaluate its best checkpoint on validation."""
    import json
    from config import Config, set_random_seed
    from data_provider.data_factory import data_provider
    from test import evaluate
    run_dir = str(Path(run_dir).resolve())
    args = parse_defaults()
    args.cpu, args.num_device = False, 0
    config = Config()
    config.load_args(run_dir, args)
    with open(config.result_path) as handle:
        recorded = json.load(handle)['train']['best_val_loss']
    set_random_seed(config.seed)
    _, val_loader = data_provider(config, flag='val')
    model = load_mymodel(config, train=False)
    mse = evaluate(model, val_loader, config)[0]['mse']
    gate('W7', abs(mse - recorded) < 1e-9 and getattr(config, 'neuron', 'legacy') == 'legacy',
         f"{Path(run_dir).name} ({config.dataset}): validation mse {mse:.12f} vs recorded {recorded:.12f} "
         f"(diff {abs(mse - recorded):.1e}) on {config.device}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spikede', default=None, help='spikeDE clone at fcd743b, for S1')
    ap.add_argument('runs', nargs='*', help='legacy run directories for W7')
    a = ap.parse_args()
    classes(a.spikede)
    with torch.enable_grad():
        gradients()
        wiring()
    for directory in a.runs:
        reproduce(directory)
    if not a.runs:
        print('[neurons] W7    not run (no legacy run directory given)')
    print(f"[neurons] {sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
