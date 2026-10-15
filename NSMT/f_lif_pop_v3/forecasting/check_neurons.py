#!/usr/bin/env python3
"""Gates for the neuron family in neurons.py (CPU; torch 1.12 is enough).

The parity with the ORIGINAL spikeDE code needs torch >= 2 and lives in
analysis/spikede_parity.py. Here:

  N1   FLIFNeuron(alpha=1, pred) == LIFNeuron                         (float64, rel 1e-12, spikes)
  N2   FLIFNeuron(alpha=1, gl)   == LIFNeuron
  N3   SelectiveFLIF(q=1) == FLIFNeuron                               bitwise, float32 and float64
  N4   PopulationFLIF(K=1, soma='none') == FLIFNeuron                 bitwise
  N5   SelectivePopulationFLIF(q=1) == PopulationFLIF                 bitwise, soma 'lif' and 'none'
  N6   SelectivePopulationFLIF(K=1, soma='none') == SelectiveFLIF     bitwise (q=0.5)
  N7   PopulationFLIF(K=4, soma='lif') == layers.PopulationNeuron, mode 'full'       bitwise
  N8   SelectivePopulationFLIF(pearson, shared, q=.5) == PopulationNeuron, mode 'hard'  bitwise
  N9   heterogeneous=False == PopulationNeuron(heterogeneous=False)   bitwise
  N10  recent / random controls == PopulationNeuron mode 'hard' with the same seed      bitwise
  N11  part='input': q=1 is FLIF bitwise; q<1 differs from part='all'; branch states equal the
       2Q (A) diagnostic's leak-kept recursion (analysis/selection_stability.run)
  N12  memory >= T is the full history (pred and gl); memory < T changes the result
  N13  gradients reach the input current and the soma weights, all finite, every class
  N14  shapes, out_features, keyword filtering, refused configurations
  N15  select_mask keeps exactly max(1, round(qJ)) slots; 'recent' keeps the newest
  N16  SelectivePopulationFLIF is composed of a PopulationFLIF and a SelectiveFLIF instance

and the wiring into the pipeline (config.py -> model.load_mymodel -> ours.myModel):

  W1   --neuron pop_flif == --neuron legacy --mode full: forward bitwise (spike, analog and drive
       readouts, flatten and linear heads), shared initial values copied
  W2   --neuron sel_pop_flif --hard_q .5 == --neuron legacy --mode hard --hard_q .5, forward bitwise
  W3   three AdamW steps with gradient clipping: every shared parameter bitwise equal (W1 pair, W2 pair)
  W4   --soma none: the heads read D*K spike trains; forward runs
  W5   config.check_neuron_args accepts consistent settings and refuses the ones a class would ignore;
       variant names are distinct; every configuration has its own calibration tag (legacy: none)
  W6   the legacy neuron_kwargs is the dict it was before the neurons.py wiring
  W7   (run directories given on the command line) a saved legacy checkpoint reproduces the
       best validation MSE its run recorded

Usage: python check_neurons.py [legacy run directory ...]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import layers                                                    # noqa: E402
import neurons as N                                              # noqa: E402
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


def current(dtype, seed=1, scale=3.):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(T, B, D, generator=g, dtype=dtype)


def run(neuron, x, **kw):
    out, aux = neuron(x, return_aux=True, **kw)
    return out, aux['state'], aux


def same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def rel(a, b):
    return float((a - b).abs().max() / max(1., float(b.abs().max())))


def build(name, dtype=torch.float64, **kw):
    kw.setdefault('max_length', T)
    return N.build_neuron(name, D, dtype=dtype, **kw)


def legacy(dtype=torch.float64, **kw):
    kw.setdefault('max_length', T)
    return layers.PopulationNeuron(D, dtype=dtype, **kw)


def cli(**kw):
    """Parser defaults with some settings changed, as Config would hold them (no run directory)."""
    ns = parse_defaults()
    for key, value in kw.items():
        setattr(ns, key, value)
    ns.task, ns.revin, ns.input_scale, ns.cpu = kw.get('task', 'ett'), kw.get('revin', True), 6., True
    ns.num_patches = ns.seq_len // ns.patch_size
    ns.device = torch.device('cpu')

    return ns


def model_pair(old_kw, new_kw):
    """The legacy model and a neurons.py model built by model.load_mymodel, the second given the
    first's initial values; returns them with their --mode."""
    old, new = (load_mymodel(cli(**kw)) for kw in (old_kw, new_kw))
    state, mine = old.state_dict(), new.state_dict()
    # SelectivePopulationFLIF 는 .population / .selection 인스턴스를 가진다: 그 키는 legacy 의 같은 이름으로 맞춘다
    legacy = lambda k: k.replace('embedding.neuron.population.', 'embedding.neuron.') \
                        .replace('embedding.neuron.selection.', 'embedding.neuron.')
    if [k for k in mine if legacy(k) not in state]:
        raise AssertionError(f'parameters without a legacy counterpart: {[k for k in mine if legacy(k) not in state]}')
    for k in mine:                                                   # 버퍼(tau, b)는 복사하지 않고 같아야 한다
        if k.endswith(('.tau', '.b')) and not torch.equal(state[legacy(k)], mine[k]):
            raise AssertionError(f'{k} differs before any copy')
    new.load_state_dict({k: state[legacy(k)] for k in mine}, strict=True)

    return (old, old_kw.get('mode', 'sparse')), (new, new_kw.get('mode', 'sparse'))


def ett_batch(seed=3, B=4, C=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.cumsum(torch.randn(B, 336, C, generator=g), dim=1)    # 수준이 움직이는 시계열
    y = x[:, -1:, :] + torch.randn(B, 96, C, generator=g)

    return x, y


@torch.no_grad()
def main():
    x = current(torch.float64)

    # N1, N2 ---------------------------------------------------------------------------------
    for gate_name, method in (('N1', 'pred'), ('N2', 'gl')):
        lif = run(build('lif', tau=2.), x)
        frac = run(build('flif', tau=2., alpha=1., method=method), x)
        gate(gate_name, rel(frac[1], lif[1]) < 1e-12 and torch.equal(frac[0], lif[0]),
             f"f-LIF(alpha=1, {method}) vs LIF: rel state err {rel(frac[1], lif[1]):.1e}, "
             f"spikes equal {torch.equal(frac[0], lif[0])} ({int(lif[0].sum())} spikes)")

    # N3, N4 ---------------------------------------------------------------------------------
    ok3 = []
    for dtype in (torch.float32, torch.float64):
        xd = current(dtype)
        ok3.append(same(run(build('sel_flif', dtype, tau=2., q=1.), xd), run(build('flif', dtype, tau=2.), xd)))
    gate('N3', all(ok3), f"SelectiveFLIF(q=1) == FLIF bitwise in float32/float64: {ok3}")
    a = run(build('pop_flif', tau=(2.,), soma='none'), x)
    b = run(build('flif', tau=2.), x)
    gate('N4', same(a, b), f"PopulationFLIF(K=1, soma='none') == FLIF bitwise: {same(a, b)}")

    # N5, N6 ---------------------------------------------------------------------------------
    ok5 = {soma: same(run(build('sel_pop_flif', soma=soma, q=1.), x), run(build('pop_flif', soma=soma), x))
           for soma in ('lif', 'none')}
    gate('N5', all(ok5.values()), f"SelectivePopulationFLIF(q=1) == PopulationFLIF bitwise: {ok5}")
    a = run(build('sel_pop_flif', tau=(2.,), soma='none', q=.5), x)
    b = run(build('sel_flif', tau=2., q=.5), x)
    gate('N6', same(a, b), f"SelectivePopulationFLIF(K=1, soma='none', q=.5) == SelectiveFLIF bitwise: {same(a, b)}")

    # N7, N8, N9 -----------------------------------------------------------------------------
    for gate_name, kw_new, kw_old, mode in (
            ('N7', dict(), dict(), 'full'),
            ('N8', dict(q=.5), dict(hard_q=.5), 'hard'),
            ('N9', dict(heterogeneous=False), dict(heterogeneous=False), 'full')):
        ok = []
        for dtype in (torch.float32, torch.float64):
            xd = current(dtype)
            name = 'sel_pop_flif' if 'q' in kw_new else 'pop_flif'
            new = run(build(name, dtype, **kw_new), xd)
            old = run(legacy(dtype, **kw_old), xd, mode=mode)
            coeff = all((cn is None and co is None) or torch.equal(cn, co)
                        for cn, co in zip(new[2]['coeff'], old[2]['coeff']))
            ok.append(same(new, old) and coeff and torch.equal(new[2]['voltage'], old[2]['voltage']))
        gate(gate_name, all(ok), f"== layers.PopulationNeuron(mode='{mode}', {kw_old}) bitwise "
                                 f"(spikes, states, coefficients, soma voltage) float32/float64: {ok}")

    # N10 ------------------------------------------------------------------------------------
    ok10 = {}
    for stat in ('recent', 'random'):
        new = run(build('sel_pop_flif', q=.5, stat=stat, seed=11), x)
        old = run(legacy(hard_q=.5, hard_stat=stat, hard_seed=11), x, mode='hard')
        ok10[stat] = same(new, old)
    gate('N10', all(ok10.values()), f"content-free controls == legacy mode 'hard' with seed 11: {ok10}")

    # N11 ------------------------------------------------------------------------------------
    import selection_stability as S
    one = same(run(build('sel_flif', tau=2., q=1., part='input'), x), run(build('flif', tau=2.), x))
    full_sel = run(build('sel_pop_flif', q=.5, part='all'), x)
    in_sel = run(build('sel_pop_flif', q=.5, part='input'), x)
    differs = not torch.equal(full_sel[1], in_sel[1])
    neuron = build('sel_pop_flif', q=.5, part='input')
    diag, _, _ = S.run(x, neuron.b.double(), neuron.tau.double(), 'pearson', .5, leak_keep=True)
    err = rel(in_sel[1], diag)
    gate('N11', one and differs and err < 1e-12,
         f"part='input': q=1 == FLIF bitwise {one}; differs from part='all' {differs}; "
         f"branch states vs the 2Q (A) leak-kept recursion rel err {err:.1e}")

    # N12 ------------------------------------------------------------------------------------
    ok12 = {}
    for method in ('pred', 'gl'):
        full = run(build('flif', tau=2., method=method), x)
        long = run(build('flif', tau=2., method=method, memory=T), x)
        short = run(build('flif', tau=2., method=method, memory=5), x)
        ok12[method] = same(full, long) and not torch.equal(full[1], short[1])
    gate('N12', all(ok12.values()), f"memory >= T is the full history and memory=5 is not: {ok12}")

    # N14, N15 -------------------------------------------------------------------------------
    shapes = {}
    for name, kw in (('lif', dict(tau=2.)), ('flif', dict(tau=2.)), ('sel_flif', dict(tau=2.)),
                     ('pop_flif', dict()), ('pop_flif', dict(soma='none')), ('sel_pop_flif', dict()),
                     ('sel_pop_flif', dict(soma='none'))):
        n = build(name, **kw, alpha=.7, q=.5, unrelated_key=1)                # 모르는 키는 걸러진다
        out = n(x)
        shapes[f"{name}{'/none' if kw.get('soma') == 'none' else ''}"] = (out.shape[-1], n.out_features)
    shape_ok = all(o == f for o, f in shapes.values()) and shapes['pop_flif/none'][0] == 4 * D \
        and shapes['pop_flif'][0] == D
    refused = []
    for bad in (lambda: build('flif', tau=(2., 4.)), lambda: build('sel_flif', tau=2., method='gl'),
                lambda: build('lif', tau=0.), lambda: build('flif', tau=2.)(torch.zeros(T + 1, B, D)),
                lambda: build('pop_flif', soma='both'), lambda: build('flif', tau=2.)(x, analog='analog')):
        try:
            bad()
            refused.append(False)
        except ValueError:
            refused.append(True)
    gate('N14', shape_ok and all(refused), f"out_features {shapes}; refused configurations {refused}")
    J, ok15 = 41, []
    xi, hist = torch.randn(B, D, 5, dtype=torch.float64), torch.randn(B, D, J, 5, dtype=torch.float64)
    for q in (.1, .25, .5, .75):
        for stat in ('pearson', 'cosine', 'recent', 'random'):
            m, _ = N.select_mask(xi, hist, q, stat, 'shared', torch.Generator().manual_seed(0))
            ok15.append(bool((m.sum(-1) == max(1, int(round(q * J)))).all()))
    m, _ = N.select_mask(xi, hist, .25, 'recent')
    newest = bool((m[..., -10:] == 1).all() and (m[..., :-10] == 0).all())
    gate('N15', all(ok15) and newest, f"kept exactly round(qJ) for 16 cases: {all(ok15)}; recent keeps the newest 10/41: {newest}")

    # N13 (needs gradients) --------------------------------------------------------------------
    with torch.enable_grad():
        ok13 = {}
        for name, kw in (('lif', dict(tau=2.)), ('flif', dict(tau=2.)), ('sel_flif', dict(tau=2., q=.5)),
                         ('pop_flif', dict()), ('pop_flif', dict(soma='none')), ('sel_pop_flif', dict(q=.5)),
                         ('sel_pop_flif', dict(q=.5, part='input'))):
            n = build(name, **kw)
            xg = current(torch.float64).requires_grad_(True)
            out = n(xg)
            (out * torch.linspace(.5, 1.5, out.shape[-1], dtype=torch.float64)).sum().backward()
            grads = [xg.grad] + ([n.soma.weight.grad] if getattr(n, 'has_soma', False) else [])
            ok13[f"{name}{'/'+kw['part'] if 'part' in kw else ''}{'/none' if kw.get('soma') == 'none' else ''}"] = \
                all(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in grads)
        gate('N13', all(ok13.values()), f"finite, nonzero gradients to the input (and soma weights): {ok13}")

    # N16 ------------------------------------------------------------------------------------
    n = build('sel_pop_flif', q=.5)
    comp = (not isinstance(n, (N.SelectiveFLIF, N.PopulationFLIF)) and isinstance(n, N.FLIFNeuron)
            and isinstance(n.population, N.PopulationFLIF) and isinstance(n.selection, N.SelectiveFLIF)
            and n.soma is n.population.soma and torch.equal(n.tau, n.population.tau)
            and torch.equal(n.tau, n.selection.tau) and torch.equal(n.b, n.selection.b)
            and n.q == .5 and n.generator is n.selection.generator)
    homog = build('sel_pop_flif', q=.5, heterogeneous=False)
    comp = comp and torch.equal(homog.tau, homog.population.tau) and torch.equal(homog.tau, homog.selection.tau)
    members = build('sel_flif', tau=(2., 8.), q=.5)                  # K selective members, no soma
    out = members(x)
    comp = comp and out.shape[-1] == 2 * D and members.out_features == 2 * D
    gate('N16', comp, f"SelectivePopulationFLIF composes PopulationFLIF + SelectiveFLIF (no inheritance), shares "
                      f"tau/b/soma/generator with them: {comp}; SelectiveFLIF with 2 taus emits {out.shape[-1]} trains")

    with torch.enable_grad():                                        # W3가 학습 스텝을 밟는다
        wiring()
    for directory in sys.argv[1:]:
        reproduce(directory)
    if len(sys.argv) < 2:
        print('[neurons] W7    not run (no legacy run directory given)')

    print(f"[neurons] {sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    return 0 if all(RESULTS) else 1


PAIRS = {'W1': (dict(mode='full'), dict(neuron='pop_flif', mode='full')),
         'W2': (dict(mode='hard', hard_q=.5), dict(neuron='sel_pop_flif', mode='hard', hard_q=.5))}


def wiring():
    x, y = ett_batch()

    # W1, W2 ---------------------------------------------------------------------------------
    for name, (old_kw, new_kw) in PAIRS.items():
        ok, spikes = {}, 0
        readouts = ('spike', 'analog', 'drive') if name == 'W1' else ('spike',)
        for head in ('flatten', 'linear'):
            for readout in readouts:
                (old, om), (new, nm) = model_pair(dict(old_kw, head_mode=head, readout=readout),
                                                  dict(new_kw, head_mode=head, readout=readout))
                with torch.no_grad():
                    a, aux_a = old(x, mode=om, return_aux=True)
                    b, aux_b = new(x, mode=nm, return_aux=True)
                ok[f'{head}/{readout}'] = torch.equal(a, b) and torch.equal(aux_a['spikes'], aux_b['spikes']) \
                    and torch.equal(aux_a['state'], aux_b['state'])
                spikes = max(spikes, int(aux_a['spikes'].sum()))
        gate(name, all(ok.values()), f"{new_kw} == legacy {old_kw}, forward bitwise: {ok} "
                                     f"(up to {spikes} spikes in the batch)")

    # W3 -------------------------------------------------------------------------------------
    ok3 = {}
    for name, (old_kw, new_kw) in PAIRS.items():
        (old, om), (new, nm) = model_pair(old_kw, new_kw)
        before = {k: p.detach().clone() for k, p in new.named_parameters()}
        for model, mode in ((old, om), (new, nm)):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=.01)
            for _ in range(3):
                loss = torch.nn.functional.mse_loss(model(x, mode=mode), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
        theirs = dict(old.named_parameters())
        legacy = lambda k: k.replace('embedding.neuron.population.', 'embedding.neuron.')
        moved = [k for k, p in new.named_parameters() if not torch.equal(p, before[k])]
        ok3[name] = (all(torch.equal(p, theirs[legacy(k)]) for k, p in new.named_parameters()) and len(moved) > 0,
                     f'{len(moved)}/{len(before)} moved')
    gate('W3', all(v[0] for v in ok3.values()), f"after 3 AdamW steps (lr 1e-2, clip 1) every shared parameter "
                                                f"bitwise equal: {ok3}")

    # W4 -------------------------------------------------------------------------------------
    ok4 = {}
    for head in ('flatten', 'linear'):
        model = load_mymodel(cli(neuron='pop_flif', mode='full', soma='none', head_mode=head))
        K, Dm = 4, model.embedding.emb_linear.out_features
        with torch.no_grad():
            out = model(x)
        widths = (model.recall_head.in_features, getattr(model.head_compress, 'in_features', None),
                  model.head.in_features)
        want = (Dm * K, Dm * K if head == 'flatten' else None,
                model.head.in_features if head == 'flatten' else Dm * K * model.num_patches)
        ok4[head] = widths == want and tuple(out.shape) == (x.shape[0], 96, x.shape[2]) \
            and bool(torch.isfinite(out).all())
    gate('W4', all(ok4.values()), f"--soma none: heads read {Dm}*{K} spike trains, forward finite: {ok4}")

    # W5 -------------------------------------------------------------------------------------
    accepted = [dict(neuron='lif', mode='full', alpha=1., tau=[2.], num_population=1),
                dict(neuron='flif', mode='full', tau=[8.], num_population=1),
                dict(neuron='flif', mode='full', tau=[8.], num_population=1, fde_method='gl', fde_memory=10),
                dict(neuron='sel_flif', mode='hard', tau=[8.], num_population=1, hard_q=.5),
                dict(neuron='pop_flif', mode='full'),
                dict(neuron='pop_flif', mode='full', soma='none'),
                dict(neuron='pop_flif', mode='full', tau=[8.], num_population=1),
                dict(neuron='pop_flif', mode='full', readout='analog'),
                dict(neuron='sel_pop_flif', mode='hard', hard_q=.5),
                dict(neuron='sel_pop_flif', mode='hard', hard_q=.5, select_part='input'),
                dict(neuron='sel_pop_flif', mode='hard', hard_q=.5, hard_stat='random'),
                dict(neuron='sel_pop_flif', mode='hard', hard_q=.5, soma='none')]
    refused = [dict(neuron='lif', mode='full', tau=[2.], num_population=1),                # alpha .7
               dict(neuron='flif', mode='full'),                                           # tau 4개
               dict(neuron='pop_flif', mode='hard'),
               dict(neuron='sel_pop_flif', mode='full', hard_q=.5),
               dict(neuron='sel_pop_flif', mode='hard', hard_q=.5, hard_stat='oracle'),
               dict(neuron='sel_flif', mode='hard', tau=[8.], num_population=1, fde_method='gl'),
               dict(neuron='flif', mode='full', tau=[8.], num_population=1, readout='analog'),
               dict(neuron='pop_flif', mode='full', soma='none', readout='drive'),
               dict(neuron='pop_flif', mode='full', cap=False),
               dict(neuron='pop_flif', mode='full', hard_q=.5),
               dict(neuron='flif', mode='full', tau=[8.], num_population=1, model='GRU'),
               dict(neuron='flif', mode='full', tau=[8.], num_population=1, fde_step=0.),
               dict(neuron='flif', mode='full', tau=[8.], num_population=1, heterogeneous=False)]
    take, names = [], set()
    for kw in accepted:
        try:
            check_neuron_args(cli(**kw))
            take.append(True)
            names.add(f"{kw.get('readout', 'spike')}_{neuron_variant(cli(**kw))}")   # run_id에 들어가는 두 조각
        except ValueError:
            take.append(False)
    refuse = []
    for kw in refused:
        try:
            check_neuron_args(cli(**kw))
            refuse.append(False)
        except ValueError:
            refuse.append(True)
    tags = {calibration_tag(cli(**kw)) for kw in accepted if kw.get('readout', 'spike') == 'spike'}
    own = len(tags) == len([kw for kw in accepted if kw.get('readout', 'spike') == 'spike']) \
        and calibration_tag(cli()) == ''
    gate('W5', all(take) and all(refuse) and len(names) == len(accepted) and own,
         f"accepted {sum(take)}/{len(take)}, refused {sum(refuse)}/{len(refuse)}, distinct names "
         f"{len(names)}/{len(accepted)}, one calibration tag per configuration and none for legacy {own} "
         f"(e.g. {calibration_tag(cli(neuron='sel_pop_flif', mode='hard', hard_q=.5))!r})")

    # W6 -------------------------------------------------------------------------------------
    keys = list(neuron_kwargs(cli()).keys())
    gate('W6', keys == LEGACY_KEYS, f"legacy neuron_kwargs keys unchanged: {keys == LEGACY_KEYS}")


def reproduce(run):
    """W7: rebuild a legacy run from its config.pt and evaluate its best checkpoint on validation."""
    import json
    from config import Config, set_random_seed
    from data_provider.data_factory import data_provider
    from test import evaluate
    run = str(Path(run).resolve())
    args = parse_defaults()
    args.cpu, args.num_device = False, 0
    config = Config()
    config.load_args(run, args)
    with open(config.result_path) as handle:
        recorded = json.load(handle)['train']['best_val_loss']
    set_random_seed(config.seed)
    _, val_loader = data_provider(config, flag='val')
    model = load_mymodel(config, train=False)
    mse = evaluate(model, val_loader, config)[0]['mse']
    gate('W7', abs(mse - recorded) < 1e-9 and getattr(config, 'neuron', 'legacy') == 'legacy',
         f"{Path(run).name} ({config.dataset}): validation mse {mse:.12f} vs recorded {recorded:.12f} "
         f"(diff {abs(mse - recorded):.1e}) on {config.device}")


if __name__ == '__main__':
    raise SystemExit(main())
