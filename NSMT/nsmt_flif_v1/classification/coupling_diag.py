"""MODI 2026-11-18 (coupling attention test, no training, descriptive): what the coupling attention of a trained ETT checkpoint does,
for the current score (dot = q.k * 64^-0.5 + learned relative bias) and the XNOR + Log-PE score (--attn_score xnor_log).
Stage 0 of the test reads the current checkpoints (arms 'now': the msn base pop LIF / pop f-LIF, suite flif-v1-ett-ms, and
Spikformer-style spk_ssa, suite flif-v1-gen-ett; input 96); after training the same script reads arms 'new' (suite flif-v1-ett-xnor).
Frozen best-validation weights (model_state/checkpoint.pth), ETT test split, hooks only. Per checkpoint:
  intact    : must reproduce the run's logged test MSE / MAE / n (|diff| <= 2e-6) or the script stops
  neo0      : (base only) the Neocortex output replaced by zeros -> every query is the fixed q0 of a silent Neocortex token
  nopos     : the position term removed from the attention-neuron input (dot: learned bias; xnor: Log-PE)
  nocontent : the content term removed (dot: q.k; xnor: the matching-channel count); only the position term is left
  (test-time interventions: they measure reliance of a model trained with the term, read them as differences between arms)
Diagnostics on the intact pass (sums over the test split):
  terms     : the attention-neuron input split as  fixed + varying + position  (mean |.| per element; the shares are of the three
              means, as in neo_path_probe.py 11-08). base: fixed = the content with q0 (silent Neocortex), varying = what the Neocortex
              changes. spk_ssa: fixed = the content with q = 0 (dot: 0; xnor: d - |k_j|), varying = what the query changes.
              The split must add up to the real input (max |diff| <= 1e-5 * max(1, max |input|)) or the script stops.
  rates     : q, k, v, attention neuron (a), attention-branch output y, MLP output, Hippocampus spikes entering the block,
              Neocortex output (base); gate: share of the Hippocampus spikes the attention gate x (1 - y) removes, MLP gate, survive
  scores    : histogram of the integer content score per (query, key) pair (dot: q.k in 0..d; xnor: matching channels in 0..d),
              learned bias |b| mean / max over the table (dot), sigmoid(s) (xnor)
  ops       : attention core operations per test window (all channel rows, T_h steps, heads), none of which syops counts:
              score adds (dot: coincident q = k = 1 spikes; xnor: matching channels), position adds (pairs with a non-zero position
              term), score x v adds (pairs with a non-zero score times v spikes); the scale is folded into the attention-neuron
              threshold (hard reset to 0, so input scale and threshold are interchangeable) and costs 0.
              E_core = 0.9 pJ x all adds (the syops AC constant); E_core_int = 0.03 pJ x (score + position adds) + 0.9 pJ x score x v adds
              (8-bit integer adds for the popcount / integer parts; the dot bias is real, so its adds stay 0.9 pJ in both)
  batch     : the first test batch evaluated at once, row by row (first 8 windows) and shuffled; max |diff| must be <= 1e-6 (else the
              run is flagged batch_ok False and its result is not used)
usage (classification folder): python coupling_diag.py --arms now|new --device cuda:0 --out_json F --out_md F [--only ETTh2_96] [--seeds 42]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.getcwd()
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                    # noqa: E402
from load_model import LOAD_MODEL                                    # noqa: E402
from forecast_wrap import Forecaster                                 # noqa: E402

FC = os.path.join(HERE, '..', 'forecasting')
ARMS = {'now': (('base_lif', 'flif-v1-ett-ms', 'nsmt4q_pop_lif_msn'), ('base_flif', 'flif-v1-ett-ms', 'nsmt4q_pop_flif_msn'),
                ('spk', 'flif-v1-gen-ett', 'spk_ssa')),
        'new': (('base_lif', 'flif-v1-ett-xnor', 'nsmt4q_pop_lif_msn_xl'), ('base_flif', 'flif-v1-ett-xnor', 'nsmt4q_pop_flif_msn_xl'),
                ('spk', 'flif-v1-ett-xnor', 'spk_ssa_xl')),
        # MODI 2026-11-19 (aux loss test, report only): the msn base trained with NSMT's reconstruction loss (spikingjelly hippocampus)
        'aux': (('base_lif', 'flif-v1-ett-aux', 'nsmt4q_pop_lif_msn_aux'), ('base_flif', 'flif-v1-ett-aux', 'nsmt4q_pop_flif_msn_aux')),
        # MODI 2026-11-19 (full-signal aux test, report only): the same with the raw patches as the reconstruction target
        'auxf': (('base_lif', 'flif-v1-ett-auxf', 'nsmt4q_pop_lif_msn_auxf'), ('base_flif', 'flif-v1-ett-auxf', 'nsmt4q_pop_flif_msn_auxf'))}
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SEEDS = (42, 7, 2026)
AC, AC_INT = 0.9e-6, 0.03e-6                                         # µJ per add


def refs(arms):
    out = {}
    for _, suite, cond in arms:
        for ln in open(os.path.join(FC, 'results', suite, 'raw.txt')):
            if not ln.startswith('RESULT'):
                continue
            f = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            if f['cond'] != cond or (f['ds'], int(f['pl'])) not in CELLS or int(f.get('seq_len', 96)) != 96:
                continue
            k = (cond, f['ds'], int(f['pl']), int(f['seed']))
            if k in out:
                raise RuntimeError(f'duplicate {k}')
            out[k] = f
    return out


def fixed_q(ssa, T, C, ref):
    """q0 [T, C] of a silent Neocortex token: q_lif(q_bn(0)) by hand (eval BN gives every row and token the same current)."""
    if ssa.q_linear.bias is not None or ssa.q_bn.training:
        raise RuntimeError('expected a bias-free q_linear and BN in eval mode')
    I = ssa.q_bn(ref.new_zeros(1, C, 1))[0, :, 0]
    lif = ssa.q_lif
    th, vr, r = float(lif.v_threshold), float(lif.v_reset), 1. / float(lif.tau)
    v, out = torch.full_like(I, vr), []
    for _ in range(T):
        H = v + r * (I - v + vr)
        out.append((H >= th).to(I.dtype))
        v = torch.where(H >= th, torch.full_like(H, vr), H)
    return torch.stack(out)


class Tap:
    def __init__(self, ssa, query):
        if ssa.attn_bn is not None or getattr(ssa, 'attn_window', 0) or ssa.kv_cat:
            raise RuntimeError('attn_bn / band / kv_cat not supported')
        self.ssa, self.query, self.mode = ssa, query, 'intact'
        self.cap, self.s, self.hist, self.q0 = {}, {}, None, None
        self.max_split = 0.
        self.hs = [getattr(ssa, f'{k}_lif').register_forward_hook(self._grab(k)) for k in 'qkv']
        self.hs.append(ssa.attn_lif.register_forward_pre_hook(self._pre))

    def _grab(self, k):
        def fn(mod, inp, out):
            self.cap[k] = out.detach()
        return fn

    def add(self, k, v):
        self.s[k] = self.s.get(k, 0.) + float(v)

    def _content(self, q, k, sc):
        ssa = self.ssa
        if ssa.score == 'xnor_log':
            d = q.shape[-1]
            return (d - q.sum(-1).unsqueeze(-1) - k.sum(-1).unsqueeze(-2) + 2 * (q @ k.transpose(-2, -1))) * sc
        return (q @ k.transpose(-2, -1)) * sc

    def _pre(self, mod, inp):
        ssa, a_in = self.ssa, inp[0].detach()
        T, B, N, C = a_in.shape
        h = ssa.num_heads
        split = lambda z: z.reshape(T, B, N, h, C // h).permute(0, 1, 3, 2, 4)
        q, k, v = (split(self.cap[n]) for n in 'qkv')
        if ssa.score == 'xnor_log':
            sc = torch.sigmoid(ssa.xl_scale.detach())
            P = (ssa.log_pe * sc)[None, None, None]                    # [1, 1, 1, N, N]
            raw = (q.shape[-1] - q.sum(-1).unsqueeze(-1) - k.sum(-1).unsqueeze(-2) + 2 * (q @ k.transpose(-2, -1)))   # matching channels
        else:
            sc = ssa.scale
            P = ssa.relative_bias_table.detach()[ssa.rel_pos_idx].permute(2, 0, 1)[None, None] if ssa.pe else torch.zeros(1, 1, h, N, N, device=a_in.device)
            raw = q @ k.transpose(-2, -1)                              # coincident spikes
        S = self._content(q, k, sc)
        merge = lambda z: z.transpose(2, 3).reshape(T, B, N, C)
        if self.mode == 'nopos':
            return (merge(S @ v),)
        if self.mode == 'nocontent':
            return (merge(P.expand_as(S) @ v),)
        if self.mode != 'intact':
            return None
        whole = merge((S + P) @ v)
        self.max_split = max(self.max_split, float((whole - a_in).abs().max()) / max(1., float(a_in.abs().max())))
        if self.query:
            if self.q0 is None:
                self.q0 = fixed_q(ssa, T, C, a_in)
            q_ref = split(self.q0[:, None, None, :].expand(T, B, N, C))
        else:
            q_ref = torch.zeros_like(q)
        S0 = self._content(q_ref, k, sc)
        t0, td, tp = S0 @ v, (S - S0) @ v, P.expand_as(S) @ v
        self.add('term_fixed', t0.abs().sum()); self.add('term_var', td.abs().sum()); self.add('term_pos', tp.abs().sum())
        self.add('term_n', t0.numel())
        d = q.shape[-1]
        hb = torch.bincount(raw.round().long().flatten(), minlength=d + 1).double().cpu()
        self.hist = hb if self.hist is None else self.hist + hb
        tot = S + P
        self.add('ops_score', raw.sum()); self.add('ops_pos', (P != 0).expand_as(S).sum())
        self.add('ops_sv', ((tot != 0).to(v.dtype) @ v).sum())
        for n in 'qkv':
            self.add(n + '_sum', self.cap[n].sum()); self.add(n + '_n', self.cap[n].numel())
        return None

    def remove(self):
        for hk in self.hs:
            hk.remove()


def load(run, device):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**dict(cfg, device=device))
    model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, args.pred_len, bias=args.bias)
    model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu'))
    model = model.to(device).eval()
    model.train_mode = 'testing'
    functional.reset_net(model)
    return args, model, model.inner


def run_one(arm, cond, ds, pl, seed, ref, device):
    args, model, inner = load(ref['dir'], device)
    if len(inner.data_block) != 1 or inner.T_h != 4:
        raise RuntimeError(f'{cond}: not the expected model')
    query = cond.startswith('nsmt4q_')
    want_score = 'xnor_log' if cond.endswith('_xl') else 'dot'
    blk = inner.data_block[0]
    if blk.ssa.score != want_score or (query and (inner.hippo_xattn_mode != 'query' or getattr(inner, 'neo_res', 'none') != 'ms')) \
            or (not query and not inner.hippo_ssa):
        raise RuntimeError(f'{cond}: unexpected configuration')
    sys.path.append(FC)
    from train_forecast import data_args
    from data_provider.data_factory import data_provider
    _, dl = data_provider(data_args(args, args), 'test')
    tap = Tap(blk.ssa, query)
    acc = {}
    st = {'mode': 'intact'}

    def add(k, v):
        acc[k] = acc.get(k, 0.) + float(v)

    def neo_hook(m, i, o):
        if st['mode'] == 'intact':
            add('neo_sum', o.sum()); add('neo_n', o.numel())
            return None
        return torch.zeros_like(o) if st['mode'] == 'neo0' else None

    def ssa_hook(m, i, o):
        if st['mode'] == 'intact':
            x = i[0]
            add('x_sum', x.sum()); add('x_n', x.numel()); add('y_sum', o.sum()); add('xy_sum', (x * o).sum())

    def mlp_hook(m, i, o):
        if st['mode'] == 'intact':
            x = i[0]
            add('xm_in_sum', x.sum()); add('m_sum', o.sum()); add('xm_sum', (x * o).sum())

    def blk_hook(m, i, o):
        if st['mode'] == 'intact':
            add('fin_sum', o.sum())

    def a_hook(m, i, o):
        if st['mode'] == 'intact':
            add('a_sum', o.sum()); add('a_n', o.numel())

    hs = [blk.ssa.register_forward_hook(ssa_hook), blk.mlp.register_forward_hook(mlp_hook), blk.register_forward_hook(blk_hook),
          blk.ssa.attn_lif.register_forward_hook(a_hook)]
    if query:
        hs.append(inner.time_block.register_forward_hook(neo_hook))
    res = dict(arm=arm, cond=cond, ds=ds, pl=pl, seed=seed, score=want_score)
    modes = ('intact', 'neo0', 'nopos', 'nocontent') if query else ('intact', 'nopos', 'nocontent')
    n_win = 0
    with torch.no_grad():
        for mode in modes:
            st['mode'] = tap.mode = mode
            se = ae = 0.; cnt = 0
            for x, y, _, _ in dl:
                x = x.float().to(device); y = y.float().to(device)[:, -pl:, :]
                d = (model(x) - y).double()
                functional.reset_net(model)
                se += float((d ** 2).sum()); ae += float(d.abs().sum()); cnt += d.numel()
                if mode == 'intact':
                    n_win += x.shape[0]
            mse, mae = se / cnt, ae / cnt
            if not (np.isfinite(mse) and np.isfinite(mae)):
                raise RuntimeError('non-finite metric')
            if mode == 'intact' and (abs(mse - float(ref['mse'])) > 2e-6 or abs(mae - float(ref['mae'])) > 2e-6 or cnt != int(ref['n'])):
                raise RuntimeError(f"reproduction {cond} {ds} {pl} {seed}: {mse:.6f} / {mae:.6f} / {cnt} vs {ref['mse']} / {ref['mae']} / {ref['n']}")
            res[mode] = dict(mse=mse, mae=mae, n=cnt)
        st['mode'] = tap.mode = 'off'
        xb = next(iter(dl))[0].float().to(device)
        full = model(xb); functional.reset_net(model)
        rows = []
        for i in range(min(8, xb.shape[0])):
            rows.append(model(xb[i:i + 1])); functional.reset_net(model)
        perm = torch.randperm(xb.shape[0], generator=torch.Generator().manual_seed(0)).to(device)
        pm = model(xb[perm]); functional.reset_net(model)
        bdiff = max(float((full[:len(rows)] - torch.cat(rows)).abs().max()), float((full[perm] - pm).abs().max()))
    tap.remove()
    for hk in hs:
        hk.remove()
    if tap.max_split > 1e-5:
        raise RuntimeError(f'attention split differs by {tap.max_split:.3g}')
    s, t = acc, tap.s
    if s.get('x_sum', 0.) <= 0 or (query and s.get('neo_sum', 0.) <= 0):
        raise RuntimeError('no Neocortex or Hippocampus spikes')
    res['batch'] = dict(max_diff=bdiff, ok=bdiff <= 1e-6)
    res['rate'] = dict(q=t['q_sum'] / t['q_n'], k=t['k_sum'] / t['k_n'], v=t['v_sum'] / t['v_n'], a=s['a_sum'] / s['a_n'],
                       y=s['y_sum'] / s['x_n'], m=s['m_sum'] / s['x_n'], hip_in=s['x_sum'] / s['x_n'],
                       neo=(s['neo_sum'] / s['neo_n']) if query else None)
    res['gate'] = dict(attn_removed=s['xy_sum'] / s['x_sum'], mlp_removed=s['xm_sum'] / s['xm_in_sum'], survive=s['fin_sum'] / s['x_sum'])
    tf, tv, tp = (t[k] / t['term_n'] for k in ('term_fixed', 'term_var', 'term_pos'))
    res['terms'] = dict(fixed=tf, var=tv, pos=tp, var_share=tv / (tf + tv + tp), pos_share=tp / (tf + tv + tp))
    h = tap.hist
    res['score_hist'] = (h / h.sum()).tolist()
    ssa = blk.ssa
    if ssa.score == 'xnor_log':
        res['scale'] = float(torch.sigmoid(ssa.xl_scale.detach()))
        res['bias_abs'] = None
    else:
        b = ssa.relative_bias_table.detach().abs() if ssa.pe else None
        res['scale'] = float(ssa.scale)
        res['bias_abs'] = dict(mean=float(b.mean()), max=float(b.max())) if b is not None else None
    ops = {k: t['ops_' + k] / n_win for k in ('score', 'pos', 'sv')}
    pos_cost = AC if ssa.score == 'dot' else AC_INT
    ops['E_core_uJ'] = AC * (ops['score'] + ops['pos'] + ops['sv'])
    ops['E_core_int_uJ'] = AC_INT * ops['score'] + pos_cost * ops['pos'] + AC * ops['sv']
    res['ops'] = ops
    for k in modes[1:]:
        res[k]['d_mse_pct'] = 100. * (res[k]['mse'] / res['intact']['mse'] - 1.)
    del model, inner
    torch.cuda.empty_cache()
    return res


def m3(g, f):
    v = [f(r) for r in g]
    return float(np.mean(v)) if all(x is not None for x in v) else None


def report(rs, arms_key):
    L = [f'# coupling_diag: arms {arms_key} (means over seeds; n = seeds per cell)', '',
         '| arm | cell | n | test MSE | Neocortex / query varying share | position share | ΔMSE % Neocortex 0 | ΔMSE % position 0 | '
         'ΔMSE % content 0 | q rate | a rate | y rate | attention gate removes | E_core µJ / window | E_core_int | batch ok |', '|' + '---|' * 16]
    f2 = lambda v, s='{:+.2f}': '—' if v is None else s.format(v)
    for arm, _, cond in ARMS[arms_key]:
        for ds, pl in CELLS:
            g = [r for r in rs if r['cond'] == cond and r['ds'] == ds and r['pl'] == pl]
            if not g:
                continue
            L.append(f"| {arm} ({cond}) | {ds} H{pl} | {len(g)} | {m3(g, lambda r: r['intact']['mse']):.6f} | "
                     f"{m3(g, lambda r: r['terms']['var_share']):.3f} | {m3(g, lambda r: r['terms']['pos_share']):.3f} | "
                     f"{f2(m3(g, lambda r: r['neo0']['d_mse_pct'] if 'neo0' in r else None))} | {m3(g, lambda r: r['nopos']['d_mse_pct']):+.2f} | "
                     f"{m3(g, lambda r: r['nocontent']['d_mse_pct']):+.2f} | {m3(g, lambda r: r['rate']['q']):.4f} | {m3(g, lambda r: r['rate']['a']):.4f} | "
                     f"{m3(g, lambda r: r['rate']['y']):.4f} | {100 * m3(g, lambda r: r['gate']['attn_removed']):.2f} % | "
                     f"{m3(g, lambda r: r['ops']['E_core_uJ']):.3f} | {m3(g, lambda r: r['ops']['E_core_int_uJ']):.3f} | "
                     f"{all(r['batch']['ok'] for r in g)} |")
    L += ['', '## Attention terms (mean |value| per element of the attention-neuron input) and score shape', '',
          '| arm | cell | fixed | varying | position | learned bias |b| mean / max (dot) or sigmoid(s) (xnor) | content score histogram (share of pairs at 0, 1, …, d) |',
          '|' + '---|' * 7]
    for arm, _, cond in ARMS[arms_key]:
        for ds, pl in CELLS:
            g = [r for r in rs if r['cond'] == cond and r['ds'] == ds and r['pl'] == pl]
            if not g:
                continue
            sc = (f"{m3(g, lambda r: r['bias_abs']['mean']):.4f} / {m3(g, lambda r: r['bias_abs']['max']):.4f}" if g[0]['bias_abs']
                  else f"{m3(g, lambda r: r['scale']):.4f}")
            hist = np.mean([r['score_hist'] for r in g], 0)
            L.append(f"| {arm} | {ds} H{pl} | {m3(g, lambda r: r['terms']['fixed']):.4f} | {m3(g, lambda r: r['terms']['var']):.4f} | "
                     f"{m3(g, lambda r: r['terms']['pos']):.4f} | {sc} | {' / '.join(f'{x:.3f}' for x in hist)} |")
    L += ['', '## Per run', '', '| arm | cell | seed | test MSE | varying share | position share | ΔMSE % Neocortex 0 | ΔMSE % position 0 | ΔMSE % content 0 | '
          'y rate | gate removes | ops score / position / score x v per window | batch max diff |', '|' + '---|' * 13]
    for r in rs:
        L.append(f"| {r['arm']} | {r['ds']} H{r['pl']} | {r['seed']} | {r['intact']['mse']:.6f} | {r['terms']['var_share']:.3f} | "
                 f"{r['terms']['pos_share']:.3f} | {f2(r['neo0']['d_mse_pct'] if 'neo0' in r else None)} | {r['nopos']['d_mse_pct']:+.2f} | "
                 f"{r['nocontent']['d_mse_pct']:+.2f} | {r['rate']['y']:.4f} | {100 * r['gate']['attn_removed']:.2f} % | "
                 f"{r['ops']['score']:.0f} / {r['ops']['pos']:.0f} / {r['ops']['sv']:.0f} | {r['batch']['max_diff']:.1e} |")
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', choices=['now', 'new', 'aux', 'auxf'], required=True)   # MODI 2026-11-19: + aux, auxf
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--only', default=None)
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--suite_new', default='flif-v1-ett-xnor', help='suite of the new arms (a smoke suite for testing the script)')
    ap.add_argument('--suite_aux', default='flif-v1-ett-aux', help='MODI 2026-11-19: suite of the aux arms (a smoke suite for testing)')
    ap.add_argument('--suite_auxf', default='flif-v1-ett-auxf', help='MODI 2026-11-19: suite of the auxf arms (a smoke suite for testing)')
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--out_md', required=True)
    a = ap.parse_args()
    if a.arms == 'new':
        ARMS['new'] = tuple((arm, a.suite_new, cond) for arm, _, cond in ARMS['new'])
    if a.arms == 'aux':                                                # MODI 2026-11-19
        ARMS['aux'] = tuple((arm, a.suite_aux, cond) for arm, _, cond in ARMS['aux'])
    if a.arms == 'auxf':                                               # MODI 2026-11-19
        ARMS['auxf'] = tuple((arm, a.suite_auxf, cond) for arm, _, cond in ARMS['auxf'])
    arms = ARMS[a.arms]
    R = refs(arms)
    seeds = tuple(int(s) for s in a.seeds.split(',')) if a.seeds else SEEDS
    rs = []
    for arm, _, cond in arms:
        for ds, pl in CELLS:
            if a.only and a.only != f'{ds}_{pl}':
                continue
            for s in seeds:
                if (cond, ds, pl, s) not in R:
                    raise RuntimeError(f'missing {(cond, ds, pl, s)}')
                t0 = time.time()
                r = run_one(arm, cond, ds, pl, s, R[(cond, ds, pl, s)], a.device)
                r['time_s'] = round(time.time() - t0, 1)
                rs.append(r)
                print(json.dumps({k: r[k] for k in ('arm', 'ds', 'pl', 'seed', 'time_s')}), f"mse {r['intact']['mse']:.6f}",
                      f"var {r['terms']['var_share']:.3f} pos {r['terms']['pos_share']:.3f}", {k: round(r[k]['d_mse_pct'], 2) for k in r if isinstance(r[k], dict) and 'd_mse_pct' in r[k]},
                      f"batch {r['batch']['max_diff']:.1e}", flush=True)
    json.dump(rs, open(a.out_json, 'w'), indent=1)
    open(a.out_md, 'w').write(report(rs, a.arms) + '\n')
    print(report(rs, a.arms))


if __name__ == '__main__':
    main()
