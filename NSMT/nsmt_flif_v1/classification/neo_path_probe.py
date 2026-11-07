"""MODI 2026-11-08 (no training): what the Neocortex path of [query pop LIF NSMT T=4] (nsmt4q_pop_lif) does on its trained checkpoints
(user request 11-08 "explain the Neocortex path in detail"; docs/MODI_PROJECT_LOG.md 2026-11-08). Frozen weights, test split, descriptive only:
nothing here selects a model or changes a verdict.

Runs: ETTh2 / ETTm2 x H96 / H720 (flif-v1-ett-query, flif-v1-ett-query-m2; model_state/checkpoint.pth = the reported best-validation weights)
and MIT-BIH (flif-v1-mitbih-query, the reported default version; model_state/best+model.pt), seeds 42 / 7 / 2026.
Measured with forward hooks on the model's own forward:
  every Neocortex neuron (embedding, time_block layers): spike rate per token position, share of rows with no spike at a token, units that
      never fire, input current mean / std, soma weights w_k, each branch's share of the soma input (mean |w_k v_k|)
  the query: spike rate of q / k / v per T_h step and per token; q0 = the fixed query of a silent Neocortex token (q_linear(0) = 0 without
      bias, so q = q_lif(q_bn(0)), the same vector for every row and token); share of the q spikes that q0 already has;
      attention output (before attn_lif) split into three terms, mean absolute value per token:
      fixed q (q0 . k' scale) @ v, Neocortex-varying ((q - q0) . k' scale) @ v, relative bias b @ v
      (first pass of this script labelled the sum of the first two "Neocortex-dependent"; corrected before commit, see the log)
Fail-closed checks (any failure raises; nothing is written):
  c56a the run's reported test metric is reproduced (ETT MSE / MAE |diff| <= 2e-6, n; MIT-BIH macro-F1 |diff| <= 1e-6, n)
  c56b the written equations reproduce each neuron's spikes, float64 recomputation from its input current:
         v_k <- v_k + (I - v_k) / tau_k;  a = sum_k w_k v_k;  H = V + (a - V) / tau_s;  s = [H >= 1];  V = H - s
       differing decisions <= 1e-4 of all decisions (the model sums float32 increments, so ties at the threshold may round apart);
       every neuron fires at least once; spikes are 0 / 1
  c56c q_linear's input is exactly the last Neocortex layer's output, repeated over T_h (--neo_grad 1.0 leaves the values unchanged)
  c56d (q . k' scale + b) @ v from the hooked q / k / v equals attn_lif's input (max |diff| <= 1e-5)
  c56e the mean Neocortex rate equals the run's logged rho_neo (|diff| <= 2e-6)
  c56f at every (row, token) where the Neocortex output has no spike, q equals the hand-computed q0 exactly (spikingjelly cupy kernel,
       as head_probe.Charge: H_t = v + (I - v + v_reset) / tau, s = [H >= v_th], v = v_reset if s else H)
usage (classification folder): python neo_path_probe.py [--device cuda:0] [--only ETTh2_96] [--seeds 42] [--out_md FILE] [--out_json FILE]
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                    # noqa: E402
from head_probe import N_CLASSES, load, macro_f1                     # noqa: E402

FC = os.path.join(HERE, '..', 'forecasting', 'results')
CL = os.path.join(HERE, 'results')
COND = 'nsmt4q_pop_lif'
SEEDS = (42, 7, 2026)
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720), ('MITBIH', 0))


def name(cell):
    return 'MITBIH' if cell[0] == 'MITBIH' else f'{cell[0]}_{cell[1]}'


def runs_of(cell):
    """{seed: RESULT dict} of nsmt4q_pop_lif in one cell; exactly one line per seed."""
    ds, pl = cell
    raw = os.path.join(CL, 'flif-v1-mitbih-query', 'raw.txt') if ds == 'MITBIH' else \
        os.path.join(FC, {'ETTh2': 'flif-v1-ett-query', 'ETTm2': 'flif-v1-ett-query-m2'}[ds], 'raw.txt')
    out = {}
    for ln in open(raw):
        if not ln.startswith('RESULT'):
            continue
        d = dict(re.findall(r'(\w+)=(\S+)', ln))
        if d.get('cond') != COND or (ds != 'MITBIH' and (d.get('ds') != ds or int(d.get('pl', -1)) != pl)):
            continue
        s = int(d['seed'])
        if s in out:
            raise RuntimeError(f'two RESULT lines for {cell} seed {s}')
        out[s] = d
    if sorted(out) != sorted(SEEDS):
        raise RuntimeError(f'{cell}: seeds {sorted(out)}')
    return out


class PopTap:
    """Forward hook on a NeoNeuron holding an alpha-1 population with soma: statistics and the c56b recomputation."""
    def __init__(self, mod, tag):
        n = mod.neuron
        if type(n).__name__ != 'PopulationFLIF' or not n.has_soma or n.alpha != 1. or n.memory is not None or n.method != 'pred' or n.step != 1.:
            raise RuntimeError(f'{tag}: expected the population LIF (alpha 1, pred, full history, soma)')
        self.mod, self.tag, self.n = mod, tag, n
        self.calls, self.dec, self.diff, self.last = 0, 0, 0, None
        self.spk_t = self.sil_t = self.unit = self.share = None
        self.rows, self.i1, self.i2, self.ni = 0, 0., 0., 0
        mod.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        x, o = inp[0].detach(), out.detach()
        if mod.layout == 'DN':
            if x.shape[3] != 1:
                raise RuntimeError(f'{self.tag}: {tuple(x.shape)}')
            cur, s = x[..., 0], o[..., 0]
        else:
            if x.shape[2] != 1:
                raise RuntimeError(f'{self.tag}: {tuple(x.shape)}')
            cur, s = x[:, :, 0], o[:, :, 0]
        T, B, D = cur.shape
        if not bool(((s == 0.) | (s == 1.)).all()):
            raise RuntimeError(f'c56b {self.tag}: non-binary output')
        if self.spk_t is None:
            K = self.n.num_members
            self.spk_t, self.sil_t = torch.zeros(T).double(), torch.zeros(T).double()
            self.unit, self.share = torch.zeros(D).double(), torch.zeros(K).double()
        tau, w = self.n.tau.double(), self.n.soma.weight.detach().double()
        ts, th = float(self.n.soma.tau_s), float(self.n.soma.threshold)
        u, V = torch.zeros(B, D, self.n.num_members, dtype=torch.float64, device=x.device), torch.zeros(B, D, dtype=torch.float64, device=x.device)
        for t in range(T):
            I = cur[t].double()
            u = u + (I.unsqueeze(-1) - u) / tau
            wu = w * u
            H = V + (wu.sum(-1) - V) / ts
            sp = (H >= th).double()
            V = H - th * sp
            self.share += wu.abs().sum((0, 1)).cpu()
            self.diff += int((sp != s[t].double()).sum()); self.dec += sp.numel()
            self.spk_t[t] += float(s[t].sum()); self.sil_t[t] += float((s[t].sum(-1) == 0).sum())
            self.unit += s[t].sum(0).double().cpu()
            self.i1 += float(I.sum()); self.i2 += float(I.square().sum()); self.ni += I.numel()
        self.rows += B
        self.calls += 1
        self.last = s

    def result(self, D):
        if self.dec == 0 or self.spk_t.sum() <= 0:
            raise RuntimeError(f'c56b {self.tag}: no spikes')
        if self.diff > 1e-4 * self.dec:
            raise RuntimeError(f'c56b {self.tag}: {self.diff} / {self.dec} decisions differ')
        n = self.rows * D
        mi = self.i1 / self.ni
        w = self.n.soma.weight.detach().cpu().double()
        tau = self.n.tau.cpu().double()
        return dict(rate=float(self.spk_t.sum() / (n * len(self.spk_t))), rate_token=(self.spk_t / n).tolist(), silent_token=(self.sil_t / self.rows).tolist(),
                    dead_units=float((self.unit == 0).double().mean()), cur_mean=mi, cur_std=math.sqrt(max(self.i2 / self.ni - mi * mi, 0.)),
                    w_mean=w.mean(0).tolist(), w_sum_mean=float(w.sum(1).mean()), w_neg=float((w < 0).double().mean()),
                    w_abs_dev=float((w - .5).abs().mean()), w_std=float(w.std(unbiased=False)), w_q=[float(q) for q in torch.quantile(w.flatten(), torch.tensor([.01, .99], dtype=w.dtype))],
                    branch_share=(self.share / self.share.sum()).tolist(),
                    first_token_current_median=float((th_over(w, tau)).median()), steady_current_median=float((1. / w.sum(1)).median()),
                    diff_frac=self.diff / self.dec)


def th_over(w, tau, tau_s=2.):
    """Current a unit needs to fire at the very first token (v = I / tau_k, H = a / tau_s), threshold 1; units with non-positive gain = inf."""
    g = (w / tau).sum(1) / tau_s
    return torch.where(g > 0, 1. / g, torch.full_like(g, float('inf')))


class QueryTap:
    def __init__(self, ssa, neo_tap):
        self.ssa, self.neo = ssa, neo_tap
        self.cap = {}
        for k in ('q', 'k', 'v'):
            getattr(ssa, f'{k}_lif').register_forward_hook(self._grab(k))
        ssa.q_linear.register_forward_hook(self._q_in)
        ssa.attn_lif.register_forward_hook(self._attn)
        self.s = {}
        self.max_d, self.rows = 0., 0
        self.q0, self.silent_checked = None, 0

    def _grab(self, k):
        def fn(mod, inp, out):
            self.cap[k] = out.detach()
        return fn

    def _q_in(self, mod, inp, out):
        x = inp[0].detach()                                          # [T_h * B, N, D]
        o = self.neo.last                                            # [N, B, D]
        T_h = x.shape[0] // o.shape[1]
        ref = o.transpose(0, 1).unsqueeze(0).expand(T_h, -1, -1, -1).flatten(0, 1)
        if not torch.equal(x, ref):
            raise RuntimeError('c56c: q_linear input is not the Neocortex output')

    def _fixed_q(self, T, C, ref):
        """q0 [T_h, D]: q_lif(q_bn(0)) by hand; BN in eval mode gives the same current to every row and token."""
        ssa = self.ssa
        if ssa.q_linear.bias is not None or ssa.q_bn.training:
            raise RuntimeError('c56f: expected a bias-free q_linear and BN in eval mode')
        I = ssa.q_bn(ref.new_zeros(1, C, 1))[0, :, 0]
        lif = ssa.q_lif
        th, vr, r = float(lif.v_threshold), float(lif.v_reset), 1. / float(lif.tau)
        v, out = torch.full_like(I, vr), []
        for _ in range(T):
            H = v + r * (I - v + vr)
            sp = (H >= th).to(I.dtype)
            v = torch.where(H >= th, torch.full_like(H, vr), H)
            out.append(sp)
        return torch.stack(out)

    def _attn(self, mod, inp, out):
        ssa, a_in = self.ssa, inp[0].detach()
        T, B, N, C = a_in.shape
        h = ssa.num_heads
        split = lambda z: z.reshape(T, B, N, h, C // h).permute(0, 1, 3, 2, 4)
        q, k, v = (split(self.cap[n]) for n in 'qkv')
        S = (q @ k.transpose(-2, -1)) * ssa.scale
        b = ssa.relative_bias_table.detach()[ssa.rel_pos_idx].permute(2, 0, 1)[None, None]   # [1, 1, h, N, N]
        whole = ((S + b) @ v).transpose(2, 3).reshape(T, B, N, C)
        self.max_d = max(self.max_d, float((whole - a_in).abs().max()))
        if self.q0 is None:
            self.q0 = self._fixed_q(T, C, a_in)
        q0 = self.q0[:, None, None, :].expand(T, B, N, C)            # [T, B, N, C]
        sil = (self.neo.last.sum(-1) == 0).transpose(0, 1)           # [B, N], the Neocortex output of this batch
        if sil.any():
            if not torch.equal(self.cap['q'][:, sil], q0[:, sil]):    # [T, n silent, C]
                raise RuntimeError('c56f: q of a silent Neocortex token differs from q0')
            self.silent_checked += int(sil.sum())
        S0 = (split(q0) @ k.transpose(-2, -1)) * ssa.scale
        t0, td, tb = S0 @ v, (S - S0) @ v, b.expand_as(S) @ v        # [T, B, h, N, d]
        add = dict(term_fixed=t0.abs().sum((0, 1, 2, 4)).double().cpu(), term_var=td.abs().sum((0, 1, 2, 4)).double().cpu(), term_pos=tb.abs().sum((0, 1, 2, 4)).double().cpu(),
                   q_in_q0=(self.cap['q'] * q0).sum().double().cpu(), q_diff_token=(self.cap['q'] - q0).abs().sum((0, 1, 3)).double().cpu(),
                   q_token=self.cap['q'].sum((0, 1, 3)).double().cpu(), q_step=self.cap['q'].sum((1, 2, 3)).double().cpu(),
                   k_all=self.cap['k'].sum().double().cpu(), v_all=self.cap['v'].sum().double().cpu(),
                   q_silent_token=(self.cap['q'].sum((0, 3)) == 0).sum(0).double().cpu())
        for key, val in add.items():
            self.s[key] = self.s.get(key, 0.) + val
        self.rows += B
        self.T, self.N, self.C = T, N, C

    def result(self):
        if self.max_d > 1e-5:
            raise RuntimeError(f'c56d: attention split differs by {self.max_d:.3g}')
        T, N, C, R = self.T, self.N, self.C, self.rows
        one = T * R * C                                              # elements per token over all rows
        if self.silent_checked == 0:
            print('note: no silent Neocortex token in this run (c56f empty)', flush=True)
        return dict(q_rate=float(self.s['q_token'].sum() / (one * N)), k_rate=float(self.s['k_all'] / (one * N)), v_rate=float(self.s['v_all'] / (one * N)),
                    q_rate_token=(self.s['q_token'] / one).tolist(), q_rate_step=(self.s['q_step'] / (R * N * C)).tolist(),
                    q_silent_token=(self.s['q_silent_token'] / R).tolist(),
                    q0_rate=float(self.q0.mean()), q0_rate_step=self.q0.mean(1).tolist(), q0_active_units=float((self.q0.sum(0) > 0).double().mean()),
                    q_spikes_in_q0=float(self.s['q_in_q0'] / self.s['q_token'].sum()), q_diff_token=(self.s['q_diff_token'] / one).tolist(),
                    silent_checked=self.silent_checked,
                    term_fixed_token=(self.s['term_fixed'] / one).tolist(), term_var_token=(self.s['term_var'] / one).tolist(),
                    term_pos_token=(self.s['term_pos'] / one).tolist(), max_diff=self.max_d)


def params(model, inner):
    c = lambda m: sum(p.numel() for p in m.parameters())
    ssa = inner.data_block[0].ssa
    out = dict(total=c(model), neo_embedding=c(inner.encoding_neo), neo_time=c(inner.time_block),
               query_proj=c(ssa.q_linear) + c(ssa.q_bn), decoder=c(inner.weak_decoder) if hasattr(inner, 'weak_decoder') else 0)
    return out


def run_one(cell, seed, ref, device):
    ds, pl = cell
    task = 'mitbih' if ds == 'MITBIH' else 'ett'
    args, model, inner = load(ref['dir'], task, device)
    if len(inner.data_block) != 1 or inner.hippo_xattn_mode != 'query' or inner.neo_grad != 1. or inner.neo_neuron != 'pop_lif' or inner.T_h != 4:
        raise RuntimeError(f'{cell} seed {seed}: not the expected model')
    taps = {'emb': PopTap(inner.encoding_neo.emb_lif, 'emb')}
    for i, lay in enumerate(inner.time_block.Mem):
        taps[f'time{i + 1}'] = PopTap(lay.lif, f'time{i + 1}')
    last = list(taps.values())[-1]
    qt = QueryTap(inner.data_block[0].ssa, last)
    res = dict(cell=name(cell), seed=seed, T=inner.T, T_h=inner.T_h, N=inner.num_patches, D=args.embed_dim, heads=args.num_heads,
               patch=args.patch_size, time_layers=len(inner.time_block.Mem), taus=[float(t) for t in last.n.tau], tau_s=float(last.n.soma.tau_s),
               params=params(model, inner), rho_neo_log=float(ref['rho_neo']), rho_hippo_log=float(ref['rho_hippo']))
    with torch.no_grad():
        if task == 'ett':
            sys.path.append(os.path.join(HERE, '..', 'forecasting'))
            from train_forecast import data_args
            from data_provider.data_factory import data_provider
            _, dl = data_provider(data_args(args, args), 'test')
            se, ae, cnt = 0., 0., 0
            for x, y, _, _ in dl:
                x = x.float().to(device); y = y.float().to(device)[:, -pl:, :]
                d = (model(x) - y).double()
                functional.reset_net(model)
                se += float((d ** 2).sum()); ae += float(d.abs().sum()); cnt += d.numel()
            mse, mae = se / cnt, ae / cnt
            if abs(mse - float(ref['mse'])) > 2e-6 or abs(mae - float(ref['mae'])) > 2e-6 or cnt != int(ref['n']):
                raise RuntimeError(f"c56a {cell} seed {seed}: {mse:.6f} / {mae:.6f} / {cnt} vs {ref['mse']} / {ref['mae']} / {ref['n']}")
            res['metric'] = dict(mse=mse, mae=mae, n=cnt)
        else:
            from dataloader import create_loader
            dl, _ = create_loader(train=False, batch_size=args.batch_size, num_classes=N_CLASSES,
                                  data_root=os.path.join(args.root_path, '5class_mitbih_test_3_187.csv'), data_window_size=187, num_workers=0)
            zs, ys = [], []
            for xb, yb in dl:
                zs.append(model(xb.float().to(device))[0].cpu()); ys.append(yb.long().reshape(-1))
                functional.reset_net(model)
            z, yy = torch.cat(zs), torch.cat(ys)
            f1, _ = macro_f1(z.argmax(1), yy)
            if abs(f1 - float(ref['f1'])) > 1e-6 or len(yy) != int(ref['n_test']):
                raise RuntimeError(f"c56a {cell} seed {seed}: {f1:.6f} / {len(yy)} vs {ref['f1']} / {ref['n_test']}")
            res['metric'] = dict(f1=f1, n=len(yy))
    res['neurons'] = {k: t.result(args.embed_dim) for k, t in taps.items()}
    res['query'] = qt.result()
    mean_rate = float(np.mean([res['neurons'][k]['rate'] for k in taps]))
    res['rho_neo_probe'] = mean_rate
    if abs(mean_rate - res['rho_neo_log']) > 2e-6:
        raise RuntimeError(f"c56e {cell} seed {seed}: {mean_rate:.6f} vs logged rho_neo {res['rho_neo_log']:.6f}")
    for k in taps:
        if not np.isfinite(res['neurons'][k]['rate']):
            raise RuntimeError('non-finite rate')
    del model, inner
    torch.cuda.empty_cache()
    return res


def quarters(v):
    """Means over the first / second / third / fourth quarter of the tokens."""
    return [float(np.mean(p)) for p in np.array_split(np.asarray(v), 4)]


def ms(vals, f=4):
    return f'{np.mean(vals):.{f}f} ± {np.std(vals):.{f}f}'


def report(res):
    L = ['# Neocortex path of [query pop LIF NSMT T=4], measured on the trained checkpoints', '',
         'See the module docstring of neo_path_probe.py. Test split, frozen weights, mean ± sd over seeds 42 / 7 / 2026 (seed sd, 0 ddof). Descriptive only.', '']
    by = {name(c): [r for r in res if r['cell'] == name(c)] for c in CELLS}
    by = {k: v for k, v in by.items() if v}
    L += ['## Structure and parameters (seed 42)', '', '| cell | N tokens | P | D | heads | time layers | total params | Neocortex embedding | Neocortex time layers | query projection | unused decoder |', '|' + '---|' * 11]
    for k, rr in by.items():
        r, p = rr[0], rr[0]['params']
        L.append(f"| {k} | {r['N']} | {r['patch']} | {r['D']} | {r['heads']} | {r['time_layers']} | {p['total']} | {p['neo_embedding']} | {p['neo_time']} | {p['query_proj']} | {p['decoder']} |")
    L += ['', '## Spike rate per layer', '', '| cell | embedding neuron | time layer 1 | time layer 2 | q | k | v | Neocortex mean (= logged rho_neo) | logged rho_hippo |', '|' + '---|' * 9]
    for k, rr in by.items():
        t2 = ms([r['neurons']['time2']['rate'] for r in rr]) if 'time2' in rr[0]['neurons'] else 'none'
        L.append(f"| {k} | {ms([r['neurons']['emb']['rate'] for r in rr])} | {ms([r['neurons']['time1']['rate'] for r in rr])} | {t2} | {ms([r['query']['q_rate'] for r in rr])} | "
                 f"{ms([r['query']['k_rate'] for r in rr])} | {ms([r['query']['v_rate'] for r in rr])} | {ms([r['rho_neo_probe'] for r in rr])} | {ms([r['rho_hippo_log'] for r in rr])} |")
    L += ['', '## Last Neocortex layer (the query input), by token position (quarters of the token axis)', '',
          '| cell | rate Q1 | Q2 | Q3 | Q4 | first token rate | last token rate | rows silent Q1 | Q2 | Q3 | Q4 | units that never fire |', '|' + '---|' * 12]
    for k, rr in by.items():
        last = [r['neurons'][f"time{r['time_layers']}"] for r in rr]
        q = np.mean([quarters(n['rate_token']) for n in last], 0); s = np.mean([quarters(n['silent_token']) for n in last], 0)
        L.append(f"| {k} | " + ' | '.join(f'{v:.4f}' for v in q) + f" | {np.mean([n['rate_token'][0] for n in last]):.4f} | {np.mean([n['rate_token'][-1] for n in last]):.4f} | "
                 + ' | '.join(f'{v:.3f}' for v in s) + f" | {np.mean([n['dead_units'] for n in last]):.3f} |")
    L += ['', '## The query, by token position and by T_h step', '', '| cell | q rate Q1 | Q2 | Q3 | Q4 | no q spike in all T_h steps: Q1 | Q2 | Q3 | Q4 | q rate per T_h step 1 / 2 / 3 / 4 |', '|' + '---|' * 10]
    for k, rr in by.items():
        q = np.mean([quarters(r['query']['q_rate_token']) for r in rr], 0); s = np.mean([quarters(r['query']['q_silent_token']) for r in rr], 0)
        st = np.mean([r['query']['q_rate_step'] for r in rr], 0)
        L.append(f"| {k} | " + ' | '.join(f'{v:.4f}' for v in q) + ' | ' + ' | '.join(f'{v:.3f}' for v in s) + ' | ' + ' / '.join(f'{v:.4f}' for v in st) + ' |')
    L += ['', '## The fixed query q0 of a silent Neocortex token', '', '| cell | q0 rate | units with a q0 spike | q rate | share of q spikes already in q0 | mean abs(q - q0) | tokens (of N) with >= 90 % of rows silent |', '|' + '---|' * 7]
    for k, rr in by.items():
        last = [r['neurons'][f"time{r['time_layers']}"] for r in rr]
        L.append(f"| {k} | {ms([r['query']['q0_rate'] for r in rr])} | {ms([r['query']['q0_active_units'] for r in rr], 3)} | {ms([r['query']['q_rate'] for r in rr])} | "
                 f"{ms([r['query']['q_spikes_in_q0'] for r in rr], 3)} | {ms([np.mean(r['query']['q_diff_token']) for r in rr])} | {np.mean([np.sum(np.asarray(n['silent_token']) >= .9) for n in last]):.1f} of {rr[0]['N']} |")
    L += ['', '## Attention output before attn_lif: three terms (mean abs value per element; mean |value| is not additive, shares are of the three means)', '',
          '| cell | fixed q term | Neocortex-varying term | bias term | varying share overall | Q1 | Q2 | Q3 | Q4 |', '|' + '---|' * 9]
    for k, rr in by.items():
        f = [np.asarray(r['query']['term_fixed_token']) for r in rr]; a = [np.asarray(r['query']['term_var_token']) for r in rr]; b = [np.asarray(r['query']['term_pos_token']) for r in rr]
        sh = [y.mean() / (x.mean() + y.mean() + z.mean()) for x, y, z in zip(f, a, b)]
        qs = np.mean([[pa.mean() / (pf.mean() + pa.mean() + pb.mean()) for pf, pa, pb in zip(np.array_split(x, 4), np.array_split(y, 4), np.array_split(z, 4))] for x, y, z in zip(f, a, b)], 0)
        L.append(f"| {k} | {ms([x.mean() for x in f])} | {ms([y.mean() for y in a])} | {ms([z.mean() for z in b])} | {ms(sh, 3)} | " + ' | '.join(f'{v:.3f}' for v in qs) + ' |')
    L += ['', '## Soma of each Neocortex neuron (branch time constants 4 / 8 / 16 / 32, soma 2, threshold 1; init w = 0.5)', '',
          '| cell | neuron | mean w_k (4 / 8 / 16 / 32) | sum w mean | w < 0 share | mean abs(w - 0.5) | w percentile 1 .. 99 | branch share of soma input % | current mean ± std | current to fire at token 1 (median) | constant current to fire (median) |', '|' + '---|' * 11]
    for k, rr in by.items():
        for nn in rr[0]['neurons']:
            x = [r['neurons'][nn] for r in rr]
            L.append(f"| {k} | {nn} | " + ' / '.join(f'{v:.3f}' for v in np.mean([n['w_mean'] for n in x], 0)) + f" | {np.mean([n['w_sum_mean'] for n in x]):.3f} | {np.mean([n['w_neg'] for n in x]):.3f} | {np.mean([n['w_abs_dev'] for n in x]):.3f} | {np.mean([n['w_q'][0] for n in x]):.3f} .. {np.mean([n['w_q'][1] for n in x]):.3f} | "
                     + ' / '.join(f'{100 * v:.1f}' for v in np.mean([n['branch_share'] for n in x], 0))
                     + f" | {np.mean([n['cur_mean'] for n in x]):+.3f} ± {np.mean([n['cur_std'] for n in x]):.3f} | {np.median([n['first_token_current_median'] for n in x]):.2f} | {np.median([n['steady_current_median'] for n in x]):.2f} |")
    L += ['', '## Share of a branch left after d tokens, (1 - 1 / tau)^d', '', '| d | tau 4 | tau 8 | tau 16 | tau 32 |', '|---|---|---|---|---|']
    for d in (1, 4, 10, 23):
        L.append(f'| {d} | ' + ' | '.join(f'{(1 - 1 / t) ** d:.3f}' for t in (4, 8, 16, 32)) + ' |')
    L += ['', '## Checks', '']
    w = max(n['diff_frac'] for r in res for n in r['neurons'].values())
    L += [f'- c56a reported test metric reproduced: {len(res)} / {len(res)} runs.',
          f'- c56b equations reproduce the spikes: largest share of differing decisions {w:.2e} (limit 1e-4).',
          '- c56c q_linear input == last Neocortex output: exact, every batch.',
          f"- c56d attention split: largest |diff| {max(r['query']['max_diff'] for r in res):.2e} (limit 1e-5).",
          f"- c56e probe mean rate vs logged rho_neo: largest |diff| {max(abs(r['rho_neo_probe'] - r['rho_neo_log']) for r in res):.2e} (limit 2e-6).",
          f"- c56f q of a silent Neocortex token == q0: exact, {sum(r['query']['silent_checked'] for r in res)} (row, token) pairs over {len(res)} runs; fewest in one run {min(r['query']['silent_checked'] for r in res)}.", '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--only', default=None, help='one cell, e.g. ETTh2_96 or MITBIH')
    ap.add_argument('--seeds', default=None, help='comma list; default all')
    ap.add_argument('--out_md', default=os.path.join(CL, 'flif-v1-neo-path-probe.md'))
    ap.add_argument('--out_json', default=os.path.join(CL, 'flif-v1-neo-path-probe.json'))
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(',')) if a.seeds else SEEDS
    res = []
    for c in CELLS:
        if a.only and name(c) != a.only:
            continue
        refs = runs_of(c)
        for s in seeds:
            res.append(run_one(c, s, refs[s], a.device))
            print(res[-1]['cell'], s, res[-1]['metric'], 'rho_neo', round(res[-1]['rho_neo_probe'], 6), flush=True)
    if not res:
        raise RuntimeError('nothing ran')
    text = report(res)
    if a.only or a.seeds:                                            # partial run: print only, never overwrite the full artifacts
        print(text)
        return
    with open(a.out_md, 'w') as f:
        f.write(text)
    with open(a.out_json, 'w') as f:
        json.dump(res, f, indent=1)
    print(text)


if __name__ == '__main__':
    main()
