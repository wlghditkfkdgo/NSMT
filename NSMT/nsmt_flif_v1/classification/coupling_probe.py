"""MODI 2026-11-17 (no training, descriptive): how much does the coupling of the msn base model (Neocortex-MS query population,
nsmt4q_pop_<lif|flif>_msn, input 96, suite flif-v1-ett-ms) use the Neocortex, and how much of the Hippocampus does its gate remove?
Asked in the context of the user question 11-17 "is the Spikformer-style cross-attention the best coupling?".
Frozen weights (model_state/checkpoint.pth = the reported best-validation weights), ETT test split, forward hooks only.
Passes per checkpoint:
  intact : the model as trained; must reproduce the run's reported test MSE / MAE / n (|diff| <= 2e-6) or the script stops
  zero   : the Neocortex output (time_block output, the query source) replaced by zeros -> q is the fixed q0 everywhere
  swap   : the Neocortex output of row r replaced by that of row r-1 in the batch (rows are window x channel, channel-major
           inside a window: mostly the same window's neighbouring channel) -> realistic spikes, wrong series
  (zero / swap are test-time interventions on a model trained with the Neocortex: they measure reliance, not what a model
   trained without the Neocortex would reach)
Diagnostics on the intact pass (sums over the whole test split): firing rates of the Neocortex output, q, the attention LIF (a),
the attention-branch output y, the MLP-branch output m; share of the Hippocampus spikes entering the block that the attention
gate x*(1-y) removes, share removed by the MLP gate, share surviving to the head.
usage (from nsmt_flif_v1/classification): python coupling_probe.py --device cuda:0 [--only ETTh2_96] [--seeds 42] --out_json F --out_md F
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.getcwd()
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                    # noqa: E402
from head_probe import load                                          # noqa: E402

FC = os.path.join(HERE, '..', 'forecasting')
RAW = os.path.join(FC, 'results', 'flif-v1-ett-ms', 'raw.txt')
CONDS = ('nsmt4q_pop_lif_msn', 'nsmt4q_pop_flif_msn')
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SEEDS = (42, 7, 2026)


def refs():
    out = {}
    for ln in open(RAW):
        if not ln.startswith('RESULT'):
            continue
        f = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
        if f['cond'] not in CONDS:
            continue
        k = (f['cond'], f['ds'], int(f['pl']), int(f['seed']))
        if k in out:
            raise RuntimeError(f'duplicate {k}')
        out[k] = f
    for c in CONDS:
        for ds, pl in CELLS:
            for s in SEEDS:
                if (c, ds, pl, s) not in out:
                    raise RuntimeError(f'missing {(c, ds, pl, s)}')
    return out


class Acc:
    def __init__(self):
        self.s = {}

    def add(self, k, v):
        self.s[k] = self.s.get(k, 0.) + float(v)


def run_one(cond, ds, pl, seed, ref, device):
    args, model, inner = load(ref['dir'], 'ett', device)
    blk = inner.data_block[0]
    if (len(inner.data_block) != 1 or inner.hippo_xattn_mode != 'query' or inner.neo_grad != 1. or inner.T_h != 4
            or getattr(inner, 'neo_res', 'none') != 'ms' or inner.neo_neuron != cond.split('nsmt4q_')[1].rsplit('_msn')[0]
            or getattr(inner, 'res_mode', 'iand') != 'iand' or inner.neo_memoryless or inner.neo_shuffle is not None):
        raise RuntimeError(f'{cond} {ds} {pl} {seed}: not the expected model')
    sys.path.append(FC)
    from train_forecast import data_args
    from data_provider.data_factory import data_provider
    _, dl = data_provider(data_args(args, args), 'test')
    st = {'mode': 'intact'}
    acc = Acc()

    def neo_hook(m, i, o):
        if st['mode'] == 'intact':
            acc.add('neo_sum', o.sum()); acc.add('neo_n', o.numel())
            return None
        if st['mode'] == 'zero':
            return torch.zeros_like(o)
        return o.roll(1, dims=1)                                     # [N, B', 1, D]: row r gets row r-1

    def ssa_hook(m, i, o):
        if st['mode'] != 'intact':
            return
        x = i[0]
        acc.add('x_sum', x.sum()); acc.add('x_n', x.numel())
        acc.add('y_sum', o.sum()); acc.add('xy_sum', (x * o).sum())

    def mlp_hook(m, i, o):
        if st['mode'] != 'intact':
            return
        x = i[0]
        acc.add('xm_in_sum', x.sum()); acc.add('m_sum', o.sum()); acc.add('xm_sum', (x * o).sum())

    def blk_hook(m, i, o):
        if st['mode'] == 'intact':
            acc.add('fin_sum', o.sum())

    def rate_hook(name):
        def f(m, i, o):
            if st['mode'] == 'intact':
                acc.add(name + '_sum', o.sum()); acc.add(name + '_n', o.numel())
        return f

    hs = [inner.time_block.register_forward_hook(neo_hook), blk.ssa.register_forward_hook(ssa_hook),
          blk.mlp.register_forward_hook(mlp_hook), blk.register_forward_hook(blk_hook),
          blk.ssa.q_lif.register_forward_hook(rate_hook('q')), blk.ssa.attn_lif.register_forward_hook(rate_hook('a'))]
    res = dict(cond=cond, ds=ds, pl=pl, seed=seed)
    with torch.no_grad():
        for mode in ('intact', 'zero', 'swap'):
            st['mode'] = mode
            se = ae = 0.; cnt = 0
            for x, y, _, _ in dl:
                x = x.float().to(device); y = y.float().to(device)[:, -pl:, :]
                d = (model(x) - y).double()
                functional.reset_net(model)
                se += float((d ** 2).sum()); ae += float(d.abs().sum()); cnt += d.numel()
            mse, mae = se / cnt, ae / cnt
            if not (np.isfinite(mse) and np.isfinite(mae)):
                raise RuntimeError('non-finite metric')
            if mode == 'intact' and (abs(mse - float(ref['mse'])) > 2e-6 or abs(mae - float(ref['mae'])) > 2e-6 or cnt != int(ref['n'])):
                raise RuntimeError(f"reproduction {cond} {ds} {pl} {seed}: {mse:.6f} / {mae:.6f} / {cnt} vs {ref['mse']} / {ref['mae']} / {ref['n']}")
            res[mode] = dict(mse=mse, mae=mae, n=cnt)
    for h in hs:
        h.remove()
    s = acc.s
    if s.get('neo_sum', 0.) <= 0 or s.get('x_sum', 0.) <= 0:
        raise RuntimeError('no Neocortex or Hippocampus spikes')
    T_h = 4
    res['rate'] = dict(neo=s['neo_sum'] / s['neo_n'], q=s['q_sum'] / s['q_n'], a=s['a_sum'] / s['a_n'],
                       y=s['y_sum'] / s['x_n'], m=s['m_sum'] / s['x_n'], hip_in=s['x_sum'] / s['x_n'])
    res['gate'] = dict(attn_removed=s['xy_sum'] / s['x_sum'], mlp_removed=s['xm_sum'] / s['xm_in_sum'], survive=s['fin_sum'] / s['x_sum'])
    for k in ('zero', 'swap'):
        res[k]['d_mse_pct'] = 100. * (res[k]['mse'] / res['intact']['mse'] - 1.)
        res[k]['d_mae_pct'] = 100. * (res[k]['mae'] / res['intact']['mae'] - 1.)
    del model, inner
    torch.cuda.empty_cache()
    return res


def report(rs):
    L = ['| model | cell | seed | test MSE intact | Neocortex 0: ΔMSE % | other row\'s Neocortex: ΔMSE % | Neocortex rate | q rate | y rate | '
         'Hippocampus spikes removed by attention gate | removed by MLP gate | surviving to head |', '|' + '---|' * 12]
    for r in rs:
        L.append(f"| {r['cond'].replace('nsmt4q_', '').replace('_msn', '')} | {r['ds']} H{r['pl']} | {r['seed']} | {r['intact']['mse']:.6f} | "
                 f"{r['zero']['d_mse_pct']:+.2f} | {r['swap']['d_mse_pct']:+.2f} | {r['rate']['neo']:.4f} | {r['rate']['q']:.4f} | "
                 f"{r['rate']['y']:.4f} | {100 * r['gate']['attn_removed']:.2f} % | {100 * r['gate']['mlp_removed']:.2f} % | {100 * r['gate']['survive']:.2f} % |")
    L.append('')
    L.append('Means over 3 seeds (per model and cell): ΔMSE zero / swap, attention-gate removal')
    for c in CONDS:
        for ds, pl in CELLS:
            g = [r for r in rs if r['cond'] == c and r['ds'] == ds and r['pl'] == pl]
            if len(g) == 3:
                L.append(f"- {c} {ds} H{pl}: zero {np.mean([r['zero']['d_mse_pct'] for r in g]):+.2f} %, swap "
                         f"{np.mean([r['swap']['d_mse_pct'] for r in g]):+.2f} %, attention gate removes "
                         f"{100 * np.mean([r['gate']['attn_removed'] for r in g]):.2f} %, MLP gate "
                         f"{100 * np.mean([r['gate']['mlp_removed'] for r in g]):.2f} %, y rate {np.mean([r['rate']['y'] for r in g]):.4f}")
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--only', default=None)
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--out_md', required=True)
    a = ap.parse_args()
    R = refs()
    seeds = tuple(int(s) for s in a.seeds.split(',')) if a.seeds else SEEDS
    rs = []
    for c in CONDS:
        for ds, pl in CELLS:
            if a.only and a.only != f'{ds}_{pl}':
                continue
            for s in seeds:
                t0 = time.time()
                r = run_one(c, ds, pl, s, R[(c, ds, pl, s)], a.device)
                r['time_s'] = round(time.time() - t0, 1)
                rs.append(r)
                print(json.dumps({k: r[k] for k in ('cond', 'ds', 'pl', 'seed', 'time_s')}), r['intact']['mse'], r['zero']['d_mse_pct'],
                      r['swap']['d_mse_pct'], r['gate'], flush=True)
    json.dump(rs, open(a.out_json, 'w'), indent=1)
    open(a.out_md, 'w').write(report(rs) + '\n')
    print(report(rs))


if __name__ == '__main__':
    main()
