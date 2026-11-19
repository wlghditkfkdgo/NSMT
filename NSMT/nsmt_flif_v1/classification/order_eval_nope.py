"""MODI 2026-11-19 (post hoc, NOT pre-registered, no training): how much the stored MIT-BIH models of the positional-encoding-off
test use the token order at test time. For every run of flif-v1-mitbih-sde (bias on) and flif-v1-mitbih-nope (bias off) the test
set is evaluated with the stored last-epoch weights (best+model.pt, the tested weights):
  normal     the true token order; the macro-F1 must reproduce the logged f1 (|diff| <= 1e-6), else the script stops
  all        the N tokens in one fixed shuffled order for both paths (numpy default_rng(0).permutation(N), the evaluation
             permutation of token_order.TokenShuffle); the hippocampus input [T_h, B, N, P] and the Neocortex input [N, B, 1, P] get
             the same permutation
  neo        (base only) the model's own neo_shuffle control at evaluation: the Neocortex reads the tokens in that shuffled order
             and its outputs are put back at the original positions; the hippocampus keeps the true order
Reported: macro-F1 of each, the drop from normal (%p), the share of test samples whose prediction changes. Checks (fail-closed):
normal reproduces the log; Spikformer(DE) without the bias must give exactly the same predictions under 'all' (it is order-blind);
every value finite; 9 + 9 rows.
usage (classification folder): python order_eval_nope.py   -> results/flif-v1-nope-order-eval.{json,md}
"""
import json
import math
import os
import sys
from types import SimpleNamespace as NS

import torch
from torcheval.metrics import MulticlassF1Score

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional

from dataloader import create_loader
from load_model import LOAD_MODEL
from token_order import TokenShuffle
from utils import get_pred

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
CONDS = ('spkall_lif', 'nsmt4q_pop_lif_msn_sde', 'nsmt4q_pop_flif_msn_sde')
NAMES = {'spkall_lif': 'Spikformer(DE)', 'nsmt4q_pop_lif_msn_sde': 'base pop LIF', 'nsmt4q_pop_flif_msn_sde': 'base pop f-LIF'}
SEEDS = (42, 7, 2026)
SUITES = (('on', 'flif-v1-mitbih-sde'), ('off', 'flif-v1-mitbih-nope'))


def rows(suite):
    out = {}
    for ln in open(os.path.join(RES, suite, 'raw.txt')):
        if ln.startswith('RESULT'):
            d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            k = (d['cond'], int(d['seed']))
            if k in out:
                raise RuntimeError(f'{suite}: duplicate {k}')
            out[k] = d
    miss = [(c, s) for c in CONDS for s in SEEDS if (c, s) not in out]
    if miss or len(out) != 9:
        raise RuntimeError(f'{suite}: {len(out)} rows, missing {miss}')
    return out


def build(d):
    cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda'))
    torch.manual_seed(0)
    m = LOAD_MODEL[a.model](a, True).cuda()
    w = torch.load(os.path.join(d, 'model_state', 'best+model.pt'), map_location='cuda')
    r = m.load_state_dict(w, strict=False)
    if r.missing_keys or not all('weak_decoder' in k for k in r.unexpected_keys):
        raise RuntimeError(f'{d}: weights do not fit (missing {r.missing_keys[:3]}, unexpected {r.unexpected_keys[:3]})')
    m.eval(); m.train_mode = 'testing'
    return m, a


def predict(m, loader, ncls):
    f1 = MulticlassF1Score(num_classes=ncls, average='macro').to('cuda')
    preds = []
    functional.reset_net(m)
    with torch.no_grad():
        for batch in loader:
            x = batch[0].float().cuda(); y = batch[1].long().squeeze(-1).cuda()
            o = m(x)
            o = o[0] if isinstance(o, tuple) else o
            functional.reset_net(m)
            p = get_pred(o)
            f1.update(p, y); preds.append(p.detach().cpu())
    return f1.compute().item(), torch.cat(preds)


def main():
    out, loader = [], None
    for arm, suite in SUITES:
        rs = rows(suite)
        for c in CONDS:
            for s in SEEDS:
                d = rs[(c, s)]
                if (arm == 'off') != (d.get('pe') == '0') or (arm == 'off') != ('_nope' in os.path.basename(d['dir'])):
                    raise RuntimeError(f'{suite} {c} {s}: bias arm does not match the run (pe={d.get("pe")}, {os.path.basename(d["dir"])})')
                m, a = build(d['dir'])
                if int(a.ssa_pe) != (0 if arm == 'off' else 1):
                    raise RuntimeError(f'{suite} {c} {s}: ssa_pe {a.ssa_pe}')
                if loader is None:
                    loader, _ = create_loader(train=False, num_classes=a.num_classes, batch_size=a.batch_size,
                                              data_root=a.root_path + '/5class_mitbih_test_3_187.csv', data_window_size=187, num_workers=4)
                f_n, p_n = predict(m, loader, a.num_classes)
                if not abs(f_n - float(d['f1'])) <= 1e-6:
                    raise RuntimeError(f'{suite} {c} {s}: normal-order f1 {f_n:.6f} != logged {d["f1"]}')
                N = m.num_patches
                perm = TokenShuffle(N, 0).eval_perm.cuda()
                if bool((perm == torch.arange(N, device='cuda')).all()):
                    raise RuntimeError('identity permutation')
                orig = m._tokens_mc

                def shuffled(x, orig=orig, perm=perm):
                    h, n = orig(x)
                    return h[:, :, perm].contiguous(), (n[perm].contiguous() if n is not None else None)
                m._tokens_mc = shuffled
                f_a, p_a = predict(m, loader, a.num_classes)
                m._tokens_mc = orig
                row = dict(arm=arm, cond=c, seed=s, n_test=int(p_n.numel()), n_tokens=int(N), f1_normal=f_n, f1_all=f_a,
                           changed_all=float((p_n != p_a).float().mean()))
                if c != 'spkall_lif':
                    m.neo_shuffle = TokenShuffle(N, 0)
                    f_o, p_o = predict(m, loader, a.num_classes)
                    m.neo_shuffle = None
                    f_back, p_back = predict(m, loader, a.num_classes)
                    if not torch.equal(p_back, p_n):
                        raise RuntimeError(f'{suite} {c} {s}: predictions not restored after the neo control')
                    row.update(f1_neo=f_o, changed_neo=float((p_n != p_o).float().mean()))
                if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                    raise RuntimeError(f'{suite} {c} {s}: non-finite value {row}')
                if arm == 'off' and c == 'spkall_lif' and not torch.equal(p_n, p_a):
                    raise RuntimeError(f'{suite} {c} {s}: Spikformer(DE) without the bias changed {row["changed_all"]:.4%} predictions under the shuffle')
                print(row, flush=True)
                out.append(row)
                del m
                torch.cuda.empty_cache()
    if len(out) != 18:
        raise RuntimeError(f'{len(out)} rows')
    json.dump(out, open(os.path.join(RES, 'flif-v1-nope-order-eval.json'), 'w'), indent=1)

    def mean(xs):
        xs = list(xs)
        if not xs or any(not math.isfinite(v) for v in xs):
            raise RuntimeError('mean of nothing / non-finite')
        return sum(xs) / len(xs)
    L = ['# MIT-BIH: test-time token-order shuffle of the stored models (post hoc, not pre-registered, no training)', '',
         'See the docstring of order_eval_nope.py. macro-F1 %, mean of seeds 42 / 7 / 2026 (per-seed drops in brackets); "changed" = share of '
         f'the {out[0]["n_test"]} test samples whose prediction changes. {out[0]["n_tokens"]} tokens.', '',
         '| model | bias | normal order | all tokens shuffled | drop %p (per seed) | changed % | Neocortex order only shuffled | drop %p (per seed) | changed % |',
         '|---|---|---|---|---|---|---|---|---|']
    for c in CONDS:
        for arm, _ in SUITES:
            g = [r for r in out if r['cond'] == c and r['arm'] == arm]
            if [r['seed'] for r in g] != list(SEEDS):
                raise RuntimeError(f'{c} {arm}: seeds {[r["seed"] for r in g]}')
            da = [100 * (r['f1_all'] - r['f1_normal']) for r in g]
            cell = [NAMES[c], arm, f"{100 * mean(r['f1_normal'] for r in g):.2f}", f"{100 * mean(r['f1_all'] for r in g):.2f}",
                    f"{mean(da):+.2f} ({' / '.join(f'{v:+.2f}' for v in da)})", f"{100 * mean(r['changed_all'] for r in g):.2f}"]
            if c == 'spkall_lif':
                cell += ['—', '—', '—']
            else:
                dn = [100 * (r['f1_neo'] - r['f1_normal']) for r in g]
                cell += [f"{100 * mean(r['f1_neo'] for r in g):.2f}", f"{mean(dn):+.2f} ({' / '.join(f'{v:+.2f}' for v in dn)})",
                         f"{100 * mean(r['changed_neo'] for r in g):.2f}"]
            L.append('| ' + ' | '.join(cell) + ' |')
    open(os.path.join(RES, 'flif-v1-nope-order-eval.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('all checks passed')


if __name__ == '__main__':
    main()
