"""MIT-BIH token-order shuffle evaluation (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "MIT-BIH 실험에 대해서
추가실험: 순서 실험 진행" / "분류 과제 통제를 먼저 진행"; no training). The classifiers read the TOKEN MEAN (order-blind head), so any
use of token order has to happen in the body. Checkpoints = the last-epoch weights test.py evaluated (model_state/best+model.pt), the
30 % MIT-BIH test file, every RESULT row of the chosen arm (3 seeds):
  mux   LIF SpkMixer, muxlin fusion, ADD residual, lr 0.02, mixer layers 2 (flif-v1-mitbih-spkmux-tune, tag spkmx_lif_muxlin_add_lr0.02_L2)
  sfde  SF-de (Spikformer-style, SSA with relative position bias), lr 0.01 (flif-v1-mitbih-sfde, tag spk_ssa_lr0.01_L2_nalif)
  muxoff   (2026-11-28) the memoryless SpkMixer (flif-v1-mitbih-memoff)
  sfdenope (2026-11-28, user "SF-de의 위치 정보 끄기") SF-de with --ssa_pe 0 (flif-v1-mitbih-sfde-nope, tag spk_ssa_lr0.01_L2_nalif_pe0)
Per sample a random permutation of the N = 11 tokens (generator seeded by permutation seed and batch index):
  none   no shuffle; macro-F1 must equal the logged one (1e-6), else stop (exit 2)
  mixer  the token-mixing module's input (SpkMixer: block mix; SF-de: SSA) is shuffled and its output put back in order
  full   the patch tokens are shuffled before the embedding
Wrapper check (fail closed): on the first run, the identity permutation through every variant gives exactly the unshuffled macro-F1.
Each variant uses permutation seeds 0, 1, 2 (averaged). Output: macro-F1 / accuracy per run and variant, drops in %p; --out JSON.
usage (from cinc_mixer_v1/): CUDA_VISIBLE_DEVICES=<g> python -B results/check-mitorder-20261128/cls_order_eval.py --arm <mux|sfde> --out <json>
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
os.chdir(C)
from spikingjelly.clock_driven import functional                 # noqa: E402
from torcheval.metrics import MulticlassF1Score                  # noqa: E402
from load_model import LOAD_MODEL                               # noqa: E402
from dataloader import create_loader                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arm', choices=['mux', 'sfde', 'muxoff', 'sfdenope'], required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
SRC = {'mux': ('flif-v1-mitbih-spkmux-tune', 'spkmx_lif_muxlin_add_lr0.02_L2'), 'sfde': ('flif-v1-mitbih-sfde', 'spk_ssa_lr0.01_L2_nalif'),
       'muxoff': ('flif-v1-mitbih-memoff', 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff'),
       'sfdenope': ('flif-v1-mitbih-sfde-nope', 'spk_ssa_lr0.01_L2_nalif_pe0')}   # 2026-11-28 SF-de without the SSA position bias
suite, tag = SRC[a.arm]
rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(os.path.join(C, 'results', suite, 'raw.txt')) if l.startswith('RESULT')]
rows = [r for r in rows if r.get('tag') == tag]
if len(rows) != 3:
    sys.exit(f'STOP: {len(rows)} rows of {tag} (want 3)')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
state = {'pseed': 0, 'batch': 0, 'identity': False}


def perm_for(Bp, N, device):
    if state['identity']:
        p = torch.arange(N, device=device).unsqueeze(0).expand(Bp, N).contiguous()
        return p, p
    g = torch.Generator().manual_seed(100003 * state['pseed'] + state['batch'])
    p = torch.argsort(torch.rand(Bp, N, generator=g), dim=1)
    return p.to(device), torch.argsort(p, dim=1).to(device)


def take(x, idx, dim):                                            # per-row permutation along `dim` (row axis = 1)
    shape = [1] * x.dim()
    shape[1], shape[dim] = idx.shape[0], idx.shape[1]
    full = idx.view(*shape).expand(*[x.shape[i] if i not in (1, dim) else (idx.shape[0] if i == 1 else idx.shape[1]) for i in range(x.dim())])
    return torch.gather(x, dim, full)


def evaluate(model, loader, nc, variant):
    ssa = hasattr(model, 'data_block')                            # myModel (SF-de) vs SpkMixer
    blk = model.data_block[0] if ssa else model.blocks[0]
    mod = blk.ssa if ssa else blk
    o_mix = mod.forward if ssa else blk.mix
    o_tok = model._tokens_mc if ssa else model.tokens
    if variant == 'mixer':
        def mix(x, *args, **kw):
            p, ip = perm_for(x.shape[1], x.shape[2], x.device)
            return take(o_mix(take(x, p, 2), *args, **kw), ip, 2)
        if ssa:
            mod.forward = mix
        else:
            blk.mix = mix
    elif variant == 'full':
        def tokens(x):
            t = o_tok(x)
            pt = t[0] if ssa else t
            p, _ = perm_for(pt.shape[1], pt.shape[2], pt.device)
            return (take(pt, p, 2), t[1]) if ssa else take(pt, p, 2)
        if ssa:
            model._tokens_mc = tokens
        else:
            model.tokens = tokens
    f1 = MulticlassF1Score(num_classes=nc, average='macro').to(dev)
    hit = n = 0
    try:
        functional.reset_net(model)
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                state['batch'] = bi
                x = batch[0].float().to(dev); y = batch[1].long().squeeze(-1).to(dev)
                out = model(x)
                out = out[0] if isinstance(out, tuple) else out
                functional.reset_net(model)
                pr = out.argmax(1)
                f1.update(pr, y); hit += int((pr == y).sum()); n += y.numel()
    finally:
        blk.__dict__.pop('mix', None)
        mod.__dict__.pop('forward', None) if ssa else None
        model.__dict__.pop('tokens', None)
        model.__dict__.pop('_tokens_mc', None)
    return 100. * float(f1.compute()), 100. * hit / n


res = []
for r in sorted(rows, key=lambda r: r['seed']):
    cfg = torch.load(os.path.join(r['dir'], 'model_state', 'config.pt'), map_location='cpu')
    args = SimpleNamespace(**cfg); args.device = dev
    model = LOAD_MODEL[args.model](args, True)
    model.load_state_dict(torch.load(os.path.join(r['dir'], 'model_state', 'best+model.pt'), map_location='cpu'))
    model = model.to(dev).eval(); model.train_mode = 'testing'
    loader, _ = create_loader(train=False, num_classes=args.num_classes, batch_size=args.batch_size,
                              data_root=args.root_path + '/5class_mitbih_test_3_187.csv', data_window_size=187, num_workers=args.num_workers)
    f0, a0 = evaluate(model, loader, args.num_classes, 'none')
    if abs(f0 / 100. - float(r['f1'])) > 1e-6:
        sys.exit(f'STOP: seed {r["seed"]}: recomputed macro-F1 {f0 / 100:.6f} != logged {r["f1"]}')
    if not res:
        state['identity'] = True
        for var in ('mixer', 'full'):
            fi, _ = evaluate(model, loader, args.num_classes, var)
            if fi != f0:
                sys.exit(f'STOP: identity permutation through {var} gives {fi} != {f0}')
        state['identity'] = False
        print('identity-permutation check passed (mixer, full give the unshuffled macro-F1 exactly)', flush=True)
    out = dict(seed=r['seed'], f1=f0, acc=a0)
    for var in ('mixer', 'full'):
        fs, as_ = [], []
        for ps in (0, 1, 2):
            state['pseed'] = ps
            f, ac = evaluate(model, loader, args.num_classes, var)
            fs.append(f); as_.append(ac)
        out[var] = dict(f1=sum(fs) / 3, acc=sum(as_) / 3, drop_f1_pp=f0 - sum(fs) / 3, drop_acc_pp=a0 - sum(as_) / 3, f1_by_seed=fs)
    res.append(out)
    print(f'{a.arm} seed {r["seed"]}: macro-F1 {f0:.3f} (logged {100 * float(r["f1"]):.3f}), acc {a0:.3f} | ' +
          ' | '.join(f'{v} macro-F1 {out[v]["f1"]:.3f} (drop {out[v]["drop_f1_pp"]:+.3f} %p), acc drop {out[v]["drop_acc_pp"]:+.3f} %p' for v in ('mixer', 'full')), flush=True)
    del model
json.dump(dict(arm=a.arm, suite=suite, tag=tag, runs=res), open(a.out, 'w'), indent=1)
print('done', len(res))
