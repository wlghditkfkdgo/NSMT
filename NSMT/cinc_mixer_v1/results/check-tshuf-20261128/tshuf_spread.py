"""Evaluation-order spread of the order-shuffled SpkMixer (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; no training).
For every run of flif-v1-mitbih-tshuf (tag spkmx_lif_muxlin_add_lr0.02_L2_tshuf): the last-epoch weights on the 30 % MIT-BIH test file with
  default  the evaluation permutations of the logged test (token_shuffle_eval_seed 1000003); its macro-F1 must equal the logged one
           (1e-6), else stop (exit 2)
  seed 1, seed 2  other evaluation permutations (token_shuffle_eval_seed 1 / 2)
  natural  token_shuffle switched off at evaluation (the tokens in their true order, never seen in training)
Report only: macro-F1 per run and reading, the spread (max - min over default / seed 1 / seed 2) and natural - default.
usage (from cinc_mixer_v1/): CUDA_VISIBLE_DEVICES=<g> python -B results/check-tshuf-20261128/tshuf_spread.py --out <json>
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
ap.add_argument('--out', required=True)
ap.add_argument('--suite', default='flif-v1-mitbih-tshuf')                # --suite / --n: the smoke test of this script
ap.add_argument('--n', type=int, default=3)
a = ap.parse_args()
TAG = 'spkmx_lif_muxlin_add_lr0.02_L2_tshuf'
rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(os.path.join(C, 'results', a.suite, 'raw.txt')) if l.startswith('RESULT')]
rows = [r for r in rows if r.get('tag') == TAG]
if len(rows) != a.n:
    sys.exit(f'STOP: {len(rows)} rows of {TAG} (want {a.n})')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(model, loader, nc, eval_seed=None, natural=False):
    model.token_shuffle = not natural
    if eval_seed is not None:
        model.token_shuffle_eval_seed = eval_seed
    model.eval(); model.train_mode = 'testing'                   # eval() re-seeds the evaluation stream
    f1 = MulticlassF1Score(num_classes=nc, average='macro').to(dev)
    functional.reset_net(model)
    with torch.no_grad():
        for batch in loader:
            x = batch[0].float().to(dev); y = batch[1].long().squeeze(-1).to(dev)
            out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            functional.reset_net(model)
            f1.update(out.argmax(1), y)
    model.token_shuffle = True
    return 100. * float(f1.compute())


res = []
for r in sorted(rows, key=lambda r: r['seed']):
    cfg = torch.load(os.path.join(r['dir'], 'model_state', 'config.pt'), map_location='cpu')
    args = SimpleNamespace(**cfg); args.device = dev
    model = LOAD_MODEL[args.model](args, True)
    model.load_state_dict(torch.load(os.path.join(r['dir'], 'model_state', 'best+model.pt'), map_location='cpu'))
    model = model.to(dev)
    if not getattr(model, 'token_shuffle', False):
        sys.exit(f'STOP: seed {r["seed"]}: model built without token_shuffle')
    default_seed = model.token_shuffle_eval_seed
    loader, _ = create_loader(train=False, num_classes=args.num_classes, batch_size=args.batch_size,
                              data_root=args.root_path + '/5class_mitbih_test_3_187.csv', data_window_size=187, num_workers=args.num_workers)
    f0 = evaluate(model, loader, args.num_classes, default_seed)
    if abs(f0 / 100. - float(r['f1'])) > 1e-6:
        sys.exit(f'STOP: seed {r["seed"]}: default evaluation macro-F1 {f0 / 100:.6f} != logged {r["f1"]}')
    f1s = evaluate(model, loader, args.num_classes, 1)
    f2s = evaluate(model, loader, args.num_classes, 2)
    fn = evaluate(model, loader, args.num_classes, default_seed, natural=True)
    out = dict(seed=r['seed'], default=f0, seed1=f1s, seed2=f2s, natural=fn, spread=max(f0, f1s, f2s) - min(f0, f1s, f2s), natural_minus_default=fn - f0)
    res.append(out)
    print(f'seed {r["seed"]}: macro-F1 default {f0:.3f} (logged {100 * float(r["f1"]):.3f}), eval seed 1 {f1s:.3f}, seed 2 {f2s:.3f}, '
          f'natural order {fn:.3f}; spread {out["spread"]:.3f} %p, natural - default {out["natural_minus_default"]:+.3f} %p', flush=True)
    del model
json.dump(dict(tag=TAG, runs=res), open(a.out, 'w'), indent=1)
print('done', len(res))
