# MODI 2026-11-18 (x0 of check_xnor.py): default-path identity around the --attn_score change. Run "save" before the change and
# "compare" after it (the saved file lives in the session scratchpad). For 4 reference ETT runs (spk_ssa and the msn base,
# ETTh2 / ETTm2 H96 seed 42): fresh-model init state_dict, train-mode output + every gradient of an MSE loss on a fixed input,
# and the eval output of the stored best-validation checkpoint on the same input. run from nsmt_flif_v1/classification.
import argparse, os, sys
import torch
sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster
mode, out = sys.argv[1], sys.argv[2]
FC = os.path.join(os.getcwd(), '..', 'forecasting', 'results')
def find(suite, cond, ds, pl, seed):
    for ln in open(os.path.join(FC, suite, 'raw.txt')):
        if ln.startswith('RESULT') and f' cond={cond} ' in ln and f' ds={ds} ' in ln and f' pl={pl} ' in ln and f' seed={seed} ' in ln:
            return dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)['dir']
    raise RuntimeError((suite, cond, ds, pl, seed))
RUNS = {'spk_ssa_h2': ('flif-v1-gen-ett', 'spk_ssa', 'ETTh2'), 'spk_ssa_m2': ('flif-v1-gen-ett', 'spk_ssa', 'ETTm2'),
        'base_lif_h2': ('flif-v1-ett-ms', 'nsmt4q_pop_lif_msn', 'ETTh2'), 'base_flif_m2': ('flif-v1-ett-ms', 'nsmt4q_pop_flif_msn', 'ETTm2')}
res = {}
for key, (suite, cond, ds) in RUNS.items():
    run = find(suite, cond, ds, 96, 42)
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = argparse.Namespace(**dict(cfg, device='cuda'))
    x = torch.randn(4, a.seq_len, 7, generator=torch.Generator().manual_seed(5)).cuda()
    y = torch.randn(4, a.pred_len, 7, generator=torch.Generator().manual_seed(6)).cuda()
    torch.manual_seed(0); m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
    res[key + '/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    o = m(x); torch.nn.functional.mse_loss(o, y).backward()
    res[key + '/train_out'] = o.detach().cpu()
    res[key + '/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
    functional.reset_net(m)
    torch.manual_seed(0); m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
    m.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu')); m.eval(); m.train_mode = 'testing'
    with torch.no_grad():
        functional.reset_net(m); res[key + '/eval_out'] = m(x).detach().cpu(); functional.reset_net(m)
if mode == 'save':
    torch.save(res, out); print('saved', len(res), 'entries')
else:
    ref = torch.load(out); bad = []
    for k, v in ref.items():
        w = res[k]
        same = (v.keys() == w.keys() and all(torch.equal(v[n], w[n]) for n in v)) if isinstance(v, dict) else torch.equal(v, w)
        if not same: bad.append(k)
    print('identical' if not bad else f'DIFFERENT: {bad}', f'({len(ref)} entries)')
