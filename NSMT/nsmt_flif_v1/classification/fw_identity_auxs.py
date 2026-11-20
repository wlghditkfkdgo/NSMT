# MODI 2026-11-20 (x0 of check_auxs.py): aux-path identity around the --aux_shift change (aux-target memory check). Run "save"
# before the change and "compare" after it (the saved file lives in the session scratchpad). fw_identity_xnor.py covers the plain
# forecasting path; this file covers the two aux paths with aux_shift left at 0 (the default):
#   ett: the all-SpikeDE pop LIF base with the raw-patch aux (flif-v1-ett-auxf, ETTh2 H96 seed 42): fresh-model init state_dict,
#        train-mode forecast, aux_rec_loss() and every gradient of (1 - r) L1 + r rec; the eval forecast of the stored checkpoint
#   mb:  the all-SpikeDE pop f-LIF base with the raw-patch aux (flif-v1-mitbih-auxf, seed 42): fresh-model init state_dict, the
#        train-mode tuple (logits, org_x, rec_x), the reconstruction MSE and every gradient of 0.9 CE + 0.1 MSE; the eval logits of
#        the stored last-epoch weights
# usage (nsmt_flif_v1/classification): python fw_identity_auxs.py save|compare <file>
import argparse, os, sys
import torch
sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster

mode, out = sys.argv[1], sys.argv[2]
HERE = os.getcwd()


def find(path, **kv):
    for ln in open(path):
        if ln.startswith('RESULT') and all(f' {k}={v} ' in ln for k, v in kv.items()):
            return dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)['dir']
    raise RuntimeError((path, kv))


res = {}
# ETT
run = find(os.path.join(HERE, '..', 'forecasting', 'results', 'flif-v1-ett-auxf', 'raw.txt'),
           cond='nsmt4q_pop_lif_msn_sde_auxf', ds='ETTh2', pl=96, seed=42)
cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
a = argparse.Namespace(**dict(cfg, device='cuda'))
x = torch.randn(4, a.seq_len, 7, generator=torch.Generator().manual_seed(5)).cuda()
y = torch.randn(4, a.pred_len, 7, generator=torch.Generator().manual_seed(6)).cuda()
torch.manual_seed(0); m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
res['ett/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
m.aux = True; m.train(); m.train_mode = 'training'; functional.reset_net(m)
o = m(x); rec = m.aux_rec_loss(); r = float(a.aux_alpha) * a.pred_len / 720.
((1 - r) * (o - y).abs().mean() + r * rec).backward()
res['ett/train_out'] = o.detach().cpu(); res['ett/rec'] = rec.detach().cpu()
res['ett/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
functional.reset_net(m)
torch.manual_seed(0); m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
m.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu')); m.eval(); m.train_mode = 'testing'
with torch.no_grad():
    functional.reset_net(m); res['ett/eval_out'] = m(x).detach().cpu(); functional.reset_net(m)
# MIT-BIH
run = find(os.path.join(HERE, 'results', 'flif-v1-mitbih-auxf', 'raw.txt'), cond='nsmt4q_pop_flif_msn_sde_auxf', seed=42)
cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
a = argparse.Namespace(**dict(cfg, device='cuda'))
x = torch.rand(4, 1, 187, generator=torch.Generator().manual_seed(7)).cuda()
lab = torch.tensor([0, 1, 2, 4]).cuda()
torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
res['mb/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
m.train(); m.train_mode = 'training'; functional.reset_net(m)
z, _, _, org_x, rec_x = m(x)
mse = ((org_x - rec_x) ** 2).mean()
(0.9 * torch.nn.functional.cross_entropy(z, lab) + 0.1 * mse).backward()
res['mb/logits'] = z.detach().cpu(); res['mb/org_x'] = org_x.detach().cpu(); res['mb/rec_x'] = rec_x.detach().cpu(); res['mb/mse'] = mse.detach().cpu()
res['mb/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
functional.reset_net(m)
torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
w = torch.load(os.path.join(run, 'model_state', 'best+model.pt'), map_location='cuda')
rr = m.load_state_dict(w, strict=False)
if rr.missing_keys or rr.unexpected_keys:
    raise RuntimeError(f'MIT-BIH weights do not fit: {rr}')
m.eval(); m.train_mode = 'testing'
with torch.no_grad():
    functional.reset_net(m); o = m(x); res['mb/eval_out'] = (o[0] if isinstance(o, tuple) else o).detach().cpu(); functional.reset_net(m)

if mode == 'save':
    torch.save(res, out); print('saved', len(res), 'entries')
else:
    ref = torch.load(out)
    bad = 0
    if sorted(ref) != sorted(res):
        raise SystemExit(f'entry sets differ: {sorted(set(ref) ^ set(res))}')
    for k in sorted(ref):
        a_, b_ = ref[k], res[k]
        if isinstance(a_, dict):
            same = sorted(a_) == sorted(b_) and all(torch.equal(a_[n], b_[n]) for n in a_)
        else:
            same = torch.equal(a_, b_)
        print(('SAME ' if same else 'DIFF ') + k)
        bad += 0 if same else 1
    print(f'{len(ref) - bad}/{len(ref)} identical')
    sys.exit(1 if bad else 0)
