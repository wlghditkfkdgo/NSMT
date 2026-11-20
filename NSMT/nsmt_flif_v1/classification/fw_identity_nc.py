# MODI 2026-11-21 (4-2 coupling placement (가) / (나), user decision 11-20): default-path identity around the --neo_current /
# --hippo_attn change. Run "save" before the change and "compare" after it (the saved file lives in the session scratchpad).
# Covers the range-1 reference runs (new default training; suites flif-v1-ett-r1 / flif-v1-mitbih-r1), flags left at their defaults:
#   ett/<cond>: ETTh2 H96 seed 42 of the base pop LIF / pop f-LIF (nsmt4q_pop_<n>_msn_sde) and Spikformer(DE) (spkall_lif):
#        fresh-model init state_dict, train-mode forecast, every gradient of the L1 loss; the eval forecast of the stored checkpoint
#   mb/<cond>: MIT-BIH seed 42 of the base pop LIF / pop f-LIF: fresh-model init state_dict, train-mode logits and every gradient
#        of the CE loss; the eval logits of the stored last-epoch weights
# usage (nsmt_flif_v1/classification): python fw_identity_nc.py save|compare <file>
import argparse, os, sys
import torch
sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster

mode, out = sys.argv[1], sys.argv[2]
HERE = os.getcwd()


def find(path, **kv):
    hits = [ln for ln in open(path) if ln.startswith('RESULT') and all(f' {k}={v} ' in ln for k, v in kv.items())]
    if len(hits) != 1:
        raise RuntimeError(f'{len(hits)} RESULT lines for {kv} in {path}')
    return dict(t.split('=', 1) for t in hits[0].split()[1:] if '=' in t)['dir']


res = {}
for cond in ('nsmt4q_pop_lif_msn_sde', 'nsmt4q_pop_flif_msn_sde', 'spkall_lif'):
    run = find(os.path.join(HERE, '..', 'forecasting', 'results', 'flif-v1-ett-r1', 'raw.txt'), cond=cond, ds='ETTh2', pl=96, seed=42)
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = argparse.Namespace(**dict(cfg, device='cuda'))
    x = torch.randn(4, a.seq_len, 7, generator=torch.Generator().manual_seed(5)).cuda()
    y = torch.randn(4, a.pred_len, 7, generator=torch.Generator().manual_seed(6)).cuda()
    torch.manual_seed(0); m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
    res[f'ett/{cond}/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    o = m(x); (o - y).abs().mean().backward()
    res[f'ett/{cond}/train_out'] = o.detach().cpu()
    res[f'ett/{cond}/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
    functional.reset_net(m)
    torch.manual_seed(0); m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
    m.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu')); m.eval(); m.train_mode = 'testing'
    with torch.no_grad():
        functional.reset_net(m); res[f'ett/{cond}/eval_out'] = m(x).detach().cpu(); functional.reset_net(m)
for cond in ('nsmt4q_pop_lif_msn_sde', 'nsmt4q_pop_flif_msn_sde'):
    run = find(os.path.join(HERE, 'results', 'flif-v1-mitbih-r1', 'raw.txt'), cond=cond, seed=42)
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = argparse.Namespace(**dict(cfg, device='cuda'))
    x = torch.rand(4, 1, 187, generator=torch.Generator().manual_seed(7)).cuda()
    lab = torch.tensor([0, 1, 2, 4]).cuda()
    torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
    res[f'mb/{cond}/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    z = m(x)[0]
    torch.nn.functional.cross_entropy(z, lab).backward()
    res[f'mb/{cond}/logits'] = z.detach().cpu()
    res[f'mb/{cond}/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
    functional.reset_net(m)
    torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
    w = torch.load(os.path.join(run, 'model_state', 'best+model.pt'), map_location='cuda')
    rr = m.load_state_dict(w, strict=False)
    if rr.missing_keys or rr.unexpected_keys:
        raise RuntimeError(f'MIT-BIH weights do not fit: {rr}')
    m.eval(); m.train_mode = 'testing'
    with torch.no_grad():
        functional.reset_net(m); o = m(x); res[f'mb/{cond}/eval_out'] = (o[0] if isinstance(o, tuple) else o).detach().cpu(); functional.reset_net(m)

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
