# usage (classification folder): python fw_identity_head.py save|compare <file.pt>
# MODI 2026-11-04: default-path identity around the H2 (--head_pool) / H6 (--fc_head) change. Saves (or compares) init state_dicts,
# train-mode outputs and every gradient of fresh MIT-BIH-size query models and ETTh2-size query forecasters, and eval outputs of
# real query checkpoints (MIT-BIH best+model.pt, ETTh2 H96 / H720 checkpoint.pth), on fixed inputs.
import glob, os, sys
from types import SimpleNamespace as NS
import torch
sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster
mode, out = sys.argv[1], sys.argv[2]
mb = glob.glob('log/flif-v1-mitbih-query/MITBIH/*/*+nsmt4q_pop_lif_lr0.01_v0+seed42+model+*/seed42_*')
ett = {pl: glob.glob(f'../forecasting/log/flif-v1-ett-query/ETTh2/p{pl}/*/*+nsmt4q_pop_lif+seed42+*/seed42_*') for pl in (96, 720)}
assert len(mb) == 1 and all(len(v) == 1 for v in ett.values()), (mb, ett)
res = {}

cfg = torch.load(os.path.join(mb[0], 'model_state', 'config.pt'), map_location='cpu')
x = torch.randn(8, 1, 187, generator=torch.Generator().manual_seed(5)).cuda()
y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).cuda()
for k, ov in {'mb_query_pop_lif': {}, 'mb_query_pop_flif': dict(neo_neuron='pop_flif', pop_soma_init=1.0)}.items():
    a = NS(**dict(cfg, device='cuda', **ov))
    torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
    res[k + '/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    o = m(x); logit = o[0] if isinstance(o, tuple) else o
    torch.nn.functional.cross_entropy(logit, y).backward()
    res[k + '/train_logit'] = logit.detach().cpu()
    res[k + '/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
    functional.reset_net(m)
a = NS(**dict(cfg, device='cuda'))
torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
m.load_state_dict(torch.load(os.path.join(mb[0], 'model_state', 'best+model.pt'), map_location='cpu')); m.eval(); m.train_mode = 'testing'
with torch.no_grad():
    functional.reset_net(m); o = m(x); res['mb_ckpt/eval_logit'] = (o[0] if isinstance(o, tuple) else o).detach().cpu()

for pl, run in ett.items():
    cfg = torch.load(os.path.join(run[0], 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda'))
    xe = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).cuda()
    ye = torch.randn(4, pl, 7, generator=torch.Generator().manual_seed(7)).cuda()
    torch.manual_seed(0); f = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, pl, bias=a.bias).cuda()
    res[f'ett{pl}/state'] = {n: t.detach().cpu().clone() for n, t in f.state_dict().items()}
    f.train(); f.train_mode = 'training'; functional.reset_net(f)
    o = f(xe); torch.nn.functional.l1_loss(o, ye).backward()
    res[f'ett{pl}/train_out'] = o.detach().cpu()
    res[f'ett{pl}/grads'] = {n: p.grad.detach().cpu().clone() for n, p in f.named_parameters() if p.grad is not None}
    functional.reset_net(f)
    torch.manual_seed(0); f = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, pl, bias=a.bias).cuda()
    f.load_state_dict(torch.load(os.path.join(run[0], 'model_state', 'checkpoint.pth'), map_location='cpu')); f.eval(); f.train_mode = 'testing'
    with torch.no_grad():
        functional.reset_net(f); res[f'ett{pl}_ckpt/eval_out'] = f(xe).detach().cpu()

if mode == 'save':
    torch.save(res, out); print('saved', len(res), 'entries')
else:
    ref = torch.load(out)
    bad = []
    for k, v in ref.items():
        w = res[k]
        if isinstance(v, dict):
            same = v.keys() == w.keys() and all(torch.equal(v[n], w[n]) for n in v)
        else:
            same = torch.equal(v, w)
        if not same: bad.append(k)
    print('identical' if not bad else f'DIFFERENT: {bad}', f'({len(ref)} entries)')
