# usage (classification folder): python fw_identity_ms.py save|compare <file.pt>
# MODI 2026-11-08: default-path identity around the membrane shortcut change (--res_mode / --neo_res). Saves (or compares) init
# state_dicts, train-mode outputs and every gradient of fresh models on fixed inputs, and eval outputs of real checkpoints:
#   MIT-BIH-size query pop LIF / pop f-LIF (fresh), the query pop LIF MIT-BIH checkpoint (best+model.pt)
#   ETTh2 H96 / H720 and ETTm2 H96 query pop LIF forecasters (fresh + checkpoint.pth; ETTm2 has 2 Neocortex layers)
#   ETTh2 H96 Spikformer-style and Neo-mixer (mixall_pop_lif_bidir) forecasters (fresh; they share Embedding / SpikLinearLayer / MLP)
import glob, os, sys
from types import SimpleNamespace as NS
import torch
sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster
mode, out = sys.argv[1], sys.argv[2]
mb = glob.glob('log/flif-v1-mitbih-query/MITBIH/*/*+nsmt4q_pop_lif_lr0.01_v0+seed42+model+*/seed42_*')
ett = {('ETTh2', pl): glob.glob(f'../forecasting/log/flif-v1-ett-query/ETTh2/p{pl}/*/*+nsmt4q_pop_lif+seed42+*/seed42_*') for pl in (96, 720)}
ett[('ETTm2', 96)] = glob.glob('../forecasting/log/flif-v1-ett-query-m2/ETTm2/p96/*/*+nsmt4q_pop_lif+seed42+*/seed42_*')
spk = glob.glob('../forecasting/log/flif-v1-gen-ett/ETTh2/p96/*/*+spk_ssa+seed42+*/seed42_*')
mix = glob.glob('../forecasting/log/flif-v1-ett-nall/ETTh2/p96/*/*+mixall_pop_lif_bidir+seed42+*/seed42_*')
assert len(mb) == 1 and all(len(v) == 1 for v in ett.values()) and len(spk) == 1 and len(mix) >= 1, (mb, ett, spk, mix)
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


def forecaster(run, key, pl, ckpt):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda'))
    xe = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).cuda()
    ye = torch.randn(4, pl, 7, generator=torch.Generator().manual_seed(7)).cuda()
    torch.manual_seed(0); f = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, pl, bias=a.bias).cuda()
    res[f'{key}/state'] = {n: t.detach().cpu().clone() for n, t in f.state_dict().items()}
    f.train(); f.train_mode = 'training'; functional.reset_net(f)
    o = f(xe); torch.nn.functional.l1_loss(o, ye).backward()
    res[f'{key}/train_out'] = o.detach().cpu()
    res[f'{key}/grads'] = {n: p.grad.detach().cpu().clone() for n, p in f.named_parameters() if p.grad is not None}
    functional.reset_net(f)
    if ckpt:
        torch.manual_seed(0); f = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, pl, bias=a.bias).cuda()
        f.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu')); f.eval(); f.train_mode = 'testing'
        with torch.no_grad():
            functional.reset_net(f); res[f'{key}_ckpt/eval_out'] = f(xe).detach().cpu()
            if f.kind != 'mixer':
                functional.reset_net(f)
                z = torch.randn(28, 1, 96, generator=torch.Generator().manual_seed(8)).cuda()
                res[f'{key}_ckpt/step_features'] = f.step_features(z).detach().cpu()   # used by energy_recount_ett.py


for (ds, pl), run in ett.items():
    forecaster(run[0], f'{ds}{pl}', pl, True)
forecaster(spk[0], 'spk_ssa96', 96, False)
forecaster(sorted(mix)[0], 'mixall96', 96, False)

if mode == 'save':
    torch.save(res, out); print('saved', len(res), 'entries')
else:
    ref = torch.load(out)
    bad = []
    if ref.keys() != res.keys():
        bad.append(f'keys {sorted(set(ref) ^ set(res))}')
    for k, v in ref.items():
        w = res.get(k)
        if w is None:
            continue
        if isinstance(v, dict):
            same = v.keys() == w.keys() and all(torch.equal(v[n], w[n]) for n in v)
        else:
            same = torch.equal(v, w)
        if not same: bad.append(k)
    print('identical' if not bad else f'DIFFERENT: {bad}', f'({len(ref)} entries)')
