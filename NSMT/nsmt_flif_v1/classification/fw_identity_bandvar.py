# usage (classification folder): python fw_identity_bandvar.py save|compare <file.pt>
# MODI 2026-11-13: identity of the frequency-band paths (--neo_band freq / same / rev, the runs of 2026-11-13) around the STFT /
# soma-less change. Saves (or compares) init state_dicts, train-mode outputs and every gradient of fresh models (query pop LIF and
# pop f-LIF; MIT-BIH and ETTm2 H96 sizes), and eval outputs / step_features of the trained pop LIF checkpoints (MIT-BIH, ETTm2 H96,
# ETTh2 H96; seed 42). The paths without --neo_band are covered by fw_identity_ms.py (28 entries) and fw_identity_band.py (38).
import glob, os, sys
from types import SimpleNamespace as NS
import torch
sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster
mode, out = sys.argv[1], sys.argv[2]
res = {}
x = torch.randn(8, 1, 187, generator=torch.Generator().manual_seed(5)).cuda()
y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).cuda()
xe = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).cuda()
ye = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(7)).cuda()
z = torch.randn(28, 1, 96, generator=torch.Generator().manual_seed(8)).cuda()


def one(p):
    g = glob.glob(p)
    assert len(g) == 1, (p, g)
    return g[0]


for n in ('pop_lif', 'pop_flif'):
    mb = one(f'log/flif-v1-mitbih-ms/MITBIH/*/*+nsmt4q_{n}_msn_lr0.01_v0+seed42+model+*/seed42_*')
    em = one(f'../forecasting/log/flif-v1-ett-ms/ETTm2/p96/*/*+nsmt4q_{n}_msn+seed42+*/seed42_*')
    for band in ('freq', 'same', 'rev'):
        cfg = torch.load(os.path.join(mb, 'model_state', 'config.pt'), map_location='cpu')
        a = NS(**dict(cfg, device='cuda', neo_band=band)); k = f'mb_{n}_{band}'
        torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
        res[k + '/state'] = {q: t.detach().cpu().clone() for q, t in m.state_dict().items()}
        m.train(); m.train_mode = 'training'; functional.reset_net(m)
        o = m(x); logit = o[0] if isinstance(o, tuple) else o
        torch.nn.functional.cross_entropy(logit, y).backward()
        res[k + '/train_logit'] = logit.detach().cpu()
        res[k + '/grads'] = {q: p.grad.detach().cpu().clone() for q, p in m.named_parameters() if p.grad is not None}
        functional.reset_net(m)
        cfg = torch.load(os.path.join(em, 'model_state', 'config.pt'), map_location='cpu')
        a = NS(**dict(cfg, device='cuda', neo_band=band)); k = f'ETTm296_{n}_{band}'
        torch.manual_seed(0); f = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, 96, bias=a.bias).cuda()
        res[k + '/state'] = {q: t.detach().cpu().clone() for q, t in f.state_dict().items()}
        f.train(); f.train_mode = 'training'; functional.reset_net(f)
        o = f(xe); torch.nn.functional.l1_loss(o, ye).backward()
        res[k + '/train_out'] = o.detach().cpu()
        res[k + '/grads'] = {q: p.grad.detach().cpu().clone() for q, p in f.named_parameters() if p.grad is not None}
        functional.reset_net(f)
for band in ('freq', 'same', 'rev'):                                   # trained pop LIF checkpoints of the band runs
    run = one(f'log/flif-v1-mitbih-band/MITBIH/*/*+nsmt4q_pop_lif_msn_{band}_lr0.01_v0+seed42+model+*/seed42_*')
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda')); assert a.neo_band == band
    torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
    m.load_state_dict(torch.load(os.path.join(run, 'model_state', 'best+model.pt'), map_location='cpu')); m.eval(); m.train_mode = 'testing'
    with torch.no_grad():
        functional.reset_net(m); o = m(x); res[f'mb_pop_lif_{band}_ckpt/eval_logit'] = (o[0] if isinstance(o, tuple) else o).detach().cpu()
    for ds in ('ETTm2', 'ETTh2'):
        run = one(f'../forecasting/log/flif-v1-ett-band/{ds}/p96/*/*+nsmt4q_pop_lif_msn_{band}+seed42+*/seed42_*')
        cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
        a = NS(**dict(cfg, device='cuda')); assert a.neo_band == band
        torch.manual_seed(0); f = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, 96, bias=a.bias).cuda()
        f.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu')); f.eval(); f.train_mode = 'testing'
        with torch.no_grad():
            functional.reset_net(f); res[f'{ds}96_pop_lif_{band}_ckpt/eval_out'] = f(xe).detach().cpu()
            functional.reset_net(f); res[f'{ds}96_pop_lif_{band}_ckpt/step_features'] = f.step_features(z).detach().cpu()

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
            same = v.keys() == w.keys() and all(torch.equal(v[q], w[q]) for q in v)
        else:
            same = torch.equal(v, w)
        if not same: bad.append(k)
    print('identical' if not bad else f'DIFFERENT: {bad}', f'({len(ref)} entries)')
    sys.exit(1 if bad else 0)
