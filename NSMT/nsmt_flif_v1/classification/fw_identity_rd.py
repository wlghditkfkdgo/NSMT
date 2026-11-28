# usage (classification folder): python fw_identity_rd.py save|compare <file.pt>
# MODI 2026-11-27 (approved plan 1-snuggly-galaxy.md, step 1): default-path identity around the changes for --neo_readout memd,
# the past-reconstruction aux on ETT (past_pair refactor, Forecaster / train_forecast --past_aux_w) and the rd_* runner family.
# Fresh models built from the r1 reference configs (ETTh2 H96 seed 42 and MIT-BIH seed 42: query pop LIF msn SpikeDE and
# Spikformer(DE)), with the Neocortex neuron switched to plain LIF / f-LIF, and the existing --neo_readout mem / --aux_past rand
# variants (MIT-BIH and Sleep-size; the aux draws its lags from the CUDA RNG, seeded here): init state_dict, train-mode output and
# every gradient, eval output, on fixed inputs.
import os
import sys
from types import SimpleNamespace as NS

import torch
from spikingjelly.clock_driven import functional

sys.path.insert(0, os.getcwd())
from config import Config, parse_arguments
from forecast_wrap import Forecaster
from load_model import LOAD_MODEL

mode, out = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))


def find(path, cond, seed='42', **kw):
    for ln in open(path):
        if ln.startswith('RESULT') and f' cond={cond} ' in ln and f' seed={seed} ' in ln and all(f' {k}={v} ' in ln for k, v in kw.items()):
            return dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)['dir']
    raise RuntimeError((path, cond, kw))


NEU = {'pop_lif': {}, 'lif': dict(neo_neuron='lif', pop_soma_init=None), 'flif': dict(neo_neuron='flif', neo_alpha=0.5, pop_soma_init=None)}
res = {}

ETT = os.path.join(HERE, '..', 'forecasting', 'results', 'flif-v1-ett-r1', 'raw.txt')
xe = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(5)).cuda()
ye = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).cuda()
for base, variants in (('nsmt4q_pop_lif_msn_sde', [(f'ett_{n}{"_mem" if m else ""}', dict(NEU[n], **({'neo_readout': 'mem'} if m else {})))
                                                   for n in NEU for m in (0, 1)]),
                       ('spkall_lif', [('ett_spkall', {})])):
    cfg = torch.load(os.path.join(find(ETT, base, ds='ETTh2', pl='96'), 'model_state', 'config.pt'), map_location='cpu')
    for key, ov in variants:
        a = NS(**dict(cfg, device='cuda', **ov))
        torch.manual_seed(0); m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
        res[key + '/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
        m.train(); m.train_mode = 'training'; functional.reset_net(m)
        o = m(xe); torch.nn.functional.l1_loss(o, ye).backward()
        res[key + '/train_out'] = o.detach().cpu()
        res[key + '/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
        m.eval(); m.train_mode = 'testing'; functional.reset_net(m)
        with torch.no_grad():
            res[key + '/eval_out'] = m(xe).detach().cpu()
        functional.reset_net(m)

MB = os.path.join(HERE, 'results', 'flif-v1-mitbih-r1', 'raw.txt')
xm = torch.randn(8, 1, 187, generator=torch.Generator().manual_seed(5)).cuda()
ym = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).cuda()


def cls_entry(key, m, x, y, aux_w):
    res[key + '/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    torch.manual_seed(1)                                                   # the aux lags come from the CUDA RNG
    o = m(x); logit = o[0] if isinstance(o, tuple) else o
    loss = torch.nn.functional.cross_entropy(logit, y)
    if aux_w:
        loss = loss * (1 - aux_w) + ((o[3] - o[4]) ** 2).mean() * aux_w
        res[key + '/aux_pair'] = (o[3].detach().cpu(), o[4].detach().cpu())
    loss.backward()
    res[key + '/train_logit'] = logit.detach().cpu()
    res[key + '/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
    m.eval(); m.train_mode = 'testing'; functional.reset_net(m)
    with torch.no_grad():
        o = m(x)
    res[key + '/eval_logit'] = (o[0] if isinstance(o, tuple) else o).detach().cpu()
    functional.reset_net(m)


for base, variants in (('nsmt4q_pop_lif_msn_sde', [(f'mb_{n}{s}', dict(NEU[n], **ov), w) for n in NEU
                                                   for s, ov, w in (('', {}, 0), ('_mem', {'neo_readout': 'mem'}, 0),
                                                                    ('_acc', {'aux_past': 'rand', 'alpha': 0.1}, 0.1))]),
                       ('spkall_lif', [('mb_spkall', {}, 0)])):
    cfg = torch.load(os.path.join(find(MB, base), 'model_state', 'config.pt'), map_location='cpu')
    for key, ov, w in variants:
        a = NS(**dict(cfg, device='cuda', **ov))
        torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
        cls_entry(key, m, xm, ym, w)

COMMON = ("--data SLEEPEDF --seed 42 --log_dir /tmp/claude-1001/fw_rd_cfg --gating attn -s -nd 0 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 "
          "--time_layers 2 --keep_ratio 0.15 --tau 2.0 --no-bias --patch_size 1 --token_step 1 --head_pool last --ssa_pe 0 --fold 0 "
          "--ctx_epochs 16 --model myModel --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0 --neo_res ms --neuron_all lif "
          "--hippo_xattn_mode query --neo_neuron lif")
xs = torch.randn(8, 60, 16, generator=torch.Generator().manual_seed(5)).cuda()
for key, extra, w in (('sleep_lif_acc', '--aux_past rand --alpha 0.1', 0.1), ('sleep_lif_mem_acc', '--neo_readout mem --aux_past rand --alpha 0.1', 0.1)):
    sys.argv = ['x'] + (COMMON + ' ' + extra).split()
    a = Config(); a.set_args(parse_arguments()); a.num_classes, a.num_channels, a.seq_len = 5, 60, 16
    torch.manual_seed(0); m = LOAD_MODEL['myModel'](a, True).cuda()
    cls_entry(key, m, xs, ym, w)

if mode == 'save':
    torch.save(res, out); print('saved', len(res), 'entries')
else:
    ref = torch.load(out); bad = []

    def same(p, q):
        if isinstance(p, dict):
            return p.keys() == q.keys() and all(torch.equal(p[i], q[i]) for i in p)
        if isinstance(p, tuple):
            return len(p) == len(q) and all(torch.equal(i, j) for i, j in zip(p, q))
        return torch.equal(p, q)
    bad = [k for k in ref if k not in res or not same(ref[k], res[k])]
    print(f'{len(ref) - len(bad)}/{len(ref)} identical' + (f'; DIFFER: {bad}' if bad else ''))
    sys.exit(1 if bad or set(res) != set(ref) else 0)
