# usage (classification folder): python fw_identity_nshuf.py save|compare <file.pt>
# MODI 2026-11-02: default-path identity around the --neo_shuffle change. Saves (or compares) init state_dicts, train-mode outputs
# and every gradient of fresh CinC-size dual-path models, and eval logits of a real query checkpoint, on fixed inputs.
import glob, os, sys
from types import SimpleNamespace as NS
import torch
sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
mode, out = sys.argv[1], sys.argv[2]
run = glob.glob('log/flif-v1-rhythm-query-cinc/CINC2017/*/*nsmt4q_pop_lif+lr0.001+seed42+*/seed42_*_f0')[0]
cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
variants = {'query_pop_lif': {}, 'replace_lif': dict(hippo_xattn_mode='replace', neo_neuron='lif', pop_soma_init=None),
            'cat_lif': dict(hippo_xattn_mode='cat', neo_neuron='lif', pop_soma_init=None),
            'query_lif_nomem': dict(neo_neuron='lif', pop_soma_init=None, neo_memoryless=True)}
x = torch.randn(4, 1, 3750, generator=torch.Generator().manual_seed(5)).cuda()
y = torch.tensor([0, 1, 2, 3]).cuda()
res = {}
for k, ov in variants.items():
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
m.load_state_dict(torch.load(os.path.join(run, 'model_state', 'best+model.pt'), map_location='cpu')); m.eval(); m.train_mode = 'testing'
with torch.no_grad():
    functional.reset_net(m); o = m(x); res['ckpt/eval_logit'] = (o[0] if isinstance(o, tuple) else o).detach().cpu(); functional.reset_net(m)
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
