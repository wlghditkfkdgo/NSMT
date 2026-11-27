# usage (classification folder): python fw_identity_lifsleep.py save|compare <file.pt>
# MODI 2026-11-27 (user decision 11-27: plain LIF / f-LIF batch): default-path identity around the change that lets --neo_readout mem /
# --aux_past rand run with a plain LIF / f-LIF Neocortex (neurons.py analog 'pre' without a soma, model.py permission). For fresh
# Sleep-EDF-size models of every runner base (Spikformer(DE), query / key-value form with LIF, f-LIF, pop LIF, pop f-LIF) on a fixed
# input: init state_dict, train-mode logits and every gradient of a CE loss, eval logits.
import sys

import torch
from spikingjelly.clock_driven import functional

from config import Config, parse_arguments
from load_model import LOAD_MODEL

mode, out = sys.argv[1], sys.argv[2]
COMMON = ("--data SLEEPEDF --seed 42 --log_dir /tmp/claude-1001/fw_lifsleep_cfg --gating attn -s -nd 0 -emb 384 -nh 8 --mlp_ratios 1 "
          "--layers 1 --time_layers 2 --keep_ratio 0.15 --alpha 0 --tau 2.0 --no-bias --patch_size 1 --token_step 1 --head_pool last "
          "--ssa_pe 0 --fold 0 --ctx_epochs 16")
Q = "--model myModel --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0 --neo_res ms --neuron_all lif"
MODELS = {'spkall_lif': "--model myModel --hippo_ssa --hippo_T 4 --neuron_all lif",
          'q_lif': f"{Q} --neo_neuron lif --hippo_xattn_mode query",
          'q_flif': f"{Q} --neo_neuron flif --neo_alpha 0.5 --hippo_xattn_mode query",
          'kv_lif': f"{Q} --neo_neuron lif",
          'q_pop_lif': f"{Q} --neo_neuron pop_lif --pop_soma_init 0.5 --hippo_xattn_mode query",
          'q_pop_flif': f"{Q} --neo_neuron pop_flif --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_xattn_mode query",
          'kv_pop_lif': f"{Q} --neo_neuron pop_lif --pop_soma_init 0.5",
          'q_lif_nml': f"{Q} --neo_neuron lif --hippo_xattn_mode query --neo_memoryless"}
x = torch.randn(8, 60, 16, generator=torch.Generator().manual_seed(5)).cuda()
y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).cuda()
res = {}
for k, cf in MODELS.items():
    sys.argv = ['x'] + (COMMON + ' ' + cf).split()
    a = Config(); a.set_args(parse_arguments()); a.num_classes, a.num_channels, a.seq_len = 5, 60, 16
    torch.manual_seed(0); m = LOAD_MODEL['myModel'](a, True).cuda()
    res[k + '/state'] = {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    o = m(x); logit = o[0] if isinstance(o, tuple) else o
    torch.nn.functional.cross_entropy(logit, y).backward()
    res[k + '/train_logit'] = logit.detach().cpu()
    res[k + '/grads'] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
    m.eval(); m.train_mode = 'testing'; functional.reset_net(m)
    with torch.no_grad():
        o = m(x)
    res[k + '/eval_logit'] = (o[0] if isinstance(o, tuple) else o).detach().cpu()
    functional.reset_net(m)
if mode == 'save':
    torch.save(res, out); print('saved', len(res), 'entries')
else:
    ref = torch.load(out); bad = []
    def same(p, q):
        if isinstance(p, dict):
            return p.keys() == q.keys() and all(torch.equal(p[i], q[i]) for i in p)
        return torch.equal(p, q)
    for key in ref:
        ok = key in res and same(ref[key], res[key])
        bad += [] if ok else [key]
    print(f'{len(ref) - len(bad)}/{len(ref)} identical' + (f'; DIFFER: {bad}' if bad else ''))
    sys.exit(1 if bad or set(res) != set(ref) else 0)
