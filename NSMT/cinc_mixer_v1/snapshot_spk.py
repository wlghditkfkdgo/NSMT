"""Default-path snapshot for edit checks (2026-10-30, PROJECT_LOG, exp/f-lif-pop-v3; moved from the 10-29 scratch snapshot.py and
extended with the SpkMixer). Saves logits, testing output and state_dict of
  - NeoMixer 6 neurons x {or, linsd} and the Spikformer-style myModel (hippo_ssa)           (13, as the 10-29 check s0)
  - SpkMixer 6 neurons x {or, sdor, sdlin, linmx} with the OR residual                      (24: every B2 / C condition)
  - SpkMixer LIF x {or, sdor, sdlin, linmx} x {IAND, ADD}                                    (8)
on one fixed random batch, every model built at seed 0. Run once before and once after an edit; check_mux.py m0 compares
the two files bitwise.
usage: python snapshot_spk.py <out.pt>
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from types import SimpleNamespace as NS                                   # noqa: E402
from spikingjelly.clock_driven import functional                          # noqa: E402
from config import set_random_seed                                        # noqa: E402
from load_model import LOAD_MODEL                                         # noqa: E402

KW = dict(keep_ratio=0.15, gating='attn', embed_dim=384, num_heads=8, mlp_ratios=1.0, num_layers=1, time_num_layers=2, bias=False,
          tau=2.0, spk_encoding=False, neo_alpha=0.5, pop_taus='4,8,16,32', neo_tau=2.0, fde_method='pred', fde_step=1.0, fde_memory=0,
          mixer_bidir=True, pos_emb=False, tau_calib=True, patch_size=64, seq_len=3750, num_channels=1, num_classes=4,
          hippo_T=4, hippo_ssa=False, hippo_xattn=False, ssa_pe=1, attn_bn=False, neo_neuron='sj_lif', neo_where='both', neo_bidir=False,
          roll_mode='none', neo_recurrent=False, neo_rec_gate=False, neo_legacy=False, no_neo=False, hippo_only=False, grad_clip=0.,
          alpha=0.1, stream_rev=False, neo_stream_fuse='none', pop_soma_init=None, drop_rate=0.)
NEURONS = (('lif', None), ('flif', None), ('pop_lif', 0.5), ('pop_flif', 1.0), ('popg_lif', None), ('popg_flif', None))
SPK_COMBS = ('or', 'sdor', 'sdlin', 'linmx')


def snap(m, x):
    m.train(); m.train_mode = 'testing'
    with torch.no_grad():
        z, h = m(x)
    functional.reset_net(m)
    return dict(z=z.cpu(), h=h.cpu(), sd={k: v.cpu() for k, v in m.state_dict().items()})


def main():
    out = sys.argv[1]
    set_random_seed(0)
    x = torch.randn(4, 1, 3750).cuda()
    res = {}
    for mixer, psi in NEURONS:
        for comb in ('or', 'linsd'):
            set_random_seed(0)
            m = LOAD_MODEL['neoMixer'](NS(**dict(KW, model='neoMixer', mixer=mixer, mixer_comb=comb, pop_soma_init=psi)), True).cuda()
            res[f'neoMixer/{mixer}/{comb}'] = snap(m, x); del m
    set_random_seed(0)
    m = LOAD_MODEL['myModel'](NS(**dict(KW, model='myModel', hippo_ssa=True, mixer='lif', mixer_comb='or')), True).cuda()
    res['myModel/spk_ssa'] = snap(m, x); del m
    todo = [(n, psi, c, 'OR') for n, psi in NEURONS for c in SPK_COMBS] + [('lif', None, c, cf) for c in SPK_COMBS for cf in ('IAND', 'ADD')]
    for mixer, psi, comb, cf in todo:
        set_random_seed(0)
        m = LOAD_MODEL['spkMixer'](NS(**dict(KW, model='spkMixer', mixer=mixer, mixer_comb=comb, connect_f=cf, pop_soma_init=psi)), True).cuda()
        res[f'spkMixer/{mixer}/{comb}/{cf}'] = snap(m, x); del m
    torch.save(res, out)
    print('saved', len(res), out)


if __name__ == '__main__':
    main()
