"""MODI 2026-11-26 ((나) Sleep-EDF SNN comparison): checks of the SNN data path and the new model options. Fail closed.
usage (classification folder): python check_sleep.py
"""
import sys
from types import SimpleNamespace as NS

import numpy as np
import torch

import order_check as O
import rhythm_data as rd

FAILS = []


def ok(name, cond, info=''):
    print(f'{"PASS" if cond else "FAIL"} {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in (0, 3):
        d = rd.build(NS(data='SLEEPEDF', fold=fold, ctx_epochs=16, patch_size=1, crop_tokens=0), 42)
        o = O.Data('SLEEPPSD', O.parse_cond('SLEEPPSD', 'gru_orig_L16'), fold, dev)
        same_split = all(np.array_equal(getattr(d[k], 'idx'), o.split[{'train': 'train', 'val': 'val', 'test': 'test'}[k]]) for k in ('train', 'val', 'test'))
        ok(f'a1 fold {fold}: SNN splits = diagnostic SLEEPPSD splits', same_split, str(d['info']['split_sizes']))
        i = int(d['test'].idx[7])
        x, y, j = d['test'][7]
        xo, yo = o.tokens(np.array([i]))
        ok(f'a2 fold {fold}: SNN input [60 bins, 16 epochs] = diagnostic tokens transposed, same label', tuple(x.shape) == (60, 16)
           and np.allclose(x.numpy(), xo[0].cpu().numpy().T, atol=1e-5) and y == int(yo[0]) and j == i)
        ok(f'a3 fold {fold}: shapes for the model', (d['num_classes'], d['num_channels'], d['seq_len']) == (5, 60, 16))
    # b. model options: 16 tokens = 16 epochs (patch 1, step 1, no padding); head 'last' reads the last token
    from load_model import LOAD_MODEL
    from spikingjelly.clock_driven import functional
    a = NS(model='myModel', num_classes=5, num_channels=60, seq_len=16, patch_size=1, token_step=1, head_pool='last', ssa_pe=0,
           hippo_ssa=True, hippo_T=4, neuron_all='lif', embed_dim=384, num_heads=8, mlp_ratios=1.0, num_layers=1, time_num_layers=2,
           keep_ratio=0.15, gating='attn', bias=False, tau=2.0, alpha=0.0, attn='SSA', spk_encoding=False, device='cuda')
    try:
        m = LOAD_MODEL['myModel'](a, True).to(dev)
    except Exception as e:                                              # the loader may need more fields; fall back to the CLI path
        m = None
        print('    (direct build failed, using parse_arguments)', type(e).__name__, str(e)[:80])
    if m is None:
        from config import Config, parse_arguments
        sys.argv = ['x'] + ('--data SLEEPEDF --seed 42 --log_dir /tmp/claude-1001/check_sleep_cfg --gating attn -s -nd 0 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 '
                            '--time_layers 2 --keep_ratio 0.15 --alpha 0 --tau 2.0 --no-bias --patch_size 1 --token_step 1 --head_pool last --ssa_pe 0 '
                            '--fold 0 --ctx_epochs 16 --model myModel --hippo_ssa --hippo_T 4 --neuron_all lif').split()
        a = Config(); a.set_args(parse_arguments()); a.num_classes, a.num_channels, a.seq_len = 5, 60, 16
        m = LOAD_MODEL['myModel'](a, True).to(dev)
    ok('b1 16 tokens, step 1, no end padding', m.num_patches == 16 and m.stride == 1 and isinstance(m.padding_patch_layer, torch.nn.Identity))
    xb = torch.randn(2, 60, 16, device=dev)
    hippo, neo = m._tokens_mc(xb)
    ok('b2 token n = epoch n (all 60 bins)', torch.equal(hippo[0], xb.transpose(1, 2)))
    z = torch.randn(4, 2, 16, 384, device=dev)
    ok('b3 head_pool last reads the last token', torch.equal(m._pool(z), z[:, :, -1]))
    # b4 (first version used eval mode at initialisation, where no spike reaches the head and the output is 0 for any small change;
    # corrected 11-26): train mode (BatchNorm batch statistics), 32 samples, the target epoch / the first epoch replaced by new values.
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    xb = torch.randn(32, 60, 16, device=dev)
    def out(z):
        with torch.no_grad():
            o = m(z); functional.reset_net(m)
        return o[0] if isinstance(o, tuple) else o
    o1 = out(xb)
    x2 = xb.clone(); x2[..., -1] = 3 * torch.randn(32, 60, device=dev)
    x3 = xb.clone(); x3[..., 0] = 3 * torch.randn(32, 60, device=dev)
    d_last, d_first = (o1 - out(x2)).abs().max().item(), (o1 - out(x3)).abs().max().item()
    ok('b4 output reacts to the target epoch and (through attention) to the first context epoch', d_last > 1e-4 and d_first > 1e-4,
       f'last {d_last:.3g}, first {d_first:.3g}')
    # c. MODI 2026-11-27: context shuffle (--ctx_shuffle) -- same epochs, target last, a new order per sample and per training epoch,
    # a fixed order for validation / test; without the flag the samples are unchanged
    d0 = rd.build(NS(data='SLEEPEDF', fold=1, ctx_epochs=16, patch_size=1, crop_tokens=0, ctx_shuffle=False), 42)
    d1 = rd.build(NS(data='SLEEPEDF', fold=1, ctx_epochs=16, patch_size=1, crop_tokens=0, ctx_shuffle=True), 42)
    a0, a1 = d0['test'][5][0].numpy(), d1['test'][5][0].numpy()
    same_set = sorted(map(tuple, a0[:, :-1].T.round(6))) == sorted(map(tuple, a1[:, :-1].T.round(6)))
    ok('c1 shuffled test sample: same 15 context epochs, target last, order changed', same_set and np.array_equal(a0[:, -1], a1[:, -1])
       and not np.array_equal(a0, a1))
    ok('c2 test order fixed (two reads equal)', np.array_equal(d1['test'][5][0].numpy(), a1))
    t0_ = d1['train'][3][0].numpy(); d1['train'].set_epoch(0); t1_ = d1['train'][3][0].numpy(); d1['train'].set_epoch(1); t2_ = d1['train'][3][0].numpy()
    ok('c3 training order changes with the epoch', not np.array_equal(t1_, t2_) and np.array_equal(t1_[:, -1], t2_[:, -1]))
    perms = {tuple(np.argsort(d1['test'][k][0].numpy()[0, :-1])) for k in range(40)}
    ok('c4 orders differ across samples', len(perms) > 35, f'{len(perms)} of 40')
    d0['train'].set_epoch(3)
    ok('c5 without --ctx_shuffle samples are unchanged (also after set_epoch)', np.array_equal(d0['train'][3][0].numpy(), rd.build(
        NS(data='SLEEPEDF', fold=1, ctx_epochs=16, patch_size=1, crop_tokens=0), 42)['train'][3][0].numpy()))
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
