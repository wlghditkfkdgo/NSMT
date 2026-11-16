"""MODI 2026-11-16: fail-closed checks of the input-length test (docs/MODI_PROJECT_LOG.md 2026-11-16, user decision 11-16): the ETT
forecasters at input length 96 / 336 / 720 and the two controls of the msn base model that are new with the Neocortex membrane
shortcut (--neo_res ms with --neo_memoryless / --neo_shuffle). c60a (every earlier path bit-identical) is fw_identity_ms.py +
fw_identity_band.py + fw_identity_bandvar.py.
  c60b sizes: tokens N = (L - P) / stride + 2 = 24 / 84 / 180 for L = 96 / 336 / 720; head = Linear(N * D -> H); the output is
       [B, H, C] and finite; msn / nomem / nshuf have the same parameters (names, shapes) at each length
  c60c data: for each cell the validation and test sets have the same number of windows at every input length, the forecast target of
       window i is the same at every length, and the last 96 input samples of the longer window are the 96-sample window (so the test
       MSE of different lengths is measured on the same targets); the number of training windows is printed
  c60d memoryless + membrane shortcut (evaluation mode, the trained msn weights): the Neocortex output of token i does not change when
       another token changes; it equals the msn model run on that token alone (one-token sequence)
  c60e shuffle + membrane shortcut (evaluation mode, the trained msn weights): the Neocortex output equals the msn model run on the
       permuted tokens with the outputs put back (the fixed evaluation permutation, which is not the identity); the forecast differs
       from the unshuffled msn model whenever the Neocortex fires (printed with the Neocortex firing rates)
  c60f gradients (training mode, input length 336, fresh models): finite everywhere; non-zero on the Neocortex embedding of the shuffle
       control and on the hippocampus of both controls (a fresh memoryless population Neocortex may be silent: printed)
  any failed comparison raises (exit code 1).
usage (classification folder): CUDA_VISIBLE_DEVICES=0 python check_len.py
"""
import glob
import os
import sys
from types import SimpleNamespace as NS

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(HERE, '..', 'forecasting'))
from spikingjelly.clock_driven import functional                        # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from forecast_wrap import Forecaster                                     # noqa: E402
from train_forecast import data_args                                     # noqa: E402
from data_provider.data_factory import data_provider                     # noqa: E402

LENS = (96, 336, 720)
TOK = {96: 24, 336: 84, 720: 180}
CTL = {'msn': {}, 'nomem': dict(neo_memoryless=True), 'nshuf': dict(neo_shuffle=True)}
FAIL = []


def ok(cond, what):
    print(('pass ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        FAIL.append(what)


def one(pattern):
    g = glob.glob(pattern)
    if len(g) != 1:
        raise RuntimeError(f'{pattern}: {g}')
    return g[0]


def run_dir(ds, pl, cond):
    suite = 'flif-v1-gen-ett' if cond == 'spk_ssa' else 'flif-v1-ett-ms'
    return one(f'../forecasting/log/{suite}/{ds}/p{pl}/*/*+{cond}+seed42+*/seed42_*')


def build(run, **over):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda', **over))
    torch.manual_seed(0)
    m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
    functional.reset_net(m)
    return m, a


def binary(t):
    return t.numel() > 0 and bool(((t == 0) | (t == 1)).all())


@torch.no_grad()
def c60b():
    for ds, pl in (('ETTh2', 96), ('ETTm2', 720)):
        for cond in ('spk_ssa', 'nsmt4q_pop_lif_msn', 'nsmt4q_pop_flif_msn'):
            run = run_dir(ds, pl, cond)
            for L in LENS:
                variants = ('msn',) if cond == 'spk_ssa' else tuple(CTL)
                shapes = {}
                for v in variants:
                    m, a = build(run, seq_len=L, **CTL[v])
                    m.eval(); m.train_mode = 'testing'
                    x = torch.randn(2, L, 7, generator=torch.Generator().manual_seed(L)).cuda()
                    y = m(x); functional.reset_net(m)
                    shapes[v] = {k: tuple(p.shape) for k, p in m.named_parameters()}
                    n_par = sum(p.numel() for p in m.parameters())
                    ok(m.num_patches == TOK[L] and m.head.in_features == TOK[L] * a.embed_dim and m.head.out_features == pl
                       and tuple(y.shape) == (2, pl, 7) and bool(torch.isfinite(y).all()),
                       f'c60b {ds} H{pl} {cond} {v} L={L}: tokens {m.num_patches}, head {m.head.in_features} -> {m.head.out_features}, '
                       f'output {tuple(y.shape)} finite, parameters {n_par}')
                    del m
                if len(variants) > 1:
                    ok(shapes['msn'] == shapes['nomem'] == shapes['nshuf'], f'c60b {ds} H{pl} {cond} L={L}: msn / nomem / nshuf have the same parameters')


def c60c():
    for ds in ('ETTh2', 'ETTm2'):
        for pl in (96, 720):
            cfg = torch.load(os.path.join(run_dir(ds, pl, 'nsmt4q_pop_lif_msn'), 'model_state', 'config.pt'), map_location='cpu')
            sets = {}
            for L in LENS:
                a = NS(**dict(cfg, seq_len=L))
                sets[L] = {flag: data_provider(data_args(a, a), flag)[0] for flag in ('train', 'val', 'test')}
            n = {L: {f: len(s) for f, s in sets[L].items()} for L in LENS}
            ok(n[96]['val'] == n[336]['val'] == n[720]['val'] and n[96]['test'] == n[336]['test'] == n[720]['test'] and n[96]['test'] > 0,
               f'c60c {ds} H{pl}: validation windows {n[96]["val"]}, test windows {n[96]["test"]} at every length; '
               f'training windows {n[96]["train"]} / {n[336]["train"]} / {n[720]["train"]} (L 96 / 336 / 720)')
            for flag in ('val', 'test'):
                N = len(sets[96][flag]); idx = sorted({0, 1, N // 3, N // 2, N - 2, N - 1})
                same_y = same_x = True
                for i in idx:
                    x96, y96 = sets[96][flag][i][0], sets[96][flag][i][1]
                    for L in (336, 720):
                        xl, yl = sets[L][flag][i][0], sets[L][flag][i][1]
                        same_y &= bool((torch.as_tensor(yl[-pl:]) == torch.as_tensor(y96[-pl:])).all())
                        same_x &= bool((torch.as_tensor(xl[-96:]) == torch.as_tensor(x96)).all()) and len(xl) == L
                ok(same_y and same_x, f'c60c {ds} H{pl} {flag}: target of window i identical at every length, last 96 input samples == the '
                                      f'96-sample window ({len(idx)} windows incl. first / last)')


@torch.no_grad()
def c60d_e():
    for ds, pl in (('ETTh2', 96), ('ETTm2', 96)):
        for n in ('pop_lif', 'pop_flif'):
            run = run_dir(ds, pl, f'nsmt4q_{n}_msn')
            sd = torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu')
            ms = {}
            for v in CTL:
                m, a = build(run, **CTL[v])
                m.load_state_dict(sd); m.eval(); m.train_mode = 'testing'
                ms[v] = m
            z = torch.randn(14, 1, 96, generator=torch.Generator().manual_seed(11)).cuda()
            _, neo_in = ms['msn'].inner._tokens_mc(z)                  # [N, B', 1, P]
            N, B = neo_in.shape[:2]

            def neo(m, t):
                e, o = m.inner.neo_forward(t, None)
                functional.reset_net(m)
                return e, o
            e0, o0 = neo(ms['msn'], neo_in)
            # c60d memoryless
            em, om = neo(ms['nomem'], neo_in)
            ok(binary(om) and tuple(om.shape) == tuple(o0.shape), f'c60d {ds} H{pl} {n}: memoryless output 0/1, shape {tuple(om.shape)}')
            j = N // 2
            t2 = neo_in.clone(); t2[j] = torch.randn_like(t2[j]) * 3
            e2, o2 = neo(ms['nomem'], t2)
            keep = [i for i in range(N) if i != j]
            ok(bool((o2[keep] == om[keep]).all()) and bool((e2[keep] == em[keep]).all()),
               f'c60d {ds} H{pl} {n}: memoryless outputs of the other {N - 1} tokens unchanged when token {j} changes')
            t3 = neo_in.clone(); t3[j] = torch.randn_like(t3[j]) * 3
            _, o3 = neo(ms['msn'], t3)
            diff_after = int((o3[j + 1:] != o0[j + 1:]).sum())
            print(f'     (msn with memory: {diff_after} output positions after token {j} change; rate msn {float(o0.mean()):.4f}, '
                  f'memoryless {float(om.mean()):.4f})', flush=True)
            mism = 0
            for i in (0, 1, j, N - 1):
                _, oi = neo(ms['msn'], neo_in[i:i + 1])
                mism += int((oi[0] != om[i]).sum())
            ok(mism == 0, f'c60d {ds} H{pl} {n}: memoryless output of token i == msn on that token alone (4 tokens, {mism} positions differ)')
            # c60e shuffle
            perm = ms['nshuf'].inner.neo_shuffle.perms(B, neo_in.device, False)   # [B, N], fixed evaluation permutation
            ok(not bool((perm[0] == torch.arange(N, device=perm.device)).all()) and sorted(perm[0].tolist()) == list(range(N)),
               f'c60e {ds} H{pl} {n}: the evaluation permutation is a permutation of the {N} tokens and not the identity')
            es, os_ = neo(ms['nshuf'], neo_in)
            ep, op = neo(ms['msn'], neo_in[perm[0]])
            back = torch.empty_like(op); back[perm[0]] = op
            eback = torch.empty_like(ep); eback[perm[0]] = ep
            ok(bool((os_ == back).all()) and bool((es == eback).all()) and binary(os_),
               f'c60e {ds} H{pl} {n}: shuffle output == msn on the permuted tokens, put back (0 positions differ)')
            y0 = ms['msn'](torch.randn(2, 96, 7, generator=torch.Generator().manual_seed(12)).cuda()); functional.reset_net(ms['msn'])
            ys = ms['nshuf'](torch.randn(2, 96, 7, generator=torch.Generator().manual_seed(12)).cuda()); functional.reset_net(ms['nshuf'])
            ym = ms['nomem'](torch.randn(2, 96, 7, generator=torch.Generator().manual_seed(12)).cuda()); functional.reset_net(ms['nomem'])
            fires = float(o0.mean()) > 0
            print(f'     Neocortex rate msn {float(o0.mean()):.4f} shuffle {float(os_.mean()):.4f} memoryless {float(om.mean()):.4f}; '
                  f'max |forecast difference| vs msn: shuffle {float((ys - y0).abs().max()):.4g}, memoryless {float((ym - y0).abs().max()):.4g}', flush=True)
            ok((not fires) or bool((os_ != o0).any()), f'c60e {ds} H{pl} {n}: the shuffled order changes the Neocortex output (msn fires: {fires})')
            del ms


def c60f():
    for ds, pl in (('ETTh2', 96), ('ETTm2', 96)):
        for n in ('pop_lif', 'pop_flif'):
            run = run_dir(ds, pl, f'nsmt4q_{n}_msn')
            for v in ('nomem', 'nshuf'):
                m, a = build(run, seq_len=336, **CTL[v])
                m.train(); m.train_mode = 'training'
                x = torch.randn(4, 336, 7, generator=torch.Generator().manual_seed(21)).cuda()
                y = torch.randn(4, pl, 7, generator=torch.Generator().manual_seed(22)).cuda()
                loss = torch.nn.functional.l1_loss(m(x), y); loss.backward(); functional.reset_net(m)
                g = {k: p.grad for k, p in m.named_parameters() if p.grad is not None}
                fin = all(bool(torch.isfinite(t).all()) for t in g.values())
                neo_g = sum(float(t.abs().sum()) for k, t in g.items() if 'encoding_neo' in k)
                hip_g = sum(float(t.abs().sum()) for k, t in g.items() if k.startswith('inner.encoding.') or 'hippo' in k or 'block' in k.lower())
                head_g = float(g['head.weight'].abs().sum())
                print(f'     {ds} H{pl} {n} {v}: |grad| Neocortex embedding {neo_g:.3g}, hippocampus {hip_g:.3g}, head {head_g:.3g}', flush=True)
                ok(fin and hip_g > 0 and head_g > 0 and (v == 'nomem' or neo_g > 0),
                   f'c60f {ds} H{pl} {n} {v} L=336: gradients finite, non-zero on the hippocampus / head'
                   + ('' if v == 'nomem' else ' / Neocortex embedding'))
                del m


if __name__ == '__main__':
    c60b(); c60c(); c60d_e(); c60f()
    print(f'{len(FAIL)} failed', flush=True)
    for f in FAIL:
        print('  FAILED', f)
    sys.exit(1 if FAIL else 0)
