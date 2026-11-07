"""Checks of the membrane shortcut (2026-11-08, exp/f-lif-pop-v3; PROJECT_LOG membrane shortcut pre-registration). Fail closed: any failure stops with exit 1.
  m1  paired start: at the same seed the shortcut model has the RNG state and every shared weight of the model without it, plus exactly the new ms_mix keys.
  m2  forward: the output equals an independent recomputation from the submodules (u0, u1, s1, u2, s2), both models, classification and Forecaster paths, bit for bit;
      every Linear input is 0/1 except the embedding; the block output is 0/1; the shortcut really changes the output.
  m3  gradients: every new parameter gets a finite non-zero gradient; with the mixer or attention output zeroed the embedding still gets a gradient through the shortcut.
  m4  energy: every Linear but the embedding and the head reads 0/1, the recorder's MACs equal a direct count (embedding, head, BNs); exactly one more neuron layer than the model without the shortcut.
  m5  fail closed: wrong flag combinations raise; a block with the shortcut refuses the spike path; connect() refuses MS.
  --repro: the default paths repeated with this code equal the control rows in every shared RESULT field but time_s, sha, dir, exp, tag.
  --smoke: every smoke run has a RESULT line, no SKIP, a best epoch that is the argmin of val_loss (ETT), a 0/1 head input and ms_mix.linear with 0 MAC in the syops per-layer file.
usage: python -B check_ms.py   |   python -B check_ms.py --repro 20261108 --smoke 20261108
"""
import argparse
import csv
import math
import os
import re
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
import torch                                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                         # noqa: E402
from config import Config, set_random_seed                               # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from forecast_wrap import Forecaster                                     # noqa: E402
from energy import SynOpRecorder, FiringRateRecorder                     # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RES = os.path.join(HERE, 'results')
TASKS = {'mit': (['--seq_len', '187', '--patch_size', '32', '-emb', '384', '--mlp_ratios', '1', '--time_layers', '2'], 5),
         'ett': (['--seq_len', '96', '--patch_size', '8', '-emb', '64', '--mlp_ratios', '0.5', '--time_layers', '1'], 1)}


def fail(msg):
    print('FAIL', msg)
    sys.exit(1)


def build(task, kind, ms, seed=42, extra=()):
    import config as cfgmod
    argv0, ncls = TASKS[task]
    argv = ['x', '--data', 'X', '--seed', str(seed), '-s', '--no-bias', '--layers', '1', '--tau', '2.0', '-nh', '8', '--log_dir', '/tmp/check_ms'] + argv0 + list(extra)
    if kind == 'mx':
        argv += ['--model', 'spkMixer', '--mixer', 'lif', '--mixer_comb', 'muxlin', '--connect_f', 'MS' if ms else 'IAND', '--hippo_T', '4']
    else:
        argv += ['--model', 'myModel', '--hippo_ssa', '--hippo_T', '4'] + (['--ms_shortcut'] if ms else [])
    old = sys.argv; sys.argv = argv
    try:
        args = Config(); args.set_args(cfgmod.parse_arguments())
    finally:
        sys.argv = old
    args.num_classes, args.num_channels, args.device = ncls, 1, DEV
    set_random_seed(seed)
    inner = LOAD_MODEL[args.model](args, True)
    return inner, args, torch.get_rng_state()


def blocks(inner, kind):
    return inner.blocks if kind == 'mx' else inner.data_block


def m1():
    for task in TASKS:
        for kind in ('mx', 'sf'):
            a, _, ra = build(task, kind, True)
            b, _, rb = build(task, kind, False)
            if not torch.equal(ra, rb):
                fail(f'm1 {task} {kind}: RNG state differs after construction')
            sa, sb = a.state_dict(), b.state_dict()
            new = set(sa) - set(sb)
            prefix = 'blocks.0.ms_mix.' if kind == 'mx' else 'data_block.0.ms_mix.'
            if set(sb) - set(sa) or not new or not all(k.startswith(prefix) for k in new):
                fail(f'm1 {task} {kind}: keys {sorted(new)[:6]} missing {sorted(set(sb) - set(sa))[:6]}')
            for k in sb:
                if not torch.equal(sa[k], sb[k]):
                    fail(f'm1 {task} {kind}: {k} differs')
    print('m1 ok: same RNG state and every shared weight; only ms_mix keys are new (2 tasks x 2 models)')


def inputs(task):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(8, 1, 187, generator=g) if task == 'mit' else torch.randn(4, 96, 7, generator=g)).to(DEV)


def manual(inner, kind, tok):
    """Independent recomputation from the submodules; returns the spike output and the mixer / attention output."""
    enc = inner.encoding
    T, B, N, _ = tok.shape
    D = enc.embed_dim
    u = enc.emb_bn(enc.emb_linear(tok.flatten(0, 1)).transpose(-1, -2)).reshape(T, B, D, N)
    s = enc.emb_lif(u).transpose(-1, -2)
    u = u.transpose(-1, -2)
    blk = blocks(inner, kind)[0]
    m = blk.mix(s) if kind == 'mx' else blk.ssa(s)
    ms, f1, f2 = blk.ms_mix, (blk.fc1 if kind == 'mx' else blk.mlp.fc1), (blk.fc2 if kind == 'mx' else blk.mlp.fc2)

    def cur(l, x):
        return l.bn(l.linear(x.flatten(0, 1)).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, -1)
    u1 = u + cur(ms, m)
    s1 = ms.lif(u1)
    u2 = u1 + cur(f2, f1(s1))
    return f2.lif(u2), m


def binary(x):
    return bool(((x == 0) | (x == 1)).all())


@torch.no_grad()
def m2():
    def warm(model, fn):
        """Train mode: a fresh model in eval mode is silent (BN running statistics at their defaults); the eval path is covered by the smoke recounts."""
        model.train()
        functional.reset_net(model)
        return fn()
    for task in TASKS:
        for kind in ('mx', 'sf'):
            inner, args, _ = build(task, kind, True)
            base, _, _ = build(task, kind, False)
            if task == 'ett':
                model = Forecaster(inner, args.embed_dim, 96, bias=False).to(DEV)
                ctrl = Forecaster(base, args.embed_dim, 96, bias=False).to(DEV)
                x = inputs(task)
                B, L, C = x.shape
                z = x.permute(0, 2, 1).reshape(B * C, 1, L)
                h = warm(model, lambda: model.step_features(z))
                h0 = warm(ctrl, lambda: ctrl.step_features(z))
                body = model.inner
            else:
                model = inner.to(DEV); body = model
                ctrl = base.to(DEV)
                z = inputs(task)
                model.train_mode = ctrl.train_mode = 'testing'
                if kind == 'mx':
                    h = warm(model, lambda: model.forward_parts(z)[4])
                    h0 = warm(ctrl, lambda: ctrl.forward_parts(z)[4])
                else:
                    h = warm(model, lambda: model.ms_body(model._tokens_mc(z)[0]))
                    h0 = warm(ctrl, lambda: ctrl.data_block[0](ctrl.encoding(ctrl._tokens_mc(z)[0])))
            tok = body.tokens(z) if kind == 'mx' else body._tokens_mc(z)[0]
            functional.reset_net(model)
            got, m = manual(body, kind, tok)
            if not torch.equal(got, h):
                fail(f'm2 {task} {kind}: forward path differs from the recomputation ({(got != h).float().mean().item()})')
            if not binary(h) or not binary(m):
                fail(f'm2 {task} {kind}: output or mixing path is not 0/1')
            if h.sum() == 0:
                fail(f'm2 {task} {kind}: silent output')
            if torch.equal(h, h0):
                fail(f'm2 {task} {kind}: the shortcut does not change the output')
            print(f'   {task} {kind}: output rate {h.float().mean().item():.4f}, without the shortcut {h0.float().mean().item():.4f}')
    print('m2 ok: forward equals the recomputation bit for bit; 0/1 output; shortcut changes it (2 tasks x 2 models)')


def m3():
    for kind in ('mx', 'sf'):
        inner, _, _ = build('mit', kind, True)
        model = inner.to(DEV).train()
        model.train_mode = 'training'
        z = inputs('mit')
        blk = blocks(model, kind)[0]
        opt = torch.optim.AdamW(model.parameters(), lr=.005)
        first = None
        for it in range(40):                                           # a fresh attention path is silent (rate 0, W gets no gradient) until training makes it fire
            functional.reset_net(model)
            out = model(z)
            out = out[0] if isinstance(out, tuple) else out
            opt.zero_grad()
            (out - 1.).pow(2).mean().backward()
            for n, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    fail(f'm3 {kind}: {n} has a non-finite gradient at step {it}')
            if it == 0 and kind == 'sf' and blk.ssa.q_linear.weight.grad.abs().sum() == 0:
                fail('m3 sf: no gradient to the attention path at the first step (it could never start firing)')
            if all(p.grad is not None and p.grad.abs().sum() > 0 for p in blk.ms_mix.parameters()):
                first = it
                break
            opt.step()
        if first is None:
            fail(f'm3 {kind}: ms_mix has no gradient within 40 steps')
        print(f'   {kind}: first non-zero gradient of ms_mix at step {first}')
        model.zero_grad()
        functional.reset_net(model)
        if kind == 'mx':
            h = model.forward_parts(z, mix_fns=[lambda m: torch.zeros_like(m)])[4]
        else:
            orig = blk.ssa.forward
            blk.ssa.forward = lambda s: torch.zeros_like(s)
            h = model.ms_body(model._tokens_mc(z)[0])
            blk.ssa.forward = orig
        h.float().mean().backward()
        g = model.encoding.emb_linear.weight.grad
        if g is None or not torch.isfinite(g).all() or g.abs().sum() == 0:
            fail(f'm3 {kind}: no gradient to the embedding through the shortcut with the mixing path zero')
    print('m3 ok: new parameters have gradients; the embedding gets a gradient through the shortcut alone')


def prime(m, kind, ms, z):
    """A few steps pushing the output up: a fresh attention path is silent, and the recorder counts an all-zero BN input as 0/1 (no MAC), which hides the count this check needs."""
    opt = torch.optim.AdamW(m.parameters(), lr=.005)
    for _ in range(8):
        functional.reset_net(m)
        if kind == 'mx':
            h = m.forward_parts(z)[4]
        else:
            tok = m._tokens_mc(z)[0]
            h = m.ms_body(tok) if ms else m.data_block[0](m.encoding(tok))
        opt.zero_grad()
        (h.float() - 1.).pow(2).mean().backward()
        opt.step()


def m4():
    """Direct count on the shortcut model: every Linear but the embedding and the head reads 0/1, so the recorder's MACs are exactly the embedding, the head and the BNs."""
    for task in TASKS:
        for kind in ('mx', 'sf'):
            inner, args, _ = build(task, kind, True)
            base, _, _ = build(task, kind, False)
            if task == 'ett':
                inner = Forecaster(inner, args.embed_dim, 96, bias=False).inner
                base = Forecaster(base, args.embed_dim, 96, bias=False).inner
                z = inputs(task).permute(0, 2, 1).reshape(28, 1, 96)
            else:
                z = inputs(task)
            m = inner.to(DEV).train(); m.train_mode = 'testing'        # train mode: BN normalises, so the fresh model fires
            prime(m, kind, True, z)
            nl = len(FiringRateRecorder(m).stats) - len(FiringRateRecorder(base).stats)
            if nl != 1:
                fail(f'm4 {task} {kind}: {nl} more neuron layers, expected 1')
            rec = SynOpRecorder(m)
            seen, handles = {}, []

            def tap(name, lin):
                def fn(mod, inp, out):
                    x = inp[0].detach()
                    b = bool(x.min() >= 0 and x.max() <= 1 and torch.unique(x).numel() <= 2)
                    seen[name] = dict(lin=lin, binary=b, numel=x.numel(), fan=out.shape[-1], bias=getattr(mod, 'bias', None) is not None)
                return fn
            for name, mod in m.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    handles.append(mod.register_forward_hook(tap(name, True)))
                elif isinstance(mod, torch.nn.BatchNorm1d):
                    handles.append(mod.register_forward_hook(tap(name, False)))
            functional.reset_net(m)
            with torch.no_grad():
                if kind == 'mx':
                    m.forward_parts(z)
                else:
                    m.ms_body(m._tokens_mc(z)[0])
            rec.add_samples(z.shape[0])
            rec.detach()
            for h in handles:
                h.remove()
            real = sorted(n for n, v in seen.items() if v['lin'] and not v['binary'])
            allowed = {'encoding.emb_linear'} | ({'head'} if kind == 'mx' and task == 'mit' else set())
            if not set(real) <= allowed or 'encoding.emb_linear' not in real:
                fail(f'm4 {task} {kind}: Linear layers with a real input: {real}')
            ms_key = ('blocks.0' if kind == 'mx' else 'data_block.0') + '.ms_mix'
            if seen[ms_key + '.linear']['binary'] is not True or seen[ms_key + '.bn']['binary'] is True:
                fail(f'm4 {task} {kind}: ms_mix inputs {seen[ms_key + ".linear"]} {seen[ms_key + ".bn"]}')
            exp = sum(v['numel'] * v['fan'] for v in seen.values() if v['lin'] and not v['binary']) + sum(2 * v['numel'] for v in seen.values() if not v['lin'] and not v['binary'])
            got = rec.summary()['rec_MAC_M'] * 1e6 * z.shape[0]
            if abs(got - exp) > 1e-3 * exp:
                fail(f'm4 {task} {kind}: recorder MAC {got} vs direct count {exp}')
    print('m4 ok: only the embedding and the head read real values; ms_mix.linear reads 0/1 (AC), its BN is MAC; recorder equals the direct count; one more neuron layer')


def m5():
    cases = [('sf + connect_f MS', lambda: build('mit', 'sf', False, extra=['--connect_f', 'MS'])),
             ('mx + ms_shortcut', lambda: build('mit', 'mx', True, extra=['--ms_shortcut'])),
             ('mx ms + head_mem', lambda: build('mit', 'mx', True, extra=['--head_mem'])),
             ('sf ms + head_pool', lambda: build('mit', 'sf', True, extra=['--head_pool', 'max']))]
    for name, fn in cases:
        try:
            fn()
        except (ValueError, NotImplementedError):
            continue
        fail(f'm5 {name} did not raise')
    x = torch.zeros(4, 2, 11, 384)
    inner, _, _ = build('mit', 'sf', True)
    try:
        inner.data_block[0](x)
        fail('m5 Block with the shortcut accepted the spike path')
    except RuntimeError:
        pass
    inner, _, _ = build('mit', 'mx', True)
    try:
        inner.blocks[0].connect(x, x)
        fail('m5 connect() accepted MS')
    except RuntimeError:
        pass
    print('m5 ok: wrong combinations raise; the spike path refuses a shortcut block')


def rows(path, skip_bad=True):
    out = []
    for line in open(path):
        if line.startswith('SKIP'):
            fail(f'SKIP line in {path}: {line[:160]}')
        if line.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in line.split()[1:] if '=' in t))
    return out


def repro(date):
    ignore = {'time_s', 'sha', 'dir', 'exp', 'tag', 'lr', 'layers'}
    pairs = [(f'check-ms-repro-{date}', lambda r: r['cond'], os.path.join(RES, 'flif-v1-mitbih-spkmux-tune', 'raw.txt'), lambda r, q: q.get('tag') == r['tag'] and q['seed'] == r['seed'] and q.get('val_fold') == '-1'),
             (f'check-ms-repro-{date}-ett', lambda r: r['cond'], None, None)]
    n = 0
    for r in rows(os.path.join(RES, pairs[0][0], 'raw.txt')):
        ref = [q for q in rows(pairs[0][2]) if pairs[0][3](r, q)]
        if len(ref) != 1:
            fail(f'repro: {len(ref)} control rows for {r["tag"]}')
        n += cmp_fields(r, ref[0], ignore)
    for r in rows(os.path.join(RES, pairs[1][0], 'raw.txt')):
        if r['cond'] == 'spk_ssa':
            ref = [q for q in rows(os.path.join(RES, 'flif-v1-ett-spkmux', 'baseline_spk_ssa_raw.txt')) if q['ds'] == r['ds'] and q['pl'] == r['pl'] and q['seed'] == r['seed']]
        else:
            ref = [q for q in rows(os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'raw.txt')) if q['tag'] == r['tag'] and q['ds'] == r['ds'] and q['pl'] == r['pl'] and q['seed'] == r['seed']]
        if len(ref) != 1:
            fail(f'repro: {len(ref)} control rows for {r["cond"]} {r["ds"]} {r["pl"]}')
        n += cmp_fields(r, ref[0], ignore)
    print(f'repro ok: 4 default-path runs equal their control rows ({n} shared fields compared)')


def cmp_fields(r, ref, ignore):
    keys = (set(r) & set(ref)) - ignore
    if len(keys) < 10:
        fail(f'repro: only {len(keys)} shared fields')
    bad = {k: (r[k], ref[k]) for k in keys if r[k] != ref[k]}
    if bad:
        fail(f'repro {r["cond"]}: {bad}')
    print(f"   {r['cond']} {r.get('ds', 'MITBIH')} {r.get('pl', '')}: {len(keys)} fields equal (mse {r.get('mse', '-')} f1 {r.get('f1', '-')})")
    return len(keys)


def smoke(date):
    for suite, want in ((f'smoke-ms-ett-{date}', 4), (f'smoke-ms-mitbih-{date}', 2)):
        rr = rows(os.path.join(RES, suite, 'raw.txt'))
        if len(rr) != want:
            fail(f'smoke {suite}: {len(rr)} runs, expected {want}')
        for r in rr:
            for k in ('total_fft_uJ', 'rho_hippo'):
                if not math.isfinite(float(r[k])) or float(r[k]) <= 0:
                    fail(f'smoke {r["tag"]}: {k} = {r[k]}')
            if 'best_epoch_val_epochs' in r:
                be, val, ep = r['best_epoch_val_epochs'].split(':')
                v = [float(x['val_loss']) for x in csv.DictReader(open(os.path.join(r['dir'], 'log', 'best_log_0.csv')))]
                if len(v) != int(ep) or abs(v[int(be)] - float(val)) > 1e-6:
                    fail(f'smoke {r["tag"]}: best {be} {val} vs {v}')
            p = os.path.join(r['dir'], 'log', 'model+info+per+layer.txt')
            found = 0
            if not os.path.exists(p):
                fail(f'smoke {r["tag"]}: no per-layer file')
            txt = open(p, encoding='utf-8').read().split('\n')
            for i, line in enumerate(txt):
                if re.match(r'\s*\(ms_mix\): SpikLinearLayer', line):
                    lin = next((x for x in txt[i + 1:i + 6] if '(linear): Linear' in x), None)
                    if lin is None:
                        fail(f'smoke {r["tag"]}: no linear line under ms_mix')
                    mac = re.search(r'ACs,\s*([\d.]+)\s*([KMG]?)\s*Ops', lin)
                    if mac is None or float(mac.group(1)) != 0:
                        fail(f'smoke {r["tag"]}: ms_mix.linear MAC: {lin.strip()[:160]}')
                    found += 1
            if found != 1:
                fail(f'smoke {r["tag"]}: {found} ms_mix blocks in the per-layer file')
    print('smoke ok: 6 runs, no SKIP, best epoch is the argmin of val_loss, ms_mix.linear has 0 MAC in syops')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repro')
    ap.add_argument('--smoke')
    a = ap.parse_args()
    if a.repro or a.smoke:
        if a.repro:
            repro(a.repro)
        if a.smoke:
            smoke(a.smoke)
    else:
        m1(); m2(); m3(); m4(); m5()
