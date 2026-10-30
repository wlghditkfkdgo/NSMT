"""Checks for the ETT check of muxlin (2026-10-31, PROJECT_LOG, exp/f-lif-pop-v3). Fail-closed: a check that cannot run, or whose
test is vacuous, FAILs.

  t0  copies: the unchanged copies (train_forecast.py, data_provider/*.py) have the git blobs of exp/nsmt-flif-neo dcfdc4811
      (ETT_PROVENANCE.md); forecast_wrap.py / energy_recount_ett.py exist with the 2026-10-31 SpkMixer additions
  t1  reused baseline: the Spikformer-style ETT forecaster (spk_ssa, ETTh2 H96, seed 42) built by this branch's code and by the code
      of the commit that produced the reused baseline runs (exp/nsmt-flif-neo f0950c9a1, exported with git archive to --src) has
      the same initial state_dict, test batch, eval output, training output and L1 gradients, bitwise (ett_t1_snapshot.py in two
      processes; the other session's code is only read, `python -B`)
  t2  SpkMixer forecaster (LIF, muxlin and sdlin, OR residual, ETTh2 H96 arguments): output [B, 96, 7]; changing one input channel
      changes only that output channel (eval mode, BN statistics set from the batch); adding a constant to the input adds it to
      the output; muxlin m == where(s_d, L, F ^ R) on the forecaster's rows; mixer / s_d / fusion neurons T = 24, embedding / MLP
      T = 4; step_features 0/1 and its T_h mean == features
  t3  energy recount of the smoke suite: energy_recount_ett.py passed (file present, every spkmx_ row recounted, binary, MSE
      recomputed) and total_fft_uJ_corr == total_fft_uJ - head_MAC * 4.6e-6 + head_AC_corr * 0.9e-6 recomputed here
  t4  reproduction: the spk_ssa ETTh2 H96 seed 42 run of this branch (suite check-ett-spkssa-<date>) is within 1 % relative test
      MSE of the reused baseline row (0.291610)
usage: python check_ett_mux.py --src <exported f0950c9a1 classification dir> --scratch <dir> [--only t0 t2] [--smoke <suite>] [--repro <suite>]
"""
import argparse
import csv
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ETT = os.path.abspath(os.path.join(HERE, '..', 'forecasting', 'dataset', 'ETT-small'))
BLOBS = {'train_forecast.py': '8c4ff92c5c91a06fe3f489c7c4933748d0f3dcb0',
         'data_provider/augmentation.py': '178d584f4b1110cb9d1986f87ad97adfb0ce3572',
         'data_provider/data_factory.py': '24367ce2a5d7804c210cebf029716e679e21cbd7',
         'data_provider/data_loader.py': '75b9bd5119a8c465ef653d1c930697fc863e5b32',
         'data_provider/download_data.py': 'c2a846e5131c88b86d44952256bba425f0e97d4f',
         'data_provider/dtw.py': '941eae8e227c441da4e7f472e310d3f7bb999a5b',
         'data_provider/m4.py': '7a40d237ef6a9d3615558eda6dcf9a5c88b92990',
         'data_provider/timefeatures.py': '913ff08859a7a2f626a76953b183e4da9dfcb10b',
         'data_provider/uea.py': 'b43d26f81d0bbea0e8b3a1ec9fa2c267650195d7'}
RESULTS = []


def report(name, ok, msg):
    RESULTS.append((name, bool(ok), msg)); print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
            except Exception as e:                                      # noqa: BLE001
                report(name, False, f'{type(e).__name__}: {e}')
        run.__name__ = name
        return run
    return deco


@check('t0')
def t0():
    bad = []
    for f, blob in BLOBS.items():
        h = subprocess.check_output(['git', 'hash-object', os.path.join(HERE, f)]).decode().strip()
        if h != blob:
            bad.append(f'{f} {h[:10]} != {blob[:10]}')
    fw = open(os.path.join(HERE, 'forecast_wrap.py')).read(); er = open(os.path.join(HERE, 'energy_recount_ett.py')).read()
    adds = "self.kind = 'spkmix'" in fw and "'spkmx_'" in er and "'spkmix'" in er
    report('t0', not bad and adds, f'{len(BLOBS)} unchanged copies match their source blobs: {not bad} {bad}; SpkMixer additions present: {adds}')


@check('t1')
def t1():
    os.makedirs(ARGS.scratch, exist_ok=True)
    outs = {}
    for tag, root in (('mine', HERE), ('src', ARGS.src)):
        out = os.path.join(ARGS.scratch, f't1_{tag}.pt')
        subprocess.run([sys.executable, '-B', os.path.join(HERE, 'ett_t1_snapshot.py'), root, out, os.path.join(ARGS.scratch, f't1log_{tag}'), ETT],
                       check=True, cwd=ARGS.scratch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outs[tag] = torch.load(out)
    a, b = outs['mine'], outs['src']
    sd = a['sd'].keys() == b['sd'].keys() and all(torch.equal(a['sd'][k], b['sd'][k]) for k in a['sd'])
    same = {k: bool(torch.equal(a[k], b[k])) for k in ('x', 'y', 'out_eval', 'out_train')}
    gr = a['grads'].keys() == b['grads'].keys() and all(torch.equal(a['grads'][k], b['grads'][k]) for k in a['grads'])
    nz = sum(float(v.abs().sum()) > 0 for v in a['grads'].values())
    src_ok = os.path.realpath(b['source']).startswith(os.path.realpath(ARGS.src)) and os.path.realpath(a['source']).startswith(HERE)
    ok = sd and all(same.values()) and gr and nz > 10 and a['n'] == b['n'] and a['kind'] == b['kind'] == 'ssa' and src_ok
    report('t1', ok, f"state_dict ({len(a['sd'])}) {sd}; {same}; grads ({len(a['grads'])}, {nz} non-zero) {gr}; windows {a['n']} / {b['n']}; "
                     f"sources ok {src_ok}")


def build_fc(comb):
    """the SpkMixer forecaster of run_ett_spk.sh (ETTh2 H96 arguments), seed 42, and the first ETTh2 test batch."""
    import train_forecast as TF
    argv0 = sys.argv
    sys.argv = ['train_forecast.py', '--data', 'ETTh2', '--data_path', 'ETTh2.csv', '--root_path', ETT, '--seq_len', '96', '--pred_len', '96',
                '--features', 'M', '--target', 'OT', '--log_dir', os.path.join(ARGS.scratch, 't2log'), '--seed', '42', '-s', '-nd', '0',
                '--gating', 'attn', '--no-bias', '--scheduler', 'reduce', '-e', '100', '--warm_up_epoch', '0', '--layers', '1', '--tau', '2.0',
                '--weight_decay', '0.06', '--patience', '10', '--patch_size', '8', '-emb', '64', '-nh', '8', '--mlp_ratios', '0.5',
                '--time_layers', '1', '-lr', '0.0005', '-bs', '64', '--model', 'spkMixer', '--mixer', 'lif', '--mixer_comb', comb,
                '--connect_f', 'OR', '--hippo_T', '4', '--tag', f'spkmx_lif_{comb}_or', 'seed42']
    try:
        fc = TF.split_args()
        args = TF.Config(); args.set_args(TF.parse_arguments())
    finally:
        sys.argv = argv0
    for k, v in vars(fc).items():
        setattr(args, k, v)
    args.num_classes, args.num_channels = 1, 1
    TF.set_random_seed(42)
    _, te = TF.data_provider(TF.data_args(args, fc), 'test')
    model = TF.Forecaster(TF.LOAD_MODEL[args.model](args, True), args.embed_dim, fc.pred_len, bias=args.bias).to(args.device)
    x = next(iter(te))[0].float().to(args.device)
    return model, x


@torch.no_grad()
def warm_bn(model, x):
    """BN running statistics := the statistics of x (momentum 1, one training-mode pass), then eval (c16 lesson)."""
    from spikingjelly.clock_driven import functional
    for md in model.modules():
        if isinstance(md, torch.nn.BatchNorm1d):
            md.momentum = 1.0
    model.train(); model.train_mode = 'training'; model(x); functional.reset_net(model)
    model.eval(); model.train_mode = 'testing'


@check('t2')
def t2():
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    rows, ok = [], True
    for comb in ('muxlin', 'sdlin'):
        model, x = build_fc(comb)
        warm_bn(model, x)
        with torch.no_grad():
            y = model(x); functional.reset_net(model)
            x2 = x.clone(); x2[:, :, 3] = x2[:, :, 3] + torch.randn_like(x2[:, :, 3])
            y2 = model(x2); functional.reset_net(model)
            other = [c for c in range(x.shape[2]) if c != 3]
            indep = torch.equal(y[:, :, other], y2[:, :, other]) and not torch.equal(y[:, :, 3], y2[:, :, 3])
            y3 = model(x + 5.0); functional.reset_net(model)
            shift = bool(torch.allclose(y3, y + 5.0, atol=1e-3, rtol=0))
            B, L, C = x.shape
            means = x.mean(1, keepdim=True); xc = x - means
            stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            z = (xc / stdev).permute(0, 2, 1).reshape(B * C, 1, L)
            st = model.step_features(z); functional.reset_net(model)
            ft = model.features(z); functional.reset_net(model)
            binary = bool(((st == 0) | (st == 1)).all()) and float(st.mean()) > 0
            mean_eq = torch.equal(st.mean(0), ft)
            inner = model.inner; blk = inner.blocks[0]
            e = inner.encoding(inner.tokens(z)); functional.reset_net(model)
            xm = e.permute(2, 1, 0, 3).contiguous()
            f, hf = blk.mixer.forward_charge(xm); r, hr = blk.mixer_rev.forward_charge(xm.flip(0)); r, hr = r.flip(0), hr.flip(0)
            functional.reset_net(model)
            d = (hf - hr).abs(); N, Bq, K, D = d.shape
            sd = blk.sd_neuron(blk.sd_bn(d.reshape(-1, D)).reshape(N, Bq, K, D).contiguous()); functional.reset_net(model)
            L_ = blk.fuse(torch.cat([torch.maximum(f, r), sd], -1)); functional.reset_net(model)
            mo = blk.mix(e).permute(2, 1, 0, 3); functional.reset_net(model)
            exp = torch.where(sd.bool(), L_, torch.minimum(f, r)) if comb == 'muxlin' else L_
            form = torch.equal(mo, exp) and float(sd.mean()) > 0
            rec = FiringRateRecorder(inner)
            model(x); rec.add_samples(x.shape[0]); rec.detach(); functional.reset_net(model)
            Ts = {}
            for rr in rec.layers():
                n = rr['module']
                Ts.setdefault('rep' if ('encoding' in n or '.fc1.' in n or '.fc2.' in n) else 'tok', set()).add(int(rr['T']))
        good = (tuple(y.shape) == (x.shape[0], 96, 7) and indep and shift and binary and mean_eq and form
                and Ts.get('rep') == {4} and Ts.get('tok') == {24} and model.kind == 'spkmix')
        ok &= good
        rows.append(f'{comb}: shape {tuple(y.shape)}, channel-independent {indep}, shift {shift}, step 0/1 {binary} (rate {float(st.mean()):.4f}), '
                    f'mean == features {mean_eq}, fusion formula {form} (s_d {float(sd.mean()):.4f}, F {float(f.mean()):.4f}, m {float(mo.mean()):.4f}), '
                    f'T {Ts}, {"ok" if good else "BAD"}')
        del model
    report('t2', ok, ' | '.join(rows))


@check('t3')
def t3():
    p = os.path.join(HERE, 'results', ARGS.smoke, 'energy_recount.csv')
    rows = list(csv.DictReader(open(p)))
    raw = [l for l in open(os.path.join(HERE, 'results', ARGS.smoke, 'raw.txt')) if l.startswith('RESULT ')]
    ok = len(rows) == len(raw) > 0
    msgs = []
    for r in rows:
        corr = float(r['total_fft_uJ']) - float(r['head_MAC']) * 4.6e-6 + float(r['head_AC_corr']) * 0.9e-6
        good = (r['cond'].startswith('spkmx_') and r['binary'] == 'True' and abs(corr - float(r['total_fft_uJ_corr'])) < 1e-4
                and abs(float(r['mse_recomputed']) - float(r['mse_logged'])) <= 1e-5 * float(r['mse_logged']))
        ok &= good
        msgs.append(f"{r['ds']} H{r['pl']} {r['cond']}: {float(r['total_fft_uJ']):.2f} -> {float(r['total_fft_uJ_corr']):.2f} uJ "
                    f"(rate {float(r['rate']):.4f}) {'ok' if good else 'BAD'}")
    report('t3', ok, f'{len(rows)} rows / {len(raw)} RESULT | ' + ' | '.join(msgs))


@check('t4')
def t4():
    import summarize_rhythm as S
    mine = [r for r in S.parse(os.path.join(HERE, 'results', ARGS.repro, 'raw.txt')) if r['cond'] == 'spk_ssa']
    base = [r for r in S.parse(os.path.join(HERE, 'results', 'flif-v1-ett-spkmux', 'baseline_spk_ssa_raw.txt'))
            if r['cond'] == 'spk_ssa' and r['ds'] == 'ETTh2' and int(r['pl']) == 96 and int(r['seed']) == 42]
    if len(mine) != 1 or len(base) != 1:
        raise RuntimeError(f'{len(mine)} reproduction rows, {len(base)} baseline rows')
    a, b = float(mine[0]['mse']), float(base[0]['mse'])
    rel = abs(a - b) / b
    report('t4', rel <= 0.01, f'reproduction MSE {a:.6f} vs reused {b:.6f} (relative {100 * rel:.3f} %), MAE {float(mine[0]["mae"]):.6f} vs '
                              f'{float(base[0]["mae"]):.6f}, best epoch {mine[0].get("best_epoch_val_epochs")} vs {base[0].get("best_epoch_val_epochs")}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks', 'check_ett_mux.txt'))
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--src', default=None, help='exported f0950c9a1 nsmt_flif_v1/classification (t1)')
    ap.add_argument('--scratch', default=None)
    ap.add_argument('--smoke', default='smoke-ett-spkmux-20261031')
    ap.add_argument('--repro', default='check-ett-spkssa-20261031')
    ARGS = ap.parse_args()
    for fn in (t0, t1, t2, t3, t4):
        if ARGS.only and fn.__name__ not in ARGS.only:
            continue
        fn()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    with open(ARGS.out, 'a' if ARGS.only else 'w') as f:
        for name, ok, msg in RESULTS:
            f.write(f"{'PASS' if ok else 'FAIL'} {name}: {msg}\n")
        f.write(f'{n_ok}/{len(RESULTS)} passed\n')
    print(f'{n_ok}/{len(RESULTS)} passed -> {ARGS.out}')
    sys.exit(0 if n_ok == len(RESULTS) and len(RESULTS) > 0 else 1)
