"""Checks of the anomaly-detection extension (2026-11-02, PROJECT_LOG AD pre-registration, exp/f-lif-pop-v3). Fail closed: an
exception, a NaN or a missing input is a failure.
    d0  the existing code paths are unchanged: no diff against HEAD in forecast_wrap.py, spk_mixer.py, model.py, train_forecast.py,
        train.py, train_rhythm.py, load_model.py, config.py, data_provider/ (the AD code is new files only)
    d1  data: window counts of MSL / SMAP / PSM = the TSLib formula; holdout training windows lie inside the first 80 %, the
        validation windows inside the last 20 % (no shared step); test labels: one label per step of every window, values 0/1
    d2  Reconstructor: output [B, 100, C]; a feature only at token i lands exactly on steps [4i, 4i+4); its features equal the
        body output (SpkMixer.forward_parts) and, for C = 1, Forecaster.features on the same model and input (bitwise)
    d3  metrics: raw / point-adjusted F1 and the threshold equal the computation of neorecall_ad_v1/anomaly_detection/test.py
        (its utils.adjustment loaded from that file); AUC-PR equals a direct step-wise computation
    d4  Spikformer-style with 55 channels: forward / backward finite; C = 1 equals Forecaster 'ssa' (inside d2)
    d5  energy of the smoke runs (results/<smoke suite>/raw.txt): energy_check ok; head MAC = N*D*4*C; the corrected energy
        recomputed by hand from the logged fields (0/1 rule and the ADD integer rule)
usage: python check_ad.py --smoke smoke-ad-spkmux-20261102 [--device cuda]
"""
import argparse
import importlib.util
import math
import os
import subprocess
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import Config                                                # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from forecast_wrap import Forecaster                                     # noqa: E402
from ad_wrap import Reconstructor                                        # noqa: E402
from ad_metrics import ad_metrics                                        # noqa: E402
from train_ad import Holdout                                             # noqa: E402
from summarize_rhythm import parse                                       # noqa: E402
from data_provider.data_factory import data_provider                     # noqa: E402
from spikingjelly.clock_driven import functional                         # noqa: E402

DATA = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/anomaly_detection/dataset'
REF_UTILS = os.path.join(HERE, '..', 'neorecall_ad_v1', 'anomaly_detection', 'utils.py')
SCRATCH = os.environ.get('CHECK_AD_SCRATCH', os.path.join(HERE, 'log', 'check-ad-scratch'))   # model folders of build() (no training)
results = []
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # the spiking neurons of these models need CUDA


def check(name, fn):
    try:
        ok, msg = fn()
    except Exception as e:                                               # fail closed
        ok, msg = False, f'{type(e).__name__}: {e}'
    results.append((name, bool(ok), msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)


def d0():
    files = ['forecast_wrap.py', 'spk_mixer.py', 'model.py', 'train_forecast.py', 'train.py', 'train_rhythm.py', 'load_model.py',
             'config.py', 'data_provider']
    r = subprocess.run(['git', 'diff', '--stat', 'HEAD', '--'] + files, cwd=HERE, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == '', f'git diff HEAD: {r.stdout.strip() or "empty"}'


def dargs(ds):
    return types.SimpleNamespace(data=ds, task_name='anomaly_detection', root_path=os.path.join(DATA, ds), seq_len=100, batch_size=128,
                                 num_workers=0, embed='timeF', freq='h')


def d1():
    msg = []
    for ds in ('MSL', 'SMAP', 'PSM'):
        a = dargs(ds)
        tr, _ = data_provider(a, 'train'); va, _ = data_provider(a, 'val'); te, _ = data_provider(a, 'test')
        n_tr, n_va, n_te = len(tr.train), len(tr.val), len(te.test)
        if len(tr) != n_tr - 100 + 1 or len(va) != n_va - 100 + 1 or len(te) != n_te - 100 + 1:
            return False, f'{ds}: window counts {len(tr)} / {len(va)} / {len(te)}'
        cut = int(n_tr * 0.8)
        if n_va != n_tr - cut or not np.array_equal(tr.val, tr.train[cut:]):
            return False, f'{ds}: TSLib validation is not the last 20 %'
        ho = Holdout(tr)
        last_end = (len(ho) - 1) * tr.step + 100                        # end (exclusive) of the last holdout training window
        if last_end > cut or (len(ho)) * tr.step + 100 <= cut:
            return False, f'{ds}: holdout windows end at {last_end}, cut {cut}'
        x_last = ho[len(ho) - 1][0]
        if not np.array_equal(x_last, tr.train[last_end - 100:last_end].astype(np.float32)):
            return False, f'{ds}: holdout window content'
        try:
            ho[len(ho)]
            return False, f'{ds}: holdout index past the end accepted'
        except IndexError:
            pass
        y = np.concatenate([np.asarray(te[i][1]).reshape(-1) for i in (0, len(te) // 2, len(te) - 1)])   # PSM: [100, 1] per window
        if y.shape != (300,) or not set(np.unique(te.test_labels)).issubset({0, 1}):
            return False, f'{ds}: labels {y.shape} {np.unique(te.test_labels)}'
        msg.append(f'{ds} C={tr.train.shape[1]} train {len(tr)} (holdout {len(ho)}, cut {cut}) val {len(va)} test {len(te)} '
                   f'anomalies {te.test_labels.mean():.4f}')
    return True, '; '.join(msg)


def build(model, C, comb='muxlin', cf='OR', device='cpu'):
    import config as cfgmod
    sys_argv = sys.argv
    argv = ['x', '--data', 'MSL', '--seed', '42', '-s', '--no-bias', '--layers', '1', '--tau', '2.0', '--patch_size', '8', '-emb', '64',
            '-nh', '8', '--mlp_ratios', '1', '--time_layers', '2', '--seq_len', '100', '--log_dir', SCRATCH]
    argv += ['--model', 'spkMixer', '--mixer', 'lif', '--mixer_comb', comb, '--connect_f', cf, '--hippo_T', '4'] if model == 'spkmix' else \
            ['--model', 'myModel', '--hippo_ssa', '--hippo_T', '4']
    sys.argv = argv
    try:
        args = Config(); args.set_args(cfgmod.parse_arguments())
    finally:
        sys.argv = sys_argv
    args.num_classes, args.num_channels, args.device = 1, C, torch.device(device)
    torch.manual_seed(0)
    return LOAD_MODEL[args.model](args, True).to(DEV), args


def d2():
    torch.manual_seed(1)
    inner, args = build('spkmix', 55)
    rec = Reconstructor(inner, 64, 100, 55).to(DEV).eval()
    rec.train_mode = 'testing'
    x = torch.randn(3, 100, 55, device=DEV)
    with torch.no_grad():
        y = rec(x); functional.reset_net(rec)
        if tuple(y.shape) != (3, 100, 55) or not torch.isfinite(y).all():
            return False, f'output {tuple(y.shape)}'
        for i in (0, 7, 24):
            f = torch.zeros(1, rec.num_patches, 64, device=DEV); f[0, i] = torch.randn(64, device=DEV)
            o = rec.decode(f)[0]
            nz = (o.abs().sum(1) > 0).nonzero().flatten().tolist()
            want = [t for t in range(4 * i, 4 * i + 4) if t < 100]
            if nz != want:
                return False, f'token {i} -> steps {nz}, expected {want}'
        xn, _, _ = rec.norm(x)
        z = xn.permute(0, 2, 1)
        f1 = rec.features(z); functional.reset_net(rec)
        f2 = inner.forward_parts(z)[4].mean(0); functional.reset_net(rec)
        if not torch.equal(f1, f2):
            return False, 'features != SpkMixer.forward_parts body output'
        for kind in ('spkmix', 'ssa'):
            torch.manual_seed(2)
            inner1, _ = build(kind, 1)
            r1 = Reconstructor(inner1, 64, 100, 1).to(DEV).eval(); r1.train_mode = 'testing'
            fc = Forecaster.__new__(Forecaster); torch.nn.Module.__init__(fc)
            fc.kind, fc.inner, fc.num_patches, fc.T = r1.kind, inner1, inner1.num_patches, getattr(inner1, 'T', None)
            z1 = torch.randn(4, 1, 100, device=DEV)
            a = r1.features(z1); functional.reset_net(r1)
            b = fc.features(z1); functional.reset_net(fc)
            if not torch.equal(a, b):
                return False, f'C=1 {kind}: Reconstructor features != Forecaster features'
    return True, 'shape [3, 100, 55]; token i -> steps [4i, 4i+4) for i = 0, 7, 24; body output and C=1 Forecaster features bitwise equal (spkmix, ssa)'


def d3():
    spec = importlib.util.spec_from_file_location('ad_ref_utils', REF_UTILS)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    from sklearn.metrics import precision_recall_fscore_support
    rng = np.random.default_rng(0)
    msgs = []
    for trial in range(3):
        n = 20000
        gt = np.zeros(n, int)
        for s in rng.integers(0, n - 300, 25):
            gt[s:s + rng.integers(5, 300)] = 1
        test = rng.gamma(2., 1., n) + 1.5 * gt * rng.random(n)
        if trial == 2:
            test = np.round(test, 1)                                    # ties
        train = rng.gamma(2., 1., 30000)
        m = ad_metrics(train, test, gt, 1.0)
        thr = np.percentile(np.concatenate([train, test]), 100 - 1.0)
        pred = (test > thr).astype(int)
        _, _, rf, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)
        g2, p2 = ref.adjustment(gt.copy(), pred.copy())
        _, _, pf, _ = precision_recall_fscore_support(np.array(g2), np.array(p2), average='binary', zero_division=0)
        order = np.argsort(-test, kind='mergesort'); st, yt = test[order], gt[order]
        tp = np.cumsum(yt); k = np.r_[np.nonzero(np.diff(st))[0], n - 1]   # last index of each distinct score
        prec, recl = tp[k] / (k + 1), tp[k] / gt.sum()
        ap = float(np.sum(np.diff(np.r_[0., recl]) * prec))
        if not (m['thr'] == thr and m['raw_f1'] == rf and m['pa_f1'] == pf and abs(m['aucpr'] - ap) < 1e-10):
            return False, f"trial {trial}: thr {m['thr']} / {thr}, raw {m['raw_f1']} / {rf}, pa {m['pa_f1']} / {pf}, ap {m['aucpr']} / {ap}"
        msgs.append(f"raw {rf:.4f} pa {pf:.4f} ap {ap:.4f}")
    return True, 'reference test.py computation reproduced (3 trials, ties included): ' + '; '.join(msgs)


def d4():
    inner, _ = build('ssa', 55)
    rec = Reconstructor(inner, 64, 100, 55).to(DEV).train(); rec.train_mode = 'training'
    x = torch.randn(4, 100, 55, device=DEV)
    loss = torch.nn.functional.mse_loss(rec(x), x); loss.backward(); functional.reset_net(rec)
    g = [p.grad for p in rec.parameters() if p.grad is not None]
    ok = math.isfinite(loss.item()) and len(g) > 0 and all(torch.isfinite(t).all() for t in g)
    return ok, f'loss {loss.item():.4f}, {len(g)} parameter gradients finite'


def d5(suite):
    rows = [r for r in parse(os.path.join(HERE, 'results', suite, 'raw.txt')) if int(r['holdout']) == 0]
    if not rows:
        return False, 'no whole-file smoke rows'
    C = {'MSL': 55, 'SMAP': 25, 'PSM': 25}
    msg = []
    for r in rows:
        if r.get('energy_check') != 'ok':
            return False, f"{r['ds']} {r['cond']}: energy_check {r.get('energy_check')}"
        mac = 25 * 64 * 4 * C[r['ds']]
        if int(r['head_MAC']) != mac:
            return False, f"{r['ds']} {r['cond']}: head MAC {r['head_MAC']} != {mac}"
        if not (0 <= r['head_syops_MAC'] <= mac and r['head_syops_AC'] >= 0):
            return False, f"{r['ds']} {r['cond']}: syops head AC {r['head_syops_AC']} / MAC {r['head_syops_MAC']}"
        if r['rule'] == 'integer':
            want = r['total_fft_uJ'] - r['syn_energy_uJ'] + r['body_syn_int_uJ'] + r['head_AC_corr'] * 0.9e-6
        else:                                                           # syops' own head count is replaced by the recount
            want = r['total_fft_uJ'] - (r['head_syops_AC'] * 0.9e-6 + r['head_syops_MAC'] * 4.6e-6) + r['head_AC_corr'] * 0.9e-6
        if not abs(want - r['total_fft_uJ_corr']) <= 1e-4 * want:
            return False, f"{r['ds']} {r['cond']}: corrected {r['total_fft_uJ_corr']} != {want}"
        msg.append(f"{r['ds']} {r['cond']} ({r['rule']}, syops head MAC {r['head_syops_MAC'] / mac:.2f} of {mac}) "
                   f"{r['total_fft_uJ']:.2f} -> {r['total_fft_uJ_corr']:.2f}")
    if not any(r['rule'] == 'integer' for r in rows):
        return False, 'no ADD smoke row: ' + '; '.join(msg)
    return True, '; '.join(msg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', required=True); a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    check('d0 existing code unchanged', d0)
    check('d1 data and holdout split', d1)
    check('d2 Reconstructor', d2)
    check('d3 metrics', d3)
    check('d4 Spikformer-style 55 channels', d4)
    check('d5 energy of the smoke runs', lambda: d5(a.smoke))
    n_ok = sum(ok for _, ok, _ in results)
    print(f'{n_ok}/{len(results)} checks passed')
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == '__main__':
    main()
