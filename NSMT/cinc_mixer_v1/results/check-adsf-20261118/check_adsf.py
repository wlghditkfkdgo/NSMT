"""Run-level checks of the AD SF-de test (2026-11-18, PROJECT_LOG, exp/f-lif-pop-v3). Fail closed (exit 1 on any failure).
  repro : check-adsf-repro-20261118 (the edited runner's default path: spikingjelly Spikformer-style, MSL seed 42, lr 2.5e-4, whole file) equals
          its 11-03 row in flif-v1-ad-spkmux-tune in every shared RESULT field but time_s, sha, dir
  smoke : smoke-adsf-20261118 holds 6 RESULT rows (SF-de, 1 epoch, seed 42, MSL / SMAP / PSM x holdout 0 / 1), no SKIP, finite AUC-PR (whole
          file) or validation MSE, energy_check ok (whole file); logargs.txt differs from the 11-02 spikingjelly smoke run of the same dataset /
          holdout (smoke-ad-spkmux-20261102, 1 epoch, lr 5e-4) only in paths / date / tag and neuron_all = lif; options added after the control
          hold their default value.
usage (from cinc_mixer_v1/): python -B results/check-adsf-20261118/check_adsf.py
"""
import math
import os
import re
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(C, 'results')
BAD = []


def fail(msg):
    BAD.append(msg); print('FAIL  ' + msg, flush=True)


def rows(p):
    out = []
    for ln in open(p):
        if ln.startswith('SKIP'):
            fail(f'SKIP line in {p}: {ln[:120]}')
        if ln.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t))
    return out


def logargs(d):
    out = {}
    for line in open(os.path.join(d, 'logargs.txt')):
        m = re.match(r'^(\w+)-+(.*)$', line.rstrip('\n'))
        if m:
            out[m.group(1)] = m.group(2)
    return out


PATH_KEYS = {'date', 'save_result_path', 'save_log_path', 'save_model_state_path', 'log_dir', 'tag', 'best_save', 'saved_epoch'}
LATER_DEFAULT = {'gain_table': 'None', 'head_mem': 'False', 'head_pool': 'none', 'ms_shortcut': 'False', 'spk_band': 'none', 'spk_taxis': 'rep',
                 'train_loss': 'l1', 'spk_mixpool': 'none', 'mixer_ms': 'False', 'mixer_reset': 'spikede', 'neuron_all': 'sj_lif', 'spk_neuron_scope': 'all'}


def repro():
    r = rows(os.path.join(RES, 'check-adsf-repro-20261118', 'raw.txt'))
    if len(r) != 1:
        fail(f'repro rows {len(r)} (want 1)'); return
    r = r[0]
    ref = [q for q in rows(os.path.join(RES, 'flif-v1-ad-spkmux-tune', 'raw.txt'))
           if q['cond'] == 'spk_ssa' and q['tag'] == r['tag'] and q['ds'] == 'MSL' and q['seed'] == '42' and q['holdout'] == '0']
    if len(ref) != 1:
        fail(f'{len(ref)} control rows for the repro run'); return
    ref = ref[0]
    keys = (set(r) & set(ref)) - {'time_s', 'sha', 'dir'}
    bad = {k: (r[k], ref[k]) for k in keys if r[k] != ref[k]}
    if bad or len(keys) < 30:
        fail(f'repro: {len(keys)} shared fields, differences {bad}')
    else:
        print(f"repro: {r['tag']} MSL seed 42 equals its 11-03 row in {len(keys)} fields (AUC-PR {r['aucpr']}, energy {r['total_fft_uJ_corr']})")


def smoke():
    rr = rows(os.path.join(RES, 'smoke-adsf-20261118', 'raw.txt'))
    ctl = rows(os.path.join(RES, 'smoke-ad-spkmux-20261102', 'raw.txt'))
    if len(rr) != 6:
        fail(f'smoke rows {len(rr)} (want 6)')
    seen = set()
    for r in rr:
        k = (r['ds'], r['holdout'])
        if k in seen:
            fail(f'duplicate smoke run {k}')
        seen.add(k)
        if not r['tag'].endswith('_nalif'):
            fail(f'smoke tag without _nalif: {r["tag"]}')
        v = float(r['val_mse_best'])
        if not math.isfinite(v):
            fail(f'smoke {k}: validation MSE {v}')
        if r['holdout'] == '0' and (not math.isfinite(float(r['aucpr'])) or r.get('energy_check') != 'ok'):
            fail(f'smoke {k}: AUC-PR {r.get("aucpr")} energy_check {r.get("energy_check")}')
        c = [q for q in ctl if q['cond'] == 'spk_ssa' and q['ds'] == r['ds'] and q['holdout'] == r['holdout'] and q['seed'] == '42']
        if not c:
            fail(f'no control for {k}'); continue
        A, B = logargs(r['dir']), logargs(c[0]['dir'])
        bad, later = [], []
        for key in sorted(set(A) | set(B)):
            if key in PATH_KEYS:
                continue
            if key == 'neuron_all':
                if A.get(key) != 'lif':
                    bad.append(f'neuron_all={A.get(key)}')
            elif key not in B:
                later.append(key)
                if key not in LATER_DEFAULT or A.get(key) != LATER_DEFAULT[key]:
                    bad.append(f'{key}={A.get(key)} (absent in the control)')
            elif A.get(key) != B.get(key):
                bad.append(f'{key}={A.get(key)} / control {B.get(key)}')
        if bad:
            fail(f'smoke {k}: {bad}')
        else:
            print(f"   smoke {r['ds']} holdout {r['holdout']}: {len(set(A) | set(B))} logargs keys; differs from the 11-02 control only in paths / date / tag "
                  f"and neuron_all = lif; {len(later)} later options at their defaults; val MSE {r['val_mse_best']}"
                  + (f", AUC-PR {r['aucpr']}, energy {r['total_fft_uJ_corr']} ({r['energy_check']})" if r['holdout'] == '0' else ''))
    print(f'smoke: {len(rr)} RESULT rows, {len(seen)} distinct dataset x holdout')


if __name__ == '__main__':
    repro()
    smoke()
    if BAD:
        print(f'{len(BAD)} FAILED'); sys.exit(1)
    print('all AD SF-de run-level checks passed')
