"""Run-level checks of the mixer-variant test (2026-11-18, PROJECT_LOG, exp/f-lif-pop-v3). Fail closed (exit 1 on any failure).
  repro : the two default-path runs (check-mx3-repro-20261118[-ett]) equal the tuned control rows in every shared RESULT field
          but time_s, sha, dir, exp, tag, lr, layers (check_ms.rows / cmp_fields)
  smoke : smoke-mx3-{ett,mitbih}-20261118 hold 12 / 5 RESULT rows, no SKIP, one per arm and cell; finite metrics; the logged training
          arguments (logargs.txt) differ from the tuned control run of the same cell and seed only in the paths / date / epochs and in the
          arm's own options (spk_mixpool, mixer_ms, mixer_reset; time_num_layers for the 2-layer ETT arms); options added after the
          control runs (absent from their logargs) must hold their default value
usage (from cinc_mixer_v1/): python -B results/check-mx3-20261118/check_runs_mx3.py
"""
import math
import os
import re
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
from check_ms import rows, cmp_fields   # noqa: E402

RES = os.path.join(C, 'results')
D = '20261118'
BAD = []


def fail(msg):
    BAD.append(msg)
    print('FAIL  ' + msg, flush=True)


def logargs(d):
    out = {}
    for line in open(os.path.join(d, 'logargs.txt')):
        m = re.match(r'^(\w+)-+(.*)$', line.rstrip('\n'))
        if m:
            out[m.group(1)] = m.group(2)
    return out


def tune_row(task, r):
    if task == 'ett':
        ref = [q for q in rows(os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'raw.txt')) if q['tag'] == f"spkmx_lif_muxlin_iand_lr{r['lr']}_L1"
               and q['ds'] == r['ds'] and q['pl'] == r['pl'] and q['seed'] == r['seed']]
    else:
        ref = [q for q in rows(os.path.join(RES, 'flif-v1-mitbih-spkmux-tune', 'raw.txt')) if q['tag'] == 'spkmx_lif_muxlin_add_lr0.02_L2'
               and q['seed'] == r['seed'] and q.get('val_fold') == '-1']
    if len(ref) != 1:
        fail(f'{len(ref)} control rows for {r["tag"]} {r.get("ds", "")} {r.get("pl", "")}')
        return None
    return ref[0]


def repro():
    ignore = {'time_s', 'sha', 'dir', 'exp', 'tag', 'lr', 'layers'}
    n = 0
    for suite, task in ((f'check-mx3-repro-{D}-ett', 'ett'), (f'check-mx3-repro-{D}', 'mitbih')):
        rr = rows(os.path.join(RES, suite, 'raw.txt'))
        if len(rr) != 1:
            fail(f'repro {suite}: {len(rr)} rows, expected 1')
            continue
        ref = tune_row(task, rr[0])
        if ref is not None:
            n += cmp_fields(rr[0], ref, ignore)
    print(f'repro: 2 default-path runs compared with their control rows ({n} shared fields)')


ARM_DEFAULT = {'spk_mixpool': 'none', 'mixer_ms': 'False', 'mixer_reset': 'spikede'}
ARM_VALUE = {'_mpmax': ('spk_mixpool', 'max'), '_mpmean': ('spk_mixpool', 'mean'), '_mms': ('mixer_ms', 'True'),
             '_mrsubtract': ('mixer_reset', 'subtract'), '_mrzero': ('mixer_reset', 'zero')}
PATH_KEYS = {'date', 'save_result_path', 'save_log_path', 'save_model_state_path', 'log_dir', 'epoch', 'epochs', 'tag', 'best_save', 'saved_epoch'}
# options added after the tuned control runs (11-02); absent from their logargs, so they must hold their default value here
LATER_DEFAULT = {'gain_table': 'None', 'head_mem': 'False', 'head_pool': 'none', 'ms_shortcut': 'False', 'spk_band': 'none', 'spk_taxis': 'rep',
                 'train_loss': 'l1'}


def smoke():
    for task, want, arms in (('ett', 12, {'_L1_mpmax', '_L1_mpmean', '_L2', '_L2_mms', '_L1_mrsubtract', '_L1_mrzero'}),
                             ('mitbih', 5, {'_L2_mpmax', '_L2_mpmean', '_L2_mms', '_L2_mrsubtract', '_L2_mrzero'})):
        suite = f'smoke-mx3-{task}-{D}'
        rr = rows(os.path.join(RES, suite, 'raw.txt'))     # rows() fails on a SKIP line
        if len(rr) != want:
            fail(f'smoke {suite}: {len(rr)} RESULT rows, expected {want}')
        seen = {}
        for r in rr:
            suf = re.sub(r'^spkmx_lif_muxlin_(iand|add)_lr[0-9.]+', '', r['tag'])
            key = (suf, r.get('ds', ''), r.get('pl', ''))
            seen[key] = seen.get(key, 0) + 1
            if suf not in arms:
                fail(f'smoke {suite}: unexpected tag {r["tag"]}')
            for k in ('mse', 'f1', 'total_fft_uJ', 'rho_hippo', 'rho_neo'):
                if k in r and not math.isfinite(float(r[k])):
                    fail(f'smoke {suite} {r["tag"]}: {k} = {r[k]}')
            ref = tune_row(task, r)
            if ref is None:
                continue
            A, B = logargs(r['dir']), logargs(ref['dir'])
            arm = next((v for s, v in ARM_VALUE.items() if suf.endswith(s)), None)
            two = task == 'ett' and suf.startswith('_L2')
            bad, later = [], []
            for k in sorted(set(A) | set(B)):
                if k in PATH_KEYS:
                    continue
                if k in ARM_DEFAULT:
                    want_v = arm[1] if arm and arm[0] == k else ARM_DEFAULT[k]
                    if A.get(k) != want_v:
                        bad.append(f'{k}={A.get(k)} (want {want_v})')
                elif k not in B:
                    later.append(k)
                    if k not in LATER_DEFAULT or A.get(k) != LATER_DEFAULT[k]:
                        bad.append(f'{k}={A.get(k)} (absent in the control; want default {LATER_DEFAULT.get(k)})')
                elif k == 'time_num_layers' and two:
                    if (A.get(k), B.get(k)) != ('2', '1'):
                        bad.append(f'{k}={A.get(k)} / control {B.get(k)} (want 2 / 1)')
                elif A.get(k) != B.get(k):
                    bad.append(f'{k}={A.get(k)} / control {B.get(k)}')
            if bad:
                fail(f'smoke {suite} {r["tag"]} {r.get("ds", "")} {r.get("pl", "")}: {bad}')
            else:
                print(f"   {suite} {r['tag']} {r.get('ds', 'MITBIH')} {r.get('pl', '')}: {len(set(A) | set(B))} logargs keys; differs from the control "
                      f"only in paths / date / epoch and {dict([arm]) if arm else {}}{' + time_num_layers 2' if two else ''}; "
                      f"{len(later)} later options at their defaults", flush=True)
        dup = [k for k, v in seen.items() if v != 1]
        if dup:
            fail(f'smoke {suite}: duplicates {dup}')
        print(f'smoke {suite}: {len(rr)} RESULT rows, 0 SKIP, {len(seen)} distinct arm x cell')


if __name__ == '__main__':
    repro()
    smoke()
    if BAD:
        print(f'{len(BAD)} FAILED')
        sys.exit(1)
    print('all run-level checks passed')
