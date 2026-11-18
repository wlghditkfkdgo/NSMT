"""Run-level checks of the neuron implementation / placement test (2026-11-18, PROJECT_LOG, exp/f-lif-pop-v3). Fail closed (exit 1 on any failure).
  repro : the five default-path runs (check-pn-repro-20261118[-ett]) equal their earlier rows in every shared RESULT field but time_s, sha, dir, exp,
          tag, lr, layers (check_ms.rows / cmp_fields): LIF / pop LIF / Spikformer-style (spikingjelly) ETTh2 H96 seed 42, LIF and Spikformer-style MIT-BIH
  smoke : smoke-pn-{ett,mitbih}-20261118 hold 22 / 11 RESULT rows, no SKIP, one per combination and cell; finite metrics; the logged training arguments
          (logargs.txt) differ from a control run of the same cell and seed only in paths / date / epochs and in the combination's own options
          (mixer, pop_soma_init, gain_table = calibrate_pn.runner_table, spk_mixpool, spk_neuron_scope; neuron_all for SF-de); options added after the
          control runs (absent from their logargs) hold their default value. Controls: the tuned LIF SpkMixer row (grid), this branch's
          spikingjelly Spikformer-style tuning row (SF-de; lr is then an expected difference), the 11-18 input-336 rows (input-336 runs).
usage (from cinc_mixer_v1/): python -B results/check-pn-20261118/check_runs_pn.py
"""
import math
import os
import re
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
from check_ms import rows, cmp_fields   # noqa: E402
import calibrate_pn as cp               # noqa: E402

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


def one(path, keep):
    rr = [r for r in rows(os.path.join(RES, path)) if keep(r)]
    if len(rr) != 1:
        fail(f'{len(rr)} control rows in {path}')
        return None
    return rr[0]


LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}


def repro():
    ignore = {'time_s', 'sha', 'dir', 'exp', 'tag', 'lr', 'layers'}
    n = 0
    e = rows(os.path.join(RES, f'check-pn-repro-{D}-ett', 'raw.txt'))
    m = rows(os.path.join(RES, f'check-pn-repro-{D}', 'raw.txt'))
    if len(e) != 3 or len(m) != 2:
        fail(f'repro rows ETT {len(e)} (want 3), MIT-BIH {len(m)} (want 2)')
    for r in e:
        if r['tag'] == 'spkmx_lif_muxlin_iand_lr0.0005_L1':
            ref = one('flif-v1-ett-spkmux-tune/raw.txt', lambda q: q.get('tag') == r['tag'] and q['ds'] == 'ETTh2' and q['pl'] == '96' and q['seed'] == '42')
        elif r['tag'] == 'spkmx_pop_lif_muxlin_iand_lr0.0005_L1':
            ref = one('flif-v1-ett-spkmux-stft/raw.txt', lambda q: q.get('tag') == r['tag'] and q['ds'] == 'ETTh2' and q['pl'] == '96' and q['seed'] == '42')
        elif r['tag'] == 'spk_ssa_lr0.0005_L1':
            ref = one('flif-v1-ett-spkmux/baseline_spk_ssa_raw.txt', lambda q: q['ds'] == 'ETTh2' and q['pl'] == '96' and q['seed'] == '42')
        else:
            fail(f'unexpected repro tag {r["tag"]}'); continue
        if ref is not None:
            n += cmp_fields(r, ref, ignore)
            print(f'   repro ETT {r["tag"]}: mse {r["mse"]} = {ref["mse"]}', flush=True)
    for r in m:
        ref = one('flif-v1-mitbih-spkmux-tune/raw.txt', lambda q: q.get('tag') == r['tag'] and q['seed'] == '42' and q.get('val_fold') == '-1')
        if ref is not None:
            n += cmp_fields(r, ref, ignore)
            print(f'   repro MIT-BIH {r["tag"]}: f1 {r["f1"]} = {ref["f1"]}', flush=True)
    print(f'repro: 5 default-path runs compared with their earlier rows ({n} shared fields)')


PATH_KEYS = {'date', 'save_result_path', 'save_log_path', 'save_model_state_path', 'log_dir', 'epoch', 'epochs', 'tag', 'best_save', 'saved_epoch'}
LATER_DEFAULT = {'gain_table': 'None', 'head_mem': 'False', 'head_pool': 'none', 'ms_shortcut': 'False', 'spk_band': 'none', 'spk_taxis': 'rep',
                 'train_loss': 'l1', 'spk_mixpool': 'none', 'mixer_ms': 'False', 'mixer_reset': 'spikede', 'neuron_all': 'sj_lif', 'spk_neuron_scope': 'all'}
SOMA = {'pop_lif': '0.5', 'pop_flif': '1.0'}


def parse(tag):
    """tag -> (kind, mixer, pool, scope, seq): kind 'mx' (SpkMixer) or 'sf' (SF-de)."""
    seq = 336 if '_s336' in tag else None
    if tag.startswith('spk_ssa_'):
        return 'sf', None, 'none', 'all', seq
    m = re.match(r'^spkmx_(lif|flif|pop_lif|pop_flif)_muxlin_(iand|add)_lr[0-9.]+_L[0-9]+(_mpmean)?(_nsmixer)?(_s336)?$', tag)
    if not m:
        return None
    return 'mx', m[1], 'mean' if m[3] else 'none', 'mixer' if m[4] else 'all', seq


def control(task, kind, r, seq):
    if task == 'mit':
        if kind == 'mx':
            return one('flif-v1-mitbih-spkmux-tune/raw.txt', lambda q: q.get('tag') == 'spkmx_lif_muxlin_add_lr0.02_L2' and q['seed'] == r['seed'] and q.get('val_fold') == '-1')
        if r.get('val_fold') == '0':
            return one('flif-v1-mitbih-spkmux-tune/raw.txt', lambda q: q.get('tag') == f'spk_ssa_lr{r["lr"]}_L2_v0' and q['seed'] == r['seed'])
        # no tuned spikingjelly row at lr 0.01 in this branch's logs: the tuned one (lr 0.005) is the control and lr is an expected difference
        return one('flif-v1-mitbih-spkmux-tune/raw.txt', lambda q: q.get('tag') == 'spk_ssa_lr0.005_L2' and q['seed'] == r['seed'] and q.get('val_fold') == '-1')
    if seq:
        t = f'spk_ssa_lr{r["lr"]}_L{r["layers"]}_s336' if kind == 'sf' else f'spkmx_lif_muxlin_iand_lr{LR[r["ds"]]}_L1_s336'
        return one('flif-v1-ett-spkmux-len336/raw.txt', lambda q: q.get('tag') == t and q['ds'] == r['ds'] and q['pl'] == r['pl'] and q['seed'] == r['seed'])
    if kind == 'sf':   # this branch's spikingjelly tuning row (lr 2.5e-4, same layers); the 10-31 baseline rows come from the other session's code
        return one('flif-v1-ett-spkmux-tune/raw.txt', lambda q: q.get('tag') == f'spk_ssa_lr0.00025_L{r["layers"]}' and q['ds'] == r['ds'] and q['pl'] == r['pl'] and q['seed'] == r['seed'])
    return one('flif-v1-ett-spkmux-tune/raw.txt', lambda q: q.get('tag') == f'spkmx_lif_muxlin_iand_lr{LR[r["ds"]]}_L1' and q['ds'] == r['ds'] and q['pl'] == r['pl'] and q['seed'] == r['seed'])


def smoke():
    for task, want in (('ett', 22), ('mitbih', 11)):
        suite = f'smoke-pn-{task}-{D}'
        rr = rows(os.path.join(RES, suite, 'raw.txt'))     # rows() fails on a SKIP line
        if len(rr) != want:
            fail(f'smoke {suite}: {len(rr)} RESULT rows, expected {want}')
        seen = {}
        for r in rr:
            key = (r['tag'], r.get('ds', ''), r.get('pl', ''), r.get('val_fold', ''))
            seen[key] = seen.get(key, 0) + 1
            p = parse(r['tag'].replace('_nalif', '').replace('_v0', ''))
            if p is None:
                fail(f'smoke {suite}: unexpected tag {r["tag"]}'); continue
            kind, mixer, pool, scope, seq = p
            for k in ('mse', 'f1', 'total_fft_uJ'):
                if k in r and not math.isfinite(float(r[k])):
                    fail(f'smoke {suite} {r["tag"]}: {k} = {r[k]}')
            ref = control('mit' if task == 'mitbih' else 'ett', kind, r, seq)
            if ref is None:
                continue
            A, B = logargs(r['dir']), logargs(ref['dir'])
            exp = {}
            if kind == 'sf':
                exp['neuron_all'] = 'lif'
                if (task == 'mitbih' and r.get('val_fold') != '0') or (task == 'ett' and not seq):
                    exp['lr'] = r['lr']
                if not r['tag'].endswith('_nalif'):
                    fail(f'{r["tag"]}: SF-de tag without _nalif')
            else:
                exp['mixer'] = mixer
                if mixer.startswith('pop_'):
                    exp['pop_soma_init'] = SOMA[mixer]
                    task_ = 'MITBIH' if task == 'mitbih' else r['ds']
                    exp['gain_table'] = cp.runner_table(task_, mixer, scope, pool, seq)
                if pool != 'none':
                    exp['spk_mixpool'] = pool
                if scope != 'all':
                    exp['spk_neuron_scope'] = scope
            bad, later = [], []
            for k in sorted(set(A) | set(B)):
                if k in PATH_KEYS:
                    continue
                if k in exp:
                    if A.get(k) != exp[k]:
                        bad.append(f'{k}={A.get(k)} (want {exp[k]})')
                elif k not in B:
                    later.append(k)
                    if k not in LATER_DEFAULT or A.get(k) != LATER_DEFAULT[k]:
                        bad.append(f'{k}={A.get(k)} (absent in the control; want default {LATER_DEFAULT.get(k)})')
                elif A.get(k) != B.get(k):
                    bad.append(f'{k}={A.get(k)} / control {B.get(k)}')
            if bad:
                fail(f'smoke {suite} {r["tag"]} {r.get("ds", "")} {r.get("pl", "")}: {bad}')
            else:
                print(f"   {suite} {r['tag']} {r.get('ds', 'MITBIH')} {r.get('pl', '')}{' fold ' + r['val_fold'] if r.get('val_fold') == '0' else ''}: "
                      f"{len(set(A) | set(B))} logargs keys; differs from the control only in paths / date / epoch and {exp}; {len(later)} later options at their defaults", flush=True)
        dup = [k for k, v in seen.items() if v != 1]
        if dup:
            fail(f'smoke {suite}: duplicates {dup}')
        print(f'smoke {suite}: {len(rr)} RESULT rows, 0 SKIP, {len(seen)} distinct combination x cell')


if __name__ == '__main__':
    repro()
    smoke()
    if BAD:
        print(f'{len(BAD)} FAILED')
        sys.exit(1)
    print('all run-level checks passed')
