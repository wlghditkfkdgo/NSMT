"""Reproduction check of the mixer-only population ETT arms (2026-11-19 evening, exp/f-lif-pop-v3; PROJECT_LOG pre-registration).
Today's code ran pop LIF mixer and pop f-LIF mixer, ETTh2 H96, seed 42, 2 epochs (suite check-mxett-repro-20261119); the 18:07 short test runs
(suite smoke-pn-ett-20261118, code 5701db48c) ran the same settings. Every shared RESULT field must be equal except time_s, sha and dir.
usage (from cinc_mixer_v1/): python -B results/check-mxett-20261119/compare_repro.py > results/check-mxett-20261119/compare_repro.txt
"""
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rows(suite):
    out = {}
    for l in open(os.path.join(C, 'results', suite, 'raw.txt')):
        if l.startswith('SKIP'):
            sys.exit(f'STOP: SKIP in {suite}')
        if l.startswith('RESULT'):
            d = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            k = (d['tag'], d['ds'], d['pl'], d['seed'])
            if k in out:
                sys.exit(f'STOP: duplicate {k} in {suite}')
            out[k] = d
    return out


new, old = rows('check-mxett-repro-20261119'), rows('smoke-pn-ett-20261118')
bad = 0
for tag in ('spkmx_pop_lif_muxlin_iand_lr0.0005_L1_nsmixer', 'spkmx_pop_flif_muxlin_iand_lr0.0005_L1_nsmixer'):
    k = (tag, 'ETTh2', '96', '42')
    if k not in new or k not in old:
        print('MISSING', k, k in new, k in old); bad += 1; continue
    a, b = new[k], old[k]
    shared = [x for x in a if x in b and x not in ('time_s', 'sha', 'dir')]
    diff = [x for x in shared if a[x] != b[x]]
    print(f'{tag}: {len(shared)} shared fields, {len(diff)} differ; mse {a["mse"]} vs {b["mse"]}; total_fft_uJ {a["total_fft_uJ"]} vs {b["total_fft_uJ"]}; '
          f'sha {a["sha"]} vs {b["sha"]}' + (f'; DIFFER: {diff}' if diff else ''))
    bad += bool(diff) or len(shared) < 30
print('REPRO OK' if bad == 0 else f'{bad} PROBLEM(S)')
sys.exit(0 if bad == 0 else 1)
