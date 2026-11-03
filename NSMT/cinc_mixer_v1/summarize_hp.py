"""Head input form results (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG head input form pre-registration; head_pool.py). Per task
and form, against the tuned final runs of that task's tuning (control rows selected exactly as summarize_hm.py does, with the task's
own summarizer):
    G1 muxlin+form vs muxlin
    G2 Spikformer-style+form vs Spikformer-style
    G3 muxlin+form vs Spikformer-style+form  (criterion (2))
Verdict rule (CinC score / MIT-BIH macro-F1): mean >= +0.5 %p and >= 2 of 3 seeds better, mirror worse (summarize_hm.cls_compare).
Forms max / seg4 / layers are candidates; attn is a reference only (real-valued, no verdict use; its dot products and softmax are
not in the energy). Adoption (pre-registered, written with --task mitbih, which needs both tasks): a candidate form whose CinC G1 or
G2 is better and whose MIT-BIH comparison of the same G is not worse.
Energy: CinC total_fft_int_uJ (else total_fft_uJ), MIT-BIH energy_recount.csv total_fft_uJ_corr. Fail closed: a missing run, a
non-finite metric or a missing energy -> incomplete (exit 2).
usage: python summarize_hp.py --task cinc|mitbih   ->  results/flif-v1-headpool/summary_<task>.md
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402
from summarize_hm import cls_table, RES                                   # noqa: E402

OUT = os.path.join(RES, 'flif-v1-headpool')
FORMS = ('max', 'seg4', 'layers', 'attn')
CANDIDATES = ('max', 'seg4', 'layers')
DESC = {'max': 'max — W(any spike over the tokens)', 'seg4': 'seg4 — W(4 contiguous token-segment means)',
        'layers': 'layers — W(embedding and after-mixing token means)', 'attn': 'attn — learned token weights (reference only, real-valued)'}


def hp_raw(task):
    p = os.path.join(RES, f'flif-v1-headpool-{task}', 'raw.txt')
    rows = parse(p)
    by = {f: [] for f in FORMS}
    for r in rows:
        f = next((f for f in FORMS if str(r.get('tag', '')).endswith(f'_hp{f}')), None)
        if f is None:
            raise SystemExit(f'{p}: a row without an _hp<form> tag')
        by[f].append(r)
    return by


def unit(name, rr, key, cls_names):
    if rr is None:
        return f'| {name} | — | | |'
    return (f"| {name} | {np.mean([100 * float(x[key]) for x in rr]):.2f} ± {np.std([100 * float(x[key]) for x in rr]):.2f} | "
            + ' / '.join(f"{np.mean([100 * float(x[f'f1_{i}']) for x in rr]):.2f}" for i in range(len(cls_names)))
            + f" | {np.mean([float(x['E']) for x in rr]):.2f} |")


def task_cinc(L):
    import summarize_cinc_tune as sc
    D = sc.load()
    sel = {s: json.load(open(os.path.join(sc.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    def cfg(b):
        ck = sc.CONDKEY[b]
        return ('-', sel['lr'][ck], sc.L0) if b == sc.REF else (sel['res'][ck], sel['lr'][ck], sel['layers'][ck])
    c_m, c_r = sc.seeds(D, sc.MUX, cfg(sc.MUX)), sc.seeds(D, sc.REF, cfg(sc.REF))
    L += [f'Settings (res, lr, mixer layers): muxlin {cfg(sc.MUX)}, Spikformer-style {cfg(sc.REF)}; CinC fold 0, test CinC score.', '']
    out = {}
    for form, rows in hp_raw('cinc').items():
        H = {}
        for r in rows:
            base, res = sc.split(r['cond'])
            k = (base, res, float(r['lr']), int(r['layers']), int(r['seed']))
            if k in H:
                raise SystemExit(f'duplicate {form} run {k}')
            r['E'] = float(r['total_fft_int_uJ']) if 'total_fft_int_uJ' in r else float(r['total_fft_uJ'])
            H[k] = r
        h_m, h_r = sc.seeds(H, sc.MUX, cfg(sc.MUX)), sc.seeds(H, sc.REF, cfg(sc.REF))
        L += [f'## {DESC[form]}', '']
        names = ('N', 'A', 'O', '~')
        rows_u = ['| condition | CinC score % | F1 class 0 / 1 / 2 / 3 % | energy µJ |', '|---|---:|---|---:|',
                  unit('Spikformer-style', c_r, 'score', names), unit(f'Spikformer-style + {form}', h_r, 'score', names),
                  unit('muxlin', c_m, 'score', names), unit(f'muxlin + {form}', h_m, 'score', names), '']
        out[form] = cls_table(L, ((f'G1 muxlin+{form} vs muxlin', h_m, c_m, False), (f'G2 Spikformer-style+{form} vs Spikformer-style', h_r, c_r, False),
                                  (f'G3 muxlin+{form} vs Spikformer-style+{form}', h_m, h_r, True)), 'score', rows_u)
    return out


def task_mitbih(L):
    import csv
    import summarize_mitbih_tune as sm
    D = sm.load()
    sel = {s: json.load(open(os.path.join(sm.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    p = os.path.join(RES, 'flif-v1-headpool-mitbih', 'energy_recount.csv')
    if not os.path.exists(p):
        raise SystemExit(f'{p} missing (run the energy recount first)')
    en = {(e['cond'], e.get('tag', ''), int(float(e['seed']))): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(p))}
    c_m, c_r = sm.seeds(D, sm.MUX, sm.tuned_cfg(sm.MUX, sel), -1), sm.seeds(D, sm.REF, sm.tuned_cfg(sm.REF, sel), -1)
    L += [f'Settings (res, lr, mixer layers): muxlin {sm.tuned_cfg(sm.MUX, sel)}, Spikformer-style {sm.tuned_cfg(sm.REF, sel)}; '
          'test macro-F1 of the last-epoch weights.', '']
    out = {}
    for form, rows in hp_raw('mitbih').items():
        H = {}
        for r in rows:
            base, res = sm.split(r['cond'])
            k = (base, res, float(r['lr']), int(r['layers']), int(r['val_fold']), int(r['seed']))
            if k in H:
                raise SystemExit(f'duplicate {form} run {k}')
            r['E'] = en.get((r['cond'], str(r['tag']), int(r['seed'])), float('nan'))
            H[k] = r
        h_m, h_r = sm.seeds(H, sm.MUX, sm.tuned_cfg(sm.MUX, sel), -1), sm.seeds(H, sm.REF, sm.tuned_cfg(sm.REF, sel), -1)
        L += [f'## {DESC[form]}', '']
        names = ('N', 'S', 'V', 'F', 'Q')
        rows_u = ['| condition | macro-F1 % | F1 N / S / V / F / Q % | energy µJ |', '|---|---:|---|---:|',
                  unit('Spikformer-style', c_r, 'f1', names), unit(f'Spikformer-style + {form}', h_r, 'f1', names),
                  unit('muxlin', c_m, 'f1', names), unit(f'muxlin + {form}', h_m, 'f1', names), '']
        out[form] = cls_table(L, ((f'G1 muxlin+{form} vs muxlin', h_m, c_m, False), (f'G2 Spikformer-style+{form} vs Spikformer-style', h_r, c_r, False),
                                  (f'G3 muxlin+{form} vs Spikformer-style+{form}', h_m, h_r, True)), 'f1', rows_u)
    return out


def adoption(L, cinc, mit):
    L += ['## Adoption (pre-registered): CinC G1 or G2 better and the same G on MIT-BIH not worse; attn excluded', '',
          '| form | CinC G1 | CinC G2 | MIT-BIH G1 | MIT-BIH G2 | adopted |', '|---|---|---|---|---|---|']
    for f in CANDIDATES:
        c, m = cinc[f], mit[f]
        ok = any(c[g]['verdict'] == 'better' and m[g]['verdict'] != 'worse' for g in ('G1', 'G2'))
        L.append(f"| {f} | {c['G1']['verdict']} | {c['G2']['verdict']} | {m['G1']['verdict']} | {m['G2']['verdict']} | {'yes' if ok else 'no'} |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task', choices=['cinc', 'mitbih'], required=True); a = ap.parse_args()
    L = [f'# Head input forms (a zero-initialised head term on another input form of the last block) — {a.task}', '']
    res = {'cinc': task_cinc, 'mitbih': task_mitbih}[a.task](L)
    bad = [f for f, r in res.items() if r is None or any(v is None for v in r.values())] + [f for f in FORMS if f not in res]
    if a.task == 'mitbih' and not bad:
        cres = task_cinc([])
        if any(r is None or any(v is None for v in r.values()) for r in cres.values()):
            bad.append('cinc')
        else:
            adoption(L, cres, res)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f'summary_{a.task}.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    if bad:
        print(f'incomplete: {bad}')
        sys.exit(2)                                                      # fail closed: an incomplete comparison stops the driver


if __name__ == '__main__':
    main()
