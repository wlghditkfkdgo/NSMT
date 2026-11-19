"""ETT pass criteria of the user (2026-11-19 evening, exp/f-lif-pop-v3; PROJECT_LOG decision entry). No training.
User (verbatim): "통과 기준은 **평균적으로** - SNN 예측 모델(SpikF) 보다 0.001이라도 낮은 MSE/MAE - (우리가 수정하고 비교해왔던) Spikformer 보다
0.001이라도 낮은 MSE/MAE + 비슷한 energy consumption - DNN 예측 모델보다 0.001이라도 낮은 MSE/MAE"; answers: DNN = every row of our
9-row look-back-96 DNN table (incl. TimeMixer); "비슷한 energy" = at most 1.05 x SF-de.
Definitions (fixed before any new run):
  mean      = 4-cell mean (ETTh2 / ETTm2 x H96 / H720, input 96) of the seed means (42 / 7 / 2026) of the test value; MSE and MAE separately.
  passes vs a reference  <=>  ref_mean - ours_mean >= 0.001 for MSE AND for MAE (my reading of "MSE/MAE" as both).
  (1) SNN: SpikF (SpikF Table 12 = NSMT Table 14, transcription verified 11-19 17:51).
  (2) Spikformer: SF-de (ours; lr-only tuned, user decision 11-19 "아니오" = not tuned further) on MSE and MAE, AND energy A/B <= 1.05, where
      A/B = mean over the 4 cells of seed-mean(ours) / seed-mean(SF-de) under the syops rule as published (user 11-19).
  (3) DNN: every one of the 9 rows below.  Pass = (1) and (2) and (3).
Reference values (test MSE / MAE at look-back 96, cells ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720), transcribed from the papers:
  iTransformer Table 10 (ICLR 2024, arXiv 2310.06625v4) "Full results of the long-term forecasting task ... input sequence length is set to
  96": iTransformer, RLinear, PatchTST, TimesNet, DLinear (the same numbers are in SpikF Table 11, checked by --verify);
  SpikF Tables 9 / 10 (ICML 2025): iTransformer trained with the MAE loss; SpikF Table 12: FITS, and SpikF itself;
  TimeMixer Table 13 (ICLR 2024, arXiv 2405.14616v1) "Unified hyperparameter results" (input 96): TimeMixer and its PatchTST re-run.
  Not checked: whether each paper's evaluation protocol equals ours (test batching, normalisation, training loss); see the research notes.
--verify <itransformer_layout.txt> <timemixer_layout.txt> <spikf_layout.txt> (pdftotext -layout of the PDFs, kept local) re-reads every
transcribed number from the tables and stops on any mismatch.
judge() is the shared checker for later stages (mixer-only population ETT, tuning); main() applies it to the current models without training.
usage (from cinc_mixer_v1/): python -B results/ett-pass-20261119/pass_ett.py [--verify ITR.txt TMX.txt SPIKF.txt]
"""
import math
import os
import re
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
from summarize_l2 import CELLS, SEEDS                                               # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
MARGIN, EBOUND, EPS = 0.001, 1.05, 1e-12
# name -> (MSE per cell, MAE per cell, source); cells in CELLS order
SNN = {'SpikF': ((0.290, 0.416, 0.175, 0.405), (0.336, 0.436, 0.251, 0.397), 'SpikF Table 12 = NSMT Table 14')}
DNN = {'iTransformer': ((0.297, 0.427, 0.180, 0.412), (0.349, 0.445, 0.264, 0.407), 'iTransformer Table 10'),
       'iTransformer, MAE loss': ((0.294, 0.423, 0.182, 0.411), (0.342, 0.439, 0.265, 0.403), 'SpikF Table 9, 10'),
       'RLinear': ((0.288, 0.420, 0.182, 0.407), (0.338, 0.440, 0.265, 0.398), 'iTransformer Table 10'),
       'PatchTST (re-run)': ((0.302, 0.431, 0.175, 0.402), (0.348, 0.446, 0.259, 0.400), 'iTransformer Table 10'),
       'PatchTST (other re-run)': ((0.308, 0.436, 0.183, 0.412), (0.355, 0.450, 0.270, 0.404), 'TimeMixer Table 13'),
       'TimeMixer': ((0.289, 0.412, 0.175, 0.391), (0.341, 0.434, 0.258, 0.396), 'TimeMixer Table 13'),
       'TimesNet': ((0.340, 0.462, 0.187, 0.408), (0.374, 0.468, 0.267, 0.403), 'iTransformer Table 10'),
       'FITS': ((0.295, 0.431, 0.183, 0.407), (0.350, 0.446, 0.266, 0.399), 'SpikF Table 12'),
       'DLinear (re-run)': ((0.333, 0.831, 0.193, 0.554), (0.387, 0.657, 0.292, 0.522), 'iTransformer Table 10')}
NSMT = ((0.288, 0.409, 0.175, 0.399), (0.334, 0.429, 0.251, 0.393), 'NSMT Table 14')       # report only (not a criterion)


def stop(m):
    print('STOP:', m)
    sys.exit(2)


# ---------------------------------------------------------------- transcription check
def grab(path, title, n, nrows, pm=False):
    """the first nrows rows (label 96 / 192 / 336 / 720 / Avg + n numbers) after the first line containing title."""
    L = open(path, encoding='utf-8').read().splitlines()
    i = next((k for k, l in enumerate(L) if title in l), None)
    if i is None:
        stop(f'title not found in {path}: {title}')
    num = r'(\d\.\d{3})\s*±\s*\d\.\d{3}' if pm else r'(\d\.\d{3})'
    rx = re.compile(r'(?:^|\s)(96|192|336|720|Avg)\s+' + r'\s+'.join([num] * n) + r'(?:\s|$)')
    rows = []
    for l in L[i:i + 400]:
        m = rx.search(l)
        if m:
            rows.append((m.group(1), [float(x) for x in m.group(2, *range(3, n + 2))]))
        if len(rows) == nrows:
            break
    if [h for h, _ in rows] != ['96', '192', '336', '720', 'Avg'] * (nrows // 5):
        stop(f'unexpected rows under {title}: {[h for h, _ in rows]}')
    return rows


def block(rows, order):
    return {(ds, rows[5 * k + j][0]): rows[5 * k + j][1] for k, ds in enumerate(order) for j in range(5)}


def verify(itr, tmx, spikf):
    T10 = block(grab(itr, 'Table 10: Full results of the long-term forecasting task', 22, 20), ('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2'))
    T13 = block(grab(tmx, 'Table 13: Unified hyperparameter results', 22, 40),
                ('Weather', 'Solar', 'Electricity', 'Traffic', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'))
    S11 = block(grab(spikf, 'Table 11. Full results of long-term prediction', 18, 30), ('ECL', 'Weather', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'))
    S9 = grab(spikf, 'Table 9. 95% confidence intervals', 6, 10, pm=True)        # SpikF 5 rows, then iTransformer (MAE loss) 5 rows
    S10 = grab(spikf, 'Table 10. 95% confidence intervals', 6, 10, pm=True)
    S12 = block(grab(spikf, 'Table 12. More benchmarks on ETT dataset', 10, 20), ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'))
    col = {'iTransformer': (T10, 0), 'RLinear': (T10, 1), 'PatchTST (re-run)': (T10, 2), 'TimesNet': (T10, 5), 'DLinear (re-run)': (T10, 6),
           'TimeMixer': (T13, 0), 'PatchTST (other re-run)': (T13, 1), 'FITS': (S12, 3), 'SpikF': (S12, 0)}
    s11col = {'iTransformer': 1, 'RLinear': 2, 'PatchTST (re-run)': 3, 'TimesNet': 5, 'DLinear (re-run)': 6}
    mae_loss = {('ETTh2', '96'): S9[5][1][2:4], ('ETTh2', '720'): S9[8][1][2:4], ('ETTm2', '96'): S10[5][1][0:2], ('ETTm2', '720'): S10[8][1][0:2]}
    bad, n = 0, 0
    lines = []
    for name, (mse, mae, src) in list(DNN.items()) + list(SNN.items()):
        for c, (ds, pl) in enumerate(CELLS):
            key = (ds, str(pl))
            if name == 'iTransformer, MAE loss':
                got = tuple(mae_loss[key])
            else:
                t, j = col[name]
                got = tuple(t[key][2 * j:2 * j + 2])
            n += 1
            if got != (mse[c], mae[c]):
                bad += 1
                lines.append(f'MISMATCH {name} {ds} H{pl}: paper {got} vs transcribed {(mse[c], mae[c])}')
            if name in s11col:                                                    # SpikF Table 11 copies iTransformer Table 10
                g2 = tuple(S11[key][2 * s11col[name]:2 * s11col[name] + 2])
                n += 1
                if g2 != (mse[c], mae[c]):
                    bad += 1
                    lines.append(f'MISMATCH SpikF Table 11 copy of {name} {ds} H{pl}: {g2} vs {(mse[c], mae[c])}')
    lines.append(f'verify: {n} checks (each a cell MSE / MAE pair), {bad} mismatches')
    return bad == 0, lines


# ---------------------------------------------------------------- criteria
def cellmeans(V):
    """V: {(ds, pl): {'mse': [3], 'mae': [3]}} -> (4-cell mean MSE, MAE) of seed means; fail closed."""
    out = []
    for k in ('mse', 'mae'):
        cm = []
        for c in CELLS:
            v = V.get(c, {}).get(k)
            if v is None or len(v) != len(SEEDS) or not all(math.isfinite(x) for x in v):
                stop(f'missing or non-finite {k} for {c}')
            cm.append(float(np.mean(v)))
        out.append(float(np.mean(cm)))
    return tuple(out)


def eratio(EA, EB):
    r = []
    for c in CELLS:
        a, b = EA.get(c), EB.get(c)
        if not a or not b or len(a) != len(SEEDS) or len(b) != len(SEEDS) or not all(math.isfinite(x) and x > 0 for x in a + b):
            stop(f'missing or bad energy for {c}')
        r.append(np.mean(a) / np.mean(b))
    return float(np.mean(r))


def ref_mean(mse, mae):
    return float(np.mean(mse)), float(np.mean(mae))


def judge(name, V, VS, EA, ES):
    """V / VS: ours / SF-de values; EA / ES: syops energy per cell (3 seeds). Returns (pass dict, markdown lines)."""
    om = cellmeans(V)
    sm = cellmeans(VS)
    er = eratio(EA, ES)
    rows, res = [], {}

    def one(ref, rm, extra=''):
        d = (rm[0] - om[0], rm[1] - om[1])
        ok = d[0] >= MARGIN - EPS and d[1] >= MARGIN - EPS
        rows.append(f'| {ref} | {rm[0]:.4f} / {rm[1]:.4f} | {d[0]:+.4f} / {d[1]:+.4f} | {"yes" if ok else "no"}{extra} |')
        return ok

    res['(1) SNN'] = one('SpikF', ref_mean(*SNN['SpikF'][:2]))
    ok2 = one('SF-de (ours)', sm)
    res['(2) Spikformer'] = ok2 and er <= EBOUND + EPS
    rows.append(f'| energy A/B vs SF-de (syops; bound {EBOUND}) | {er:.4f} | | {"yes" if er <= EBOUND + EPS else "no"} |')
    dnn = [one(m, ref_mean(*DNN[m][:2])) for m in DNN]
    res['(3) DNN'] = all(dnn)
    res['pass'] = all(res[k] for k in ('(1) SNN', '(2) Spikformer', '(3) DNN'))
    refs =[ref_mean(*SNN['SpikF'][:2]), sm] + [ref_mean(*DNN[m][:2]) for m in DNN]
    tgt = (min(r[0] for r in refs) - MARGIN, min(r[1] for r in refs) - MARGIN)
    L = [f'### {name}', '', f'4-cell mean MSE / MAE: **{om[0]:.4f} / {om[1]:.4f}**; energy A/B vs SF-de (syops) {er:.4f}', '',
         '| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |', '|---|---|---|---|'] + rows + [
        '', f'- (1) SNN: **{"pass" if res["(1) SNN"] else "fail"}**; (2) Spikformer: **{"pass" if res["(2) Spikformer"] else "fail"}**; '
        f'(3) DNN (all 9 rows): **{"pass" if res["(3) DNN"] else "fail"}** ({sum(dnn)}/9 rows); overall: **{"PASS" if res["pass"] else "NOT PASSED"}**',
        f'- Needed for all three: 4-cell mean MSE ≤ {tgt[0]:.4f} (now {om[0]:.4f}, short by {max(0., om[0] - tgt[0]):.4f}) and MAE ≤ {tgt[1]:.4f} '
        f'(now {om[1]:.4f}, short by {max(0., om[1] - tgt[1]):.4f}); plus energy A/B ≤ {EBOUND}.',
        f'- Report (not a criterion): NSMT {ref_mean(*NSMT[:2])[0]:.4f} / {ref_mean(*NSMT[:2])[1]:.4f}, ref − ours '
        f'{ref_mean(*NSMT[:2])[0] - om[0]:+.4f} / {ref_mean(*NSMT[:2])[1] - om[1]:+.4f}.', '']
    return res, L


def vals(T):
    V = {c: {k: [float(T[(c[0], c[1], s)][k]) for s in SEEDS] for k in ('mse', 'mae')} for c in CELLS}
    E = {c: [float(T[(c[0], c[1], s)]['total_fft_uJ']) for s in SEEDS] for c in CELLS}
    return V, E


def main():
    import summarize_pn as sp                                                       # noqa: E402
    vlines = ['Transcription not verified in this run.']
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        ok, vlines = verify(*sys.argv[2:5])
        print('\n'.join(vlines))
        if not ok:
            sys.exit(1)
    LIF, SF = vals(sp.arms_ett_one('LIF')), vals(sp.load_sfde_ett())
    L = ['# ETT pass criteria (user 11-19): current models, input 96, no training', '',
         'Rules and sources: module docstring of pass_ett.py. Ours: test values of the best-validation weights, seeds 42 / 7 / 2026; '
         'energy = logged total_fft_uJ (syops as published + f-LIF-paper neuron energy).', '', '## Transcription check', ''] + \
        [f'- {x}' for x in vlines] + ['', '## Reference rows (look-back 96; MSE / MAE per cell: ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720)', '',
                                       '| row | kind | source | MSE | MAE | 4-cell mean MSE / MAE |', '|---|---|---|---|---|---|']
    for kind, D in (('SNN', SNN), ('DNN', DNN)):
        for m, (mse, mae, src) in D.items():
            L.append(f'| {m} | {kind} | {src} | {", ".join("%.3f" % x for x in mse)} | {", ".join("%.3f" % x for x in mae)} | '
                     f'{np.mean(mse):.4f} / {np.mean(mae):.4f} |')
    L += ['', '## Criteria', '']
    for name, (V, E) in (('LIF SpkMixer (tuned 11-01: muxlin, IAND, 1 mixer layer per direction)', LIF),
                         ('SF-de (report: the same criteria applied to SF-de against itself is trivially "no" for (2))', SF)):
        _, Lj = judge(name, V, SF[0], E, SF[1])
        L += Lj
    open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
