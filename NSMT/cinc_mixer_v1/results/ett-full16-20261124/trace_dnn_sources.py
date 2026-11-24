"""Trace the DNN rows of NSMT Table 13 to their published sources (2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG; user 11-24: "NSMT는 내
논문이라는 뜻은, NSMT의 논문을 작성했지만 '최종적으로는 reject되어' 아직 완성된 연구라고 할 수 없어. 그러니 차이를 보고할 필요도 없어.").
No training. The NSMT paper is not published, so its Table 13 cannot be cited as the source of the DNN numbers. Its text says the
forecasting baselines (L = 96) are "taken from [43] and [41]" = SpikF (ICML 2025) and TimesNet (ICLR 2023). This script checks every one
of the 2 x 16 x 5 = 160 DNN numbers in refs16.json against those two papers' own tables:
  TimesNet Table 13 "Full results for the long-term forecasting task" (input 96): ETSformer, FEDformer, Autoformer, Informer
  SpikF Table 11 "Full results of long-term prediction" (input 96): PatchTST, Autoformer
  SpikF Table 12 "More benchmarks on ETT dataset" (input 96): FEDformer
A number counts as traced when at least one listed source has exactly the same value in the same cell. Report per model: which source(s)
match all 32 numbers, and every mismatch.
Sources (pdftotext -layout of the arXiv PDFs, kept local): TimesNet arXiv 2210.02186v3; SpikF as in make_refs16.py.
usage (from cinc_mixer_v1/): python -B results/ett-full16-20261124/trace_dnn_sources.py <timesnet_layout.txt> <spikf_layout.txt>
      -> results/ett-full16-20261124/trace_dnn_sources.txt
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DSO, HO = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), ('96', '192', '336', '720')


def stop(m):
    print('STOP:', m); sys.exit(2)


def rows_after(path, title, n, k):
    """the first k rows with a horizon label and n numbers after the line containing title."""
    L = open(path, encoding='utf-8').read().splitlines()
    i = next((j for j, l in enumerate(L) if title in l), None)
    if i is None:
        stop(f'{title!r} not found in {path}')
    rx = re.compile(r'(?:^|\s)(96|192|336|720|Avg)\s+((?:\d+\.\d+\s+){%d}\d+\.\d+)' % (n - 1))
    out = []
    for l in L[i:i + 400]:
        m = rx.search(l)
        if m:
            out.append((m.group(1), [float(x) for x in m.group(2).split()]))
        if len(out) == k:
            break
    if len(out) != k:
        stop(f'{title!r}: {len(out)} rows parsed, want {k}')
    return out


def by_dataset(rows, order, first):
    """map (ds, H) -> numbers; rows come in blocks of 5 (96 / 192 / 336 / 720 / Avg) in the given dataset order."""
    T = {}
    for b, ds in enumerate(order):
        blk = rows[5 * b:5 * b + 5]
        if [h for h, _ in blk] != list(HO) + ['Avg']:
            stop(f'{ds}: horizon labels {[h for h, _ in blk]}')
        if ds in DSO:
            for h, v in blk[:4]:
                T[(ds, h)] = v
    if sorted(T) != sorted((d, h) for d in DSO for h in HO):
        stop(f'{first}: ETT cells incomplete')
    return T


tn, sk = sys.argv[1], sys.argv[2]
# TimesNet Table 13: 13 models x (MSE, MAE); dataset order ETTm1, ETTm2, ETTh1, ETTh2, ...
TN = by_dataset(rows_after(tn, 'Table 13: Full results for the long-term forecasting task', 26, 20), ('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2'), 'TimesNet T13')
TN_COL = {'ETSformer': 1, 'FEDformer': 4, 'Autoformer': 6, 'Informer': 8}
# SpikF Table 11: 9 models; dataset order ECL, Weather, ETTh1, ETTh2, ETTm1, ETTm2, ...
SK11 = by_dataset(rows_after(sk, 'Table 11. Full results of long-term prediction', 18, 30), ('ECL', 'Weather', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), 'SpikF T11')
SK11_COL = {'PatchTST': 3, 'Autoformer': 8}
# SpikF Table 12: SpikF, SpikeRNN, iSpikformer, FITS, FEDformer; dataset order ETTh1, ETTh2, ETTm1, ETTm2
SK12 = by_dataset(rows_after(sk, 'Table 12. More benchmarks on ETT dataset', 10, 20), ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), 'SpikF T12')
SK12_COL = {'FEDformer': 4}
SRC = {'TimesNet Table 13': (TN, TN_COL), 'SpikF Table 11': (SK11, SK11_COL), 'SpikF Table 12': (SK12, SK12_COL)}
REF = json.load(open(os.path.join(HERE, 'refs16.json')))['models']
L = ['# DNN rows of NSMT Table 13 traced to their published sources (input 96, 16 ETT cells, MSE and MAE)', '',
     f'TimesNet text md5 {hashlib.md5(open(tn, "rb").read()).hexdigest()}, SpikF text md5 {hashlib.md5(open(sk, "rb").read()).hexdigest()}', '',
     '| model (NSMT Table 13) | source checked | numbers equal (of 32) | mismatches |', '|---|---|--:|---|']
untraced = 0
for m in ('Informer', 'FEDformer', 'PatchTST', 'ETSformer', 'Autoformer'):
    if REF[m]['role'] != 'DNN':
        stop(f'{m} is not a DNN row in refs16.json')
    anyhit = {(d, h, k): False for d in DSO for h in HO for k in (0, 1)}
    for sn, (T, col) in SRC.items():
        if m not in col:
            continue
        j = col[m]; eq, bad = 0, []
        for d in DSO:
            for h in HO:
                for k in (0, 1):
                    a, b = REF[m]['rows'][f'{d} {h}'][k], T[(d, h)][2 * j + k]
                    if a == b:
                        eq += 1; anyhit[(d, h, k)] = True
                    else:
                        bad.append(f'{d} H{h} {"MSE" if k == 0 else "MAE"}: NSMT {a} vs {b}')
        L.append(f'| {m} | {sn} | {eq} | {"none" if not bad else "; ".join(bad)} |')
    miss = [k for k, v in anyhit.items() if not v]
    untraced += len(miss)
    if miss:
        L.append(f'| {m} | **not traced to any source** | — | ' + '; '.join(f'{d} H{h} {"MSE" if k == 0 else "MAE"}' for d, h, k in miss) + ' |')
L += ['', f'Numbers not found in any published source: {untraced} of 160.']
open(os.path.join(HERE, 'trace_dnn_sources.txt'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
