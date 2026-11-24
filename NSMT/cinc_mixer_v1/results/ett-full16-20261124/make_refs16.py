"""Reference rows for the full ETT table (4 datasets x H 96 / 192 / 336 / 720, input 96) (2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG
full-ETT-table pre-registration; user decisions 11-23 / 11-24). No training.
Reads the pdftotext -layout text of the two papers (kept local) and writes refs16.json (kept in Git):
  SNN   NSMT Table 14 "Full comparisons on SNN-based models" (SpikF, SpikeRNN, iSpikformer, NSMT) and SpikF Table 12 "More benchmarks on
        ETT dataset" (SpikF, SpikeRNN, iSpikformer, ...): the three models present in both must agree on all 20 rows
  DNN   NSMT Table 13 "Full results on full data" (Informer, FEDformer, PatchTST, ETSformer, Autoformer, NSMT); its NSMT column must
        equal the NSMT column of Table 14 on all 20 rows
Checks (fail closed, exit 2): 20 rows parsed per table in the order ETTh1 / ETTh2 / ETTm1 / ETTm2 x 96 / 192 / 336 / 720 / Avg; the two
agreements above; the 4 cells used so far equal pass_ett_nsmt.DNN_NSMT and the 11-19 transcription (ref-snn-lb96-20261119).
Report only: for every model and dataset, |Avg row - mean of the four horizons| (the papers' own rounding; > 0.0015 is listed).
usage (from cinc_mixer_v1/): python -B results/ett-full16-20261124/make_refs16.py <nsmt_layout.txt> <spikf_layout.txt>
      -> results/ett-full16-20261124/{refs16.json, refs16_verify.txt}
"""
import hashlib
import json
import os
import re
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, C)
sys.path.insert(0, os.path.join(C, 'results', 'ett-pass-20261123'))
import pass_ett_nsmt as pn                                                           # noqa: E402
DSO, HO = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), ('96', '192', '336', '720', 'Avg')


def stop(m):
    print('STOP:', m); sys.exit(2)


def table_rows(path, title, n):
    L = open(path, encoding='utf-8').read().splitlines()
    i = next(k for k, l in enumerate(L) if title in l)
    rx = re.compile(r'(?:^|\s)(96|192|336|720|Avg)\s+((?:\d\.\d{3}\s+){%d}\d\.\d{3})' % (n - 1))
    rows = []
    for l in L[i:i + 120]:
        m = rx.search(l)
        if m:
            rows.append((m.group(1), [float(x) for x in m.group(2).split()]))
        if len(rows) == 20:
            break
    if [h for h, _ in rows] != list(HO) * 4:
        stop(f'{title}: rows not parsed: {[h for h, _ in rows]}')
    return {(DSO[k // 5], rows[k][0]): rows[k][1] for k in range(20)}


nsmt, spikf = sys.argv[1], sys.argv[2]
T14 = table_rows(nsmt, 'Table 14: Full comparisons on SNN-based models', 8)          # SpikF, SpikeRNN, iSpikformer, NSMT
T12 = table_rows(spikf, 'Table 12. More benchmarks on ETT dataset', 10)              # SpikF, SpikeRNN, iSpikformer, FITS, FEDformer
T13 = table_rows(nsmt, 'Table 13: Full results on full data', 12)                    # Informer, FEDformer, PatchTST, ETSformer, Autoformer, NSMT
V = []
if not all(T14[k][:6] == T12[k][:6] for k in T14):
    stop('NSMT Table 14 != SpikF Table 12 for SpikF / SpikeRNN / iSpikformer: ' + str([k for k in T14 if T14[k][:6] != T12[k][:6]]))
V.append('NSMT Table 14 = SpikF Table 12 for SpikF / SpikeRNN / iSpikformer on all 20 rows: True')
if not all(T13[k][10:12] == T14[k][6:8] for k in T13):
    stop('NSMT column of Table 13 != Table 14: ' + str([k for k in T13 if T13[k][10:12] != T14[k][6:8]]))
V.append('NSMT column of Table 13 = NSMT column of Table 14 on all 20 rows: True')
R = {}
for j, m in enumerate(('SpikF', 'SpikeRNN', 'iSpikformer', 'NSMT')):
    R[m] = {'role': 'SNN' if m != 'NSMT' else 'own', 'source': 'SpikF Table 12 = NSMT Table 14' if m != 'NSMT' else 'NSMT Table 14 = Table 13',
            'rows': {f'{ds} {h}': T14[(ds, h)][2 * j:2 * j + 2] for ds in DSO for h in HO}}
for j, m in enumerate(('Informer', 'FEDformer', 'PatchTST', 'ETSformer', 'Autoformer')):
    R[m] = {'role': 'DNN', 'source': 'NSMT Table 13', 'rows': {f'{ds} {h}': T13[(ds, h)][2 * j:2 * j + 2] for ds in DSO for h in HO}}
n = 0
for m, (ms, ma, _) in pn.DNN_NSMT.items():                                            # the 4 cells judged so far
    for c, (ds, pl) in enumerate(pn.pe.CELLS):
        n += 2
        if R[m]['rows'][f'{ds} {pl}'] != [ms[c], ma[c]]:
            stop(f'{m} {ds} H{pl}: {R[m]["rows"][f"{ds} {pl}"]} != pass_ett_nsmt {[ms[c], ma[c]]}')
ref = open(os.path.join(C, 'results', 'ref-snn-lb96-20261119', 'compare_snn_lb96.py')).read()
ns = {}
exec(ref[ref.index('P = {'):ref.index('MODELS =')], ns)
for (ds, pl), mm in ns['P'].items():
    for m, v in mm.items():
        n += 2
        if tuple(R[m]['rows'][f'{ds} {pl}']) != v:
            stop(f'{m} {ds} H{pl}: {R[m]["rows"][f"{ds} {pl}"]} != 11-19 transcription {v}')
V.append(f'4 cells used so far (ETTh2 / ETTm2 x H96 / H720) equal pass_ett_nsmt.DNN_NSMT and the 11-19 transcription: {n} numbers, 0 mismatches')
odd = []
for m in R:
    for ds in DSO:
        for k in (0, 1):
            avg = sum(R[m]['rows'][f'{ds} {h}'][k] for h in HO[:4]) / 4
            d = abs(avg - R[m]['rows'][f'{ds} Avg'][k])
            if d > 0.0015:
                odd.append(f'{m} {ds} {"MSE" if k == 0 else "MAE"}: Avg row {R[m]["rows"][f"{ds} Avg"][k]} vs mean of horizons {avg:.4f}')
V.append(f'Avg row vs mean of the four horizons (report; the papers\' own numbers): {len(odd)} of {len(R) * 8} differ by > 0.0015'
         + ('' if not odd else ': ' + '; '.join(odd)))
md5 = {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest() for p in (nsmt, spikf)}
V.append(f'source text md5: nsmt {md5[os.path.basename(nsmt)]}, spikf {md5[os.path.basename(spikf)]}')
json.dump({'datasets': DSO, 'horizons': HO, 'models': R, 'source_md5': {'nsmt_layout': hashlib.md5(open(nsmt, 'rb').read()).hexdigest(),
                                                                         'spikf_layout': hashlib.md5(open(spikf, 'rb').read()).hexdigest()}},
          open(os.path.join(OUT, 'refs16.json'), 'w'), indent=1)
open(os.path.join(OUT, 'refs16_verify.txt'), 'w').write('\n'.join(V) + '\n')
print('\n'.join(V))
