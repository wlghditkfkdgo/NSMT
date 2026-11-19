"""Independent recheck of the AD energy test summary (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG AD energy result entry).
Imports nothing from summarize_*: parses the RESULT lines of the three suites itself, picks the rows by their tags, and recomputes
G1-G5 and the two report rows (R, pairs higher, verdict, energy A/B under syops = total_fft_uJ and the old rule = total_fft_uJ_corr,
paper bar), then compares them with the main table of results/flif-v1-ad-energy/summary.md (4-decimal / 3-decimal printed values).
Rows (test AUC-PR, holdout=0, seeds 42 / 7 / 2026):
  (A) mean  = this suite, the 11-03 tuned muxlin setting + _mpmean          (B) M1 = this suite, selected_res.json "res", M 1
  (B) spike = this suite, selected_res.json "spike" where it differs         SF-de = flif-v1-ad-sfde, its selected lr, tag _nalif
  muxlin tuned / default = flif-v1-ad-spkmux-tune (11-03 selections / OR, 5e-4, M 2)
Rule: r = 100 (A - B) / B per seed, R = mean over MSL / SMAP / PSM of the seed-mean r; better = R >= +2 and >= 6 of 9 higher, worse =
mirror. Energy A/B = mean over datasets of seed-mean A / seed-mean B. Paper bar = R > 0 and syops A/B < 1.
Fail closed: missing / duplicate row, SKIP line, non-finite value, energy_check != ok -> exit 2.
usage (from cinc_mixer_v1/): python -B results/check-ade-20261119/recheck_ade.py > results/check-ade-20261119/recheck_ade.txt
"""
import json
import math
import os
import re
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R_ = os.path.join(C, 'results')
DSS, SEEDS = ('MSL', 'SMAP', 'PSM'), ('42', '7', '2026')


def stop(m):
    print('STOP:', m)
    sys.exit(2)


def rows(suite):
    p = os.path.join(R_, suite, 'raw.txt')
    out = []
    for l in open(p):
        if l.startswith('SKIP') and suite == 'flif-v1-ad-energy':
            stop(f'SKIP in {suite}')
        if l.startswith('RESULT'):
            out.append(dict(kv.split('=', 1) for kv in l.split()[1:] if '=' in kv))
    return out


def pick(rs, ds, tag, seed):
    m = [r for r in rs if r['ds'] == ds and r['tag'] == tag and r['holdout'] == '0' and r['seed'] == seed]
    if len(m) != 1:
        stop(f'{len(m)} rows for {ds} {tag} seed {seed}')
    r = m[0]
    for k in ('aucpr', 'total_fft_uJ', 'total_fft_uJ_corr'):
        if not math.isfinite(float(r[k])):
            stop(f'non-finite {k} {ds} {tag} {seed}')
    if r.get('energy_check') != 'ok':
        stop(f'energy_check {r.get("energy_check")} {ds} {tag} {seed}')
    return r


E, SF, TU = rows('flif-v1-ad-energy'), rows('flif-v1-ad-sfde'), rows('flif-v1-ad-spkmux-tune')
sel = lambda f: json.load(open(os.path.join(R_, f)))['selected']
t_lr, t_ly, t_rs = (sel(f'flif-v1-ad-spkmux-tune/selected_{s}.json')['spkmx_lif_muxlin_or'] for s in ('lr', 'layers', 'res'))
e_lr, e_rs, s_lr = sel('flif-v1-ad-energy/selected_lr.json'), sel('flif-v1-ad-energy/selected_res.json'), sel('flif-v1-ad-sfde/selected_lr.json')
g = lambda v: f'{float(v):g}'
ARM = {
    'A_mean': (E, lambda d: f'spkmx_lif_muxlin_{t_rs[d]}_lr{g(t_lr[d])}_L{t_ly[d]}_mpmean'),
    'B_M1': (E, lambda d: f'spkmx_lif_muxlin_{e_rs["res"][d]}_lr{g(e_lr[d])}_L1'),
    'B_spike': (E, lambda d: f'spkmx_lif_muxlin_{e_rs["spike"][d]}_lr{g(e_lr[d])}_L1'),
    'SFde': (SF, lambda d: f'spk_ssa_lr{g(s_lr[d])}_L2_nalif'),
    'mux_tuned': (TU, lambda d: f'spkmx_lif_muxlin_{t_rs[d]}_lr{g(t_lr[d])}_L{t_ly[d]}'),
    'mux_default': (TU, lambda d: 'spkmx_lif_muxlin_or_lr0.0005_L2'),
}
V = {a: {d: [pick(src, d, f(d), s) for s in SEEDS] for d in DSS} for a, (src, f) in ARM.items()}
n_e = sum(1 for r in E)
if n_e != 90:
    stop(f'{n_e} RESULT rows in flif-v1-ad-energy, want 90')


def comp(a, b):
    rd, hi = {}, 0
    for d in DSS:
        rs = []
        for x, y in zip(V[a][d], V[b][d]):
            rs.append(100 * (float(x['aucpr']) - float(y['aucpr'])) / float(y['aucpr']))
            hi += float(x['aucpr']) > float(y['aucpr'])
        rd[d] = sum(rs) / 3
    R = sum(rd.values()) / 3
    verdict = 'better' if R >= 2 and hi >= 6 else 'worse' if R <= -2 and (9 - hi) >= 6 else 'no clear difference'
    en = {}
    for k in ('total_fft_uJ', 'total_fft_uJ_corr'):
        en[k] = sum((sum(float(x[k]) for x in V[a][d]) / 3) / (sum(float(y[k]) for y in V[b][d]) / 3) for d in DSS) / 3
    return R, hi, verdict, en['total_fft_uJ'], en['total_fft_uJ_corr'], rd


CMP = [('G1', 'B_M1', 'SFde'), ('G2', 'B_M1', 'mux_tuned'), ('G3', 'A_mean', 'SFde'), ('G4', 'A_mean', 'mux_tuned'),
       ('G5', 'B_spike', 'SFde'), ('report tuned', 'mux_tuned', 'SFde'), ('report default', 'mux_default', 'SFde')]
summ = open(os.path.join(R_, 'flif-v1-ad-energy', 'summary.md')).read().split('\n')
trow = [l for l in summ if l.startswith('| G') or l.startswith('| (report) muxlin')]
if len(trow) < 7:
    stop('summary main table not found')
bad = 0
print('comparison | R % | pairs | verdict | syops A/B | old A/B | paper bar (syops) | summary row | match')
for (name, a, b), srow in zip(CMP, trow[:7]):
    R, hi, vd, es, eo, rd = comp(a, b)
    pb = 'met' if (R > 0 and es < 1) else 'not met'
    c = [x.strip() for x in srow.strip('|').split('|')]
    ok = (f'{R:+.3f}' == c[1] and c[2] == f'{hi}/9' and c[3] == vd and f'{es:.4f}' == c[4] and f'{eo:.4f}' == c[6]
          and (c[5] == '' or c[5] == pb))
    bad += not ok
    print(f'{name} | {R:+.3f} | {hi}/9 | {vd} | {es:.4f} | {eo:.4f} | {pb} | {c[0]} | {"ok" if ok else "MISMATCH"}')
    print('   per dataset r:', ', '.join(f'{d} {rd[d]:+.2f}' for d in DSS))
# SMAP (B) M1 is the 11-03 tuned setting (ADD, 2e-3, M 1): the rows must be bit-identical (training is bit-reproducible)
same = all(x['aucpr'] == y['aucpr'] and x['total_fft_uJ'] == y['total_fft_uJ'] for x, y in zip(V['B_M1']['SMAP'], V['mux_tuned']['SMAP']))
print('SMAP (B) M1 rows identical to the 11-03 tuned rows (aucpr, total_fft_uJ):', 'yes' if same else 'NO')
bad += not same
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} MISMATCH(ES)')
sys.exit(0 if bad == 0 else 2)
