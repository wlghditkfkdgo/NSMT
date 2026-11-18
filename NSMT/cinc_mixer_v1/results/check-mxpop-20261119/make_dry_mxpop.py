"""Dry suite for the mixer-only population summary (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG mixer-only population pre-registration).
Writes <out>/flif-v1-mitbih-spkmux-mxpop from the 11-17 pop LIF all MIT-BIH rows (flif-v1-mitbih-spkmux-stft, tag spkmx_pop_lif_muxlin_add_lr0.02_L2)
renamed to the 4 mixer-only arm tags (12 RESULT rows, energy_recount.csv rows, energy_pool.csv with E_pool = E_old for runs without pooling and
0.5 E_old for mean runs), and copies flif-v1-mitbih-sfde (raw.txt, energy_recount.csv, selected_lr.json) and flif-v1-ett-sfde/selected_lr.json.
Expected with PN_DRY=<out>: B3 = 0 exactly (pop LIF mixer = pop LIF all), C3 = C5 = 0, pop f-LIF mixer vs pop LIF mixer = 0, B1 = the 11-17
pop LIF all vs LIF difference, every mixer-only arm vs SF-de = pop LIF all vs SF-de; pooled-rule energy of the mean arms = half the existing.
usage (from cinc_mixer_v1/): python -B results/check-mxpop-20261119/make_dry_mxpop.py <out folder>
"""
import csv
import os
import shutil
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(C, 'results')
SRC, TAG0 = os.path.join(RES, 'flif-v1-mitbih-spkmux-stft'), 'spkmx_pop_lif_muxlin_add_lr0.02_L2'
NEW = {'spkmx_pop_lif_muxlin_add_lr0.02_L2_nsmixer': 0, 'spkmx_pop_lif_muxlin_add_lr0.02_L2_mpmean_nsmixer': 1,
       'spkmx_pop_flif_muxlin_add_lr0.02_L2_nsmixer': 0, 'spkmx_pop_flif_muxlin_add_lr0.02_L2_mpmean_nsmixer': 1}
out = sys.argv[1]
D = os.path.join(out, 'flif-v1-mitbih-spkmux-mxpop'); os.makedirs(D, exist_ok=False)
src = [l for l in open(os.path.join(SRC, 'raw.txt')) if l.startswith('RESULT') and f' tag={TAG0} ' in l and ' val_fold=-1 ' in l]
assert len(src) == 3, len(src)
with open(os.path.join(D, 'raw.txt'), 'w') as f:
    for t in NEW:
        for l in src:
            f.write(l.replace(f' tag={TAG0} ', f' tag={t} ').replace(' cond=spkmx_pop_lif_muxlin_add ', f" cond={t.split('_lr')[0]} "))
en = [r for r in csv.DictReader(open(os.path.join(SRC, 'energy_recount.csv'))) if r['tag'] == TAG0 and str(r.get('val_fold', '-1')) == '-1']
assert len(en) == 3, len(en)
with open(os.path.join(D, 'energy_recount.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(en[0])); w.writeheader()
    for t in NEW:
        for r in en:
            w.writerow(dict(r, tag=t))
with open(os.path.join(D, 'energy_pool.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['suite', 'tag', 'seed', 'f1', 'E_old', 'E_pool', 'pooled_layers']); w.writeheader()
    for t, mean_run in NEW.items():
        for r in en:
            e = float(r['total_fft_uJ_corr'])
            w.writerow(dict(suite='dry', tag=t, seed=r['seed'], f1=r['f1_logged'], E_old=f'{e:.6f}', E_pool=f'{e * (0.5 if mean_run else 1):.6f}', pooled_layers=''))
S = os.path.join(out, 'flif-v1-mitbih-sfde'); os.makedirs(S)
for n in ('raw.txt', 'energy_recount.csv', 'selected_lr.json'):
    shutil.copy(os.path.join(RES, 'flif-v1-mitbih-sfde', n), S)
os.makedirs(os.path.join(out, 'flif-v1-ett-sfde'))                                  # recheck_pn.sel() also reads the ETT selection
shutil.copy(os.path.join(RES, 'flif-v1-ett-sfde', 'selected_lr.json'), os.path.join(out, 'flif-v1-ett-sfde'))
print('dry suites in', out)
