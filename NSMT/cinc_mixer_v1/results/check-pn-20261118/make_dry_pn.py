"""Dry-run inputs for summarize_pn.py / recheck_pn.py (2026-11-18, PROJECT_LOG, exp/f-lif-pop-v3): the new suites rebuilt from EXISTING rows with only
the tag / cond changed, so every expected value is known in advance:
  SF-de (ETT, MIT-BIH) = the spikingjelly Spikformer-style tuning rows (same lr grid)  -> selection = the 11-01 one (ETTh2 5e-4, ETTm2 2.5e-4, MIT-BIH
                         0.005); A1 = the old LIF SpkMixer vs SF-sj values; SF-de vs SF-sj = 0
  grid arms            = the tuned LIF SpkMixer rows                                  -> B / C comparisons between new arms = 0
  input 336            = the 11-18 LIF 336 rows as pop LIF mixer 336, the SF-sj 336 rows as SF-de 336 -> D1 = LIF 336 vs SF-sj 336 (= P2), D2 = 0
usage: python3 make_dry_pn.py <out folder>   then   PN_DRY=<out folder> python -B summarize_pn.py --stage ... / recheck_pn.py ...
"""
import csv
import os
import re
import sys

OUT = sys.argv[1]
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LRM = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}; LYS = {'ETTh2': '1', 'ETTm2': '2'}


def res(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(''.join(lines))


def sub(line, **kv):
    for k, v in kv.items():
        if re.search(rf'\b{k}=\S+', line):
            line = re.sub(rf'\b{k}=\S+', f'{k}={v}', line)
        else:
            line = line.replace('RESULT ', f'RESULT {k}={v} ', 1)
    return line


def en_rows(path):
    return list(csv.DictReader(open(path)))


def write_csv(rows, path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


# --- SF-de ETT: every lr of the spikingjelly grid (ETTh2 5e-4 = baseline rows without tag, the rest from the tuning suite)
raw, en = [], []
tune_raw = [l for l in open(f'{R}/flif-v1-ett-spkmux-tune/raw.txt') if l.startswith('RESULT') and ' cond=spk_ssa ' in l]
tune_en = [e for e in en_rows(f'{R}/flif-v1-ett-spkmux-tune/energy_recount.csv') if e['cond'] == 'spk_ssa']
base_raw = [l for l in open(f'{R}/flif-v1-ett-spkmux/baseline_spk_ssa_raw.txt') if l.startswith('RESULT')]
base_en = en_rows(f'{R}/flif-v1-ett-spkmux/baseline_spk_ssa_energy.csv')
F = list(en_rows(f'{R}/flif-v1-ett-spkmux-tune/energy_recount.csv')[0].keys())
for l in tune_raw:
    t = re.search(r'tag=(\S+)', l)[1]
    raw.append(sub(l, tag=t + '_nalif'))
for e in tune_en:
    raw_e = dict(e); raw_e['tag'] = e['tag'] + '_nalif'; en.append(raw_e)
for l in base_raw:
    ds = re.search(r'ds=(\S+)', l)[1]
    raw.append(sub(l, tag=f'spk_ssa_lr0.0005_L{LYS[ds]}_nalif', lr='0.0005', layers=LYS[ds]))
for e in base_en:
    x = {k: e.get(k, '') for k in F}; x['tag'] = f'spk_ssa_lr0.0005_L{LYS[e["ds"]]}_nalif'; x['rule'] = 'binary'; en.append(x)
res(raw, f'{OUT}/flif-v1-ett-sfde/raw.txt'); write_csv(en, f'{OUT}/flif-v1-ett-sfde/energy_recount.csv', F)

# --- SF-de MIT-BIH: fold-0 tuning rows of the spikingjelly model + its final rows at 0.005
raw = [l for l in open(f'{R}/flif-v1-mitbih-spkmux-tune/raw.txt') if l.startswith('RESULT') and ' cond=spk_ssa ' in l]
raw = [sub(l, tag=re.search(r'tag=(\S+)', l)[1] + '_nalif') for l in raw]
en = []
for e in en_rows(f'{R}/flif-v1-mitbih-spkmux-tune/energy_recount.csv'):
    if e['cond'] == 'spk_ssa':
        e = dict(e); e['tag'] = e['tag'] + '_nalif'; en.append(e)
res(raw, f'{OUT}/flif-v1-mitbih-sfde/raw.txt'); write_csv(en, f'{OUT}/flif-v1-mitbih-sfde/energy_recount.csv', list(en[0].keys()))

# --- grid: the tuned LIF SpkMixer rows under every new arm's tag / cond
ARMS = [('flif', ''), ('flif', '_mpmean'), ('pop_lif', '_mpmean'), ('pop_lif', '_nsmixer'), ('pop_lif', '_mpmean_nsmixer'),
        ('pop_flif', ''), ('pop_flif', '_mpmean'), ('pop_flif', '_nsmixer'), ('pop_flif', '_mpmean_nsmixer')]
lif_e = [l for l in open(f'{R}/flif-v1-ett-spkmux-tune/raw.txt') if l.startswith('RESULT') and re.search(r'tag=spkmx_lif_muxlin_iand_lr(\S+)_L1 ', l)
         and re.search(r'tag=spkmx_lif_muxlin_iand_lr(\S+)_L1 ', l)[1] == LRM[re.search(r'ds=(\S+)', l)[1]]]
lif_ee = [e for e in en_rows(f'{R}/flif-v1-ett-spkmux-tune/energy_recount.csv') if e['tag'] == f'spkmx_lif_muxlin_iand_lr{LRM[e["ds"]]}_L1']
raw, en = [], []
for mx, suf in ARMS:
    for l in lif_e:
        ds = re.search(r'ds=(\S+)', l)[1]
        raw.append(sub(l, cond=f'spkmx_{mx}_muxlin_iand', tag=f'spkmx_{mx}_muxlin_iand_lr{LRM[ds]}_L1{suf}'))
    for e in lif_ee:
        x = dict(e); x['cond'] = f'spkmx_{mx}_muxlin_iand'; x['tag'] = f'spkmx_{mx}_muxlin_iand_lr{LRM[e["ds"]]}_L1{suf}'; en.append(x)
res(raw, f'{OUT}/flif-v1-ett-spkmux-pn/raw.txt'); write_csv(en, f'{OUT}/flif-v1-ett-spkmux-pn/energy_recount.csv', F)
lif_m = [l for l in open(f'{R}/flif-v1-mitbih-spkmux-tune/raw.txt') if l.startswith('RESULT') and ' tag=spkmx_lif_muxlin_add_lr0.02_L2 ' in l]
lif_me = [e for e in en_rows(f'{R}/flif-v1-mitbih-spkmux-tune/energy_recount.csv') if e['tag'] == 'spkmx_lif_muxlin_add_lr0.02_L2']
raw, en = [], []
for mx, suf in ARMS:
    for l in lif_m:
        raw.append(sub(l, cond=f'spkmx_{mx}_muxlin_add', tag=f'spkmx_{mx}_muxlin_add_lr0.02_L2{suf}'))
    for e in lif_me:
        x = dict(e); x['cond'] = f'spkmx_{mx}_muxlin_add'; x['tag'] = f'spkmx_{mx}_muxlin_add_lr0.02_L2{suf}'; en.append(x)
res(raw, f'{OUT}/flif-v1-mitbih-spkmux-pn/raw.txt'); write_csv(en, f'{OUT}/flif-v1-mitbih-spkmux-pn/energy_recount.csv', list(lif_me[0].keys()))

# --- input 336: LIF 336 -> pop LIF mixer 336, SF-sj 336 -> SF-de 336 (the selected lr equals the SF-sj lr in this dry run)
L = f'{R}/flif-v1-ett-spkmux-len336'
raw336 = [l for l in open(f'{L}/raw.txt') if l.startswith('RESULT')]
en336 = en_rows(f'{L}/energy_recount.csv')
raw, en = [], []
for l in raw336:
    t = re.search(r'tag=(\S+)', l)[1]
    if t.startswith('spkmx_lif_'):
        raw.append(sub(l, cond='spkmx_pop_lif_muxlin_iand', tag=t.replace('spkmx_lif_', 'spkmx_pop_lif_').replace('_L1_s336', '_L1_nsmixer_s336')))
    elif t.startswith('spk_ssa_'):
        raw.append(sub(l, tag=t + '_nalif'))
for e in en336:
    t = e['tag']
    if t.startswith('spkmx_lif_'):
        x = dict(e); x['cond'] = 'spkmx_pop_lif_muxlin_iand'; x['tag'] = t.replace('spkmx_lif_', 'spkmx_pop_lif_').replace('_L1_s336', '_L1_nsmixer_s336'); en.append(x)
    elif t.startswith('spk_ssa_'):
        x = dict(e); x['tag'] = t + '_nalif'; en.append(x)
res(raw, f'{OUT}/flif-v1-ett-spkmux-pn336/raw.txt'); write_csv(en, f'{OUT}/flif-v1-ett-spkmux-pn336/energy_recount.csv', list(en336[0].keys()))
print('dry suites written to', OUT)
