"""Dry suites for summarize_ade.py (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG AD energy pre-registration). Built from the 11-03 rows of
results/flif-v1-ad-spkmux-tune/raw.txt (muxlin), relabelled; nothing is trained. Expected (checked by check_dry_ade.py):
  stage lr  : the 11-03 lr-stage rows (OR, M 2, holdout) relabelled M 1 -> selected lr = the 11-03 muxlin selection (MSL 0.002, SMAP 0.002, PSM 0.001)
  stage res : OR = those rows at the selected lr; IAND / ADD = the 11-03 residual-stage rows (MSL M 2, SMAP M 1, PSM M 3) relabelled M 1
              -> the selection equals a direct recomputation from the raw numbers (check_dry_ade.py)
  stage final (folder final/, selections written here as the 11-03 tuned settings, spike = MSL or, SMAP or, PSM iand):
              (B) M1 = the 11-03 tuned finals relabelled M 1; (A) mean = the same rows tagged _mpmean; (B) 0/1 residual for SMAP / PSM =
              the 11-03 muxlin(default) finals relabelled -> G1 = G3 = E1 of the 11-03 SF-de summary (syops energy 1.1937, current 1.0499,
              R +1.368), G2 = G4 = R 0 and energy 1 exactly, G5 present.
usage (from cinc_mixer_v1/): python -B results/check-ade-20261119/make_dry_ade.py <out folder>
"""
import json
import os
import re
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TUNE = os.path.join(C, 'results', 'flif-v1-ad-spkmux-tune')
out = sys.argv[1]
os.makedirs(os.path.join(out, 'sel'), exist_ok=False); os.makedirs(os.path.join(out, 'final'))
L = [l for l in open(os.path.join(TUNE, 'raw.txt')) if l.startswith('RESULT') and ' cond=spkmx_lif_muxlin_' in l]
f = lambda l, k: re.search(rf' {k}=(\S+)', l).group(1)


def relabel(l, res=None, lr=None, layers=1, mean=False):
    r0, lr0, la0, ho = f(l, 'cond').split('_')[3], f(l, 'lr'), f(l, 'layers'), f(l, 'holdout') == '1'
    res, lr = res or r0, lr or lr0
    tag = f"spkmx_lif_muxlin_{res}_lr{lr}_L{layers}{'_ho' if ho else ''}{'_mpmean' if mean else ''}"
    l = re.sub(r' cond=\S+', f' cond=spkmx_lif_muxlin_{res}', l)
    l = re.sub(r' tag=\S+', f' tag={tag}', l)
    l = re.sub(r' lr=\S+', f' lr={lr}', l)
    return re.sub(r' layers=\S+', f' layers={layers}', l)


sel_lr = json.load(open(os.path.join(TUNE, 'selected_lr.json')))['selected']['spkmx_lif_muxlin_or']
sel_lay = json.load(open(os.path.join(TUNE, 'selected_layers.json')))['selected']['spkmx_lif_muxlin_or']
sel_res = json.load(open(os.path.join(TUNE, 'selected_res.json')))['selected']['spkmx_lif_muxlin_or']
g = lambda l, **kw: all(f(l, k) == str(v) for k, v in kw.items())
with open(os.path.join(out, 'sel', 'raw.txt'), 'w') as w:
    for l in L:                                                                    # lr stage: OR, M 2, holdout -> M 1
        if g(l, holdout=1, layers=2) and f(l, 'cond').endswith('_or'):
            w.write(relabel(l))
    for l in L:                                                                    # res stage: IAND / ADD at the 11-03 lr and layers -> M 1
        ds = f(l, 'ds')
        if g(l, holdout=1) and f(l, 'cond').split('_')[3] in ('iand', 'add') and float(f(l, 'lr')) == sel_lr[ds] and int(f(l, 'layers')) == sel_lay[ds]:
            w.write(relabel(l))
with open(os.path.join(out, 'final', 'raw.txt'), 'w') as w:
    for l in L:
        ds = f(l, 'ds')
        if not g(l, holdout=0):
            continue
        r = f(l, 'cond').split('_')[3]
        if r == sel_res[ds] and float(f(l, 'lr')) == sel_lr[ds] and int(f(l, 'layers')) == sel_lay[ds]:
            w.write(relabel(l, lr=f'{sel_lr[ds]:g}'))                                                   # (B) M1
            w.write(relabel(l, lr=f'{sel_lr[ds]:g}', layers=sel_lay[ds], mean=True))                    # (A) mean
        if ds in ('SMAP', 'PSM') and r == 'or' and float(f(l, 'lr')) == 0.0005 and int(f(l, 'layers')) == 2:
            w.write(relabel(l, res={'SMAP': 'or', 'PSM': 'iand'}[ds], lr=f'{sel_lr[ds]:g}'))           # (B) 0/1 residual
json.dump({'selected': {ds: f'{sel_lr[ds]:g}' for ds in sel_lr}}, open(os.path.join(out, 'final', 'selected_lr.json'), 'w'))
json.dump({'selected': {'res': sel_res, 'spike': {'MSL': 'or', 'SMAP': 'or', 'PSM': 'iand'}}}, open(os.path.join(out, 'final', 'selected_res.json'), 'w'))
print('dry suites in', out, {k: sum(1 for _ in open(os.path.join(out, k, 'raw.txt'))) for k in ('sel', 'final')})
