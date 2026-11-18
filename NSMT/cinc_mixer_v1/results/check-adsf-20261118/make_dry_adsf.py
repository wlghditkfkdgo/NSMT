"""Dry-run input for summarize_adsf.py / recheck_adsf.py (2026-11-18, PROJECT_LOG AD SF-de): the spikingjelly Spikformer-style AD rows of
11-02 / 11-03 (flif-v1-ad-spkmux-tune, cond spk_ssa) with only the tag renamed (+ '_nalif'), so the expected values are known:
selection = the 11-03 SF-sj selection (MSL 2.5e-4, SMAP 2e-3, PSM 5e-4); E1 = 11-03 A1 (R +1.05 %, 6/9, energy 1.03); SF-de vs SF-sj = 0.
usage: python3 make_dry_adsf.py <out folder>   then   ADSF_DRY=<out folder> python -B summarize_adsf.py ... / recheck_adsf.py ..."""
import os, re, sys
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lines = [l for l in open(f'{R}/flif-v1-ad-spkmux-tune/raw.txt') if l.startswith('RESULT') and ' cond=spk_ssa ' in l]
with open(f'{OUT}/raw.txt', 'w') as f:
    for l in lines:
        t = re.search(r' tag=(\S+) ', l)[1]
        f.write(l.replace(f' tag={t} ', f' tag={t}_nalif ', 1))
print(len(lines), 'rows written to', OUT)
