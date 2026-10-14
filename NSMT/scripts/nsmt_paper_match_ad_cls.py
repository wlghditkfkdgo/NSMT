"""Match the NSMT paper's anomaly-detection (Table 9, P/R/F1 %) and UEA classification (Table 10,
accuracy %) cells to saved runs under <repo-root>/anomaly_detection and <repo-root>/classification.

usage: python nsmt_paper_match_ad_cls.py <repo-root>
A run = a directory holding log/final+result.csv (first data row) and logargs.txt.
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsmt_paper_match_runs import read_args                 # noqa: E402

ROOT = Path(sys.argv[1])
AD = {'SMD': (88.83, 85.13, 86.94), 'MSL': (88.24, 67.54, 76.51), 'SMAP': (91.61, 64.57, 75.75),
      'PSM': (98.54, 92.56, 95.46), 'SWAT': (92.34, 92.78, 92.56)}                 # Table 9, P R F1 (%)
CLS = {'EthanolConcentration': 30.8, 'FaceDetection': 55.3, 'Handwriting': 43.8,
       'Heartbeat': 79.0, 'JapaneseVowels': 98.9}                                 # Table 10, acc (%)
AD_KEYS = ['model', 'seed', 'gating', 'epoch', 'batch_size', 'lr', 'patience', 'alpha', 'keep_ratio', 'embed_dim',
           'num_heads', 'num_layers', 'time_num_layers', 'patch_size', 'mlp_ratios', 'seq_len', 'anomaly_ratio',
           'c_out', 'bias', 'tau']
CLS_KEYS = ['model', 'seed', 'gating', 'attn', 'epoch', 'batch_size', 'lr', 'max_lr', 'scheduler', 'alpha',
            'keep_ratio', 'embed_dim', 'num_heads', 'num_layers', 'time_num_layers', 'patch_size', 'stride',
            'time_steps', 'mlp_ratios', 'bias', 'tau', 'num_classes']


def runs_under(folder):
    for csvp in (ROOT / folder).rglob('final+result.csv'):
        if 'dataset' in csvp.parts:
            continue
        run = csvp.parent.parent
        ap = run / 'logargs.txt'
        if not ap.exists():
            continue
        rows = list(csv.reader(csvp.open()))
        if len(rows) < 2:
            continue
        header = [h.strip() for h in rows[0]]
        vals = [v.strip() for v in rows[1]]
        rec = {}
        for h, v in zip(header, vals):                       # 'xx M Ops' 같은 문자열 칸은 건너뛴다
            try:
                rec[h] = float(v)
            except ValueError:
                pass
        yield run, read_args(ap), rec


def show(run, a, keys):
    print(f'   {run.relative_to(ROOT)}')
    print('   ' + ' '.join(f'{k}={a.get(k)}' for k in keys))


print('## anomaly detection (Table 9): F1 to 2 decimals, then P and R')
ad_runs = [(r, a, m) for r, a, m in runs_under('anomaly_detection') if 'f-score' in m]
print(f'{len(ad_runs)} runs with results')
for ds, (p, r_, f1) in AD.items():
    full = [(r, a, m) for r, a, m in ad_runs if a.get('data', '').upper() == ds
            and round(100 * m['f-score'], 2) == f1 and round(100 * m['precision'], 2) == p
            and round(100 * m['recall'], 2) == r_]
    f1_only = [(r, a, m) for r, a, m in ad_runs if a.get('data', '').upper() == ds
               and round(100 * m['f-score'], 2) == f1]
    print(f'\n== {ds}: paper P {p} R {r_} F1 {f1} -> {len(full)} full match, {len(f1_only)} F1-only')
    for r, a, m in (full or f1_only):
        print(f"   P {100 * m['precision']:.2f} R {100 * m['recall']:.2f} F1 {100 * m['f-score']:.2f}")
        show(r, a, AD_KEYS)

print('\n## classification (Table 10): overall accuracy to 1 decimal')
cls_runs = [(r, a, m) for r, a, m in runs_under('classification') if 'overall_acc' in m]
print(f'{len(cls_runs)} runs with results')
for ds, acc in CLS.items():
    cands = [(r, a, m) for r, a, m in cls_runs if ds in str(r) and round(100 * m['overall_acc'], 1) == acc]
    print(f'\n== {ds}: paper acc {acc} -> {len(cands)} matching run(s)')
    for r, a, m in cands:
        print(f"   acc {100 * m['overall_acc']:.2f}")
        show(r, a, CLS_KEYS)
