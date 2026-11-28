"""Dry run of select_mitres.py, summarize_mitres.py and summarize_tshuf.py with planted rows (2026-11-28, exp/f-lif-pop-v3; no training).
A temporary results root holds the real reference suites (symlinks) and fake flif-v1-mitbih-res / flif-v1-mitbih-tshuf suites built from
the ADD base rows (fields edited), with energy recount files and a fake spread JSON. Planted readings:
  (A) IAND validation best at lr 0.005 -> selected 0.005; MS tie 0.01 = 0.02 -> keeps 0.02; final IAND = ADD - 1 %p (worse),
      MS = ADD - 0.1 %p (no clear difference), both 7.9 uJ -> candidates [MS], recommendation MS; shuffled = base - 1 %p -> S1 worse,
      S2 (vs memoryless 90.13) better
  (B) IAND = ADD -> candidates [IAND, MS], recommendation IAND
  (C) IAND worse and MS at 20 uJ (> 1.5x SF-de) -> recommendation none
Stop cases (exit 2): a missing validation run (select), mixed commits (select), a final row at a lr that was not selected, a recount
mismatch (summarize_mitres), a SKIP line (summarize_tshuf), a spread default reading != logged (summarize_tshuf).
usage (from cinc_mixer_v1/): python -B results/check-mitres-20261128/dry_mitres.py
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEL = os.path.join(C, 'results', 'check-mitres-20261128', 'select_mitres.py')
SUMR = os.path.join(C, 'results', 'check-mitres-20261128', 'summarize_mitres.py')
SUMT = os.path.join(C, 'results', 'check-tshuf-20261128', 'summarize_tshuf.py')
REFS = ('flif-v1-mitbih-spkmux-tune', 'flif-v1-mitbih-sfde', 'flif-v1-mitbih-sfde-nope', 'flif-v1-mitbih-memoff', 'flif-v1-mitbih-spkmux-ms')
SEEDS = ('42', '7', '2026')
fails = 0


def check(label, c):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}', flush=True)


def parse(l):
    return dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)


BASE = {parse(l)['seed']: l.rstrip('\n') for l in open(os.path.join(C, 'results', 'flif-v1-mitbih-spkmux-tune', 'raw.txt'))
        if l.startswith('RESULT') and parse(l)['tag'] == 'spkmx_lif_muxlin_add_lr0.02_L2' and parse(l).get('val_fold') == '-1'}


def edit(line, **kv):
    for k, v in kv.items():
        if re.search(rf' {k}=\S+', line):
            line = re.sub(rf' {k}=\S+', f' {k}={v}', line)
        else:
            line = line.replace(' dataset=', f' {k}={v} dataset=', 1)
    return line


def build(iand_val, ms_val, iand_d, ms_d, ms_e=7.9, shuf_d=-1., sel_override=None, mixed=False, drop_val=False, bad_recount=False,
          skip_tshuf=False, bad_spread=False):
    root = tempfile.mkdtemp()
    for s in REFS:
        os.symlink(os.path.join(C, 'results', s), os.path.join(root, s))
    rd = os.path.join(root, 'flif-v1-mitbih-res'); os.makedirs(rd)
    L, E = [], []
    for k, cond, vals in (('iand', 'spkmx_lif_muxlin_iand', iand_val), ('ms', 'spkmx_lif_muxlin_ms', ms_val)):
        for lr, v in vals.items():
            for s in SEEDS:
                if drop_val and k == 'iand' and lr == '0.01' and s == '7':
                    continue
                tag = f'{cond}_lr{lr}_L2_v0'
                sha = 'bbbbbbbbb' if (mixed and k == 'ms' and lr == '0.0025' and s == '42') else 'aaaaaaaaa'
                L.append(edit(BASE[s], cond=cond, tag=tag, lr=lr, val_fold='0', val_f1_last=f'{v / 100:.6f}', sha=sha))
                E.append(dict(tag=tag, seed=s, f1_recomputed=parse(BASE[s])['f1']))
    open(os.path.join(rd, 'raw.txt'), 'w').write('\n'.join(L) + '\n')
    return root, rd, L, E


def finals(root, rd, L, E, iand_d, ms_d, ms_e, sel, bad_recount=False):
    for k, cond, dd, e in (('IAND', 'spkmx_lif_muxlin_iand', iand_d, 7.9), ('MS', 'spkmx_lif_muxlin_ms', ms_d, ms_e)):
        lr = sel[k]
        for s in SEEDS:
            f1 = float(parse(BASE[s])['f1']) + dd / 100.
            tag = f'{cond}_lr{lr}_L2'
            L.append(edit(BASE[s], cond=cond, tag=tag, lr=lr, f1=f'{f1:.6f}', syn_energy_uJ=f'{e:.2f}', sha='aaaaaaaaa'))
            E.append(dict(tag=tag, seed=s, f1_recomputed=f'{f1 + (0.01 if (bad_recount and k == "MS" and s == "7") else 0.):.6f}'))
    open(os.path.join(rd, 'raw.txt'), 'w').write('\n'.join(L) + '\n')
    with open(os.path.join(rd, 'energy_recount.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['tag', 'seed', 'f1_recomputed']); w.writeheader(); w.writerows(E)


def tshuf(root, shuf_d, skip=False, bad_spread=False):
    td = os.path.join(root, 'flif-v1-mitbih-tshuf'); os.makedirs(td)
    L, E, S = [], [], []
    for s in SEEDS:
        f1 = float(parse(BASE[s])['f1']) + shuf_d / 100.
        L.append(edit(BASE[s], tag='spkmx_lif_muxlin_add_lr0.02_L2_tshuf', f1=f'{f1:.6f}', sha='ccccccccc'))
        E.append(dict(tag='spkmx_lif_muxlin_add_lr0.02_L2_tshuf', seed=s, f1_recomputed=f'{f1:.6f}'))
        S.append(dict(seed=s, default=100 * f1 + (0.5 if (bad_spread and s == '42') else 0.), seed1=100 * f1, seed2=100 * f1, natural=100 * f1, spread=0., natural_minus_default=0.))
    if skip:
        L.append('SKIP(rc=1, no result) cond=spkmx_lif_muxlin_add seed=7')
    open(os.path.join(td, 'raw.txt'), 'w').write('\n'.join(L) + '\n')
    with open(os.path.join(td, 'energy_recount.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['tag', 'seed', 'f1_recomputed']); w.writeheader(); w.writerows(E)
    sp = os.path.join(root, 'spread.json'); json.dump(dict(tag='spkmx_lif_muxlin_add_lr0.02_L2_tshuf', runs=S), open(sp, 'w'))
    return sp


def run(script, *args):
    p = subprocess.run([sys.executable, '-B', script] + list(args), capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


IV = {'0.0025': 90.0, '0.005': 92.0, '0.01': 91.0, '0.02': 91.7}
MV = {'0.0025': 90.0, '0.005': 91.0, '0.01': 92.5, '0.02': 92.5}
for case, iand_d, ms_d, ms_e, want_rec, want_cand in (('A', -1., -0.1, 7.9, 'MS', ['MS']), ('B', 0., -0.1, 7.9, 'IAND', ['IAND', 'MS']),
                                                      ('C', -1., 0., 20., 'none (keep ADD)', [])):
    root, rd, L, E = build(IV, MV, iand_d, ms_d)
    rc, out = run(SEL, '--root', root)
    sel = json.load(open(os.path.join(rd, 'selected_lr.json')))['selected'] if rc == 0 else {}
    check(f'({case}) select rc {rc}: IAND {sel.get("IAND")} (want 0.005), MS {sel.get("MS")} (want 0.02, tie keeps 0.02)', rc == 0 and sel == {'IAND': '0.005', 'MS': '0.02'})
    finals(root, rd, L, E, iand_d, ms_d, ms_e, sel)
    od = tempfile.mkdtemp()
    rc, out = run(SUMR, '--root', root, '--out_dir', od)
    js = json.load(open(os.path.join(od, 'summary_mitres.json'))) if rc == 0 else {}
    cd = {c['kind']: c for c in js.get('comparisons', [])}
    check(f'({case}) summarize_mitres rc {rc}: R1 {cd.get("R1", {}).get("verdict")}, R2 {cd.get("R2", {}).get("verdict")}, candidates {js.get("candidates")}, '
          f'recommendation {js.get("recommendation")}', rc == 0 and js['recommendation'] == want_rec and js['candidates'] == want_cand)
    if case == 'A':
        check(f'(A) R1 IAND vs ADD worse ({cd["R1"]["mean_pp"]:+.3f}), R2 MS vs ADD no clear difference ({cd["R2"]["mean_pp"]:+.3f})',
              cd['R1']['verdict'] == 'A worse' and abs(cd['R1']['mean_pp'] + 1) < 1e-3 and cd['R2']['verdict'] == 'no clear difference')
        sp = tshuf(root, -1.)
        rc, out = run(SUMT, '--root', root, '--spread', sp, '--out_dir', od)
        jt = json.load(open(os.path.join(od, 'summary_tshuf.json'))) if rc == 0 else {}
        ct = {c['kind']: c for c in jt.get('comparisons', [])}
        check(f'(A) summarize_tshuf rc {rc}: S1 {ct.get("S1", {}).get("verdict")} (want A worse), S2 {ct.get("S2", {}).get("verdict")} (want A better, '
              f'{ct.get("S2", {}).get("mean_pp", 0):+.2f})', rc == 0 and ct['S1']['verdict'] == 'A worse' and ct['S2']['verdict'] == 'A better'
              and abs(sum(jt['order_share_pp']) / 3 - 1.) < 1e-3)
    shutil.rmtree(root); shutil.rmtree(od)
# stop cases
for nm, kw, script, msg in [('missing validation run', dict(drop_val=True), 'sel', 'seeds'), ('mixed commits', dict(mixed=True), 'sel', 'mixed commits')]:
    root, rd, L, E = build(IV, MV, 0., 0., drop_val=kw.get('drop_val', False), mixed=kw.get('mixed', False))
    rc, out = run(SEL, '--root', root)
    check(f'select stops on {nm}: rc {rc} "{out.strip().splitlines()[-1]}"', rc == 2 and msg in out and not os.path.exists(os.path.join(rd, 'selected_lr.json')))
    shutil.rmtree(root)
root, rd, L, E = build(IV, MV, 0., 0.)
run(SEL, '--root', root)
finals(root, rd, L, E, 0., 0., 7.9, {'IAND': '0.01', 'MS': '0.02'})          # IAND final at a lr that was not selected
od = tempfile.mkdtemp()
rc, out = run(SUMR, '--root', root, '--out_dir', od)
check(f'summarize_mitres stops on a final row at an unselected lr: rc {rc} "{out.strip().splitlines()[-1]}"', rc == 2 and 'rows' in out)
shutil.rmtree(root)
root, rd, L, E = build(IV, MV, 0., 0.)
run(SEL, '--root', root)
finals(root, rd, L, E, 0., 0., 7.9, {'IAND': '0.005', 'MS': '0.02'}, bad_recount=True)
rc, out = run(SUMR, '--root', root, '--out_dir', od)
check(f'summarize_mitres stops on a recount mismatch: rc {rc} "{out.strip().splitlines()[-1]}"', rc == 2 and 'recomputed' in out)
for nm, kw, msg in [('SKIP line', dict(skip=True), 'SKIP'), ('spread default != logged', dict(bad_spread=True), 'spread')]:
    if os.path.exists(os.path.join(root, 'flif-v1-mitbih-tshuf')):
        shutil.rmtree(os.path.join(root, 'flif-v1-mitbih-tshuf'))
    sp = tshuf(root, -1., **kw)
    rc, out = run(SUMT, '--root', root, '--spread', sp, '--out_dir', od)
    check(f'summarize_tshuf stops on {nm}: rc {rc} "{out.strip().splitlines()[-1]}"', rc == 2 and msg in out)
shutil.rmtree(root); shutil.rmtree(od)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
