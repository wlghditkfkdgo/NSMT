"""Checks of the ETT tuning extension code (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning extension pre-registration). No training.
P  first chain unchanged (no ET_CHAIN): in a copy of results/flif-v1-ett-spkmux-ettune (ET_DRY)
   P1 make_queue_ettune.sh patch / dmodel / mlp / blocks / wd: the same queue as the builder of git HEAD (run on the same selections;
      the queue files the first chain ran are empty because gpu_pool2 consumes them)
   P2 summarize_ettune.py --stage <stage> on raw.txt cut to the rows that existed at that stage (first 24 / 60 / 96 / 120 / 156 lines;
      the stages ran one after another) rewrites selected_<stage>.json and selection_<stage>.md byte-identical to git HEAD; final (156
      rows) rewrites summary_final.md and final.json byte-identical
   P3 recheck_ettune.py --stage final: ALL CHECKS PASSED
X  extended chain (ET_CHAIN=x, ET_DRY): every queue line becomes a fake RESULT row (template = the row of the same cell / seed at the
   stage's current value, real or fake; validation L1, test MSE and MAE multiplied by a factor of the new value; energy row copied)
   XA patch 16 kept: queues 24 / 0 / 0 / 0 / 36; patch 4 / 16 rows are not queued again; selections = the first chain's for patch .. blocks;
      wd: ETTh2 0.02 (factor 0.999), ETTm2 0 (real rows, 0.999931); final config and tuned 4-cell MSE = hand value; report row vs the
      first chain present; recheck every stage and final
   XB ETTh2 patch 24 wins (factor 0.99): queues 24 / 18 / 18 / 12 / 48 (ETTh2 re-runs d_model .. wd at patch 24, ETTm2 only the new wd
      values); final ETTh2 {24, 64, 0.5, 1, 0.06}, ETTm2 {16, 64, 2, 1, 0}
   XC fail closed: C1 new-suite row of a configuration no stage asked for, C2 new-suite row of a configuration that the first-chain suite
      holds (duplicate), C3 a missing row, C4 a SKIP line -> summarizer exit 2 and no selected_patch.json; C5 make_queue dmodel without
      selected_patch.json -> exit != 0; C6 resume: with half of the patch rows present the queue holds only the other half;
      C7 final with one energy row missing -> exit 2, no final.json; C8 ET_CHAIN=x smoke -> refused
usage (from cinc_mixer_v1/): python -B results/check-ettunex-20261120/check_x.py > results/check-ettunex-20261120/check_x.txt
"""
import csv
import json
import os
import shutil
import subprocess
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(C, 'results')
OLD = os.path.join(RES, 'flif-v1-ett-spkmux-ettune')
PY = sys.executable
LIB = os.path.join(os.path.dirname(os.path.dirname(PY)), 'lib')
REPO = os.path.dirname(os.path.dirname(C))
LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
K = ('PATCH', 'EMB', 'MLPR', 'BLOCKS', 'WD')
FLD = dict(zip(K, ('patch', 'emb', 'mlpr', 'blocks', 'wd')))
SUF = dict(zip(K, ('_pt', '_e', '_mlp', '_blk', '_wd')))
D0 = {'ETTh2': dict(zip(K, ('8', '64', '0.5', '1', '0.06'))), 'ETTm2': dict(zip(K, ('8', '64', '1', '1', '0.06')))}
STAGES = ('patch', 'dmodel', 'mlp', 'blocks', 'wd')
KNOB = dict(zip(STAGES, K))
CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]
SEEDS = ('42', '7', '2026')
fails = 0


def check(label, cond, detail=''):
    global fails
    fails += not cond
    print(f'{"PASS" if cond else "FAIL"} {label}{(": " + detail) if detail else ""}')


def tag(ds, cfg):
    return f'spkmx_lif_muxlin_iand_lr{LR[ds]}_L1' + ''.join(SUF[k] + cfg[k] for k in K if cfg[k] != D0[ds][k])


def ck(cfg):
    return tuple(cfg[k] for k in K)


def run(args, env_add, sh=False):
    env = dict(os.environ)
    env.pop('ET_CHAIN', None); env.pop('ET_DRY', None)
    env.update(env_add)
    if sh:
        env.pop('LD_LIBRARY_PATH', None)
    else:
        env['LD_LIBRARY_PATH'] = LIB
    p = subprocess.run(args, cwd=C, env=env, capture_output=True, text=True)
    return p.returncode, (p.stderr + p.stdout).strip()


def summ(stage, env):
    return run([PY, '-B', 'summarize_ettune.py', '--stage', stage], env)


def rech(stage, env):
    return run([PY, '-B', 'recheck_ettune.py', '--stage', stage], env)


def queue(stage, env):
    rc, o = run(['bash', 'scripts/make_queue_ettune.sh', stage], env, sh=True)
    if rc != 0:
        return rc, None, o
    q = o.splitlines()[-1]
    path, nq = q.rsplit(':', 1)
    return rc, (path, int(nq)), o


# ---------------- P: first chain unchanged -----------------------------------------------------------------------------------------
P = os.path.join(OUT, 'dry_P')
shutil.rmtree(P, ignore_errors=True)
os.makedirs(P)
shutil.copytree(OLD, os.path.join(P, 'flif-v1-ett-spkmux-ettune'))
envP = {'ET_DRY': P}
OLDQ = os.path.join(P, 'old_make_queue_ettune.sh')
open(OLDQ, 'w').write(subprocess.run(['git', 'show', 'HEAD:NSMT/cinc_mixer_v1/scripts/make_queue_ettune.sh'], cwd=REPO, capture_output=True,
                                     text=True).stdout)
for st in STAGES:
    rc0, o0 = run(['bash', OLDQ, st], envP, sh=True)
    q0 = o0.splitlines()[-1].rsplit(':', 1)[0] if rc0 == 0 else None
    old_q = open(q0).read() if q0 else None
    rc, q, o = queue(st, envP)
    same = rc == 0 and old_q is not None and q[0] == q0 and open(q[0]).read() == old_q
    check(f'P1 first-chain queue {st}: new builder = HEAD builder ({q[1] if q else "-"} lines)', same, '' if same else (o0 + o)[-300:])


def head_file(f):
    return subprocess.run(['git', 'show', f'HEAD:NSMT/cinc_mixer_v1/results/flif-v1-ett-spkmux-ettune/{f}'], cwd=REPO, capture_output=True,
                          text=True).stdout


RAW = open(os.path.join(OLD, 'raw.txt')).read().splitlines(keepends=True)
for st, n in zip(STAGES + ('final',), (24, 60, 96, 120, 156, 156)):
    open(os.path.join(P, 'flif-v1-ett-spkmux-ettune', 'raw.txt'), 'w').write(''.join(RAW[:n]))
    rc, o = summ(st, envP)
    check(f'P2 summarize {st} (no ET_CHAIN, first {n} rows) exit 0', rc == 0, o[-300:] if rc else '')
    for f in ([f'selected_{st}.json', f'selection_{st}.md'] if st != 'final' else ['summary_final.md', 'final.json']):
        head = head_file(f)
        got = open(os.path.join(P, 'flif-v1-ett-spkmux-ettune', f)).read()
        check(f'P2 {f} byte-identical to git HEAD', got == head and len(head) > 0)
rc, o = rech('final', envP)
check('P3 recheck final (no ET_CHAIN)', rc == 0 and 'ALL CHECKS PASSED' in o, o.splitlines()[-1] if o else '')


# ---------------- X: extended chain, fake rows -----------------------------------------------------------------------------------
def real_rows():
    rows, en = {}, {}
    for l in open(os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'raw.txt')):
        if l.startswith('RESULT'):
            d = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            if d['cond'] == 'spkmx_lif_muxlin_iand' and d['ds'] in LR and d['tag'] == tag(d['ds'], D0[d['ds']]) and d['pl'] in ('96', '720') \
                    and d['seed'] in SEEDS:
                rows[(d['ds'], d['pl'], d['seed'], ck(D0[d['ds']]))] = d
    for l in open(os.path.join(OLD, 'raw.txt')):
        if l.startswith('RESULT'):
            d = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            rows[(d['ds'], d['pl'], d['seed'], ck({k: d[FLD[k]] for k in K}))] = d
    for path in (os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'energy_recount.csv'), os.path.join(OLD, 'energy_recount.csv')):
        for e in csv.DictReader(open(path)):
            en[(e['tag'], e['ds'], e['pl'], e['seed'])] = e
    return rows, en


REAL, REN = real_rows()


class Dry:
    def __init__(self, name, factor):
        self.dir = os.path.join(OUT, f'dry_{name}')
        shutil.rmtree(self.dir, ignore_errors=True)
        self.suite = os.path.join(self.dir, 'flif-v1-ett-spkmux-ettune2')
        os.makedirs(self.suite)
        self.env = {'ET_CHAIN': 'x', 'ET_DRY': self.dir}
        self.rows, self.en, self.factor, self.lines, self.erows = dict(REAL), dict(REN), factor, [], []

    def fake(self, qline, stage):
        tok = qline.split()
        ds, pl, seed = tok[-3], tok[-2], tok[-1]
        env = dict(t.split('=', 1) for t in tok[:tok.index('bash')])
        cfg = dict(D0[ds]); cfg.update({k: env[k] for k in K if k in env})
        knob = KNOB[stage]
        cur = self.cur[ds]
        tcfg = dict(cfg, **{knob: cur})
        t = self.rows[(ds, pl, seed, ck(tcfg))]
        f = self.factor(ds, knob, cfg[knob])
        d = dict(t)
        be = d['best_epoch_val_epochs'].split(':')
        val = float(be[1]) * f
        d.update({'tag': tag(ds, cfg), 'mse': f'{float(t["mse"]) * f:.6f}', 'mae': f'{float(t["mae"]) * f:.6f}',
                  'val_loss_best': f'{val:.9f}', 'best_epoch_val_epochs': f'{be[0]}:{val:.9f}:{be[2]}', 'sha': 'dry', 'lr': LR[ds],
                  'layers': '1', 'cond': 'spkmx_lif_muxlin_iand'})
        for k in K:
            d[FLD[k]] = cfg[k]
        self.rows[(ds, pl, seed, ck(cfg))] = d
        self.lines.append('RESULT ' + ' '.join(f'{k}={v}' for k, v in d.items()))
        e = dict(self.en[(t['tag'], ds, pl, seed)])
        e.update({'tag': d['tag'], 'mse_logged': d['mse'], 'mse_recomputed': d['mse']})
        self.en[(d['tag'], ds, pl, seed)] = e
        self.erows.append(e)

    def write(self, extra=''):
        open(os.path.join(self.suite, 'raw.txt'), 'w').write(''.join(l + '\n' for l in self.lines) + extra)

    def stage(self, st, want_n, want_sel):
        prev = {ds: dict(D0[ds]) for ds in D0}
        for s in STAGES[:STAGES.index(st)]:
            js = json.load(open(os.path.join(self.suite, f'selected_{s}.json')))
            for ds in D0:
                prev[ds][KNOB[s]] = str(js['selected'][ds])
        self.cur = {ds: prev[ds][KNOB[st]] for ds in D0}
        rc, q, o = queue(st, self.env)
        if rc != 0:
            check(f'{self.name} {st} queue', False, o[-300:]); return False
        lines = [l for l in open(q[0]).read().splitlines() if l]
        check(f'{self.name} {st} queue: {q[1]} jobs (want {want_n})', q[1] == want_n == len(lines))
        if st == 'patch':
            check(f'{self.name} patch queue has no PATCH=4 / PATCH=16 line (rows exist)', not any(('PATCH=4 ' in l or 'PATCH=16 ' in l) for l in lines))
        for l in lines:
            self.fake(l, st)
        self.write()
        rc, o = summ(st, self.env)
        if rc != 0:
            check(f'{self.name} summarize {st}', False, o[-400:]); return False
        js = json.load(open(os.path.join(self.suite, f'selected_{st}.json')))
        check(f'{self.name} {st} selected {js["selected"]} (want {want_sel})', js['selected'] == want_sel)
        rc, o = rech(st, self.env)
        check(f'{self.name} recheck {st}', rc == 0 and 'ALL CHECKS PASSED' in o, o.splitlines()[-1] if o else '')
        return True

    def final(self):
        with open(os.path.join(self.suite, 'energy_recount.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.erows[0].keys()))
            w.writeheader(); w.writerows(self.erows)
        rc, o = summ('final', self.env)
        check(f'{self.name} summarize final exit 0', rc == 0, o[-400:] if rc else '')
        rc2, o2 = rech('final', self.env)
        check(f'{self.name} recheck final', rc2 == 0 and 'ALL CHECKS PASSED' in o2, o2.splitlines()[-1] if o2 else '')
        return json.load(open(os.path.join(self.suite, 'final.json'))) if rc == 0 else None


def m4(rows, cfgs):
    return sum(sum(float(rows[(ds, pl, s, ck(cfgs[ds]))]['mse']) for s in SEEDS) / 3 for ds, pl in CELLS) / 4


FIRST = {'patch': {'ETTh2': '16', 'ETTm2': '16'}, 'dmodel': {'ETTh2': '64', 'ETTm2': '64'}, 'mlp': {'ETTh2': '0.5', 'ETTm2': '2'},
         'blocks': {'ETTh2': '1', 'ETTm2': '1'}}

# XA: patch 16 kept; ETTh2 wd 0.02 wins by factor 0.999
A = Dry('XA', lambda ds, knob, v: 0.999 if (ds, knob, v) == ('ETTh2', 'WD', '0.02') else 1.0)
A.name = 'XA'
for st, n in zip(STAGES[:4], (24, 0, 0, 0)):
    A.stage(st, n, FIRST[st])
A.stage('wd', 36, {'ETTh2': '0.02', 'ETTm2': '0'})
fa = A.final()
if fa:
    want = {'ETTh2': dict(zip(K, ('16', '64', '0.5', '1', '0.02'))), 'ETTm2': dict(zip(K, ('16', '64', '2', '1', '0')))}
    check('XA final config', fa['config'] == want, str(fa['config']))
    hand = m4(A.rows, want)
    check(f'XA tuned 4-cell MSE = hand value {hand:.6f}', abs(fa['tuned']['mse'] - hand) < 1e-9, f'{fa["tuned"]["mse"]:.6f}')
    check('XA report row vs the first chain present', '(report) tuned (extended) vs tuned (first chain, 11-20 04:02)' in fa['comparisons'])
    sm = open(os.path.join(A.suite, 'summary_final.md')).read()
    check('XA summary title marks the extended chain', 'extended chain (ET_CHAIN=x' in sm.splitlines()[0])

# XB: ETTh2 patch 24 wins (factor 0.99), everything else ties
B = Dry('XB', lambda ds, knob, v: 0.99 if (ds, knob, v) == ('ETTh2', 'PATCH', '24') else 1.0)
B.name = 'XB'
B.stage('patch', 24, {'ETTh2': '24', 'ETTm2': '16'})
B.stage('dmodel', 18, {'ETTh2': '64', 'ETTm2': '64'})
B.stage('mlp', 18, {'ETTh2': '0.5', 'ETTm2': '2'})
B.stage('blocks', 12, {'ETTh2': '1', 'ETTm2': '1'})
B.stage('wd', 48, {'ETTh2': '0.06', 'ETTm2': '0'})
fb = B.final()
if fb:
    want = {'ETTh2': dict(zip(K, ('24', '64', '0.5', '1', '0.06'))), 'ETTm2': dict(zip(K, ('16', '64', '2', '1', '0')))}
    check('XB final config', fb['config'] == want, str(fb['config']))
    hand = m4(B.rows, want)
    check(f'XB tuned 4-cell MSE = hand value {hand:.6f}', abs(fb['tuned']['mse'] - hand) < 1e-9, f'{fb["tuned"]["mse"]:.6f}')


# XC: fail closed
def patch_rows(name):
    d = Dry(name, lambda ds, knob, v: 1.0)
    d.name = name
    d.cur = {'ETTh2': '8', 'ETTm2': '8'}
    rc, q, o = queue('patch', d.env)
    for l in open(q[0]).read().splitlines():
        d.fake(l, 'patch')
    return d


def must_stop(d, label, extra='', why=''):
    d.write(extra)
    rc, o = summ('patch', d.env)
    ok = rc == 2 and not os.path.exists(os.path.join(d.suite, 'selected_patch.json')) and (why in o)
    check(f'{label}: exit 2, no selected_patch.json, reason "{why}"', ok, o.splitlines()[-1][:200] if o else '')


d = patch_rows('XC1')
r = dict(d.rows[('ETTh2', '96', '42', ck(dict(D0['ETTh2'], PATCH='24')))])
cfg = dict(D0['ETTh2'], PATCH='32', EMB='128'); r.update({'tag': tag('ETTh2', cfg), 'patch': '32', 'emb': '128'})
must_stop(d, 'XC1 new-suite row of an unrequested configuration (PATCH=32 EMB=128)',
          'RESULT ' + ' '.join(f'{k}={v}' for k, v in r.items()) + '\n', 'no stage asked for')
d = patch_rows('XC2')
r = dict(d.rows[('ETTh2', '96', '42', ck(dict(D0['ETTh2'], PATCH='16')))])
must_stop(d, 'XC2 new-suite row of a configuration in the first-chain suite (PATCH=16)',
          'RESULT ' + ' '.join(f'{k}={v}' for k, v in r.items()) + '\n', 'duplicate row')
d = patch_rows('XC3')
d.lines = d.lines[1:]
must_stop(d, 'XC3 a missing row', '', 'missing runs')
d = patch_rows('XC4')
must_stop(d, 'XC4 a SKIP line', 'SKIP rc=1 ETTh2 96 42 dry\n', 'SKIP line')
d = Dry('XC5', lambda ds, knob, v: 1.0)
rc, q, o = queue('dmodel', d.env)
check('XC5 make_queue dmodel without selected_patch.json: exit != 0, no queue', rc != 0 and q is None, o.splitlines()[-1][:160] if o else '')
d = patch_rows('XC6')
half = len(d.lines) // 2
d.lines = d.lines[:half]
d.write()
rc, q, o = queue('patch', d.env)
check(f'XC6 resume: {half} of 24 patch rows present -> queue holds the other {24 - half}', rc == 0 and q[1] == 24 - half, f'{q[1] if q else o}')
B7 = Dry('XC7', lambda ds, knob, v: 1.0)
B7.name = 'XC7'
for st, n in zip(STAGES[:4], (24, 0, 0, 0)):
    B7.stage(st, n, FIRST[st])
B7.stage('wd', 36, {'ETTh2': '0.06', 'ETTm2': '0'})
B7.erows = B7.erows[1:]
with open(os.path.join(B7.suite, 'energy_recount.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(B7.erows[0].keys())); w.writeheader(); w.writerows(B7.erows)
rc, o = summ('final', B7.env)
check('XC7 final with one energy row missing: exit 2, no final.json', rc == 2 and not os.path.exists(os.path.join(B7.suite, 'final.json'))
      and 'no energy recount row' in o, o.splitlines()[-1][:200] if o else '')
rc, q, o = queue('smoke', {'ET_CHAIN': 'x', 'ET_DRY': os.path.join(OUT, 'dry_XC8')})
check('XC8 ET_CHAIN=x smoke refused', rc != 0 and 'chain-1 stage' in o, o.splitlines()[-1][:160] if o else '')
print('ALL X CHECKS PASSED' if fails == 0 else f'{fails} FAILED')
sys.exit(0 if fails == 0 else 1)
