"""MODI 2026-11-23: unit checks of the hyperparameter range 2 options (docs/MODI_PROJECT_LOG.md 2026-11-23; no training).

  a  train_forecast.LinearWarmup: step k of the first W runs at lr * (k + 1) / W in every parameter group, step W - 1 at exactly lr,
     later calls change nothing; ReduceLROnPlateau after the warmup reduces from the full lr.
  b  train_forecast.train_epoch: warm=None leaves the lr untouched (default path); with a LinearWarmup the lr used by each optimizer
     step is the planned one (recorded by a hook on opt.step).
  c  utils.get_scheduler('cosine'): warmup_epochs=None gives the same lr sequence as the explicit int(0.2 * epochs) (30 epochs: 6);
     warmup_epochs=0 starts at the full lr; 6 starts at the warmup lr 1e-5 and reaches its cosine value at epoch 6.
  d  the runners stop before python on a bad BS / WU value (fake python: nothing recorded) and pass a good one on.
  e  every line of the range-2 search queue (select_r2.search_queue, 430 lines) run with the fake python: 430 distinct stdout files and
     430 distinct run folders (log dir, folder tag, lr), and the ETT stdout name without options is the old one. (The first launch
     on 11-23 was stopped because ETT runs that differed only in lr / batch / warmup shared a stdout file.)
  f  (added 11-24) the runner's run-folder lookup after training: the 340 ETT lines of the search queue first make their folders (fake
     python), then each line runs again with every other folder made newer than its own; the RESULT line must name the line's own
     folder. The lookup before 11-24 had no lr when LR was unset and took the newest folder of the same tag (5 search runs got a SKIP
     line); f_lookup(<old runner root>) is how the failure of the old runners was confirmed.
usage (classification folder): python check_r2.py     -> exit 1 on any failure
"""
import math
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn

import train_forecast as TF
from utils import get_scheduler

HERE = os.path.dirname(os.path.abspath(__file__))
ok_n = bad_n = 0


def check(c, msg):
    global ok_n, bad_n
    c = bool(c)
    print(('PASS ' if c else 'FAIL ') + msg); ok_n += c; bad_n += (not c)


def opt2(lr=5e-4):
    a, b = nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(3, 3))
    return torch.optim.AdamW([{'params': [a], 'weight_decay': 0.}, {'params': [b], 'weight_decay': 0.06}], lr=lr)


def a_warmup():
    opt = opt2(); w = TF.LinearWarmup(opt, 5); seen = []
    for _ in range(8):
        w(); seen.append([g['lr'] for g in opt.param_groups])
    want = [5e-4 * (k + 1) / 5 for k in range(5)] + [5e-4] * 3
    check(all(abs(s[0] - x) < 1e-18 and s[0] == s[1] for s, x in zip(seen, want)), f'a: warmup lrs {[f"{s[0]:.1e}" for s in seen]}')
    check(seen[4][0] == 5e-4 and seen[7][0] == 5e-4, 'a: step W - 1 and later run at exactly the full lr')
    sch = get_scheduler('reduce', opt)
    for v in (1.0, 1.1, 1.2, 1.3, 1.4):
        sch.step(v)
    check(abs(opt.param_groups[0]['lr'] - 5e-5) < 1e-18, f"a: the plateau scheduler reduces from the full lr ({opt.param_groups[0]['lr']:.1e})")
    try:
        TF.LinearWarmup(opt2(), 0); check(False, 'a: zero warmup steps must stop')
    except ValueError:
        check(True, 'a: zero warmup steps stop')


class Tiny(nn.Module):
    kind = 'x'

    def __init__(self):
        super().__init__(); self.l = nn.Linear(4, 2)

    def forward(self, x):
        return self.l(x[:, -1, :4]).unsqueeze(-1).expand(-1, -1, x.shape[-1])


def b_train_epoch():
    torch.manual_seed(0)
    dl = [(torch.randn(8, 6, 4), torch.randn(8, 5, 4), 0, 0) for _ in range(6)]
    for warm_steps in (None, 4):
        m = Tiny(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); used = []
        step0 = opt.step
        opt.step = lambda *a, _s=step0, **k: (used.append(opt.param_groups[0]['lr']), _s(*a, **k))[1]
        warm = None if warm_steps is None else TF.LinearWarmup(opt, warm_steps)
        TF.train_epoch(m, dl, opt, 2, 'cpu', warm=warm)
        want = [1e-3] * 6 if warm_steps is None else [1e-3 * (k + 1) / 4 for k in range(4)] + [1e-3] * 2
        check(len(used) == 6 and all(abs(u - x) < 1e-18 for u, x in zip(used, want)), f'b: lr per optimizer step, warmup {warm_steps}: {[f"{u:.2e}" for u in used]}')


def cos_seq(wu, epochs=30, lr=0.01):
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(2))], lr=lr)
    sch = get_scheduler('cosine', opt, max_lr=0.005, min_lr=1e-6, max_epochs=epochs, **({} if wu == 'default' else {'warmup_epochs': wu}))
    sch.step(0); seq = [opt.param_groups[0]['lr']]
    for e in range(epochs - 1):
        sch.step(e + 1); seq.append(opt.param_groups[0]['lr'])
    return seq


def c_cosine():
    d, s6, s0, sn = cos_seq('default'), cos_seq(6), cos_seq(0), cos_seq(None)
    check(d == s6 == sn, 'c: default (and None) = the explicit 6 warmup epochs of 30, every epoch')
    check(abs(d[0] - 1e-5) < 1e-12 and all(d[i] < d[i + 1] for i in range(6)) and d[6] == max(d), f'c: 6 warmup epochs rise from 1e-5 to the peak at epoch 6 ({d[0]:.1e} .. {d[6]:.4f})')
    check(abs(s0[0] - 0.01) < 1e-12 and all(s0[i] >= s0[i + 1] for i in range(29)), f'c: warmup 0 starts at the full lr and only decays ({s0[0]:.4f} .. {s0[-1]:.2e})')
    check(all(math.isfinite(v) and v >= 0 for v in d + s0), 'c: every lr finite and non-negative')
    print('   lr per epoch, warmup 6: ' + ' '.join(f'{v:.4f}' for v in d[:8]) + ' ... ' + f'{d[-1]:.5f}')
    print('   lr per epoch, warmup 0: ' + ' '.join(f'{v:.4f}' for v in s0[:8]) + ' ... ' + f'{s0[-1]:.5f}')


def d_runners():
    N = os.path.abspath(os.path.join(HERE, '..', '..'))
    ET, MX, M4 = (os.path.join(N, p) for p in ('nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh', 'nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh',
                                               'nsmt_flif_v1/classification/scripts/run_mitbih4.sh'))
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, 'bin', 'python'); os.makedirs(os.path.dirname(fake)); rec = os.path.join(td, 'rec.txt')
        open(fake, 'w').write('#!/usr/bin/env bash\necho "$*" >> "$FAKE_OUT"\nexit 3\n'); os.chmod(fake, 0o755)

        def run(env, cmd):
            open(rec, 'w').close()
            e = dict(os.environ, PY=fake, FAKE_OUT=rec, RES=os.path.join(td, 'res'), LOG=os.path.join(td, 'log'), **env)
            subprocess.run(['bash'] + cmd, env=e, capture_output=True, text=True)
            return open(rec).read()
        for name, sh, args, c in (('ETT dual', ET, ['ETTh2', '96', '42', '0'], 'nsmt4q_pop_lif_msn_sde_ncna'), ('ETT mixer', MX, ['ETTh2', '96', '42', '0'], 'spkall_lif')):
            for bad in (dict(BS='0'), dict(BS='64.0'), dict(BS='x'), dict(WU='0'), dict(WU='2'), dict(WU='1.5'), dict(WU='-1'), dict(WU='0.0')):
                check(run(dict(COND=c, **bad), [sh] + args) == '', f'd: {name} stops before python on {bad}')
            out = run(dict(COND=c, BS='32', WU='1', LR='0.00025'), [sh] + args)
            check(' -bs 64 ' in out and out.count(' -bs 32 ') == 1 and out.index(' -bs 32 ') > out.index(' -bs 64 ') and ' --lr_warmup 1 ' in out
                  and f' --tag {c}_b32_w1 seed42' in out and ' -lr 0.00025 ' in out, f'd: {name} passes BS / WU / LR on and tags the folder')
            out = run(dict(COND=c, WU='0.5'), [sh] + args)
            check(' --lr_warmup 0.5 ' in out and f' --tag {c}_w0.5 seed42' in out and ' -bs 32' not in out, f'd: {name} WU alone')
        for bad in (dict(BS='0'), dict(BS='x'), dict(WU='-1'), dict(WU='1.5'), dict(WU='x')):
            check(run(dict(COND='spkall_lif', **bad), [M4, '42', '0']) == '', f'd: MIT-BIH stops before python on {bad}')
        out = run(dict(COND='nsmt4q_pop_flif_msn_sde_ncna', BS='1024', WU='0', LR='0.02', VAL='1'), [M4, '42', '0'])
        check(' -bs 1024 ' in out and ' -bs 512' not in out and ' --lr_warmup 0 ' in out and ' --tag nsmt4q_pop_flif_msn_sde_ncna_lr0.02_v1_wdx_b1024_w0 seed42' in out
              and ' --mitbih_val ' in out, 'd: MIT-BIH passes BS / WU / LR / VAL on and tags the folder')
        out = run(dict(COND='spkall_lif'), [M4, '42', '0'])
        check(' -bs 512 ' in out and '--lr_warmup' not in out and ' --tag spkall_lif_lr0.01_v0_wdx seed42' in out, 'd: MIT-BIH defaults unchanged')


def e_queue():
    import re
    import select_r2 as S
    N = os.path.abspath(os.path.join(HERE, '..', '..'))
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, 'bin', 'python'); os.makedirs(os.path.dirname(fake)); rec = os.path.join(td, 'rec.txt')
        open(fake, 'w').write('#!/usr/bin/env bash\necho "$*" >> "$FAKE_OUT"\nexit 3\n'); os.chmod(fake, 0o755)

        def run(ln):
            toks = ln.split(); i = toks.index('bash')
            env = dict(t.split('=', 1) for t in toks[:i])
            raw = os.path.join(td, 'res', 'raw.txt')
            for f in (rec, raw):
                if os.path.exists(f):
                    os.remove(f)
            e = dict(os.environ, PY=fake, FAKE_OUT=rec, RES=os.path.join(td, 'res'), LOG=os.path.join(td, 'log'), **env)
            subprocess.run(toks[i:] + ['0'], env=e, cwd=N, capture_output=True, text=True)
            cmd = open(rec).read().split(); skip = open(raw).read()
            out = re.search(r' out=(\S+)', skip).group(1)
            lr = [cmd[j + 1] for j, t in enumerate(cmd) if t == '-lr'][-1]
            return out, (cmd[cmd.index('--log_dir') + 1], cmd[cmd.index('--tag') + 1], cmd[cmd.index('--tag') + 2], lr)
        Q = S.search_queue(); got = [run(ln) for ln in Q]
        check(len(Q) == 430 and len({g[0] for g in got}) == 430, f'e: 430 queue lines -> {len({g[0] for g in got})} distinct stdout files')
        check(len({g[1] for g in got}) == 430, f'e: 430 queue lines -> {len({g[1] for g in got})} distinct run folders (log dir, tag, seed, lr)')
        for sh, c in (('nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh', 'nsmt4q_pop_lif_msn_sde'), ('nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh', 'spkall_lif')):
            out, _ = run(f'COND={c} SUITE=x bash {sh} ETTh2 96 42')
            check(re.search(rf'/ett_ETTh2_p96_{c}_seed42_\d{{8}}_[0-9a-f]+\.stdout$', out), f'e: {os.path.basename(sh)} stdout name without options is the old one ({os.path.basename(out)})')
            out, _ = run(f'COND={c} SUITE=x LR=0.001 BS=32 WU=1 bash {sh} ETTh2 96 42')
            check(f'/ett_ETTh2_p96_{c}_b32_w1_lr0.001_seed42_' in out, f'e: {os.path.basename(sh)} stdout name with options ({os.path.basename(out)})')


FAKE_RUN = r"""#!/usr/bin/env bash
# fake python for f: '-' = the runner's csv parse; otherwise make the run folder the way config.py names it (the parts the lookup reads)
if [ "$1" = "-" ]; then echo "mse=1 mae=1"; exit 0; fi
a=("$@"); ld=; tg=; st=; e=; lr=; wu=; sd=
for ((i = 0; i < ${#a[@]}; i++)); do case "${a[i]}" in
  --log_dir) ld=${a[i+1]} ;; --tag) tg=${a[i+1]}; st=${a[i+2]} ;; -e) e=${a[i+1]} ;; -lr) lr=${a[i+1]} ;; --lr_warmup) wu=${a[i+1]} ;; --seed) sd=${a[i+1]} ;;
esac; done
D="$ld/261124/261124+$tg+$st+model+x+epoch+$e+lr+$lr+gating+attn/seed${sd}_x${wu:+_wu$wu}"
if [ ! -d "$D" ]; then mkdir -p "$D/log/last"; echo w > "$D/log/final+result.csv"; echo "1 1 1 1/1" > "$D/log/best_epoch.txt"; echo "$D" > "$FAKE_MADE"; fi
echo "number of parameters : 1"
"""


def f_lookup(root):
    """runs the 340 ETT search lines twice with the runners under root; returns (lines whose RESULT names their own folder, lines in all)."""
    import re
    import select_r2 as S
    ett = [ln for ln in S.search_queue() if 'forecasting/scripts' in ln]
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, 'bin', 'python'); os.makedirs(os.path.dirname(fake)); made = os.path.join(td, 'made.txt')
        open(fake, 'w').write(FAKE_RUN); os.chmod(fake, 0o755)
        raw = os.path.join(td, 'res', 'raw.txt')

        def run(ln):
            toks = ln.split(); i = toks.index('bash')
            for p in (made, raw):
                if os.path.exists(p):
                    os.remove(p)
            e = dict(os.environ, PY=fake, FAKE_MADE=made, RES=os.path.join(td, 'res'), LOG=os.path.join(td, 'log'), **dict(t.split('=', 1) for t in toks[:i]))
            subprocess.run(toks[i:] + ['0'], env=e, cwd=root, capture_output=True, text=True)
            got = re.search(r' dir=(\S+)', open(raw).read()) if os.path.exists(raw) else None
            return open(made).read().strip() if os.path.exists(made) else None, got.group(1) if got else None
        own = [run(ln)[0] for ln in ett]                               # pass 1: every line makes its own folder
        if None in own or len(set(own)) != len(ett):
            return 0, len(ett)
        ok = 0
        for k, ln in enumerate(ett):                                   # pass 2: every other folder is newer than the line's own
            for j, d in enumerate(own):
                os.utime(d, (2e9, 2e9) if j != k else (1e9, 1e9))
            m, got = run(ln)
            ok += (m is None and got == own[k])
    return ok, len(ett)


def f_check():
    n_ok, n = f_lookup(os.path.abspath(os.path.join(HERE, '..', '..')))
    check(n == 340 and n_ok == n, f'f: run-folder lookup -- {n_ok} of {n} ETT search lines find their own folder while other folders are newer')


if __name__ == '__main__':
    a_warmup(); b_train_epoch(); c_cosine(); d_runners(); e_queue(); f_check()
    print(f'{ok_n} passed, {bad_n} failed')
    sys.exit(1 if bad_n else 0)
