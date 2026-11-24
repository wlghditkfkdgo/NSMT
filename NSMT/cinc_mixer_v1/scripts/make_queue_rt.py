"""Queue of one stage of the ETTh1 / ETTm1 retuning (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG pre-registration of 2026-11-25; plan and
rules in rt_plan.py). A stage queues every run its reachable entries need (shared stage: one entry; chain stage: chain a and chain x,
their union) that has no RESULT row in the suite yet, slow cells first; a stage whose earlier selections are missing stops (exit 1).
  lr | layers | patch | dmodel | mlp | blocks | wd   H96 / H720 x seeds 42 / 7 / 2026, every candidate of the stage
  final                                               each chain's selected configuration at H192 / H336 x 3 seeds
Every queue line passes all seven knobs and EPOCHS=50 to run_ett_spk.sh (a value equal to the runner default counts as unset there).
usage (any folder): python make_queue_rt.py <stage>   ->  prints <queue file>:<jobs>
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rt_plan as P                                                              # noqa: E402

RUN = 'bash cinc_mixer_v1/scripts/run_ett_spk.sh'


def line(ds, pl, seed, cfg):
    kn = ' '.join(f'{k}={cfg[k]}' for k in P.KN)
    return f'SUITE={P.SUITE} EPOCHS={P.EPOCHS} COND={P.COND} {kn} {RUN} {ds} {pl} {seed}'


def have():
    h = set()
    p = os.path.join(P.OUT, 'raw.txt')
    for l in (open(p) if os.path.exists(p) else []):
        if l.startswith('RESULT'):
            r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            h.add((r['ds'], int(r['pl']), int(r['seed']), tuple(r[P.FIELD[k]] for k in P.KN)))
    return h


def main():
    st = sys.argv[1] if len(sys.argv) > 1 else ''
    if st not in P.STAGES:
        print(f'bad stage {st!r} ({"|".join(P.STAGES)})'); sys.exit(1)
    E, F = P.walk()
    H = have()
    want = []                                                                    # (ds, pl, seed, cfg) in queue order
    if st == 'final':
        if F is None:
            print('STOP: final needs every chain selection'); sys.exit(1)
        for ds, pl in P.SLOWF:
            for c in P.CHAINS:
                for s in P.SEEDS:
                    want.append((ds, pl, s, F[c][ds]))
    else:
        es = [e for e in E if e['name'] == st]
        need = 1 if st in [x[0] for x in P.SHARED] else len(P.CHAINS)
        if len(es) != need:
            print(f'STOP: stage {st} is not reachable yet ({len(es)} of {need} entries; an earlier selection is missing)'); sys.exit(1)
        for ds, pl in P.SLOW:
            for e in es:
                for v in e['vals']:
                    for s in P.SEEDS:
                        want.append((ds, pl, s, dict(e['base'][ds], **{e['knob']: v})))
    lines, seen = [], set()
    for ds, pl, s, cfg in want:
        k = (ds, pl, s, P.key(cfg))
        if k in seen or k in H:
            continue
        seen.add(k)
        lines.append(line(ds, pl, s, cfg))
    os.makedirs(P.QDIR, exist_ok=True)
    f = os.path.join(P.QDIR, f'retune-{st}-{datetime.date.today():%Y%m%d}.txt')
    if os.path.exists(f) and open(f).read().strip():
        print(f'STOP: {f} exists and is not empty (a pool may be using it)'); sys.exit(1)
    open(f, 'w').write(''.join(l + '\n' for l in lines))
    print(f'{f}:{len(lines)}')


if __name__ == '__main__':
    main()
