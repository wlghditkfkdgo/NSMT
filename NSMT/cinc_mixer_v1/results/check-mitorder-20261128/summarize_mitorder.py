"""Summary of the MIT-BIH order shuffle (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; rules fixed before running).
Reads order_mux.json / order_sfde.json (cls_order_eval.py). drop = macro-F1(no shuffle) - macro-F1(shuffled) in %p (mean of 3
permutation seeds), per training seed (42 / 7 / 2026). Per arm and variant: 'order used' if the mean drop >= 0.5 %p and the drop is
> 0 for >= 2 of 3 seeds (the MIT-BIH rule of this branch), else 'no evidence'. Paired contrast per seed: D = drop_mux(mixer) -
drop_SF-de(SSA); 'muxlin depends more' if mean D >= 0.5 %p and D > 0 for >= 2 of 3, 'SF-de depends more' if <= -0.5 %p and D < 0
for >= 2 of 3, else 'no difference'. Accuracy drops reported. Fails closed on missing / non-finite values or unmatched seeds.
usage (from this folder): python -B summarize_mitorder.py
"""
import json
import math
import sys

J = {arm: {r['seed']: r for r in json.load(open(f'order_{arm}.json'))['runs']} for arm in ('mux', 'sfde')}
if any(sorted(d) != ['2026', '42', '7'] for d in J.values()):
    sys.exit('STOP: seeds')
avg = lambda x: sum(x) / len(x)
out = {}
for arm in ('mux', 'sfde'):
    for var in ('mixer', 'full'):
        d = [J[arm][s][var]['drop_f1_pp'] for s in ('42', '7', '2026')]
        da = [J[arm][s][var]['drop_acc_pp'] for s in ('42', '7', '2026')]
        if not all(math.isfinite(x) for x in d + da):
            sys.exit('STOP: non-finite')
        out[f'{arm}_{var}'] = dict(mean_drop_pp=avg(d), seeds_dropped=sum(x > 0 for x in d), drops=d, mean_acc_drop_pp=avg(da),
                                   f1_none=avg([J[arm][s]['f1'] for s in J[arm]]),
                                   verdict='order used' if avg(d) >= 0.5 and sum(x > 0 for x in d) >= 2 else 'no evidence')
D = [J['mux'][s]['mixer']['drop_f1_pp'] - J['sfde'][s]['mixer']['drop_f1_pp'] for s in ('42', '7', '2026')]
mD = avg(D)
out['contrast'] = dict(mean_D=mD, D=D, verdict='muxlin depends more' if mD >= 0.5 and sum(x > 0 for x in D) >= 2 else
                       ('SF-de depends more' if mD <= -0.5 and sum(x < 0 for x in D) >= 2 else 'no difference'))
json.dump(out, open('summary_mitorder.json', 'w'), indent=1)
print('| model | shuffle | macro-F1 unshuffled | mean drop %p (seeds 42 / 7 / 2026) | seeds dropped | verdict | accuracy drop %p |')
print('|---|---|--:|---|--:|---|--:|')
for arm, nm in (('mux', 'SpkMixer muxlin ADD'), ('sfde', 'SF-de (SSA + rel. bias)')):
    for var in ('mixer', 'full'):
        m = out[f'{arm}_{var}']
        print(f'| {nm} | {"SSA" if (arm == "sfde" and var == "mixer") else var} | {m["f1_none"]:.2f} | {m["mean_drop_pp"]:+.2f} '
              f'({" / ".join(f"{x:+.2f}" for x in m["drops"])}) | {m["seeds_dropped"]} / 3 | **{m["verdict"]}** | {m["mean_acc_drop_pp"]:+.3f} |')
print(f'\nPaired contrast drop_mux(mixer) - drop_SF-de(SSA): mean {mD:+.2f} %p (' + ' / '.join(f'{x:+.2f}' for x in D) + f') -> **{out["contrast"]["verdict"]}**')
