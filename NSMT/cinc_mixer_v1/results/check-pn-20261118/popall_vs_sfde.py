"""Report-only (not pre-registered): pop LIF all SpkMixer vs SF-de / SF-sj, same rules as summarize_pn (user question 11-18 20:3x).
usage (from cinc_mixer_v1/): python -B <this> """
import os, sys
sys.path.insert(0, os.getcwd())
import summarize_pn as sp
from summarize_l2 import compare, spikformer, mean, num, SEEDS, CELLS
from summarize_ms import cmp_mit, load_mit, MITB

E = {'pop LIF all': sp.arms_ett_one('pop LIF all'), 'LIF': sp.arms_ett_one('LIF'), 'SF-de': sp.load_sfde_ett(), 'SF-sj': spikformer()}
st = sp.where('pop LIF all', 'mitbih')
M = {'pop LIF all': load_mit(os.path.join(st, 'raw.txt'), os.path.join(st, 'energy_recount.csv'), 'spkmx_pop_lif_muxlin_add_lr0.02_L2'),
     'LIF': load_mit(os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv'), 'spkmx_lif_muxlin_add_lr0.02_L2'),
     'SF-de': sp.load_sfde_mit(),
     'SF-sj': load_mit(os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv'), 'spk_ssa_lr0.005_L2')}
print('ETT test MSE (seed mean) per cell, mean, energy')
for n, X in E.items():
    v = [mean([num(X[(ds, pl, s)], 'mse') for s in SEEDS]) for ds, pl in CELLS]
    e = mean([mean([X[(ds, pl, s)]['E'] for s in SEEDS]) for ds, pl in CELLS])
    print(f'  {n:12s} ' + ' '.join(f'{z:.4f}' for z in v) + f'  mean {mean(v):.4f}  energy {e:.2f}')
for a, b in (('pop LIF all', 'SF-de'), ('pop LIF all', 'SF-sj'), ('pop LIF all', 'LIF'), ('LIF', 'SF-de')):
    c = compare(E[a], E[b], 'mse')
    print(f"ETT {a} vs {b}: R {c['R']:+.2f} lower {c['lo']}/12 {c['verdict']} energy {c['energy']:.2f} cells " + ', '.join(f'{z:+.2f}' for z in c['cells']))
    print('    pairs ' + '; '.join(f"{ds} H{pl} " + ', '.join(f'{z:+.2f}' for z in c['pairs'][3 * i:3 * i + 3]) for i, (ds, pl) in enumerate(CELLS)))
print('MIT-BIH macro-F1 % (seed 42, 7, 2026), energy')
for n, X in M.items():
    print(f'  {n:12s} {100 * mean([num(X[s], "f1") for s in SEEDS]):.2f}  ' + ', '.join(f'{100 * num(X[s], "f1"):.2f}' for s in SEEDS)
          + '  F1 N/S/V/F/Q ' + ' / '.join(f"{100 * mean([num(X[s], f'f1_{i}') for s in SEEDS]):.2f}" for i in range(5)) + f"  energy {mean([X[s]['E'] for s in SEEDS]):.2f}")
for a, b in (('pop LIF all', 'SF-de'), ('pop LIF all', 'SF-sj'), ('pop LIF all', 'LIF'), ('LIF', 'SF-de')):
    c = cmp_mit(M[a], M[b])
    print(f"MIT-BIH {a} vs {b}: {c['mean']:+.2f} %p higher {c['hi']}/3 per seed " + ', '.join(f'{z:+.2f}' for z in c['d']) + f" {c['verdict']} energy {c['energy']:.2f}")
