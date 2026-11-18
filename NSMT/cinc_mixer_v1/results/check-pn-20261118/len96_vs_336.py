"""Report-only (user request 11-18 22:3x "ETT 주 설정 선택 => 96과 336을 비교해줘"; no training): ETT input 96 vs 336 for the LIF SpkMixer,
SF-de, SF-sj and pop LIF all, with the same compare functions as summarize_pn (rules: R = mean over 4 cells of the seed-mean r, better R <= -1 %
and >= 8/12; paper bar = R < 0 and energy A/B < 1). usage (from cinc_mixer_v1/): python -B results/check-pn-20261118/len96_vs_336.py"""
import os, sys
sys.path.insert(0, os.getcwd())
import summarize_pn as sp
from summarize_l2 import compare, spikformer, mean, num, SEEDS, CELLS
from summarize_ms import load_ett, LR_MX, LR_SF, LY_SF

L336 = sp.L336
def at336(tagf, cond):
    return load_ett(os.path.join(L336, 'raw.txt'), os.path.join(L336, 'energy_recount.csv'), tagf, cond)
A96 = {'LIF SpkMixer': sp.arms_ett_one('LIF'), 'SF-de': sp.load_sfde_ett(), 'SF-sj': spikformer(), 'pop LIF all': sp.arms_ett_one('pop LIF all')}
A336 = {'LIF SpkMixer': at336(lambda ds: f'spkmx_lif_muxlin_iand_lr{LR_MX[ds]}_L1_s336', 'spkmx_lif_muxlin_iand'), 'SF-de': sp.load_sfde_ett('_s336'),
        'SF-sj': at336(lambda ds: f'spk_ssa_lr{LR_SF[ds]}_L{LY_SF[ds]}_s336', 'spk_ssa'),
        'pop LIF all': at336(lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LR_MX[ds]}_L1_s336', 'spkmx_pop_lif_muxlin_iand')}
f = lambda z: ' '.join(f'{v:+.2f}' for v in z)
print('## test MSE (seed mean), 96 -> 336')
for n in A96:
    v96 = [mean([num(A96[n][(d, p, s)], 'mse') for s in SEEDS]) for d, p in CELLS]
    v336 = [mean([num(A336[n][(d, p, s)], 'mse') for s in SEEDS]) for d, p in CELLS]
    e96 = mean([mean([A96[n][(d, p, s)]['E'] for s in SEEDS]) for d, p in CELLS]); e336 = mean([mean([A336[n][(d, p, s)]['E'] for s in SEEDS]) for d, p in CELLS])
    print(f'{n:13s} 96: ' + ' '.join(f'{x:.4f}' for x in v96) + f' mean {mean(v96):.4f} E {e96:.2f} | 336: ' + ' '.join(f'{x:.4f}' for x in v336) + f' mean {mean(v336):.4f} E {e336:.2f}')
print('## 336 vs 96, same model (R %, lower pairs, verdict, energy x, cells)')
for n in A96:
    c = compare(A336[n], A96[n], 'mse')
    print(f"{n:13s} R {c['R']:+.2f} lower {c['lo']}/12 {c['verdict']} energy {c['energy']:.2f} cells {f(c['cells'])}")
print('## vs SF-de at each length (R %, lower pairs, verdict, energy x, cells, paper bar)')
for L, A in (('96', A96), ('336', A336)):
    for n in ('LIF SpkMixer', 'pop LIF all', 'SF-sj'):
        c = compare(A[n], A['SF-de'], 'mse')
        bar = 'met' if c['R'] < 0 and c['energy'] < 1 else 'not met'
        print(f"input {L:3s} {n:13s} vs SF-de: R {c['R']:+.3f} lower {c['lo']}/12 {c['verdict']} energy {c['energy']:.4f} cells {f(c['cells'])} paper bar {bar}")
print('## cell-wise gap change (r at 336 minus r at 96, LIF SpkMixer vs SF-de; negative = the SpkMixer gains on SF-de at 336)')
c96, c336 = compare(A96['LIF SpkMixer'], A96['SF-de'], 'mse'), compare(A336['LIF SpkMixer'], A336['SF-de'], 'mse')
print('cells ' + ', '.join(f'{d} H{p}' for d, p in CELLS) + ': ' + f([b - a for a, b in zip(c96['cells'], c336['cells'])]) + f"  mean {c336['R'] - c96['R']:+.2f}")
