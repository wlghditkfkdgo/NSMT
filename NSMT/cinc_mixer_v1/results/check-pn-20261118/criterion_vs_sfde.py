"""Report-only (user goal 11-18 21:0x, not pre-registered): every trained SpkMixer variant vs SF-de on the paper bar
(on average better than the modified Spikformer, with less energy). Same compare functions as summarize_pn.
usage (from cinc_mixer_v1/): python -B <this>"""
import os, sys
sys.path.insert(0, os.getcwd())
import summarize_pn as sp
from summarize_l2 import compare, spikformer, mean, num, SEEDS, CELLS, TUNE
from summarize_ms import cmp_mit, load_mit, load_ett, MITB, LR_MX

def ett(suite, suf, mixer='lif', L='1'):
    S = os.path.join(sp.RES, suite)
    return load_ett(os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv'), lambda ds: f'spkmx_{mixer}_muxlin_iand_lr{LR_MX[ds]}_L{L}{suf}', f'spkmx_{mixer}_muxlin_iand')

def mit(suite, suf, mixer='lif'):
    S = os.path.join(sp.RES, suite)
    return load_mit(os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv'), f'spkmx_{mixer}_muxlin_add_lr0.02_L2{suf}')

E = {'LIF (base)': ett('flif-v1-ett-spkmux-tune', ''), 'LIF+mean pool': ett('flif-v1-ett-spkmux-mx3', '_mpmean'), 'LIF+max pool': ett('flif-v1-ett-spkmux-mx3', '_mpmax'),
     'LIF reset subtract': ett('flif-v1-ett-spkmux-mx3', '_mrsubtract'), 'LIF reset zero': ett('flif-v1-ett-spkmux-mx3', '_mrzero'),
     'LIF 2-layer stack': ett('flif-v1-ett-spkmux-mx3', '', L='2'), 'LIF 2-layer + MS': ett('flif-v1-ett-spkmux-mx3', '_mms', L='2'),
     'pop LIF all': ett('flif-v1-ett-spkmux-stft', '', mixer='pop_lif')}
M = {'LIF (base)': load_mit(os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv'), 'spkmx_lif_muxlin_add_lr0.02_L2'),
     'LIF+mean pool': mit('flif-v1-mitbih-spkmux-mx3', '_mpmean'), 'LIF+max pool': mit('flif-v1-mitbih-spkmux-mx3', '_mpmax'),
     'LIF reset subtract': mit('flif-v1-mitbih-spkmux-mx3', '_mrsubtract'), 'LIF reset zero': mit('flif-v1-mitbih-spkmux-mx3', '_mrzero'),
     'LIF 2-layer + MS': mit('flif-v1-mitbih-spkmux-mx3', '_mms'), 'pop LIF all': mit('flif-v1-mitbih-spkmux-stft', '', mixer='pop_lif')}
SDE, SDM = sp.load_sfde_ett(), sp.load_sfde_mit()
eS = mean([mean([SDE[(ds, pl, s)]['E'] for s in SEEDS]) for ds, pl in CELLS]); mS = mean([SDM[s]['E'] for s in SEEDS])
print(f'SF-de: ETT mean MSE {mean([num(SDE[(ds, pl, s)], "mse") for ds, pl in CELLS for s in SEEDS]):.4f} energy {eS:.2f}; MIT-BIH F1 {100 * mean([num(SDM[s], "f1") for s in SEEDS]):.2f} energy {mS:.2f}')
print('| arm | ETT R % vs SF-de | lower pairs | ETT energy x | MIT-BIH %p vs SF-de | higher seeds | MIT-BIH energy x |')
for n in E:
    c = compare(E[n], SDE, 'mse')
    row = f"| {n} | {c['R']:+.2f} | {c['lo']}/12 | {c['energy']:.2f} | "
    if n in M:
        d = cmp_mit(M[n], SDM)
        row += f"{d['mean']:+.2f} | {d['hi']}/3 | {d['energy']:.2f} |"
    else:
        row += '- | - | - |'
    print(row)
