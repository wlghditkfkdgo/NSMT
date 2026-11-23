"""ETT pass criteria with the DNN baselines of the NSMT paper only (2026-11-23, exp/f-lif-pop-v3; PROJECT_LOG user decision entry). No training.
User (verbatim, 11-23): "(중요) DNN baseline은 NSMT 원 논문에 실린 결과만 포함할 것. 따라서 TimeMixer 제외할 것."
Change to results/ett-pass-20261119/pass_ett.py: criterion (3) DNN = every DNN row of NSMT Table 13 ("Full results on full data", the
full version of Table 2 "Multivariate long-term forecasting results compared with DNN-based state-of-the-art models"; NSMT text: "we use
the same look-back window size L = 96 for all methods ... All results are taken from [43] and [41]"): Informer, FEDformer, PatchTST,
ETSformer, Autoformer (5 rows; NSMT has no other DNN forecasting results). Everything else is unchanged: 4-cell mean (ETTh2 / ETTm2 x
H96 / H720) of the seed means, MSE AND MAE each >= 0.001 lower; (1) SpikF (NSMT Table 14 = SpikF Table 12); (2) SF-de + syops energy
A/B <= 1.05; pass = (1) and (2) and (3).
  --verify <nsmt_layout.txt> (pdftotext -layout of NSMT/papers/NSMT.pdf, kept local) re-reads Table 13 and checks every transcribed
  number (5 DNN models + the NSMT column, 4 cells, MSE and MAE); also checks NSMT Table 13 = pass_ett.NSMT (Table 14) on the 4 cells.
main(): re-judges the existing ETT candidates (no new run) with the old 9-row DNN set and with the NSMT set:
  pre-registered finals: first chain (results/flif-v1-ett-spkmux-ettune, 11-20 04:02), extension (flif-v1-ett-spkmux-ettune2, 17:07);
  reference: the untuned LIF SpkMixer (11-01); report only (post hoc): the configuration after each stage of both chains.
  The generalised judge() must reproduce pass_ett.judge exactly with the old set (checked for every candidate; stops otherwise).
usage (from cinc_mixer_v1/): python -B results/ett-pass-20261123/pass_ett_nsmt.py [--verify NSMT_LAYOUT.txt] > results/ett-pass-20261123/summary.md
"""
import importlib.util
import json
import os
import re
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
sys.path.insert(0, os.path.join(C, 'results', 'ett-pass-20261119'))
import pass_ett as pe                                                               # noqa: E402

# name -> (MSE per cell, MAE per cell, source); cells ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720 (pe.CELLS order)
DNN_NSMT = {'Informer': ((3.755, 3.647, 0.365, 3.379), (1.525, 1.625, 0.453, 1.338), 'NSMT Table 13'),
            'FEDformer': ((0.358, 0.463, 0.203, 0.421), (0.397, 0.474, 0.287, 0.415), 'NSMT Table 13'),
            'PatchTST': ((0.302, 0.431, 0.175, 0.402), (0.348, 0.446, 0.259, 0.400), 'NSMT Table 13'),
            'ETSformer': ((0.340, 0.500, 0.189, 0.414), (0.391, 0.497, 0.280, 0.413), 'NSMT Table 13'),
            'Autoformer': ((0.346, 0.515, 0.255, 0.433), (0.388, 0.511, 0.339, 0.432), 'NSMT Table 13')}
T13_ORDER = ('Informer', 'FEDformer', 'PatchTST', 'ETSformer', 'Autoformer', 'NSMT')
DSO, HO = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), ('96', '192', '336', '720', 'Avg')


def verify(path):
    L = open(path, encoding='utf-8').read().splitlines()
    i = next(k for k, l in enumerate(L) if 'Table 13: Full results on full data' in l)
    rx = re.compile(r'(?:^|\s)(96|192|336|720|Avg)\s+((?:\d\.\d{3}\s+){11}\d\.\d{3})')
    rows = []
    for l in L[i:i + 120]:
        m = rx.search(l)
        if m:
            rows.append((m.group(1), [float(x) for x in m.group(2).split()]))
        if len(rows) == 20:
            break
    if [h for h, _ in rows] != list(HO) * 4:
        return False, [f'Table 13 rows not parsed: {[h for h, _ in rows]}']
    T = {(DSO[k // 5], rows[k][0]): rows[k][1] for k in range(20)}
    bad, n = [], 0
    want = dict(DNN_NSMT, NSMT=pe.NSMT)
    for j, m in enumerate(T13_ORDER):
        for c, (ds, pl) in enumerate(pe.CELLS):
            got = T[(ds, str(pl))][2 * j: 2 * j + 2]
            exp = [want[m][0][c], want[m][1][c]]
            n += 2
            if got != exp:
                bad.append(f'{m} {ds} H{pl}: table {got} vs transcribed {exp}')
    lines = [f'NSMT Table 13 transcription: {n} numbers checked, {len(bad)} mismatches' + ('' if not bad else ': ' + '; '.join(bad)),
             'NSMT column of Table 13 = pass_ett.NSMT (Table 14) on the 4 cells: ' + str(not any(b.startswith('NSMT') for b in bad)),
             'PatchTST of NSMT Table 13 = pass_ett "PatchTST (re-run)" (iTransformer Table 10), report: '
             + str(DNN_NSMT['PatchTST'][:2] == pe.DNN['PatchTST (re-run)'][:2])]
    return not bad, lines


def judge(name, V, VS, EA, ES, dnn):
    """pass_ett.judge with the DNN set as a parameter; returns (pass dict, markdown lines, means, energy ratio)."""
    om, sm, er = pe.cellmeans(V), pe.cellmeans(VS), pe.eratio(EA, ES)
    rows, res = [], {}

    def one(ref, rm):
        d = (rm[0] - om[0], rm[1] - om[1])
        ok = d[0] >= pe.MARGIN - pe.EPS and d[1] >= pe.MARGIN - pe.EPS
        rows.append(f'| {ref} | {rm[0]:.4f} / {rm[1]:.4f} | {d[0]:+.4f} / {d[1]:+.4f} | {"yes" if ok else "no"} |')
        return ok

    res['(1) SNN'] = one('SpikF', pe.ref_mean(*pe.SNN['SpikF'][:2]))
    ok2 = one('SF-de (ours)', sm)
    res['(2) Spikformer'] = ok2 and er <= pe.EBOUND + pe.EPS
    rows.append(f'| energy A/B vs SF-de (syops; bound {pe.EBOUND}) | {er:.4f} | | {"yes" if er <= pe.EBOUND + pe.EPS else "no"} |')
    d = [one(m, pe.ref_mean(*dnn[m][:2])) for m in dnn]
    res['(3) DNN'] = all(d)
    res['pass'] = all(res[k] for k in ('(1) SNN', '(2) Spikformer', '(3) DNN'))
    L = [f'#### {name}', '', f'4-cell mean MSE / MAE: **{om[0]:.4f} / {om[1]:.4f}**; energy A/B vs SF-de (syops) {er:.4f}', '',
         '| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |', '|---|---|---|---|'] + rows + [
        '', f'- (1) SNN: **{"pass" if res["(1) SNN"] else "fail"}**; (2) Spikformer: **{"pass" if res["(2) Spikformer"] else "fail"}**; '
        f'(3) DNN ({len(dnn)} rows): **{"pass" if res["(3) DNN"] else "fail"}** ({sum(d)}/{len(dnn)}); overall: '
        f'**{"PASS" if res["pass"] else "NOT PASSED"}**', '']
    return res, L, om, er


def load_chain(chain):
    os.environ.pop('ET_DRY', None)
    os.environ['ET_CHAIN'] = chain
    spec = importlib.util.spec_from_file_location(f'se_{chain or "first"}', os.path.join(C, 'summarize_ettune.py'))
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)
    return se


def chain_rows(se):
    D, E = se.load(), se.energy_rows()
    path = [{ds: dict(se.DEFAULT[ds]) for ds in se.DSS}]
    for name, knob, _ in se.STAGES:
        s = json.load(open(os.path.join(se.OUT, f'selected_{name}.json')))
        path.append({ds: dict(path[-1][ds], **{knob: str(s['selected'][ds])}) for ds in se.DSS})
    return D, E, path


def vals(se, T):
    V = {c: {k: [T[(c[0], c[1], s)][k] for s in se.SEEDS] for k in ('mse', 'mae')} for c in se.CELLS}
    E = {c: [T[(c[0], c[1], s)]['Es'] for s in se.SEEDS] for c in se.CELLS}
    return V, E


def main():
    vlines = ['Transcription not verified in this run.']
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        ok, vlines = verify(sys.argv[2])
        print('\n'.join(vlines), file=sys.stderr)
        if not ok:
            sys.exit(1)
    s1, sx = load_chain(''), load_chain('x')
    D1, E1, p1 = chain_rows(s1)
    Dx, Ex, px = chain_rows(sx)
    sf = s1.sfde_rows(E1)
    VS, ES = vals(s1, sf)
    cands = [('untuned LIF SpkMixer (11-01; reference)', s1.arm(D1, E1, p1[0]), 'reference', s1, p1[0])]
    cands += [('first chain final (11-20 04:02; pre-registered)', s1.arm(D1, E1, p1[-1]), 'pre-registered', s1, p1[-1]),
              ('extension final (11-20 17:07; pre-registered)', sx.arm(Dx, Ex, px[-1]), 'pre-registered', sx, px[-1])]
    for se, D, E, p, lab in ((s1, D1, E1, p1, 'first chain'), (sx, Dx, Ex, px, 'extension')):
        for k in range(1, len(se.STAGES)):
            cands.append((f'{lab} after stage {se.NAMES[k - 1]} (post hoc)', se.arm(D, E, p[k]), 'post hoc', se, p[k]))
    out, blocks, summ = [], [], {}
    for name, T, kind, se, cfg in cands:
        V, EA = vals(se, T)
        r_old, _, om, er = judge(name, V, VS, EA, ES, pe.DNN)
        r_ref, _ = pe.judge(name, V, VS, EA, ES)
        if r_old != r_ref:
            pe.stop(f'generalised judge != pass_ett.judge for {name}: {r_old} vs {r_ref}')
        r_new, L, _, _ = judge(name, V, VS, EA, ES, DNN_NSMT)
        cf = '; '.join(f'{ds} {se.cfgtxt(ds, cfg[ds])}' for ds in se.DSS)
        out.append((name, kind, cf, om, er, r_old, r_new))
        if kind != 'post hoc':
            blocks += L
        summ[name] = {'kind': kind, 'config': cfg, 'mse': om[0], 'mae': om[1], 'energy_ab': er, 'old_9_rows': r_old, 'nsmt_5_rows': r_new}
    yn = lambda b: 'pass' if b else 'fail'                                         # noqa: E731
    P = ['# ETT pass criteria with the NSMT-paper DNN baselines only (user decision 11-23; no training)', '',
         'Rules and sources: module docstring of pass_ett_nsmt.py. Criterion (3) now = the 5 DNN rows of NSMT Table 13 (TimeMixer and the '
         'other rows not in the NSMT paper removed); (1) and (2) unchanged.', '', '## Transcription check', ''] + [f'- {x}' for x in vlines] + [
        '', '## NSMT-paper DNN rows (look-back 96; MSE / MAE per cell: ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720)', '',
        '| row | source | MSE | MAE | 4-cell mean MSE / MAE |', '|---|---|---|---|---|']
    for m, (mse, mae, src) in DNN_NSMT.items():
        rm = pe.ref_mean(mse, mae)
        P.append(f'| {m} | {src} | {", ".join("%.3f" % x for x in mse)} | {", ".join("%.3f" % x for x in mae)} | {rm[0]:.4f} / {rm[1]:.4f} |')
    refs = [pe.ref_mean(*pe.SNN['SpikF'][:2]), pe.cellmeans(VS)] + [pe.ref_mean(*DNN_NSMT[m][:2]) for m in DNN_NSMT]
    P += ['', f'Needed for (1)-(3) accuracy: 4-cell mean MSE ≤ {min(r[0] for r in refs) - pe.MARGIN:.4f} and MAE ≤ '
          f'{min(r[1] for r in refs) - pe.MARGIN:.4f} (the binding row is now SpikF for both); plus energy A/B ≤ {pe.EBOUND}.', '',
          '## All candidates (old = 9-row DNN set incl. TimeMixer; new = NSMT 5 rows)', '',
          '| candidate | kind | configuration (non-default knobs) | MSE / MAE | energy A/B | (1) | (2) | (3) old | (3) new | overall old | overall new |',
          '|---|---|---|---|--:|---|---|---|---|---|---|']
    for name, kind, cf, om, er, ro, rn in out:
        P.append(f'| {name} | {kind} | {cf} | {om[0]:.4f} / {om[1]:.4f} | {er:.4f} | {yn(rn["(1) SNN"])} | {yn(rn["(2) Spikformer"])} | '
                 f'{yn(ro["(3) DNN"])} | {yn(rn["(3) DNN"])} | {"PASS" if ro["pass"] else "NOT PASSED"} | {"PASS" if rn["pass"] else "NOT PASSED"} |')
    P += ['', '## Detail with the NSMT set (pre-registered finals and the reference)', ''] + blocks + [
        'Report (not a criterion): NSMT itself (Table 13 / 14) 4-cell mean MSE / MAE '
        f'{pe.ref_mean(*pe.NSMT[:2])[0]:.4f} / {pe.ref_mean(*pe.NSMT[:2])[1]:.4f}.']
    print('\n'.join(P))
    json.dump(summ, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rejudge.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
