"""Range-requirement sweep figure (fig_pulses) with the scalar-CRB baseline,
overriding the pulse-count figure so the manuscript's Fig. 8 is backed by the
real solver.  Run AFTER runner_ext.py's 's' study.  Also refreshes summary.json
so make_facts.py's fig_pulses macros are consistent."""
import os, json
import numpy as np
from plots import *          # STY, LAB, band, save, W2, H1, RES
import matplotlib.pyplot as plt

NREAL = int(os.environ.get('NREAL', 50))
SR_LIST = [0.040, 0.025, 0.016, 0.012, 0.009, 0.007]
# per sr: proposed(matrix), proposed(scalar), rsma_diag, rsma_pas
ORDER = ['proposed', 'scalar', 'rsma_diag', 'rsma_pas']
LABS = {'proposed': 'Proposed GPM (matrix CRB)',
        'scalar': 'RSMA + scalar-CRB (angle only)',
        'rsma_diag': 'RSMA + diagonal hybrid RIS',
        'rsma_pas': 'RSMA + passive BD-RIS'}
COL = {'proposed': STY['proposed']['c'], 'scalar': '#000000',
       'rsma_diag': STY['rsma_diag']['c'], 'rsma_pas': STY['rsma_pas']['c']}
MK = {'proposed': 'o', 'scalar': 'd', 'rsma_diag': 'p', 'rsma_pas': 'v'}
LS = {'proposed': '-', 'scalar': '-', 'rsma_diag': ':', 'rsma_pas': '--'}


def main():
    rows = json.load(open(os.path.join(RES, 'sim_pulses_sr.json')))
    nb = len(ORDER)
    # blocks in job order: sr outer, scheme middle, seed inner
    feas = {k: [] for k in ORDER}
    rng = {k: [] for k in ORDER}
    i = 0
    for _ in SR_LIST:
        for k in ORDER:
            b = rows[i:i + NREAL]; i += NREAL
            good = [r for r in b if not r.get('failed')]
            feas[k].append(100 * np.mean([float(r['feas']) for r in good]))
            rng[k].append(np.mean([min(float(r['crb'][1]), 1e5) for r in good]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, H1 * 0.98))
    for k in ORDER:
        a1.plot(SR_LIST, feas[k], color=COL[k], marker=MK[k], ls=LS[k],
                markerfacecolor='none', markeredgewidth=0.9, label=LABS[k])
        a2.semilogy(SR_LIST, rng[k], color=COL[k], marker=MK[k], ls=LS[k],
                    markerfacecolor='none', markeredgewidth=0.9, label=LABS[k])
    a2.semilogy(SR_LIST, SR_LIST, color='0.4', ls='--', lw=0.9,
                label='requirement $\\sigma_r$')
    for ax in (a1, a2):
        ax.set_xscale('log'); ax.invert_xaxis()
        ax.set_xticks(SR_LIST); ax.set_xticklabels([str(v) for v in SR_LIST])
        ax.minorticks_off()
        ax.set_xlabel('Range requirement $\\sigma_r$ (m), tightening $\\rightarrow$')
    a1.set_ylabel('2D localisation feasibility (\\%)'); a1.set_ylim(-4, 104)
    a1.legend(loc='lower left', fontsize=5.6)
    a2.set_ylabel('Achieved range RMSE (m)')
    a2.legend(loc='upper right', fontsize=5.6)
    fig.tight_layout(pad=0.35)
    save(fig, 'fig_pulses')

    # refresh fig_pulses macros in summary.json for make_facts.py
    sp = os.path.join(RES, 'summary.json')
    summ = json.load(open(sp)) if os.path.exists(sp) else {}
    summ['fig_pulses'] = dict(
        xs=[int(1000 * s) for s in SR_LIST],           # placeholder x for macros
        proposed=dict(theta=[0.0] * len(SR_LIST), rng=rng['proposed']),
        rsma_diag=dict(theta=[0.0] * len(SR_LIST), rng=rng['rsma_diag']))
    json.dump(summ, open(sp, 'w'), indent=1, default=float)
    print('wrote fig_pulses (range-requirement sweep) and refreshed summary')


if __name__ == '__main__':
    main()
