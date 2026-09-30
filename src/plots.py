"""IEEE-compliant figure style.  Vector PDF, Type-42 fonts, 8 pt labels,
single-column width 3.45 in, double column 7.16 in."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'tex', 'figs')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 6.6, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'lines.linewidth': 1.15, 'lines.markersize': 4.0,
    'axes.grid': True, 'grid.alpha': 0.30, 'grid.linewidth': 0.4,
    'legend.frameon': True, 'legend.framealpha': 0.92,
    'legend.borderpad': 0.3, 'legend.labelspacing': 0.25,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
W1, H1 = 3.45, 2.55
W2 = 7.1

STY = {
    'proposed':  dict(c='#0072BD', m='o', ls='-'),
    'fixed':     dict(c='#4DBEEE', m='x', ls='-'),
    'sdma':      dict(c='#D95319', m='s', ls='--'),
    'noma':      dict(c='#77AC30', m='^', ls='-.'),
    'rsma_pas':  dict(c='#7E2F8E', m='v', ls='--'),
    'rsma_act':  dict(c='#A2142F', m='d', ls='-.'),
    'rsma_diag': dict(c='#EDB120', m='p', ls=':'),
    'shannon':   dict(c='#4D4D4D', m='*', ls=':'),
}
LAB = {
    'proposed': 'Proposed GPM (gated hybrid BD-RIS)',
    'fixed': 'Fixed partition, exhaustive sweep',
    'sdma': 'SDMA + gated hybrid BD-RIS',
    'noma': 'NOMA + gated hybrid BD-RIS',
    'rsma_pas': 'RSMA + passive BD-RIS',
    'rsma_act': 'RSMA + fully active BD-RIS',
    'rsma_diag': 'RSMA + diagonal hybrid RIS',
    'shannon': 'Shannon bound (no dispersion penalty)',
}


def load(tag):
    with open(os.path.join(RES, tag + '.json')) as f:
        return json.load(f)


def grid(tag, xs, names, nreal):
    """Rows are ordered as the job list: x outer, name middle, seed inner."""
    rows = load(tag)
    assert len(rows) == len(xs) * len(names) * nreal, (tag, len(rows))
    out, i = {}, 0
    for _ in xs:
        for nm in names:
            out.setdefault(nm, []).append(rows[i:i + nreal])
            i += nreal
    return out


def ci(v, level=1.96):
    """Mean and half-width of the 95 percent confidence interval of the mean."""
    v = np.asarray(v, float)
    n = max(len(v), 1)
    return float(v.mean()), float(level * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def stat(blocks, f=lambda r: r['sum_rate']):
    m = np.array([ci([f(r) for r in b])[0] for b in blocks])
    h = np.array([ci([f(r) for r in b])[1] for b in blocks])
    return m, h


def band(ax, xs, m, h, nm, label=None, marker=None, ls=None, color=None):
    s = STY.get(nm, dict(c='#333333', m='o', ls='-'))
    c = color or s['c']
    ax.fill_between(xs, m - h, m + h, color=c, alpha=0.15, lw=0)
    ax.plot(xs, m, color=c, marker=marker or s['m'], linestyle=ls or s['ls'],
            label=label if label is not None else LAB.get(nm, nm),
            markerfacecolor='none', markeredgewidth=0.9)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig)
    print('wrote', name)
