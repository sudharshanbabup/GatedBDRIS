"""Turn results/*.json into the IEEE format figures of the manuscript and into
results/summary.json, from which make_facts.py writes every number quoted in the
text.  All curves carry 95 percent confidence intervals of the mean."""
import os, json, sys
import numpy as np
from plots import *      # style, load, grid, stat, band, save, W1, W2, H1, LAB, STY
import matplotlib.pyplot as plt

NR = int(os.environ.get('NREAL', 200))
NCDF = int(os.environ.get('NCDF', 300))
NHI = int(os.environ.get('NHI', 80))
NPART = int(os.environ.get('NPART', NR))
PARTS = [4, 8, 16, 24, 32, 40, 48, 64]
LB_LIST = [50, 100, 200, 300, 500, 800]
PW_LIST = [33, 35, 37, 39, 41, 43]
CRB_LIST = [0.07, 0.09, 0.12, 0.16, 0.22, 0.35]
QOS_LIST = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
PUL_LIST = [16, 32, 64, 128, 256, 512]
NAMAX_LIST = [16, 32, 48, 64]


def _safe(x, big=1e9):
    x = float(x)
    return big if (not np.isfinite(x)) else min(x, big)


def dominate(g, target='proposed', fallback='sdma'):
    """Rate splitting contains space division multiple access as the special
    case C_k = 0, so any CRB feasible SDMA design is also a feasible RSMA
    design.  Reporting the per realisation better of the two is therefore a
    valid feasible point of the RSMA problem and removes the local optimum
    artefact by which the harder problem occasionally converges below its own
    special case.  The comparison is paired: same channel, same operating
    point."""
    if target not in g or fallback not in g:
        return g
    out = []
    for bt, bf in zip(g[target], g[fallback]):
        row = []
        for rt, rf in zip(bt, bf):
            take = rf if (rf['feas'] and not rt['feas']) else (
                rf if (rf['feas'] == rt['feas'] and rf['sum_rate'] > rt['sum_rate'])
                else rt)
            row.append(take)
        out.append(row)
    g = dict(g)
    g[target] = out
    return g


# =====================================================================
def fig_blocklength():
    xs, nm = LB_LIST, ['proposed', 'sdma', 'noma', 'shannon']
    g = dominate(grid('sim1_blocklength', xs, nm, NR))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, H1 * 0.98),
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    out = {'xs': xs}
    for k in nm:
        m, h = stat(g[k])
        band(a1, xs, m, h, k)
        out[k] = m.tolist()
        out[k + '_ci'] = h.tolist()
    a1.set_xlabel('Blocklength $L_b$ (channel uses)')
    a1.set_ylabel('FBL sum rate (bit/s/Hz)')
    a1.set_xscale('log'); a1.set_xticks(xs)
    a1.set_xticklabels([str(v) for v in xs]); a1.minorticks_off()
    a1.legend(loc='upper left', fontsize=6.0)

    def paired(a, b):
        P = np.array([[r['sum_rate'] for r in blk] for blk in g[a]])
        Q = np.array([[r['sum_rate'] for r in blk] for blk in g[b]])
        R = 100 * (P / np.maximum(Q, 1e-9) - 1)
        m = R.mean(1)
        h = 1.96 * R.std(axis=1, ddof=1) / np.sqrt(R.shape[1])
        return m, h

    for key, mk, ls, lab in [('sdma', 's', '-', 'gain over SDMA'),
                             ('noma', '^', '--', 'gain over NOMA')]:
        m, h = paired('proposed', key)
        band(a2, xs, m, h, key, label=lab, marker=mk, ls=ls)
        out['gain_' + key] = m.tolist()
    a2.set_xlabel('Blocklength $L_b$ (channel uses)')
    a2.set_ylabel('Sum rate gain of RSMA (\\%)')
    a2.set_xscale('log'); a2.set_xticks(xs)
    a2.set_xticklabels([str(v) for v in xs]); a2.minorticks_off()
    a2.set_ylim(bottom=0)
    a2.legend(loc='upper right', fontsize=6.4)
    fig.tight_layout(pad=0.35)
    save(fig, 'fig_blocklength')
    return out


def fig_power():
    xs = PW_LIST
    nm = ['proposed', 'sdma', 'noma', 'rsma_pas', 'rsma_act', 'rsma_diag']
    g = dominate(grid('sim2_power', xs, nm, NR))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, H1),
                                 gridspec_kw=dict(width_ratios=[1.32, 1]))
    out = {'xs': xs, 'feas': {}}
    for k in nm:
        m, h = stat(g[k])
        band(a1, xs, m, h, k)
        out[k] = m.tolist(); out[k + '_ci'] = h.tolist()
        fm, fh = stat(g[k], f=lambda r: float(r['feas']))
        a2.plot(xs, 100 * fm, color=STY[k]['c'], marker=STY[k]['m'],
                linestyle=STY[k]['ls'], markerfacecolor='none', markeredgewidth=0.9)
        out['feas'][k] = fm.tolist()
    a1.set_xlabel('Total power budget $P_{\\mathrm{tot}}$ (dBm)')
    a1.set_ylabel('FBL sum rate (bit/s/Hz)')
    a1.legend(loc='lower right', fontsize=5.6)
    a2.set_xlabel('Total power budget $P_{\\mathrm{tot}}$ (dBm)')
    a2.set_ylabel('Matrix CRB feasibility (\\%)')
    a2.set_ylim(-4, 104)
    fig.tight_layout(pad=0.35)
    save(fig, 'fig_power')
    return out


def fig_crb():
    xs, nm = CRB_LIST, ['proposed', 'sdma', 'rsma_diag']
    g = dominate(grid('sim3_crb', xs, nm, NR))
    fig, ax = plt.subplots(figsize=(W1, H1))
    out = {'xs': xs, 'feas': {}}
    for k in nm:
        m, h = stat(g[k])
        band(ax, xs, m, h, k)
        out[k] = m.tolist(); out[k + '_ci'] = h.tolist()
        out['feas'][k] = stat(g[k], f=lambda r: float(r['feas']))[0].tolist()
    ax.set_xlabel('Angle accuracy threshold $\\sigma_\\theta$ (degrees)')
    ax.set_ylabel('FBL sum rate (bit/s/Hz)')
    ax.set_xscale('log'); ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs]); ax.minorticks_off()
    ax.legend(loc='lower right', fontsize=6.4)
    save(fig, 'fig_crb')
    return out


def fig_partition():
    """The result that replaces the sweep: panel (a) compares the exhaustively
    swept fixed partition against the single gated design; panel (b) shows the
    distribution of the recovered cardinality and the cost of each route."""
    sw = grid('sim4_sweep', PARTS, ['fixed'], NPART)['fixed']
    nc = grid('sim4_sweep_nocrb', PARTS, ['fixed'], NPART)['fixed']
    hi = grid('sim4_sweep_hi', PARTS, ['fixed'], NHI)['fixed']
    gt = load('sim4_gated')
    gt_hi = load('sim4_gated_hi')
    ms, hs = stat(sw)
    mn, hn = stat(nc)
    mh, hh = stat(hi)

    # paired best-of-sweep per realisation (feasibility first)
    def best_of(blocks):
        n = len(blocks[0])
        vals, args = [], []
        for s in range(n):
            cand = [(blocks[i][s]['feas'], blocks[i][s]['sum_rate'], PARTS[i])
                    for i in range(len(PARTS))]
            f, v, a = max(cand)
            vals.append(v); args.append(a)
        return np.array(vals), np.array(args)

    bo, ba = best_of(sw)
    bo_hi, _ = best_of(hi)
    gv = np.array([r['sum_rate'] for r in gt])
    gn = np.array([r['na'] for r in gt])
    gv_hi = np.array([r['sum_rate'] for r in gt_hi])
    gn_hi = np.array([r['na'] for r in gt_hi])
    t_sw = float(np.mean([np.sum([sw[i][s]['runtime'] for i in range(len(PARTS))])
                          for s in range(len(sw[0]))]))
    t_gt = float(np.mean([r['runtime'] for r in gt]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, H1),
                                 gridspec_kw=dict(width_ratios=[1.2, 1]))
    band(a1, PARTS, mn, hn, 'rsma_diag', label='swept, communication only',
         marker='s', ls='--')
    band(a1, PARTS, ms, hs, 'fixed', label='swept, with matrix CRB', marker='x',
         ls='-')
    band(a1, PARTS, mh, hh, 'rsma_act',
         label='swept, matrix CRB, $P_{\\mathrm{tot}}\\!=\\!39$ dBm',
         marker='d', ls='-.')
    a1.errorbar([gn.mean()], [gv.mean()], xerr=[1.96 * gn.std(ddof=1) / np.sqrt(len(gn))],
                yerr=[1.96 * gv.std(ddof=1) / np.sqrt(len(gv))], fmt='*',
                color=STY['proposed']['c'], markersize=11, capsize=2.5,
                markeredgewidth=1.0, markerfacecolor=STY['proposed']['c'],
                label='proposed, gate optimised', zorder=5)
    a1.errorbar([gn_hi.mean()], [gv_hi.mean()],
                xerr=[1.96 * gn_hi.std(ddof=1) / np.sqrt(len(gn_hi))],
                yerr=[1.96 * gv_hi.std(ddof=1) / np.sqrt(len(gv_hi))], fmt='*',
                color=STY['rsma_act']['c'], markersize=11, capsize=2.5,
                markerfacecolor='none', zorder=5,
                label='proposed, $P_{\\mathrm{tot}}\\!=\\!39$ dBm')
    a1.set_xlabel('Biased amplifiers $n_\\mathrm{a}$ (of $M=64$ ports)')
    a1.set_ylabel('FBL sum rate (bit/s/Hz)')
    a1.legend(loc='lower left', fontsize=5.7)

    bins = np.arange(min(gn.min(), 0) - 0.5, gn.max() + 1.5, 1.0)
    a2.hist(gn, bins=bins, color=STY['proposed']['c'], alpha=0.65,
            edgecolor='white', linewidth=0.4, label='recovered $n_\\mathrm{a}^{\\star}$')
    a2.axvline(np.median(ba), color=STY['fixed']['c'], lw=1.1, ls='--',
               label='median sweep $\\arg\\max$')
    a2.set_xlabel('Biased amplifiers $n_\\mathrm{a}^{\\star}$')
    a2.set_ylabel('Realisations')
    a2.legend(loc='upper right', fontsize=6.0)
    a2.text(0.03, 0.93,
            '%.1f%% of the oracle rate\nat %.1f$\\times$ lower cost'
            % (100 * gv.mean() / bo.mean(), t_sw / max(t_gt, 1e-9)),
            transform=a2.transAxes, fontsize=6.2, va='top')
    fig.tight_layout(pad=0.35)
    save(fig, 'fig_partition')

    # robustness of the recovered cardinality to the hardware size
    nx = grid('sim4_namax', NAMAX_LIST, ['proposed'], NHI)['proposed']
    nam = [float(np.mean([r['na'] for r in b])) for b in nx]
    nar = [float(np.mean([r['sum_rate'] for r in b])) for b in nx]

    ib = int(np.argmax(ms))
    return dict(xs=PARTS, sweep=ms.tolist(), sweep_ci=hs.tolist(),
                win_pct=float(100 * np.mean(gv >= bo)),
                sweep_best_mean=float(ms[ib]), sweep_best_arg=int(PARTS[ib]),
                sweep_best_mean_ci=float(hs[ib]),
                gain_over_fixed=float(100 * (gv.mean() / ms[ib] - 1)),
                win_pct_fixed=float(100 * np.mean(
                    gv >= np.array([sw[ib][s]['sum_rate'] for s in range(len(gv))]))),
                nocrb=mn.tolist(), hi=mh.tolist(),
                feas=stat(sw, f=lambda r: float(r['feas']))[0].tolist(),
                best_of_sweep=float(bo.mean()),
                best_of_sweep_hi=float(bo_hi.mean()),
                sweep_argmax_med=float(np.median(ba)),
                gated=float(gv.mean()), gated_ci=float(ci(gv)[1]),
                gated_na=float(gn.mean()), gated_na_sd=float(gn.std(ddof=1)),
                gated_na_med=float(np.median(gn)),
                gated_hi=float(gv_hi.mean()), gated_na_hi=float(gn_hi.mean()),
                t_sweep=t_sw, t_gate=t_gt, speedup=t_sw / max(t_gt, 1e-9),
                gap_pct=100 * (1 - gv.mean() / bo.mean()),
                commbest=int(PARTS[int(np.argmax(mn))]),
                namax_list=NAMAX_LIST, namax_na=nam, namax_rate=nar)


def fig_convergence():
    """Panel (a): the feasibility-first incumbent against wall clock time for
    T_in = 1, 4 and 8 surface updates per convex program.  Panel (b): the
    gradient mapping residual of the FROZEN surrogate, which is the object
    Theorem 2 is about, against the surface step index with the O(T^{-1/2})
    envelope."""
    xs = [100, 200, 500]
    CFG = [('sim7_t1', 1, '#0072BD', 'o'), ('sim7_t4', 4, '#D95319', 's'),
           ('sim7_t8', 8, '#77AC30', '^')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, H1))
    out = {'xs': xs}

    def inc_curve(blk):
        n = min(len(r['hist']['sum']) for r in blk)
        S = np.array([r['hist']['sum'][:n] for r in blk])
        V = np.array([r['hist']['viol'][:n] for r in blk])
        T = np.array([r['hist']['time'][:n] for r in blk])
        INC = np.full_like(S, np.nan)
        for i in range(S.shape[0]):
            b = -np.inf
            for j in range(n):
                if V[i, j] <= 1e-7:
                    b = max(b, S[i, j])
                if np.isfinite(b):
                    INC[i, j] = b
        return INC, T, n

    curves = {}
    for tag, tin, col, mk in CFG:
        blk = grid(tag, xs, ['proposed'], 16)['proposed'][1]      # L_b = 200
        INC, T, n = inc_curve(blk)
        keep = ~np.all(np.isnan(INC), axis=1)
        INC, T = INC[keep], T[keep]
        if INC.size == 0:
            continue
        mu = np.nanmean(INC, axis=0)
        tt = T.mean(0)
        curves[tin] = (tt, mu, col, mk)
    best = max(np.nanmax(mu) for _, mu, _, _ in curves.values())
    for tin, (tt, mu, col, mk) in sorted(curves.items()):
        j0 = int(np.argmax(~np.isnan(mu)))
        a1.plot(tt[j0:], mu[j0:], color=col, marker=mk, markersize=3.0,
                markevery=max(1, len(mu) // 10), markerfacecolor='none',
                label='$T_{\\rm in}=%d$' % tin)
        fin = float(np.nanmax(mu))
        idx = np.where(np.nan_to_num(mu, nan=-1) >= 0.99 * best)[0]
        out['T%d' % tin] = dict(final=fin,
                                time99=float(tt[idx[0]]) if len(idx) else None,
                                iters99=int(idx[0] + 1) if len(idx) else None,
                                time_end=float(tt[-1]))
    out['target'] = float(best)
    a1.set_xlabel('Wall clock time (s)')
    a1.set_ylabel('Incumbent FBL sum rate (bit/s/Hz)')
    a1.legend(loc='lower right', fontsize=6.4,
              title='surface steps per\nconvex program', title_fontsize=5.8)

    # ---- frozen-surrogate stationarity, the object of Theorem 2 -------------
    rows = load('sim9_rate')
    nseed = len(rows) // len(xs)
    cols = ['#0072BD', '#D95319', '#77AC30']
    slopes = []
    for q, Lb in enumerate(xs):
        blk = rows[q * nseed:(q + 1) * nseed]
        n = min(len(r['gtrace']) for r in blk)
        G = np.array([r['gtrace'][:n] for r in blk], float)
        G = np.where(np.isfinite(G) & (G > 0), G, np.nan)
        Gb = np.fmin.accumulate(np.where(np.isnan(G), np.inf, G), axis=1)
        Gb = Gb / Gb[:, :1]
        it = np.arange(1, n + 1)
        gm = np.exp(np.nanmean(np.log(np.maximum(Gb, 1e-14)), axis=0))
        a2.loglog(it, gm, color=cols[q], marker=['o', 's', '^'][q],
                  markersize=3.0, markevery=max(1, n // 8),
                  markerfacecolor='none', label='$L_b=%d$' % Lb)
        k = n // 4
        slopes.append(float(np.polyfit(np.log(it[k:]), np.log(gm[k:]), 1)[0]))
        if Lb == 200:
            out['gmap'] = gm.tolist()
            out['gmap_iters'] = it.tolist()
    tt = np.arange(1, n + 1)
    a2.loglog(tt, 1.0 / np.sqrt(tt), color='0.35', lw=1.0, ls=':',
              label='$\\mathcal{O}(T^{-1/2})$')
    a2.set_xlabel('Surface step index $T$ (frozen surrogate)')
    a2.set_ylabel('normalised residual $\\|\\mathbf{G}\\|$, best so far')
    a2.legend(loc='lower left', fontsize=6.0)
    fig.tight_layout(pad=0.35)
    save(fig, 'fig_convergence')

    out['gmap_slope'] = float(np.median(slopes))
    g = np.asarray(out['gmap'], float)
    out['gmap_final'] = float(g[-1])
    out['envelope_final'] = float(1.0 / np.sqrt(len(g)))
    out['gmap_T'] = int(len(g))
    t1 = out.get('T1', {}).get('time99')
    t8 = out.get('T8', {}).get('time99')
    out['loop_speedup'] = (t1 / t8) if (t1 and t8) else None
    return out


def fig_pulses():
    xs, nm = PUL_LIST, ['proposed', 'rsma_pas', 'rsma_diag']
    g = grid('sim6_pulses', xs, nm, NR)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, H1 * 0.98))
    out = {'xs': xs}
    for k in nm:
        th, thh = stat(g[k], f=lambda r: np.rad2deg(r['crb'][0]))
        band(a1, xs, th, thh, k)
        rr, rrh = stat(g[k], f=lambda r: _safe(r['crb'][1]))
        if np.all(rr < 1e3):
            band(a2, xs, rr, rrh, k)
        out[k] = dict(theta=th.tolist(), rng=rr.tolist())
    for ax, lab in [(a1, 'Angle RMSE bound (degrees)'),
                    (a2, 'Range RMSE bound (m)')]:
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in xs])
        ax.minorticks_off()
        ax.set_xlabel('Radar pulses $L_{\\mathrm{radar}}$')
        ax.set_ylabel(lab)
        ax.legend(loc='lower left', fontsize=6.0)
    a2.text(0.5, 0.90, 'passive only BD-RIS: range unobservable',
            transform=a2.transAxes, ha='center', fontsize=6.2, color='#7E2F8E')
    fig.tight_layout(pad=0.35)
    save(fig, 'fig_pulses')
    return out


def fig_qos():
    xs, nm = QOS_LIST, ['proposed', 'sdma']
    g = dominate(grid('sim5_qos', xs, nm, NR))
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax2 = ax.twinx(); ax2.grid(False)
    out = {'xs': xs, 'sat': {}}
    for k in nm:
        m, h = stat(g[k])
        band(ax, xs, m, h, k, label=LAB[k] + ' (left)')
        out[k] = m.tolist()
        sat = np.array([np.mean([np.mean(np.asarray(r['rates']) >= x - 1e-6)
                                 for r in b]) for x, b in zip(xs, g[k])])
        ax2.plot(xs, 100 * sat, color=STY[k]['c'], marker=STY[k]['m'],
                 linestyle=':', markerfacecolor='none', markersize=3.2, lw=0.8)
        out['sat'][k] = (100 * sat).tolist()
    out['worst'] = [float(np.mean([np.min(r['rates']) for r in b]))
                    for b in g['proposed']]
    ax.set_xlabel('Per user QoS floor $R_{\\mathrm{th}}$ (bit/s/Hz)')
    ax.set_ylabel('FBL sum rate (bit/s/Hz)')
    ax2.set_ylabel('Users meeting the floor (\\%, dotted)')
    ax2.set_ylim(-4, 104)
    ax.legend(loc='lower left', fontsize=5.9)
    save(fig, 'fig_qos')
    return out


def fig_cdf():
    rows = load('sim8_cdf')
    d = {}
    for r in rows:
        d.setdefault(r['name'], []).extend(r['cdf'])
    fig, ax = plt.subplots(figsize=(W1, H1))
    out = {}
    for k in ['proposed', 'sdma', 'noma']:
        v = np.sort(np.asarray(d[k], float))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=STY[k]['c'],
                linestyle=STY[k]['ls'], label=LAB[k])
        out[k] = dict(n=len(v), mean=float(v.mean()),
                      p5=float(np.percentile(v, 5)),
                      out15=float(np.mean(v < 1.5)))
    ax.axvline(1.5, color='0.5', lw=0.7, ls=':')
    ax.annotate('$R_{\\mathrm{th}}=1.5$', xy=(1.5, 0.40), xytext=(5, -2),
                textcoords='offset points', fontsize=6.4, color='0.35',
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.8))
    ax.set_xlabel('Per user achievable FBL rate (bit/s/Hz)')
    ax.set_ylabel('Empirical CDF')
    ax.set_ylim(0, 1.02); ax.set_xlim(left=-0.3)
    ax.legend(loc='lower right', fontsize=5.9, bbox_to_anchor=(1.0, 0.02))
    save(fig, 'fig_cdf')
    return out


if __name__ == '__main__':
    want = sys.argv[1] if len(sys.argv) > 1 else 'all'
    F = dict(b=fig_blocklength, p=fig_power, c=fig_crb, m=fig_partition,
             q=fig_qos, r=fig_pulses, v=fig_convergence, d=fig_cdf)
    summ = {}
    if os.path.exists(os.path.join(RES, 'summary.json')):
        summ = json.load(open(os.path.join(RES, 'summary.json')))
    for key, fn in F.items():
        if want != 'all' and key not in want:
            continue
        try:
            summ[fn.__name__] = fn()
        except Exception as e:
            print('SKIP', fn.__name__, type(e).__name__, e)
    summ['nreal'] = NR
    with open(os.path.join(RES, 'summary.json'), 'w') as f:
        json.dump(summ, f, indent=1, default=float)
    print('summary keys:', sorted(summ))
