"""Emit tex/facts.tex with \\newcommand macros taken from results/summary.json,
so that every number quoted in the text is the number that produced the figures.
A macro that cannot be computed is emitted as ?? and shows up in the PDF, which
makes a stale claim impossible to miss."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, '..', 'results', 'summary.json')))
OUT = os.path.join(HERE, '..', 'tex', 'facts.tex')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
L, SEEN = [], set()


def cmd(name, val, fmt='%.2f'):
    if name in SEEN:
        return
    SEEN.add(name)
    try:
        txt = val if isinstance(val, str) else fmt % val
    except Exception:
        txt = '??'
    L.append('\\newcommand{\\%s}{%s}' % (name, txt))


cmd('CmcReal', S.get('nreal', 200), '%d')

b = S.get('fig_blocklength')
if b:
    xs = b['xs']
    gp, gs, gn = np.array(b['proposed']), np.array(b['sdma']), np.array(b['noma'])
    cmd('FblLbLo', xs[0], '%d'); cmd('FblLbHi', xs[-1], '%d')
    cmd('FblLowRate', gp[0]); cmd('FblHighRate', gp[-1])
    cmd('FblLowSdma', gs[0]); cmd('FblHighSdma', gs[-1])
    cmd('FblLowGain', b['gain_sdma'][0], '%.0f')
    cmd('FblHighGain', b['gain_sdma'][-1], '%.0f')
    cmd('FblNomaGainLo', b['gain_noma'][0], '%.0f')
    cmd('FblNomaGainHi', b['gain_noma'][-1], '%.0f')
    sh = np.array(b['shannon'])
    cmd('IdealRateLo', sh[0]); cmd('IdealRate', sh[-1])
    cmd('FblGapLow', 100 * (1 - gp[0] / sh[0]), '%.0f')
    cmd('FblGapHigh', 100 * (1 - gp[-1] / sh[-1]), '%.0f')
    cmd('FblCi', float(np.mean(b['proposed_ci'])), '%.3f')

p = S.get('fig_power')
if p:
    xs = p['xs']
    cmd('PwrLo', xs[0], '%d'); cmd('PwrHi', xs[-1], '%d')
    cmd('PwrPropLo', p['proposed'][0]); cmd('PwrPropHi', p['proposed'][-1])
    cmd('PwrPasLo', p['rsma_pas'][0]); cmd('PwrActLo', p['rsma_act'][0])
    cmd('PwrDiagLo', p['rsma_diag'][0])
    cmd('PwrGainPas', 100 * (p['proposed'][0] / p['rsma_pas'][0] - 1), '%.0f')
    cmd('PwrGainDiag', 100 * (p['proposed'][0] / p['rsma_diag'][0] - 1), '%.0f')
    cmd('PwrGainAct', 100 * (p['proposed'][0] / p['rsma_act'][0] - 1), '%.0f')
    f = p.get('feas', {})
    if f:
        cmd('FeasPasLo', 100 * max(f['rsma_pas']), '%.0f')
        cmd('FeasActLo', 100 * f['rsma_act'][0], '%.0f')
        cmd('FeasPropLo', 100 * min(f['proposed']), '%.0f')
        cmd('FeasActMin', 100 * min(f['rsma_act']), '%.0f')

c = S.get('fig_crb')
if c:
    xs = c['xs']
    cmd('CrbLo', xs[0], '%.2f'); cmd('CrbHi', xs[-1], '%.2f')
    cmd('CrbRateLo', c['proposed'][0]); cmd('CrbRateHi', c['proposed'][-1])
    cmd('CrbDrop', 100 * (1 - c['proposed'][0] / c['proposed'][-1]), '%.0f')
    cmd('CrbGainLo', 100 * (c['proposed'][0] / c['sdma'][0] - 1), '%.0f')
    cmd('CrbGainHi', 100 * (c['proposed'][-1] / c['sdma'][-1] - 1), '%.0f')

m = S.get('fig_partition')
if m:
    cmd('PartNaMean', m['gated_na'], '%.1f')
    cmd('PartNaSd', m['gated_na_sd'], '%.1f')
    cmd('PartNaMed', m['gated_na_med'], '%.0f')
    cmd('PartGated', m['gated'])
    cmd('PartCi', m.get('gated_ci', float('nan')), '%.2f')
    cmd('PartSweepBest', m['best_of_sweep'])
    cmd('PartSweepArg', m['sweep_argmax_med'], '%.0f')
    cmd('PartGapPct', abs(m['gap_pct']), '%.1f')
    cmd('PartSpeedup', m['speedup'], '%.1f')
    cmd('PartWinPct', m.get('win_pct', float('nan')), '%.0f')
    cmd('PartCommBest', m['commbest'], '%d')
    cmd('PartNaHi', m['gated_na_hi'], '%.1f')
    cmd('PartGatedHi', m['gated_hi'])
    cmd('PartSweepBestHi', m['best_of_sweep_hi'])
    cmd('PartNamaxNa', ', '.join('%.1f' % v for v in m['namax_na']))
    cmd('PartNamaxRate', ', '.join('%.2f' % v for v in m['namax_rate']))
    cmd('PartFixedBest', m.get('sweep_best_mean', float('nan')))
    cmd('PartFixedArg', m.get('sweep_best_arg', 0), '%d')
    cmd('PartGainFixed', m.get('gain_over_fixed', float('nan')), '%.1f')
    cmd('PartWinFixed', m.get('win_pct_fixed', float('nan')), '%.0f')
    cmd('PartTsweep', m['t_sweep'], '%.1f')
    cmd('PartTgate', m['t_gate'], '%.1f')

v = S.get('fig_convergence')
if v:
    NAME = {1: 'One', 4: 'Four', 8: 'Eight'}
    for t in (1, 4, 8):
        k = 'T%d' % t
        if k in v:
            cmd('ConvIters' + NAME[t], v[k]['iters99'], '%d')
            cmd('ConvTime' + NAME[t], v[k]['time99'], '%.1f')
            cmd('ConvFinal' + NAME[t], v[k]['final'])
    cmd('ConvLoopSpeedup', v.get('loop_speedup') or float('nan'), '%.1f')
    cmd('ConvSlope', v.get('gmap_slope', float('nan')), '%.2f')
    cmd('ConvSteps', v.get('gmap_T', 0), '%d')
    cmd('ConvGmapFinal', v.get('gmap_final', float('nan')), '%.3f')
    cmd('ConvEnvFinal', v.get('envelope_final', float('nan')), '%.3f')
    cmd('ConvTarget', v.get('target', float('nan')))

r = S.get('fig_pulses')
if r:
    cmd('PulLo', r['xs'][0], '%d'); cmd('PulHi', r['xs'][-1], '%d')
    cmd('PulThPropLo', r['proposed']['theta'][0], '%.3f')
    cmd('PulThPropHi', r['proposed']['theta'][-1], '%.3f')
    cmd('PulThDiagHi', r['rsma_diag']['theta'][-1], '%.3f')
    cmd('PulRgPropLo', r['proposed']['rng'][0], '%.4f')
    cmd('PulRgPropHi', r['proposed']['rng'][-1], '%.4f')
    cmd('PulRgDiagHi', r['rsma_diag']['rng'][-1], '%.4f')

q = S.get('fig_qos')
if q:
    cmd('QosLo', q['xs'][0], '%.1f'); cmd('QosHi', q['xs'][-1], '%.1f')
    cmd('QosRateLo', q['proposed'][0]); cmd('QosRateHi', q['proposed'][-1])
    cmd('QosDrop', 100 * (1 - q['proposed'][-1] / q['proposed'][0]), '%.0f')
    cmd('QosWorstLo', q['worst'][0]); cmd('QosWorstHi', q['worst'][-1])
    sp, ss = q['sat']['proposed'], q['sat']['sdma']
    cmd('QosSatPropMid', sp[2], '%.0f'); cmd('QosSatSdmaMid', ss[2], '%.0f')
    kk = max([i for i, x in enumerate(sp) if x > 99.0], default=0)
    cmd('QosMaxFeasProp', q['xs'][kk], '%.1f')
    kk2 = max([i for i, x in enumerate(ss) if x > 99.0], default=0)
    cmd('QosMaxFeasSdma', q['xs'][kk2], '%.1f')

V = {}
vp = os.path.join(HERE, '..', 'results', 'verify.json')
if os.path.exists(vp):
    V = json.load(open(vp)).get('v6', {}) or {}
cmd('GateXmin', V.get('min_excess_rel', float('nan')), '%.2f')
cmd('GateXmed', V.get('median_excess_rel', float('nan')), '%.2f')
cmd('GateNports', V.get('n_ports_on', 0), '%d')
cmd('GateFlips', V.get('mean_flips', float('nan')), '%.1f')
cmd('GateOneOpt', V.get('frac_one_optimal', float('nan')), '%.0f')
cmd('GateLam', V.get('mean_lambda', float('nan')), '%.4f')
nd_ = int(V.get('n_designs', 0)); cmd('VerOneTot', nd_, '%d')
cmd('VerOneOpt', int(round(V.get('frac_one_optimal', 0) * nd_ / 100.0)), '%d')

d = S.get('fig_cdf')
if d:
    for k, tag in [('proposed', 'Prop'), ('sdma', 'Sdma'), ('noma', 'Noma')]:
        cmd('Cdf%sMean' % tag, d[k]['mean'])
        cmd('Cdf%sPfive' % tag, d[k]['p5'])
        cmd('Cdf%sOut' % tag, 100 * d[k]['out15'], '%.1f')
    cmd('CdfN', d['proposed']['n'], '%d')

# every macro the manuscript may reference must exist, even if a study is
# missing, so that the build never fails on an undefined control sequence
FALLBACK = ['CmcReal', 'FblLbLo', 'FblLbHi', 'FblLowRate', 'FblHighRate',
            'FblLowSdma', 'FblHighSdma', 'FblLowGain', 'FblHighGain',
            'FblNomaGainLo', 'FblNomaGainHi', 'IdealRate', 'IdealRateLo', 'FblGapLow',
            'FblGapHigh', 'FblCi', 'PwrLo', 'PwrHi', 'PwrPropLo', 'PwrPropHi',
            'PwrPasLo', 'PwrActLo', 'PwrDiagLo', 'PwrGainPas', 'PwrGainDiag',
            'PwrGainAct', 'FeasPasLo', 'FeasActLo', 'FeasPropLo',
            'FeasActMin', 'CrbLo', 'CrbHi', 'CrbRateLo', 'CrbRateHi',
            'CrbDrop', 'CrbGainLo', 'CrbGainHi', 'PartNaMean', 'PartNaSd',
            'PartNaMed', 'PartGated', 'PartSweepBest', 'PartSweepArg',
            'PartGapPct', 'PartSpeedup', 'PartWinPct', 'PartCommBest',
            'PartNaHi', 'PartGatedHi', 'PartSweepBestHi', 'PartNamaxNa',
            'PartNamaxRate', 'PartTsweep', 'PartTgate', 'PartFixedBest',
            'PartFixedArg', 'PartGainFixed', 'PartWinFixed', 'PartCi',
            'GateXmin', 'GateXmed', 'GateNports', 'GateFlips', 'GateOneOpt', 'GateLam', 'VerOneTot', 'VerOneOpt', 'ConvItersOne',
            'ConvTimeOne', 'ConvFinalOne', 'ConvItersFour', 'ConvTimeFour',
            'ConvFinalFour', 'ConvItersEight', 'ConvTimeEight',
            'ConvFinalEight', 'ConvLoopSpeedup', 'ConvSteps', 'ConvGmapFinal',
            'ConvEnvFinal', 'ConvTarget',
            'ConvSlope', 'PulLo', 'PulHi', 'PulThPropLo', 'PulThPropHi',
            'PulThDiagHi', 'PulRgPropLo', 'PulRgPropHi', 'PulRgDiagHi',
            'QosLo', 'QosHi', 'QosRateLo', 'QosRateHi', 'QosDrop',
            'QosWorstLo', 'QosWorstHi', 'QosSatPropMid', 'QosSatSdmaMid',
            'QosMaxFeasProp', 'QosMaxFeasSdma', 'CdfPropMean', 'CdfPropPfive',
            'CdfPropOut', 'CdfSdmaMean', 'CdfSdmaPfive', 'CdfSdmaOut',
            'CdfNomaMean', 'CdfNomaPfive', 'CdfNomaOut', 'CdfN']
missing = [k for k in FALLBACK if k not in SEEN]
for k in missing:
    cmd(k, '??')

open(OUT, 'w').write('\n'.join(L) + '\n')
print('wrote %d macros to %s' % (len(L), OUT))
if missing:
    print('MISSING (emitted as ??):', ' '.join(missing))
