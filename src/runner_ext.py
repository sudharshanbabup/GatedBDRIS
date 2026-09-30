"""Extended Monte-Carlo studies added in the paper_8 revision.

These back the reviewer-requested experiments with the SAME real solver as the
base campaign (runner.py): a range-requirement sweep with a scalar-CRB baseline,
the matrix/scalar/nuisance ablation, RIS-size and user scaling, amplifier-noise
and DC-power sensitivity, and a genuine initialisation-robustness study.

Every study writes an append-only JSONL checkpoint and a condensed results
JSON, exactly like runner.py, so runs resume on relaunch.

Usage:  python runner_ext.py [letters]     (default: all)
  s  sim_pulses_sr   range-requirement sweep + scalar-CRB baseline
  a  sim_ablation    matrix / scalar / no-nuisance at the nominal point
  m  sim_mscale      RIS size M in {32,64,96,128}
  k  sim_kscale      users K in {2,4,6,8}
  f  sim_fa          amplifier noise figure F_a in {3,5,7,10} dB
  d  sim_pdc         bias power P_DC in {6,9,12,15} dBm
  i  sim_init        four initialisations, rate/na spread
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, time, warnings, hashlib
import numpy as np
import multiprocessing as mp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system_model as sm
import algorithm as al

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)

NREAL = int(os.environ.get('NREAL', 50))
NHI = int(os.environ.get('NHI', 30))
BASE_SEED = 2024
N_ITER = 20
T_IN = 8

SR_LIST = [0.040, 0.025, 0.016, 0.012, 0.009, 0.007]     # range requirement (m)
M_LIST = [32, 64, 96, 128]
K_LIST = [2, 4, 6, 8]
FA_LIST = [3.0, 5.0, 7.0, 10.0]
PDC_LIST = [6.0, 9.0, 12.0, 15.0]
INIT_LIST = ['max_gain', 'chan_align', 'passive', 'random']


def key_of(job):
    return hashlib.md5(json.dumps(job, sort_keys=True,
                                  default=str).encode()).hexdigest()[:16]


def _run_one(job):
    if job.get('kind') == 'echo':
        return _run_echo(job)
    if job.get('kind') == 'access':
        return _run_access(job)
    kw = job.get('kw', {})
    p = sm.SysParams(**kw)
    ch = sm.Channels(p, np.random.default_rng(BASE_SEED + job['seed']))
    sch = al.SCHEMES[job['name']]
    r = al.sl_gpm(ch, p, sch, n_iter=job.get('n_iter', N_ITER),
                  crb_on=job.get('crb_on', True),
                  n_inner=job.get('n_inner', T_IN),
                  part=job.get('part', None), refine=job.get('refine', True),
                  crb_mode=job.get('crb_mode', 'matrix'),
                  init=job.get('init', 'max_gain'),
                  init_seed=job.get('init_seed', 0))
    out = dict(key=job['_key'], sum_rate=float(r['sum_rate']),
               rates=np.asarray(r['rates']).tolist(),
               crb=[float(x) for x in r['crb']], feas=bool(r['feas']),
               viol=float(r['viol']), name=job['name'], seed=job['seed'],
               na=int(r['na']), namax=int(r['namax']), iters=int(r['iters']),
               runtime=float(r['runtime']))
    for tagk in ('crb_mode', 'init', 'part', 'init_seed'):
        if tagk in job:
            out[tagk] = job[tagk]
    return out


ECHO_W = [0.0, 0.25, 0.5, 1.0]


def _run_echo(job):
    """Design with the paper's model (passive echo neglected), then evaluate
    the nuisance aware matrix CRB with the lossless block's echo route
    included at weight w."""
    p = sm.SysParams(**job.get('kw', {}))
    ch = sm.Channels(p, np.random.default_rng(BASE_SEED + job['seed']))
    r = al.sl_gpm(ch, p, al.SCHEMES[job['name']], n_iter=N_ITER, n_inner=T_IN,
                  refine=False)
    Pa, Pp = r['Pa'], r['Pp']
    part = Pa.shape[0] if Pa.size else 0
    Rx = np.outer(r['wc'], r['wc'].conj()) + r['W'] @ r['W'].conj().T
    ev = []
    for w in ECHO_W:
        J = sm.efim_echo(ch, Pa, Pp, Rx, part, w)
        try:
            C = np.linalg.inv(J + 1e-30 * np.eye(2))
            rng_ = float(np.sqrt(max(C[1, 1], 0.0))) if C[1, 1] > 0 else 1e9
            th_ = float(np.sqrt(max(C[0, 0], 0.0)))
        except np.linalg.LinAlgError:
            rng_, th_ = 1e9, 1e9
        # smallest s with J^-1 <= s Cmax (s <= 1 means jointly feasible)
        try:
            Ci = np.linalg.inv(J + 1e-30 * np.eye(2))
            Lc = np.linalg.cholesky(np.linalg.inv(p.Cmax))
            scl = float(np.max(np.linalg.eigvalsh(Lc.T @ Ci @ Lc)))
        except np.linalg.LinAlgError:
            scl = 1e9
        ev.append(dict(w=w, rng=min(rng_, 1e9), th=th_, scl=min(scl, 1e9),
                       feas=bool(sm.crb_feasible(J, p.Cmax))))
    return dict(key=job['_key'], name=job['name'], seed=job['seed'],
                sum_rate=float(r['sum_rate']), na=int(r['na']), echo=ev,
                runtime=float(r['runtime']))


def precoder_only(ch, p, name, Pa, Pp, na, n_iter=N_ITER, crb=True):
    """Optimise ONLY the precoders / rate split of access scheme `name` on a
    FIXED surface (Phi_a, Phi_p) and activation n_a, with the matrix CRB."""
    sch = al.SCHEMES[name]
    part = Pa.shape[0] if Pa.size else 0
    inst = al.Instance(ch, p, part=part, na=na)
    H = inst.eff(Pa, Pp)
    order = list(np.argsort(-np.linalg.norm(H, axis=0)))
    stages, has_common = al.build_stages(sch.access, p.K, order)
    W = H / np.linalg.norm(H, axis=0, keepdims=True)
    wc = W.sum(1); wc = wc / np.linalg.norm(wc)
    pw = inst.PBS / (p.K + (1 if has_common else 0))
    W = W * np.sqrt(pw)
    wc = wc * np.sqrt(pw) if has_common else np.zeros(p.Nt, complex)
    tfl = sm.sinr_floor(p.Lb, p.eps)
    g, _, _, _ = al.stage_sinr(inst, Pa, Pp, wc, W, stages)
    tref = {k: max(v, tfl) for k, v in g.items()}
    mu, cands = 400.0, []
    for _ in range(n_iter):
        wc, W, Cv, ok, _ = al.solve_precoders(inst, Pa, Pp, wc, W, tref, sch,
                                              stages, has_common, crb_on=crb,
                                              mu_crb=mu)
        g, _, _, _ = al.stage_sinr(inst, Pa, Pp, wc, W, stages)
        tref = {k: max(v, tfl) for k, v in g.items()}
        tot = al.scheme_rate(inst, Pa, Pp, wc, W, Cv, stages, sch, order)[0]
        Rx = np.outer(wc, wc.conj()) + W @ W.conj().T
        J = sm.efim(ch, Pa, Rx, part=part)
        vio = float(sm.crb_violation(J, p.Cmax)) if crb else 0.0
        cands.append((vio <= 1e-9, tot, -vio))
        if vio > 1e-9:
            mu = min(mu * 6.0, 2e6)
    feas = [c for c in cands if c[0]]
    if feas:
        return max(c[1] for c in feas), True
    return max(cands, key=lambda c: c[2])[1], False


def _run_access(job):
    """Surface gain versus access gain: fix the proposed design's surface and
    activation, then re-optimise only the access scheme."""
    p = sm.SysParams(**job.get('kw', {}))
    ch = sm.Channels(p, np.random.default_rng(BASE_SEED + job['seed']))
    r = al.sl_gpm(ch, p, al.SCHEMES['proposed'], n_iter=N_ITER, n_inner=T_IN,
                  refine=False)
    out = dict(key=job['_key'], name='access', seed=job['seed'],
               joint=float(r['sum_rate']), na=int(r['na']))
    for nm in ('proposed', 'sdma', 'noma'):
        v, f = precoder_only(ch, p, nm, r['Pa'], r['Pp'], int(r['na']))
        out['fixed_' + nm] = float(v); out['feas_' + nm] = bool(f)
        v, _ = precoder_only(ch, p, nm, r['Pa'], r['Pp'], int(r['na']), crb=False)
        out['comm_' + nm] = float(v)
    return out


def run_one(job):
    try:
        return _run_one(job)
    except Exception as e:
        print('  JOB FAILED %s: %s: %s' % (job['_key'], type(e).__name__, e),
              flush=True)
        return dict(key=job['_key'], failed=True, sum_rate=float('nan'),
                    rates=[], crb=[float('inf'), float('inf')], feas=False,
                    viol=float('inf'), name=job['name'], seed=job['seed'],
                    na=0, namax=0, iters=0, runtime=0.0)


def sweep(tag, jobs, workers):
    ck = os.path.join(RES, tag + '.jsonl')
    done = {}
    if os.path.exists(ck):
        with open(ck) as f:
            for line in f:
                try:
                    d = json.loads(line); done[d['key']] = d
                except Exception:
                    pass
    for j in jobs:
        # the gated hardware cannot be rewired, so a gated scheme is reported
        # from its single (gated) pass only; see the paper, Section II-E
        sc = al.SCHEMES.get(j.get('name'))
        if sc is not None and sc.gate and 'refine' not in j:
            j['refine'] = False
        j['_key'] = key_of({k: v for k, v in j.items() if k != '_key'})
    todo = [j for j in jobs if j['_key'] not in done]
    print(f'[{tag}] {len(jobs)} jobs, {len(todo)} to run', flush=True)
    t0 = time.time()
    if todo:
        with open(ck, 'a') as f, mp.Pool(workers) as pool:
            for n, out in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
                done[out['key']] = out
                f.write(json.dumps(out) + '\n'); f.flush()
                if (n + 1) % 25 == 0 or n + 1 == len(todo):
                    el = time.time() - t0
                    print(f'  [{tag}] {n+1}/{len(todo)}  {el:.0f}s  '
                          f'eta {el/(n+1)*(len(todo)-n-1):.0f}s', flush=True)
    res = [done[j['_key']] for j in jobs]
    with open(os.path.join(RES, tag + '.json'), 'w') as f:
        json.dump(res, f)
    print(f'[{tag}] complete in {time.time()-t0:.0f}s', flush=True)
    return res


def main(which, workers):
    # ---- range-requirement sweep with a scalar-CRB baseline (Fig. pulses) ----
    if 's' in which:
        jobs = []
        for sr in SR_LIST:
            kw = {'crb_range': sr}
            jobs += [dict(kw=kw, name='proposed', seed=s, crb_mode='matrix')
                     for s in range(NREAL)]
            jobs += [dict(kw=kw, name='proposed', seed=s, crb_mode='scalar')
                     for s in range(NREAL)]
            jobs += [dict(kw=kw, name='rsma_diag', seed=s, crb_mode='matrix')
                     for s in range(NREAL)]
            jobs += [dict(kw=kw, name='rsma_pas', seed=s, crb_mode='matrix')
                     for s in range(NREAL)]
        sweep('sim_pulses_sr', jobs, workers)

    # ---- ablation: matrix vs scalar vs no-nuisance at the nominal point ------
    if 'a' in which:
        jobs = []
        for md in ['matrix', 'scalar', 'matrix_nonuis']:
            jobs += [dict(kw={}, name='proposed', seed=s, crb_mode=md)
                     for s in range(NREAL)]
        sweep('sim_ablation', jobs, workers)

    # ---- RIS size scaling ----------------------------------------------------
    if 'm' in which:
        jobs = []
        for M in M_LIST:
            ma = int(round(0.75 * M)); mp_ = M - ma
            jobs += [dict(kw={'Mact': ma, 'Mpas': mp_}, name='proposed', seed=s,
                          part=ma) for s in range(NHI)]
        sweep('sim_mscale', jobs, workers)

    # ---- user scaling --------------------------------------------------------
    if 'k' in which:
        jobs = []
        for K in K_LIST:
            jobs += [dict(kw={'K': K}, name='proposed', seed=s)
                     for s in range(NHI)]
        sweep('sim_kscale', jobs, workers)

    # ---- amplifier noise-figure sensitivity ---------------------------------
    if 'f' in which:
        jobs = []
        for fa in FA_LIST:
            jobs += [dict(kw={'NF_act_dB': fa}, name='proposed', seed=s)
                     for s in range(NHI)]
        sweep('sim_fa', jobs, workers)

    # ---- DC bias-power sensitivity ------------------------------------------
    if 'd' in which:
        jobs = []
        for pd in PDC_LIST:
            jobs += [dict(kw={'PDC_dBm': pd}, name='proposed', seed=s)
                     for s in range(NHI)]
        sweep('sim_pdc', jobs, workers)

    # ---- initialisation robustness (four genuine starts) --------------------
    if 'i' in which:
        jobs = []
        for ini in INIT_LIST:
            jobs += [dict(kw={}, name='proposed', seed=s, init=ini)
                     for s in range(NHI)]
        sweep('sim_init', jobs, workers)

    # ---- off-state insertion loss of an unbiased amplifier port ------------
    if 'l' in which:
        jobs = []
        for il in [0.0, 0.5, 1.0, 2.0]:
            jobs += [dict(kw={'off_loss_dB': il}, name='proposed', seed=s)
                     for s in range(NHI)]
        sweep('sim_il', jobs, workers)

    # ---- CSI-error model sensitivity (eps^2 scaled by 0.5, 1, 2) ------------
    if 'c' in which:
        jobs = []
        for cs in [0.5, 1.0, 2.0]:
            for nm in ['proposed', 'sdma']:
                jobs += [dict(kw={'csi_scale': cs}, name=nm, seed=s)
                         for s in range(NREAL)]
        sweep('sim_csi', jobs, workers)

    # ---- 20 seeded random initialisations per channel -----------------------
    if 'r' in which:
        jobs = [dict(kw={}, name='proposed', seed=s, init='random', init_seed=j)
                for s in range(NREAL) for j in range(20)]
        sweep('sim_init20', jobs, workers)

    # ---- passive echo route sensitivity ------------------------------------
    if 'e' in which:
        jobs = [dict(kind='echo', v=3, kw={}, name=nm, seed=s)
                for nm in ['proposed', 'rsma_pas'] for s in range(NREAL)]
        sweep('sim_echo', jobs, workers)

    # ---- fixed-surface access ablation -------------------------------------
    if 'x' in which:
        jobs = [dict(kind='access', v=2, kw={}, name='access', seed=s)
                for s in range(NREAL)]
        sweep('sim_access', jobs, workers)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    which = args[0] if args else 'samkfdi'
    workers = int(os.environ.get('WORKERS', 2))
    main(which, workers)
