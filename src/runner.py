"""Monte Carlo studies for the SL-GPM paper.

Every study writes an append-only JSONL checkpoint, so an interrupted run is
resumed simply by launching the script again: jobs whose key is already in the
checkpoint are skipped.  When a study is complete its JSONL is condensed into
results/<tag>.json.

Usage:  python runner.py [study digits] [--nreal N] [--workers W]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')   # small dense algebra: threads only hurt
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

NREAL = int(os.environ.get('NREAL', 200))
NCDF = int(os.environ.get('NCDF', 300))       # designs for the outage study
NHI = int(os.environ.get('NHI', 80))          # realisations for the side studies
NPART = int(os.environ.get('NPART', NREAL))   # realisations for the partition study
N_ITER = 20
T_IN = 8
BASE_SEED = 2024

PARTS = [4, 8, 16, 24, 32, 40, 48, 64]
LB_LIST = [50, 100, 200, 300, 500, 800]
PW_LIST = [33, 35, 37, 39, 41, 43]
CRB_LIST = [0.07, 0.09, 0.12, 0.16, 0.22, 0.35]
QOS_LIST = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
PUL_LIST = [16, 32, 64, 128, 256, 512]


# ----------------------------------------------------------------------------
def key_of(job):
    return hashlib.md5(json.dumps(job, sort_keys=True,
                                  default=str).encode()).hexdigest()[:16]


def run_one(job):
    """job = dict(kw=..., name=..., seed=..., crb_on=..., track=..., part=...)

    A job that raises is reported rather than killing the study: the entry is
    written with ``failed`` set so it is visible downstream and can be re-run by
    deleting it from the checkpoint."""
    try:
        return _run_one(job)
    except Exception as e:
        print('  JOB FAILED %s: %s: %s' % (job['_key'], type(e).__name__, e),
              flush=True)
        return dict(key=job['_key'], failed=True, sum_rate=float('nan'),
                    rates=[], crb=[float('inf'), float('inf')], feas=False,
                    viol=float('inf'), name=job['name'], seed=job['seed'],
                    na=0, namax=0, iters=0, runtime=0.0)


def _run_one(job):
    kw = job.get('kw', {})
    p = sm.SysParams(**kw)
    ch = sm.Channels(p, np.random.default_rng(BASE_SEED + job['seed']))
    sch = al.SCHEMES[job['name']]
    if job.get('probe'):
        pr = al.stationarity_probe(ch, p, sch, n_warm=6, T=job.get('T', 400))
        return dict(key=job['_key'], name=job['name'], seed=job['seed'],
                    gtrace=pr['g'], ftrace=pr['f'], sum_rate=float('nan'),
                    rates=[], crb=[0.0, 0.0], feas=True, viol=0.0, na=0,
                    namax=0, iters=len(pr['g']), runtime=0.0)
    r = al.sl_gpm(ch, p, sch, n_iter=job.get('n_iter', N_ITER),
                  crb_on=job.get('crb_on', True), track=job.get('track', False),
                  single_loop=job.get('single_loop', False),
                  n_inner=job.get('n_inner', T_IN),
                  part=job.get('part', None), refine=job.get('refine', True))
    out = dict(key=job['_key'], sum_rate=float(r['sum_rate']),
               rates=np.asarray(r['rates']).tolist(),
               crb=[float(x) for x in r['crb']], feas=bool(r['feas']),
               viol=float(r['viol']), name=job['name'], seed=job['seed'],
               na=int(r['na']), namax=int(r['namax']), iters=int(r['iters']),
               runtime=float(r['runtime']))
    if job.get('part') is not None:
        out['part'] = int(job['part'])
    if job.get('track'):
        out['hist'] = r['hist']
        if r.get('na_path') is not None:
            out['na_path'] = r['na_path']
    if job.get('cdf'):
        out['cdf'] = cdf_draws(r, ch, p, sch, job['seed'])
    return out


def cdf_draws(r, ch, p, sch, seed, n_draw=100):
    """Design on the estimate, evaluate on independent true channel draws."""
    inst = al.Instance(ch, p, part=r['Pa'].shape[0] if r['Pa'].size else 0,
                       na=int(r['na']))
    Hhat = inst.eff(r['Pa'], r['Pp'])
    wc, W, Cv = r['wc'], r['W'], np.asarray(r['C'], float)
    rng = np.random.default_rng(90000 + seed)
    sd = np.sqrt(inst.eps2 / 2.0)[None, :]
    vals = []
    for _ in range(n_draw):
        Ht = Hhat + sd * (rng.normal(size=Hhat.shape) + 1j * rng.normal(size=Hhat.shape))
        S = Ht.conj().T @ np.concatenate([wc[:, None], W], axis=1)
        tot = np.sum(np.abs(S[:, 1:]) ** 2, axis=1) + inst.sig2 \
            + inst.sig2a * inst.qnoise(r['Pa'])
        dg = np.abs(np.diag(S[:, 1:])) ** 2
        Rp = sm.fbl_rate(dg / np.maximum(tot - dg, 1e-12), p.Lb, p.eps)
        if sch.access == 'rsma':
            Rc = float(sm.fbl_rate(np.abs(S[:, 0]) ** 2 / tot, p.Lb, p.eps).min())
            a = Cv / Cv.sum() * Rc if Cv.sum() > 1e-12 else np.full(p.K, Rc / p.K)
            Rk = a + Rp
        else:
            Rk = Rp
        vals.extend([float(x) for x in Rk])
    return vals


# ----------------------------------------------------------------------------
def sweep(tag, jobs, workers):
    """Run ``jobs``, resuming from the JSONL checkpoint of the same tag."""
    ck = os.path.join(RES, tag + '.jsonl')
    done = {}
    if os.path.exists(ck):
        with open(ck) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done[d['key']] = d
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
                f.write(json.dumps(out) + '\n')
                f.flush()
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
    E = {}

    # ---- Study 1 : FBL sum rate versus blocklength --------------------------
    if '1' in which:
        jobs = [dict(kw={'Lb': Lb}, name=nm, seed=s)
                for Lb in LB_LIST
                for nm in ['proposed', 'sdma', 'noma', 'shannon']
                for s in range(NREAL)]
        sweep('sim1_blocklength', jobs, workers)

    # ---- Study 2 : FBL sum rate versus total transmit power -----------------
    if '2' in which:
        jobs = [dict(kw={'Ptot_dBm': P}, name=nm, seed=s)
                for P in PW_LIST
                for nm in ['proposed', 'sdma', 'noma', 'rsma_pas', 'rsma_act',
                           'rsma_diag']
                for s in range(NREAL)]
        sweep('sim2_power', jobs, workers)

    # ---- Study 3 : rate versus matrix CRB threshold -------------------------
    if '3' in which:
        jobs = [dict(kw={'crb_theta': float(np.deg2rad(d)),
                         'crb_range': 0.2 * d / 0.1}, name=nm, seed=s)
                for d in CRB_LIST
                for nm in ['proposed', 'sdma', 'rsma_diag']
                for s in range(NREAL)]
        sweep('sim3_crb', jobs, workers)

    # ---- Study 4 : aperture partition, swept versus optimised ---------------
    if '4' in which:
        jobs = [dict(kw={}, name='fixed', seed=s, part=pt)
                for pt in PARTS for s in range(NPART)]
        sweep('sim4_sweep', jobs, workers)
        jobs = [dict(kw={}, name='fixed', seed=s, part=pt, crb_on=False)
                for pt in PARTS for s in range(NPART)]
        sweep('sim4_sweep_nocrb', jobs, workers)
        jobs = [dict(kw={}, name='proposed', seed=s, track=True)
                for s in range(NPART)]
        sweep('sim4_gated', jobs, workers)
        jobs = [dict(kw={'Ptot_dBm': 39.0}, name='fixed', seed=s, part=pt)
                for pt in PARTS for s in range(NHI)]
        sweep('sim4_sweep_hi', jobs, workers)
        jobs = [dict(kw={'Ptot_dBm': 39.0}, name='proposed', seed=s, track=True)
                for s in range(NHI)]
        sweep('sim4_gated_hi', jobs, workers)
        # robustness of the recovered partition to the amplifier-equipped size
        jobs = [dict(kw={'Mact': mm, 'Mpas': 64 - mm}, name='proposed', seed=s)
                for mm in [16, 32, 48, 64] for s in range(NHI)]
        sweep('sim4_namax', jobs, workers)

    # ---- Study 5 : user fairness versus the QoS floor -----------------------
    if '5' in which:
        jobs = [dict(kw={'Rth': R}, name=nm, seed=s)
                for R in QOS_LIST for nm in ['proposed', 'sdma']
                for s in range(NREAL)]
        sweep('sim5_qos', jobs, workers)

    # ---- Study 6 : estimation accuracy versus radar pulse count -------------
    if '6' in which:
        jobs = [dict(kw={'Lradar': L, 'crb_theta': 1e-1, 'crb_range': 10.0},
                     name=nm, seed=s)
                for L in PUL_LIST for nm in ['proposed', 'rsma_pas', 'rsma_diag']
                for s in range(NREAL)]
        sweep('sim6_pulses', jobs, workers)

    # ---- Study 7 : convergence, stationarity rate and loop structure --------
    if '7' in which:
        for tag, tin, nit in [('sim7_t1', 1, 60), ('sim7_t4', 4, 30),
                              ('sim7_t8', 8, 20)]:
            jobs = [dict(kw={'Lb': Lb}, name='proposed', seed=s, track=True,
                         single_loop=(tin == 1), n_inner=tin, refine=False,
                         n_iter=nit)
                    for Lb in [100, 200, 500] for s in range(16)]
            sweep(tag, jobs, workers)

    # ---- Study 9 : stationarity rate of the surface block -------------------
    if '9' in which:
        jobs = [dict(kw={'Lb': Lb}, name='proposed', seed=s, probe=True)
                for Lb in [100, 200, 500] for s in range(24)]
        sweep('sim9_rate', jobs, workers)

    # ---- Study 8 : achievable rate CDF under imperfect CSI ------------------
    if '8' in which:
        jobs = [dict(kw={}, name=nm, seed=s, cdf=True)
                for nm in ['proposed', 'sdma', 'noma'] for s in range(NCDF)]
        sweep('sim8_cdf', jobs, workers)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    which = args[0] if args else '12345678'
    workers = int(os.environ.get('WORKERS', 2))
    main(which, workers)
