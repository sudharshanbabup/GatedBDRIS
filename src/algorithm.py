"""
GPM : Gated Penalty-Manifold algorithm for finite blocklength symbiotic RSMA
BD-RIS ISAC.

Every outer iteration performs

  (a) one convex precoder / rate-split program                          (P4)
  (b) T_in majorisation-minimisation steps on the two scattering matrices,
      including the smoothed-count amplifier activation penalty          (P5)
  (c) per step, one closed-form manifold projection, one gate retraction
      and one dual ascent                                                (P6)

and records the gradient-mapping residual of every surface step, whose
O(1/sqrt(T)) decay in the number of surface steps T is what Theorem 2
certifies (the bound is independent of T_in).

A "stage" (k, j, I) means: user k decodes message j while treating the
messages in I as noise.  All multiple-access schemes are instances of a stage
list, which is what makes every benchmark share one solver.
"""
import time
import numpy as np
import cvxpy as cp
import system_model as sm

LOG2 = np.log(2.0)
COMMON = -1                      # message index of the RSMA super-common stream


# ==========================================================================
#  Noise-normalised problem instance
# ==========================================================================
class Instance:
    """Noise normalised view of one channel realisation.

    ``part`` is the number of AMPLIFIER EQUIPPED ports, i.e. the hardware size
    M_a^max of the active sub-surface.  How many of those ports are actually
    biased is the optimisation variable ``na``: only powered ports are charged
    the bias power P_DC, so the aperture partition is decided by the algorithm
    and not by an exhaustive sweep."""

    def __init__(self, ch: sm.Channels, p: sm.SysParams, part=None, na=None):
        self.ch, self.p = ch, p
        self.part = p.Mact if part is None else int(part)
        s = np.sqrt(p.sigma2_ue)
        Ma = self.part
        self.hd = ch.hd / s
        self.hact, self.hpas = ch.hfull[:Ma] / s, ch.hfull[Ma:] / s
        self.Gact, self.Gpas = ch.Gfull[:Ma], ch.Gfull[Ma:]
        self.sig2 = 1.0
        self.sig2a = p.sigma2_act
        self._Om = None
        self.set_na(Ma if na is None else na)

    # ---- the powered-port count drives the transmit power budget -----------
    def set_na(self, na):
        na = int(min(max(na, 0), self.part))
        self.na = na
        self.PBS = self.p.PBS_of(na)
        self._set_csi()
        return self.PBS

    def _set_csi(self):
        """Normalised CSI error variance per antenna.  A block of L_b channel
        uses devotes zeta L_b symbols to training; a BD-RIS with M elements has
        O(M^2) cascaded coefficients, so linear MMSE estimation gives
        tau_k^2 = 1/(1 + zeta L_b rho_k /(M(M+1)))."""
        p = self.p
        if self._Om is None:
            self._Om = (np.sum(np.abs(self.hd) ** 2, axis=0)
                        + np.sum(np.abs(self.Gact.conj().T @ self.hact) ** 2, axis=0)
                        + np.sum(np.abs(self.Gpas.conj().T @ self.hpas) ** 2, axis=0))
        Om = self._Om
        self.Omega = Om
        rho = self.PBS * Om / p.Nt
        M = p.M
        if p.tau2 is None:
            tau2 = 1.0 / (1.0 + p.pilot_frac * p.Lb * rho / (M * (M + 1.0)))
        else:
            tau2 = np.full(p.K, float(p.tau2))
        self.tau2 = tau2
        self.eps2 = float(getattr(p, 'csi_scale', 1.0)) * tau2 * Om / p.Nt  # residual error power (scaled for sensitivity)

    def eff(self, Pa, Pp):
        return self.hd + self.Gact.conj().T @ (Pa.conj().T @ self.hact) \
                       + self.Gpas.conj().T @ (Pp.conj().T @ self.hpas)

    def qnoise(self, Pa):
        return np.sum(np.abs(Pa.conj().T @ self.hact) ** 2, axis=0)


# ==========================================================================
#  Scheme definitions
# ==========================================================================
class Scheme:
    """access: 'rsma' | 'sdma' | 'noma';  ris: 'hybrid' | 'passive' | 'active'
       | 'diagonal';  fbl: finite blocklength rate;  gate: optimise the number
       of powered amplifiers."""

    def __init__(self, name, access='rsma', ris='hybrid', fbl=True, label=None,
                 gate=False):
        self.name, self.access, self.ris, self.fbl = name, access, ris, fbl
        self.gate = gate
        self.label = label or name


SCHEMES = {
    'proposed':  Scheme('proposed', 'rsma', 'hybrid', True,
                        'Proposed: RSMA + gated hybrid BD-RIS', gate=True),
    'fixed':     Scheme('fixed', 'rsma', 'hybrid', True,
                        'RSMA + fixed partition (swept)'),
    'sdma':      Scheme('sdma', 'sdma', 'hybrid', True,
                        'SDMA + gated hybrid BD-RIS', gate=True),
    'noma':      Scheme('noma', 'noma', 'hybrid', True,
                        'NOMA + gated hybrid BD-RIS', gate=True),
    'rsma_pas':  Scheme('rsma_pas', 'rsma', 'passive', True, 'RSMA + passive BD-RIS'),
    'rsma_act':  Scheme('rsma_act', 'rsma', 'active', True, 'RSMA + fully active BD-RIS'),
    'rsma_diag': Scheme('rsma_diag', 'rsma', 'diagonal', True,
                        'RSMA + diagonal hybrid RIS', gate=True),
    'shannon':   Scheme('shannon', 'rsma', 'hybrid', False,
                        'Shannon bound (no dispersion penalty)', gate=True),
}


def build_stages(access, K, order=None):
    """Return (stages, has_common).  stage = (decoder k, message j, interf set)."""
    st = []
    if access == 'rsma':
        for k in range(K):
            st.append((k, COMMON, list(range(K))))
        for k in range(K):
            st.append((k, k, [j for j in range(K) if j != k]))
        return st, True
    if access == 'sdma':
        for k in range(K):
            st.append((k, k, [j for j in range(K) if j != k]))
        return st, False
    if access == 'noma':
        rank = {u: r for r, u in enumerate(order)}      # order[0] = strongest
        for k in range(K):
            for j in range(K):
                if rank[j] >= rank[k]:
                    st.append((k, j, [i for i in range(K) if rank[i] < rank[j]]))
        return st, False
    raise ValueError(access)


# ==========================================================================
#  Fisher information as an affine map of the transmit covariance
# ==========================================================================
def fim_operator(ch, Pa, part=None):
    a, a_dth, a_dr = sm.radar_vectors(ch, Pa, part)
    u, u_d = ch.beta_BT * ch.aNr, ch.beta_BT * ch.aNr_dth
    al = np.sqrt(ch.alpha0_sq)
    D = [[(al * u_d, 0), (al * u, 1)], [(al * u, 2)], [(u, 0)], [(1j * u, 0)]]
    return D, [a, a_dth, a_dr], 2.0 * ch.p.Lradar / ch.p.sigma2_bs


def fim_from_pairs(D, V, c, Rx):
    Q = np.array([[V[m].conj() @ Rx @ V[n] for n in range(3)] for m in range(3)])
    J = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            s = sum(np.vdot(uj, ui) * Q[mi, mj]
                    for (ui, mi) in D[i] for (uj, mj) in D[j])
            J[i, j] = J[j, i] = c * np.real(s)
    return J


# ==========================================================================
#  P4 : convex precoder / rate-split subproblem, compiled once per process
# ==========================================================================
_PCACHE = {}


def _build_program(Nt, K, access, stages, crb_on, crb_mode='matrix'):
    """Build the parametrised convex program once.

    Every quantity that changes from iteration to iteration is a cp.Parameter
    and the formulation is kept disciplined-parametrised (DPP), so
    canonicalisation is paid once per (Nt, K, access, crb_on) signature and each
    subsequent iteration only re-solves.  The two devices that make this
    possible are (i) the auxiliary signal variables S = H^H [w_c, W], which
    move every squared magnitude off the channel parameter, and (ii) folding
    all Fisher-information coefficients into a single complex matrix per LMI
    entry, so the linear matrix inequality stays parameter-affine."""
    idx = lambda j: 0 if j == COMMON else j + 1
    NS = K + 1
    keys = [s[:2] for s in stages]
    P = {'Hre': cp.Parameter((Nt, K)), 'Him': cp.Parameter((Nt, K)),
         'PBS': cp.Parameter(nonneg=True), 'Rth': cp.Parameter(),
         'ure': {k: cp.Parameter() for k in keys},
         'uim': {k: cp.Parameter() for k in keys},
         'c1': {k: cp.Parameter() for k in keys},              # |u|^2 (ne+1)
         'c2': {k: cp.Parameter(nonneg=True) for k in keys},   # |u|^2 eps2
         'c3': {k: cp.Parameter(nonneg=True) for k in keys},   # |u|^2
         'k1': {k: cp.Parameter() for k in keys},              # kap (psi - psi' t^r)
         'k2': {k: cp.Parameter() for k in keys}}              # kap psi'

    wc = cp.Variable(Nt, complex=True)
    W = cp.Variable((Nt, K), complex=True)
    Sre = cp.Variable((K, NS))
    Sim = cp.Variable((K, NS))
    tv = {k: cp.Variable(nonneg=True) for k in keys}
    sq = cp.Variable(K, nonneg=True)
    scrb = cp.Variable(nonneg=True)

    Wall = cp.hstack([cp.reshape(wc, (Nt, 1), order='F'), W])
    Hc = P['Hre'] + 1j * P['Him']
    cons = [Sre + 1j * Sim == cp.conj(Hc).T @ Wall]
    pw = cp.sum_squares(wc) + cp.sum_squares(W)
    Sabs2 = cp.square(Sre) + cp.square(Sim)
    obj = 0

    def Rt(key):
        return cp.log1p(tv[key]) / LOG2 + P['k1'][key] + P['k2'][key] * tv[key]

    for (k, j, I) in stages:
        key = (k, j)
        jj = idx(j)
        rhs = 2 * (P['ure'][key] * Sre[k, jj] + P['uim'][key] * Sim[k, jj]) \
            - P['c1'][key] - P['c2'][key] * pw
        if I:
            rhs = rhs - P['c3'][key] * cp.sum(Sabs2[k, [idx(i) for i in I]])
        cons += [tv[key] <= rhs]

    if access == 'rsma':
        Cv = cp.Variable(K, nonneg=True)
        for k in range(K):
            cons += [cp.sum(Cv) <= Rt((k, COMMON))]
            obj += Cv[k] + Rt((k, k))
            cons += [Cv[k] + Rt((k, k)) + sq[k] >= P['Rth']]
    elif access == 'sdma':
        Cv = None
        cons += [wc == 0]
        for k in range(K):
            obj += Rt((k, k))
            cons += [Rt((k, k)) + sq[k] >= P['Rth']]
    else:                                            # NOMA
        Cv = None
        cons += [wc == 0]
        rv = cp.Variable(K)
        for (k, j, I) in stages:
            cons += [rv[j] <= Rt((k, j))]
        for j in range(K):
            obj += rv[j]
            cons += [rv[j] + sq[j] >= P['Rth']]

    cpow = pw <= P['PBS']
    cons += [cpow]
    obj = obj - 60.0 * cp.sum(sq)

    if crb_on:
        P['mu'] = cp.Parameter(nonneg=True)
        P['Ere'] = [[cp.Parameter((Nt, NS)) for _ in range(4)] for _ in range(4)]
        P['Eim'] = [[cp.Parameter((Nt, NS)) for _ in range(4)] for _ in range(4)]
        P['Coff'] = cp.Parameter((4, 4), symmetric=True)
        Wre, Wim = cp.real(Wall), cp.imag(Wall)
        rows = []
        for i in range(4):
            row = []
            for j in range(4):
                e = cp.sum(cp.multiply(P['Ere'][i][j], Wre)) \
                    + cp.sum(cp.multiply(P['Eim'][i][j], Wim)) + P['Coff'][i, j]
                row.append(e)
            rows.append(row)
        Xi = cp.bmat(rows)
        if crb_mode == 'matrix_nonuis':
            # design that ignores the unknown reflectivity: constrain the 2x2
            # angle-range information block J_bb (alpha treated as known), i.e.
            # no Schur elimination.  Evaluated against the true nuisance-aware
            # requirement it over-promises accuracy.
            cons += [cp.bmat([[rows[0][0], rows[0][1]],
                              [rows[1][0], rows[1][1]]]) + scrb * np.eye(2) >> 0]
        elif crb_mode == 'scalar':
            # angle-only (scalar) CRB design: constrain only the (theta,theta)
            # information entry, leaving range and the correlation free.  This
            # reproduces the scalar-CRB baseline of the RIS/BD-RIS ISAC
            # literature and, evaluated against the 2-D matrix requirement,
            # exposes the range it does not control.
            cons += [Xi[0, 0] + scrb >= 0]
        else:
            cons += [Xi + scrb * np.eye(4) >> 0]
        obj = obj - P['mu'] * scrb

    prob = cp.Problem(cp.Maximize(obj), cons)
    return dict(prob=prob, P=P, wc=wc, W=W, Cv=Cv, cpow=cpow, crb_on=crb_on)


def _get_program(Nt, K, access, stages, crb_on, crb_mode='matrix'):
    key = (Nt, K, access, crb_on, crb_mode,
           tuple((s[0], s[1], tuple(s[2])) for s in stages))
    if key not in _PCACHE:
        _PCACHE[key] = _build_program(Nt, K, access, stages, crb_on, crb_mode)
    return _PCACHE[key]


def lmi_coefficients(ch, Pa, Wall0, part, Cmax):
    """Numerical coefficients of the parameter-affine restriction
    Xi(tilde R_x) >= 0 of Lemma 3 and Lemma 4, folded so that every LMI entry
    reads  <Re E_ij, Re Wall> + <Im E_ij, Im Wall> + Coff_ij ."""
    D, V, cst = fim_operator(ch, Pa, part)
    Jref = fim_from_pairs(D, V, cst, Wall0 @ Wall0.conj().T)
    dsc = 1.0 / np.sqrt(np.maximum(np.abs(np.diag(Jref)), 1e-30))
    Cinv = np.linalg.inv(Cmax)
    a0 = [V[m].conj() @ Wall0 for m in range(3)]           # (NS,)
    B = [[np.outer(V[n], a0[m]) for n in range(3)] for m in range(3)]
    E = [[None] * 4 for _ in range(4)]
    Coff = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            Z = np.zeros((3, 3), complex)
            for (ui, mi) in D[i]:
                for (uj, mj) in D[j]:
                    Z[mi, mj] += np.vdot(uj, ui) * cst * dsc[i] * dsc[j]
            Eij = np.zeros_like(B[0][0])
            c0 = 0.0
            for m in range(3):
                for n in range(3):
                    z = Z[m, n]
                    if z == 0:
                        continue
                    Eij = Eij + z * B[m][n] + np.conj(z) * B[n][m]
                    c0 -= float(np.real(z * np.vdot(a0[n], a0[m])))
            if i < 2 and j < 2:
                c0 -= Cinv[i, j] * dsc[i] * dsc[j]
            E[i][j] = Eij
            Coff[i, j] = c0
    # J is symmetric, so symmetrise to make the LMI exactly symmetric
    Es = [[0.5 * (E[i][j] + E[j][i]) for j in range(4)] for i in range(4)]
    return Es, 0.5 * (Coff + Coff.T)


def solve_precoders(inst, Pa, Pp, wc0, W0, tref, scheme, stages, has_common,
                    crb_on=True, mu_crb=400.0, crb_mode='matrix'):
    """Returns (wc, W, C, ok, muP) with muP the dual variable of the transmit
    power constraint, i.e. the marginal sum rate per watt.  muP calibrates the
    amplifier activation penalty of Sec. IV-C without any tuning."""
    p, ch = inst.p, inst.ch
    Nt, K = p.Nt, p.K
    kap = sm.fbl_kappa(p.Lb, p.eps) if scheme.fbl else 0.0
    H = inst.eff(Pa, Pp)
    ne = inst.sig2a * inst.qnoise(Pa)
    idx = lambda j: 0 if j == COMMON else j + 1

    G = _get_program(Nt, K, scheme.access, stages, crb_on, crb_mode)
    P = G['P']
    P['Hre'].value = np.ascontiguousarray(H.real)
    P['Him'].value = np.ascontiguousarray(H.imag)
    P['PBS'].value = float(inst.PBS)
    P['Rth'].value = float(p.Rth)

    Wall0 = np.concatenate([wc0[:, None], W0], axis=1)
    S0 = H.conj().T @ Wall0
    pw0 = float(np.linalg.norm(Wall0) ** 2)
    for (k, j, I) in stages:
        key = (k, j)
        den = (sum(abs(S0[k, idx(i)]) ** 2 for i in I) + ne[k] + 1.0
               + inst.eps2[k] * pw0)
        u = S0[k, idx(j)] / den
        ua2 = float(abs(u) ** 2)
        P['ure'][key].value = float(u.real)
        P['uim'][key].value = float(u.imag)
        P['c1'][key].value = ua2 * float(ne[k] + 1.0)
        P['c2'][key].value = ua2 * float(inst.eps2[k])
        P['c3'][key].value = ua2
        tr = float(tref[key])
        d = float(sm.dpsi(tr))
        P['k1'][key].value = kap * (float(sm.psi(tr)) - d * tr)
        P['k2'][key].value = kap * d

    if crb_on:
        P['mu'].value = float(mu_crb)
        Es, Coff = lmi_coefficients(ch, Pa, Wall0, inst.part, p.Cmax)
        for i in range(4):
            for j in range(4):
                P['Ere'][i][j].value = np.ascontiguousarray(Es[i][j].real)
                P['Eim'][i][j].value = np.ascontiguousarray(Es[i][j].imag)
        P['Coff'].value = Coff

    prob = G['prob']
    ok = True
    try:
        prob.solve(solver='CLARABEL', verbose=False)
        if G['W'].value is None:
            raise RuntimeError
    except Exception:
        try:
            prob.solve(solver='SCS', verbose=False, max_iters=6000, eps=1e-6)
        except Exception:
            ok = False
    if (not ok) or G['W'].value is None:
        return wc0, W0, np.zeros(K), False, 0.0
    Cval = np.maximum(G['Cv'].value, 0.0) if G['Cv'] is not None else np.zeros(K)
    wcv = G['wc'].value if G['wc'].value is not None else np.zeros(Nt, complex)
    try:
        muP = float(np.atleast_1d(G['cpow'].dual_value)[0])
    except Exception:
        muP = 0.0
    return wcv, np.asarray(G['W'].value), Cval, True, max(muP, 0.0)


# ==========================================================================
#  P6 : closed-form manifold projections
# ==========================================================================
def _clip(L, cap=1.0e6):
    """Keep the penalty dual decomposition multipliers bounded, which is also
    what the convergence argument assumes."""
    L = np.nan_to_num(L, nan=0.0, posinf=cap, neginf=-cap)
    n = np.linalg.norm(L)
    return L if n <= cap else L * (cap / n)


def _sane(X, cap=1e12):
    """Guard against a non-finite or wildly scaled iterate before an SVD.  A
    dual ascent with an aggressively shrinking penalty parameter can overflow,
    and LAPACK then reports non-convergence instead of returning."""
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=cap, neginf=-cap)
    n = np.max(np.abs(X)) if X.size else 0.0
    if n > cap:
        X = X * (cap / n)
    return X


def proj_sym_unitary(X):
    Xs = 0.5 * (_sane(X) + _sane(X).T)
    try:
        U, _, Vh = np.linalg.svd(Xs)
        T = 0.5 * ((U @ Vh) + (U @ Vh).T)
        U, _, Vh = np.linalg.svd(T)
    except np.linalg.LinAlgError:
        return np.eye(Xs.shape[0], dtype=complex)
    return U @ Vh


def _sigma_max(X, iters=40, tol=1e-7):
    """Top singular value by power iteration on X^H X.  Cheap enough to be used
    as a guard so that the O(n^3) singular value decomposition of
    proj_sym_bounded is skipped whenever the spectral bound is already met."""
    n = X.shape[1]
    v = np.full(n, 1.0 / np.sqrt(n), dtype=complex)
    s = 0.0
    for _ in range(iters):
        w = X @ v
        v = X.conj().T @ w
        nv = np.linalg.norm(v)
        if nv <= 1e-300:
            return 0.0
        v /= nv
        s_new = np.sqrt(nv)
        if abs(s_new - s) <= tol * max(s_new, 1e-12):
            return float(s_new)
        s = s_new
    return float(s)


def proj_sym_bounded(X, amax, guard=True):
    X = _sane(X)
    Xs = 0.5 * (X + X.T)
    if guard and _sigma_max(Xs) <= 0.995 * amax:
        return Xs                       # already inside the spectral norm ball
    try:
        U, s, Vh = np.linalg.svd(Xs)
    except np.linalg.LinAlgError:
        return np.zeros_like(Xs)
    T = (U * np.minimum(s, amax)) @ Vh
    return 0.5 * (T + T.T)


def proj_diag_unitary(X):
    d = np.diag(X).copy()
    d = np.where(np.abs(d) < 1e-12, 1.0 + 0j, d / np.abs(d))
    return np.diag(d)


def proj_diag_bounded(X, amax):
    d = np.diag(X).copy()
    m = np.abs(d)
    d = np.where(m > amax, d / np.maximum(m, 1e-12) * amax, d)
    return np.diag(d)


def project(Pa, Pp, ris, amax):
    if ris == 'diagonal':
        return proj_diag_bounded(Pa, amax), proj_diag_unitary(Pp)
    return proj_sym_bounded(Pa, amax), proj_sym_unitary(Pp)


def scale_active_power(Pa, inst, Rx):
    """Rescale the active block so that \eqref{eq:pact} holds.  The two traces
    are evaluated without forming any M_a x M_a product:
    Tr(A R A^H) = sum(A .* conj(A R)) and Tr(Phi Phi^H) = ||Phi||_F^2."""
    p = inst.p
    if Pa.size == 0:
        return Pa
    A = Pa @ inst.Gact                                  # (M_a, N_t)
    pw = float(np.real(np.sum(A.conj() * (A @ Rx)))) \
        + p.sigma2_act * float(np.sum(Pa.real ** 2 + Pa.imag ** 2))
    return Pa if (pw <= p.Pact or pw <= 0) else Pa * np.sqrt(p.Pact / pw)


# ==========================================================================
#  Amplifier activation surrogate (Sec. IV-C)
# ==========================================================================
def gate_terms(Pa, dl, lam):
    """Reference implementation of the smoothed-count relaxation discussed in
    Sec. IV-C.  The algorithm itself decides the activation pattern with the
    exact switching test ``gate_sweep``; this function is retained because the
    relaxation is what motivates the exact price and is used in the analysis.

    Value and Wirtinger gradient of the majorised activation penalty
        lam * sum_m w_m x_m ,   x_m = (||Phi_a e_m||^2 - 1)_+ ,
        w_m = phi'(x_m) = dl/(x_m+dl)^2 ,
    which majorises lam * sum_m phi(x_m), a smoothed count of powered ports."""
    if Pa.size == 0 or lam <= 0.0:
        return 0.0, np.zeros_like(Pa)
    x = sm.excess_gain(Pa)
    w = sm.gate_weight(x, dl)
    val = lam * float(np.sum(w * x))
    g = lam * (Pa * (w * (x > 0.0))[None, :])
    return val, 0.5 * (g + g.T)


OFF_GAIN = 1.0      # column gain cap of an UNBIASED port (1 = lossless off state)


def gate_cap(on, amax):
    """Per port cap: amax for a biased port, OFF_GAIN (<= 1) for an unbiased
    one; OFF_GAIN = 10^(-IL/20) models an off-state insertion loss IL (dB)."""
    return np.where(on, amax, OFF_GAIN)


def gate_apply(Pa_raw, on, amax):
    """The scattering matrix actually realised by the activation pattern ``on``:
    the unconstrained iterate retracted onto the port caps by Lemma 6."""
    if Pa_raw.size == 0:
        return Pa_raw
    return sm.gate_retract(Pa_raw, gate_cap(on, amax))


def gate_sweep(Pa_raw, on, amax, lam, value, max_flips=None):
    """One sweep of the exact single-port switching test (Proposition 2).

    For every port the penalised objective  F(S) = value(Phi(S)) - lam |S|  is
    evaluated with that port flipped.  Because Phi(S) is the closed-form
    retraction of Lemma 6, a flip costs one O(M_a^2) retraction and one
    objective evaluation, not a re-optimisation; the whole sweep is therefore
    O(M_a^max) evaluations against one full design per candidate partition for a
    grid search.  Flips are applied in decreasing order of improvement and each
    is re-tested against the current pattern, so F strictly increases at every
    accepted flip and the sweep is monotone."""
    M = Pa_raw.shape[0]
    if M == 0 or lam <= 0.0:
        return on, 0.0, 0
    F0 = value(gate_apply(Pa_raw, on, amax)) - lam * float(np.sum(on))
    gains = np.full(M, -np.inf)
    for m in range(M):
        trial = on.copy(); trial[m] = ~trial[m]
        F = value(gate_apply(Pa_raw, trial, amax)) - lam * float(np.sum(trial))
        gains[m] = F - F0
    order = np.argsort(-gains)
    nflip = 0
    cap = M if max_flips is None else max_flips
    for m in order:
        if gains[m] <= 1e-12 or nflip >= cap:
            break
        trial = on.copy(); trial[m] = ~trial[m]
        F = value(gate_apply(Pa_raw, trial, amax)) - lam * float(np.sum(trial))
        if F > F0 + 1e-12:                 # re-test against the current pattern
            on, F0 = trial, F
            nflip += 1
    return on, F0, nflip


# ==========================================================================
#  BD-RIS surrogate with analytic gradients
# ==========================================================================
def stage_sinr(inst, Pa, Pp, wc, W, stages, u=None):
    Wall = np.concatenate([wc[:, None], W], axis=1)
    H = inst.eff(Pa, Pp)
    S = H.conj().T @ Wall
    ne = inst.sig2a * inst.qnoise(Pa)
    idx = lambda j: 0 if j == COMMON else j + 1
    pw = float(np.linalg.norm(Wall) ** 2)
    g, uu = {}, {}
    for (k, j, I) in stages:
        den = (sum(abs(S[k, idx(i)]) ** 2 for i in I) + ne[k] + inst.sig2
               + inst.eps2[k] * pw)
        uu[(k, j)] = S[k, idx(j)] / den
        c = u[(k, j)] if u is not None else uu[(k, j)]
        g[(k, j)] = float(2 * np.real(np.conj(c) * S[k, idx(j)]) - abs(c) ** 2 * den)
    return g, uu, S, ne


def phi_objective(inst, Pa, Pp, wc, W, stages, weights, u, tref, kap, grad=True):
    """V(Phi) = sum_s weights[s] * Rtilde(gamma_s(Phi) | tref_s)."""
    Wall = np.concatenate([wc[:, None], W], axis=1)
    Gaw, Gpw = inst.Gact @ Wall, inst.Gpas @ Wall
    pw = float(np.linalg.norm(Wall) ** 2)
    H = inst.eff(Pa, Pp)
    S = H.conj().T @ Wall
    q = inst.qnoise(Pa)
    idx = lambda j: 0 if j == COMMON else j + 1
    val = 0.0
    Ga = np.zeros_like(Pa) if grad else None
    Gp = np.zeros_like(Pp) if grad else None
    for (k, j, I) in stages:
        w_s = weights.get((k, j), 0.0)
        if w_s == 0.0 and grad is False:
            continue
        c = u[(k, j)]
        den = (sum(abs(S[k, idx(i)]) ** 2 for i in I) + inst.sig2a * q[k]
               + inst.sig2 + inst.eps2[k] * pw)
        g = float(2 * np.real(np.conj(c) * S[k, idx(j)]) - abs(c) ** 2 * den)
        tr = tref[(k, j)]
        gg = max(g, 1e-9)
        val += w_s * (np.log1p(gg) / LOG2 + kap * (sm.psi(tr) + sm.dpsi(tr) * (g - tr)))
        if not grad or w_s == 0.0:
            continue
        dR = 1.0 / ((1 + gg) * LOG2) + kap * sm.dpsi(tr)
        ha, hp = inst.hact[:, k], inst.hpas[:, k]
        a1, a2 = w_s * dR * c, w_s * dR * abs(c) ** 2
        if Pa.size:
            Ga += a1 * np.outer(ha, Gaw[:, idx(j)].conj())
        if Pp.size:
            Gp += a1 * np.outer(hp, Gpw[:, idx(j)].conj())
        for i in I:
            if Pa.size:
                Ga -= a2 * S[k, idx(i)] * np.outer(ha, Gaw[:, idx(i)].conj())
            if Pp.size:
                Gp -= a2 * S[k, idx(i)] * np.outer(hp, Gpw[:, idx(i)].conj())
        if Pa.size:
            Ga -= a2 * inst.sig2a * np.outer(ha, ha.conj() @ Pa)
    if not grad:
        return val
    return val, 0.5 * (Ga + Ga.T), 0.5 * (Gp + Gp.T)


def scheme_rate(inst, Pa, Pp, wc, W, Cv, stages, scheme, order=None):
    """True achievable per-user rates (no surrogate) and the stage weights that
    a subgradient of the scheme utility assigns to each stage."""
    p = inst.p
    K = p.K
    g, _, _, _ = stage_sinr(inst, Pa, Pp, wc, W, stages)
    keys = [s[:2] for s in stages]
    gv = np.array([g[k] for k in keys], float)
    rv = sm.fbl_rate(gv, p.Lb, p.eps) if scheme.fbl else sm.shannon_rate(gv)
    R = {k: float(v) for k, v in zip(keys, rv)}
    wts = {s[:2]: 0.0 for s in stages}
    if scheme.access == 'rsma':
        Rcv = np.array([R[(k, COMMON)] for k in range(K)])
        Rc = float(Rcv.min())
        kmin = int(np.argmin(Rcv))
        Rp = np.array([R[(k, k)] for k in range(K)])
        Cs = np.maximum(np.asarray(Cv, float), 0.0)
        alloc = (Cs / Cs.sum() * Rc) if Cs.sum() > 1e-12 else np.full(K, Rc / K)
        per = alloc + Rp
        tot = float(Rc + Rp.sum())
        for k in range(K):
            wts[(k, k)] = 1.0
        wts[(kmin, COMMON)] = 1.0
    elif scheme.access == 'sdma':
        per = np.array([R[(k, k)] for k in range(K)])
        tot = float(per.sum())
        for k in range(K):
            wts[(k, k)] = 1.0
    else:
        rank = {u: r for r, u in enumerate(order)}
        per = np.zeros(K)
        for j in range(K):
            cand = [(R[(k, j)], k) for k in range(K) if rank[j] >= rank[k]]
            per[j], kmin = min(cand)
            wts[(kmin, j)] = 1.0
        tot = float(per.sum())
    return tot, per, R, wts


# ==========================================================================
#  Algorithm 1 : single-loop gated penalty manifold method
# ==========================================================================
def _run(ch, p, scheme, n_iter=20, rho0=1.0, crb_on=True, track=False,
         verbose=False, single_loop=False, n_inner=8, gate=None, part=None,
         crb_mode='matrix', init='max_gain', init_seed=0):
    """One pass of Algorithm 1.  ``single_loop=False`` reproduces the nested
    two-loop scheme and is used only for the runtime comparison."""
    t_start = time.time()
    global OFF_GAIN
    OFF_GAIN = 10 ** (-float(getattr(p, 'off_loss_dB', 0.0)) / 20.0)
    K, Nt = p.K, p.Nt
    amax = p.alpha_max
    gate = scheme.gate if gate is None else bool(gate)

    if scheme.ris == 'passive':
        Ma, Mp = 0, p.M
    elif scheme.ris == 'active':
        Ma, Mp = p.M, 0
    else:
        Ma = p.Mact if part is None else int(part)
        Mp = p.M - Ma
    inst = Instance(ch, p, part=Ma)
    if not scheme.fbl:            # ideal bound: L_b -> infinity, perfect CSI
        inst.eps2 = np.zeros(K)
    Xmax = p.alpha_max ** 2 - 1.0

    Pa = np.zeros((0, 0), complex)
    Pp = np.zeros((0, 0), complex)
    _rng = np.random.default_rng(20240 + Ma + Mp + 1000 * int(init_seed))
    if Mp:
        if init == 'random':
            Rr = (_rng.normal(size=(Mp, Mp)) + 1j * _rng.normal(size=(Mp, Mp)))
            Pp = proj_sym_unitary(Rr)
        else:
            Pp = _align_init(inst.hpas, inst.Gpas, Mp)
    if Ma:
        # Every surface starts at full gain.  The gated phase then PRUNES: the
        # activation penalty removes the amplifiers that do not pay for the
        # transmit power they consume.  Starting from the passive configuration
        # instead would ask the concave surrogate to switch ports on, which it
        # is deliberately bad at, since its weight is largest at the origin.
        base = _align_init(inst.hact, inst.Gact, Ma)
        if init == 'passive':
            # all-lossless start: unit-gain surface, every amplifier unbiased
            Pa = base
        elif init == 'chan_align':
            # channel-aligned start at a moderate gain rather than full gain
            Pa = (0.6 * amax) * base
        elif init == 'random':
            Rr = (_rng.normal(size=(Ma, Ma)) + 1j * _rng.normal(size=(Ma, Ma)))
            Pa = proj_sym_bounded(amax * 0.5 * (Rr + Rr.T), amax)
        else:                                    # 'max_gain' (default)
            Pa = amax * base
    # Pa_raw carries the unconstrained iterate; the realised scattering matrix
    # is Pa = T_c(Pa_raw) with the caps of the current activation pattern.
    Pa_raw = Pa.copy()
    on = (np.zeros(Ma, dtype=bool) if init == 'passive'
          else np.ones(Ma, dtype=bool))
    inst.set_na(int(np.sum(on)) if Ma else 0)

    H = inst.eff(Pa, Pp)
    order = list(np.argsort(-np.linalg.norm(H, axis=0)))
    stages, has_common = build_stages(scheme.access, K, order)
    W = H / np.linalg.norm(H, axis=0, keepdims=True)
    wc = np.sum(W, axis=1); wc /= np.linalg.norm(wc)
    pw = inst.PBS / (K + (1 if has_common else 0))
    W = W * np.sqrt(pw); wc = wc * np.sqrt(pw) if has_common else np.zeros(Nt, complex)
    Cv = np.zeros(K)
    kap = sm.fbl_kappa(p.Lb, p.eps) if scheme.fbl else 0.0

    tfloor = sm.sinr_floor(p.Lb, p.eps) if scheme.fbl else 1e-3
    g, uu, _, _ = stage_sinr(inst, Pa, Pp, wc, W, stages)
    tref = {kk: max(v, tfloor) for kk, v in g.items()}

    Th_a, Th_p = Pa.copy(), Pp.copy()
    La, Lp = np.zeros_like(Pa), np.zeros_like(Pp)
    rho, eta, eta0 = rho0, None, None
    mu_crb, mu_pen, lam = 4.0e2, 4.0e3, 0.0
    hist = {'sum': [], 'cons': [], 'crb': [], 'viol': [], 'gmap': [], 'na': [],
            'gap': [], 'time': [], 'lam': [], 'gtrace': [], 'ftrace': [],
            'nflip': []}
    best, cv = None, 0.0
    T_in = 1 if single_loop else n_inner

    for it in range(n_iter):
        # ------------- (a) precoders and rate split  (P4) -------------------
        wc, W, Cv, ok, muP = solve_precoders(inst, Pa, Pp, wc, W, tref, scheme,
                                             stages, has_common, crb_on=crb_on,
                                             mu_crb=mu_crb, crb_mode=crb_mode)
        Rx = np.outer(wc, wc.conj()) + W @ W.conj().T
        # the activation weight is not a tuning parameter: powering one
        # amplifier costs P_DC watts of transmit power, worth muP P_DC of sum
        # rate, so lam = muP P_DC makes the surrogate an exact penalty.  The
        # marginal value of power cannot exceed twice its average value on a
        # concave rate, which bounds lam; a one-pole filter removes dual jitter.
        if gate and Ma:
            rref = hist['sum'][-1] if hist['sum'] else 1.0
            cap = 2.0 * max(rref, 1.0) / inst.PBS * p.PDC
            lam = 0.8 * lam + 0.2 * min(float(muP) * p.PDC, cap)

        # ------------- (b)-(c) surface, gate and dual  (P5, P6) -------------
        gnorm = 0.0
        if Ma or Mp:
            for t in range(T_in):
                gsin, uu, _, _ = stage_sinr(inst, Pa, Pp, wc, W, stages)
                tref = {kk: max(v, tfloor) for kk, v in gsin.items()}
                _, _, _, wts = scheme_rate(inst, Pa, Pp, wc, W, Cv, stages,
                                           scheme, order)
                V0, Ga, Gp = phi_objective(inst, Pa, Pp, wc, W, stages, wts, uu,
                                           tref, kap)
                if Ma:
                    Ga = Ga - 0.5 * La - (Pa_raw - Th_a) / (2 * rho)
                if Mp:
                    Gp = Gp - 0.5 * Lp - (Pp - Th_p) / (2 * rho)
                nrm = np.sqrt(np.linalg.norm(Ga) ** 2 + np.linalg.norm(Gp) ** 2) + 1e-16
                if eta is None:
                    eta = eta0 = 1.0 / nrm     # first step fixes the scale
                # Gradient mapping at a FIXED reference step: the stationarity
                # measure of Theorem 2,  G = (Pi(Phi + eta0 grad) - Phi)/eta0.
                # Pi is the Euclidean projection onto the CONVEX set the surface
                # block actually moves in: the symmetric spectral-norm ball for
                # the active block and the whole symmetric subspace for the
                # passive block, whose nonconvex unitary manifold is carried by
                # the consensus copy Theta_p instead.
                if track:
                    d2 = 0.0
                    if Ma:
                        Pa_g = (proj_sym_bounded(Pa + eta0 * Ga, amax, guard=False)
                                if scheme.ris != 'diagonal'
                                else proj_diag_bounded(Pa + eta0 * Ga, amax))
                        d2 += np.linalg.norm(Pa_g - Pa) ** 2
                    if Mp:
                        d2 += (eta0 * np.linalg.norm(Gp)) ** 2
                    gnorm = np.sqrt(d2) / eta0
                    hist['gtrace'].append(float(gnorm))
                    hist['ftrace'].append(float(V0))
                accepted, Pa_o, Pp_o = False, Pa, Pp
                pen0 = _crb_pen(inst, Pa, Rx, mu_pen) if (crb_on and Ma and crb_mode=='matrix') else 0.0
                for _ in range(16):
                    Pr_n = Pa_raw + eta * Ga if Ma else Pa_raw
                    Pp_n = Pp + eta * Gp if Mp else Pp
                    if Ma:
                        Pr_n = (proj_sym_bounded(Pr_n, amax)
                                if scheme.ris != 'diagonal'
                                else proj_diag_bounded(Pr_n, amax))
                        Pa_n = gate_apply(Pr_n, on, amax) if gate else Pr_n
                        Pa_n = scale_active_power(Pa_n, inst, Rx)
                    else:
                        Pa_n = Pa
                    Vn = phi_objective(inst, Pa_n, Pp_n, wc, W, stages, wts, uu,
                                       tref, kap, grad=False)
                    pen = _crb_pen(inst, Pa_n, Rx, mu_pen) if (crb_on and Ma and crb_mode=='matrix') else 0.0
                    if (Vn - pen) >= (V0 - pen0) - 1e-12:
                        Pa_raw, Pa, Pp = Pr_n, Pa_n, Pp_n
                        accepted = True
                        break
                    eta *= 0.5
                if accepted:
                    eta = min(eta * 1.25, 1e4 / nrm)
                else:
                    break
                if Ma:
                    Th_a = (proj_sym_bounded(Pa_raw + rho * La, amax)
                            if scheme.ris != 'diagonal'
                            else proj_diag_bounded(Pa_raw + rho * La, amax))
                    La = _clip(La + (Pa_raw - Th_a) / rho)
                if Mp:
                    Th_p = (proj_sym_unitary(Pp + rho * Lp)
                            if scheme.ris != 'diagonal'
                            else proj_diag_unitary(Pp + rho * Lp))
                    Lp = _clip(Lp + (Pp - Th_p) / rho)
            cv = (np.linalg.norm(Pa_raw - Th_a) if Ma else 0.0) + \
                 (np.linalg.norm(Pp - Th_p) if Mp else 0.0)
            if cv > 1e-3:
                rho = max(rho * 0.7, 1e-6)
            cands = []
            for (Ar, B) in [(Th_a, Th_p), project(Pa_raw, Pp, scheme.ris, amax)]:
                A = gate_apply(Ar, on, amax) if (Ma and gate) else Ar
                A = scale_active_power(A, inst, Rx) if Ma else Pa
                B = B if Mp else Pp
                s_, _, _, _ = scheme_rate(inst, A, B, wc, W, Cv, stages, scheme, order)
                s_ -= _crb_pen(inst, A, Rx, mu_pen) if (crb_on and Ma and crb_mode=='matrix') else 0.0
                cands.append((s_, Ar, A, B))
            _, Pa_raw, Pa, Pp = max(cands, key=lambda z: z[0])

        # ------------- (d) exact switching sweep, anchors, incumbent --------
        nflip = 0
        if gate and Ma and lam > 0.0:
            def _value(A):
                # the switching test is evaluated on the TRUE objective, not on
                # the minorant: a flip needs no re-optimisation, so there is no
                # reason to pay the looseness of a surrogate anchored elsewhere
                A = scale_active_power(A, inst, Rx)
                v = scheme_rate(inst, A, Pp, wc, W, Cv, stages, scheme, order)[0]
                if crb_on and crb_mode=='matrix':
                    v -= _crb_pen(inst, A, Rx, mu_pen)
                return v
            on, _, nflip = gate_sweep(Pa_raw, on, amax, lam, _value,
                                      max_flips=max(1, Ma // 12))
            Pa = scale_active_power(gate_apply(Pa_raw, on, amax), inst, Rx)
            inst.set_na(int(np.sum(on)))
        gsin, uu, _, _ = stage_sinr(inst, Pa, Pp, wc, W, stages)
        tref = {kk: max(v, tfloor) for kk, v in gsin.items()}
        tot, per, Rtrue, wts = scheme_rate(inst, Pa, Pp, wc, W, Cv, stages,
                                           scheme, order)
        Rx = np.outer(wc, wc.conj()) + W @ W.conj().T
        J = sm.efim(ch, Pa, Rx, part=inst.part)
        sc = np.linalg.norm(np.linalg.inv(p.Cmax))
        # incumbent selection uses the DESIGN's own feasibility metric so that a
        # scalar or no-nuisance baseline is not judged by a requirement it never
        # attempts; the true nuisance-aware 2-D metrics (feas, crb) are always
        # computed from J below for reporting and plotting.
        if crb_mode == 'scalar':
            dvio = max(0.0, float(np.linalg.inv(p.Cmax)[0, 0] - J[0, 0])) / sc
        elif crb_mode == 'matrix_nonuis':
            Jbb = sm.fim_full(ch, Pa, Rx, part=inst.part)[:2, :2]
            dvio = sm.crb_violation(Jbb, p.Cmax) / sc
        else:
            dvio = sm.crb_violation(J, p.Cmax) / sc
        vio = dvio
        # surrogate tightness gap of Proposition 3
        Vs = phi_objective(inst, Pa, Pp, wc, W, stages, wts, uu, tref, kap, grad=False)
        Vt = sum(wts[s[:2]] * Rtrue[s[:2]] for s in stages)
        hist['sum'].append(tot)
        hist['cons'].append(float(cv))
        hist['viol'].append(float(vio))
        hist['gmap'].append(float(gnorm))
        hist['na'].append(int(inst.na))
        hist['nflip'].append(int(nflip))
        hist['lam'].append(float(lam))
        hist['gap'].append(float(abs(Vt - Vs)))
        hist['time'].append(time.time() - t_start)
        hist['crb'].append(np.sqrt(np.diag(sm.crb_matrix(J))).tolist())
        cand = (tot, per, Pa.copy(), Pp.copy(), wc.copy(), W.copy(), Cv.copy(),
                J, vio, int(inst.na))
        if best is None or _better(cand, best, crb_on):
            best = cand
        if verbose:
            print(f'  it{it}: sum={tot:.3f} na={inst.na} flips={nflip} '
                  f'cons={cv:.2e} viol={vio:.2e} |G|={gnorm:.3e} lam={lam:.4f}',
                  flush=True)
        if crb_on and vio > 1e-7:
            stall = (len(hist['viol']) >= 3
                     and hist['viol'][-1] > 0.97 * hist['viol'][-3])
            if not stall:
                mu_crb = min(mu_crb * 6.0, 2e6)
                mu_pen = min(mu_pen * 6.0, 2e7)
        elif it >= 4 and abs(hist['sum'][-1] - hist['sum'][-2]) < 1e-4 * max(1, tot) \
                and nflip == 0 and hist['na'][-1] == hist['na'][-2]:
            break

    tot, per, Pa, Pp, wc, W, Cv, J, vio, na = best
    out = dict(sum_rate=tot, rates=per, Pa=Pa, Pp=Pp, wc=wc, W=W, C=Cv,
               J=J, crb=np.sqrt(np.diag(sm.crb_matrix(J))),
               feas=sm.crb_feasible(J, p.Cmax), viol=vio, scheme=scheme.name,
               na=int(na), namax=int(Ma), iters=len(hist['sum']),
               runtime=time.time() - t_start)
    if track:
        out['hist'] = hist
    return out


def stationarity_probe(ch, p, scheme, n_warm=6, T=400, crb_on=True,
                       part=None, seed=0):
    """Measure the gradient-mapping residual under exactly the hypotheses of
    Theorem 2.

    The theorem is a statement about the surface block for a FIXED augmented
    Lagrangian: fixed precoders, fixed consensus copy and multipliers, fixed
    penalty and activation weights, fixed anchors.  Inside the full algorithm
    all of those move from one outer iteration to the next, so a residual
    measured there is not a residual of one function.  This probe therefore
    warms up with ``n_warm`` ordinary outer iterations, freezes everything
    except the surface, and then runs ``T`` projected majorisation minimisation
    steps, recording ||G_eta0|| at each one.

    Returns dict(g=residual trace, f=objective trace, eta0=reference step)."""
    r = _run(ch, p, scheme, n_iter=n_warm, crb_on=crb_on, part=part,
             track=False, gate=False)
    Ma = r['Pa'].shape[0] if r['Pa'].size else 0
    Mp = r['Pp'].shape[0] if r['Pp'].size else 0
    inst = Instance(ch, p, part=Ma, na=int(r['na']))
    Pa, Pp, wc, W, Cv = r['Pa'].copy(), r['Pp'].copy(), r['wc'], r['W'], r['C']
    amax = p.alpha_max
    stages, _ = build_stages(scheme.access, p.K)
    kap = sm.fbl_kappa(p.Lb, p.eps) if scheme.fbl else 0.0
    tfl = sm.sinr_floor(p.Lb, p.eps) if scheme.fbl else 1e-3
    # freeze the surrogate: anchors, receivers and stage weights are evaluated
    # once and never refreshed again, so f below is one fixed smooth function
    g0, u0, _, _ = stage_sinr(inst, Pa, Pp, wc, W, stages)
    tref = {k: max(v, tfl) for k, v in g0.items()}
    _, _, _, wts = scheme_rate(inst, Pa, Pp, wc, W, Cv, stages, scheme, None)
    Th_a, Th_p = Pa.copy(), Pp.copy()
    La = np.zeros_like(Pa)
    Lp = np.zeros_like(Pp)
    rho = 1.0

    def val_grad(A, B, grad=True):
        return phi_objective(inst, A, B, wc, W, stages, wts, u0, tref, kap,
                             grad=grad)

    V0, Ga, Gp = val_grad(Pa, Pp)
    nrm = np.sqrt(np.linalg.norm(Ga) ** 2 + np.linalg.norm(Gp) ** 2) + 1e-16
    eta = eta0 = 1.0 / nrm
    gtr, ftr = [], []
    for t in range(T):
        V0, Ga, Gp = val_grad(Pa, Pp)
        # residual at the fixed reference step, projection onto the convex set
        d2 = 0.0
        if Ma:
            d2 += np.linalg.norm(proj_sym_bounded(Pa + eta0 * Ga, amax,
                                                  guard=False) - Pa) ** 2
        if Mp:
            d2 += (eta0 * np.linalg.norm(Gp)) ** 2
        gtr.append(float(np.sqrt(d2) / eta0))
        ftr.append(float(V0))
        ok = False
        for _ in range(24):
            An = proj_sym_bounded(Pa + eta * Ga, amax) if Ma else Pa
            Bn = Pp + eta * Gp if Mp else Pp
            if val_grad(An, Bn, grad=False) >= V0 - 1e-14:
                Pa, Pp, ok = An, Bn, True
                break
            eta *= 0.5
        if not ok:
            gtr.extend([gtr[-1]] * (T - len(gtr)))
            ftr.extend([ftr[-1]] * (T - len(ftr)))
            break
        eta = min(eta * 1.05, 1e4 / nrm)
    return dict(g=gtr, f=ftr, eta0=float(eta0))


def gpm(ch, p, scheme, n_iter=20, crb_on=True, track=False, verbose=False,
        single_loop=False, n_inner=8, gate=None, part=None, refine=True,
        rho0=1.0, crb_mode='matrix', init='max_gain', init_seed=0):
    """Algorithm 1 in full.

    Phase I (identification).  If the scheme gates its amplifiers, one pass is
    run over the whole amplifier-equipped sub-aperture with the smoothed-count
    penalty active.  By Proposition 2 the pass returns an integral activation
    pattern, whose cardinality n_a* is the aperture partition.

    Phase II (refinement).  The two-block surface is instantiated at that
    partition, so that the n_a* powered ports form the active block and the
    remaining M - n_a* ports form the lossless block, and one ungated pass is
    run.  Two passes replace the exhaustive sweep over the partition, which
    needs one pass per candidate."""
    gate = scheme.gate if gate is None else bool(gate)
    kw = dict(n_iter=n_iter, crb_on=crb_on, verbose=verbose,
              single_loop=single_loop, n_inner=n_inner, rho0=rho0,
              crb_mode=crb_mode, init=init, init_seed=init_seed)
    if not (gate and refine) or scheme.ris in ('passive', 'active'):
        return _run(ch, p, scheme, track=track, gate=gate, part=part, **kw)
    r1 = _run(ch, p, scheme, track=track, gate=True, part=part, **kw)
    na = max(int(r1['na']), 1)
    r2 = _run(ch, p, scheme, track=track, gate=False, part=na, **kw)
    r2['na'] = na
    r2['namax'] = int(r1['namax'])
    r2['iters'] = int(r1['iters']) + int(r2['iters'])
    r2['runtime'] = float(r1['runtime']) + float(r2['runtime'])
    r2['na_path'] = r1['hist']['na'] if track else None
    if track:
        for k in r2['hist']:
            if k == 'time':
                off = r1['hist']['time'][-1] if r1['hist']['time'] else 0.0
                r2['hist'][k] = r1['hist'][k] + [v + off for v in r2['hist'][k]]
            else:
                r2['hist'][k] = r1['hist'][k] + r2['hist'][k]
        r2['hist']['n_phase1'] = len(r1['hist']['sum'])
    if _better((r1['sum_rate'], None, None, None, None, None, None, None,
                r1['viol'], 0),
               (r2['sum_rate'], None, None, None, None, None, None, None,
                r2['viol'], 0), crb_on):
        keep = dict(r2)
        for k in ('sum_rate', 'rates', 'Pa', 'Pp', 'wc', 'W', 'C', 'J', 'crb',
                  'feas', 'viol'):
            keep[k] = r1[k]
        keep['namax'] = int(r1['namax'])
        return keep
    return r2


sl_gpm = gpm                    # backwards compatible aliases
hpm_wmmse_fbl = gpm


def _align_init(hR, GR, M):
    """Symmetric unitary start that co-phases the dominant cascaded direction."""
    u = hR @ np.ones(hR.shape[1]) if hR.size else np.ones(M, complex)
    v = GR @ np.ones(GR.shape[1]) if GR.size else np.ones(M, complex)
    u = u / (np.linalg.norm(u) + 1e-16)
    v = v / (np.linalg.norm(v) + 1e-16)
    X = np.eye(M, dtype=complex) + 2.0 * np.outer(u.conj(), v.conj())
    return proj_sym_unitary(X)


def _better(cand, best, crb_on, tol=1e-7):
    """Feasibility-first incumbent update for the matrix CRB constraint."""
    if not crb_on:
        return cand[0] > best[0]
    fc, fb = cand[8] <= tol, best[8] <= tol
    if fc and fb:
        return cand[0] > best[0]
    if fc != fb:
        return fc
    return cand[8] < best[8]


def _crb_pen(inst, Pa, Rx, mu=4000.0):
    p = inst.p
    J = sm.efim(inst.ch, Pa, Rx, part=inst.part)
    v = sm.crb_violation(J, p.Cmax)
    sc = np.linalg.norm(np.linalg.inv(p.Cmax))
    return mu * (v / sc) ** 2
