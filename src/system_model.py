"""
System model for the symbiotic RSMA-enabled hybrid active-passive BD-RIS ISAC
system with finite blocklength (FBL) URLLC and matrix CRB constraints.

Implements:
  * geometry, ULA/near-field steering vectors, path loss
  * hybrid active-passive BD-RIS scattering model (block-diagonal, reciprocal)
  * effective end-to-end channels and RSMA SINRs with active-RIS noise
  * Polyanskiy finite blocklength normal approximation rates
  * exact 4x4 Fisher information matrix and matrix Cramer-Rao bound
  * per-port reflection gains, the smoothed-count amplifier activation
    surrogate and the symmetry preserving gate retraction (Sec. IV-C)

Authors: O. G. and P. S. Babu
"""
import numpy as np
from scipy.stats import norm

C_LIGHT = 3e8


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class SysParams:
    def __init__(self, **kw):
        self.fc = 5.8e9                # carrier frequency [Hz]
        self.B = 20e6                  # bandwidth [Hz]
        self.Nt = 16                   # BS transmit antennas
        self.Nr = 8                    # BS radar receive antennas
        self.K = 4                     # number of URLLC users
        self.Mact = 48                 # amplifier-equipped ports  M_a^max
        self.Mpas = 16                 # lossless passive ports     M_p
        self.gate_delta = 0.05         # smoothing constant of phi(x)=x/(x+delta)
        self.gate_tol = 1e-3           # excess gain above which a port counts
        self.Lb = 200                  # blocklength (channel uses)
        self.eps = 1e-5                # target block error probability
        self.Lradar = 256              # coherent radar pulses
        self.Ptot_dBm = 33.0           # TOTAL system power budget [dBm]
        self.Pact_dBm = 10.0           # active sub-surface RF output budget [dBm]
        self.PDC_dBm = 12.0            # DC bias drawn by ONE active element [dBm]
        self.alpha_max_dB = 12.0       # active element max amplification [dB]
        self.NF_act_dB = 5.0           # noise figure at active sub-surface
        self.NF_ue_dB = 7.0            # noise figure at UE / radar receiver
        self.N0_dBmHz = -174.0         # thermal noise PSD
        self.Rth = 1.5                 # per-user minimum FBL rate [bit/s/Hz]
        self.pilot_frac = 0.5          # fraction of the block spent on training
        self.tau2 = None               # override the CSI error variance if set
        self.crb_theta = 1.745e-3      # angle RMSE threshold [rad]  (0.1 deg)
        self.crb_range = 0.02          # range RMSE threshold [m]
        self.rcs_dbsm = 20.0           # target radar cross section [dBsm]
        # geometry (2-D, metres)
        self.pBS = np.array([0.0, 0.0])
        self.pRIS = np.array([48.0, 12.0])
        self.ris_axis = np.array([1.0, 0.0])   # RIS array axis
        self.theta0 = np.deg2rad(20.0)         # target AoA from BS
        self.r0 = 35.0                         # target range from BS
        self.user_radius = (55.0, 70.0)
        # channel model
        self.kappa_BR = 10.0           # Rician K-factor BS->RIS [linear]
        self.kappa_RU = 3.0            # Rician K-factor RIS->UE
        self.pl_exp_BR = 2.2
        self.pl_exp_RU = 2.4
        self.pl_exp_BU = 3.6           # blocked direct link
        self.direct_block_dB = 15.0    # extra blockage loss on the direct link
        self.C0_dB = -30.0             # reference path loss at 1 m
        for k, v in kw.items():
            setattr(self, k, v)

    # ---- derived quantities ------------------------------------------------
    @property
    def lam(self):
        return C_LIGHT / self.fc

    @property
    def M(self):
        return self.Mact + self.Mpas

    @property
    def Ptot(self):
        return 10 ** ((self.Ptot_dBm - 30) / 10.0)

    @property
    def PDC(self):
        return 10 ** ((self.PDC_dBm - 30) / 10.0)

    def PBS_of(self, na):
        """BS radiated power left once the amplifier bias of the ``na`` POWERED
        ports has been paid for:  P_tot = P_BS + P_act^RF + n_a P_DC.

        Only ports whose excess gain is strictly positive draw bias power, so
        n_a is an optimisation variable and not a hardware constant."""
        return max(self.Ptot - self.Pact - float(na) * self.PDC, 1e-4)

    @property
    def PBS(self):
        return self.PBS_of(self.Mact)

    @property
    def Pact(self):
        return 10 ** ((self.Pact_dBm - 30) / 10.0)

    @property
    def alpha_max(self):
        return 10 ** (self.alpha_max_dB / 20.0)

    @property
    def gate_dl(self):
        """Absolute smoothing constant delta of the activation surrogate."""
        return self.gate_delta * (self.alpha_max ** 2 - 1.0)

    @property
    def sigma2_ue(self):
        return 10 ** ((self.N0_dBmHz + 10 * np.log10(self.B) + self.NF_ue_dB - 30) / 10.0)

    @property
    def sigma2_act(self):
        return 10 ** ((self.N0_dBmHz + 10 * np.log10(self.B) + self.NF_act_dB - 30) / 10.0)

    @property
    def sigma2_bs(self):
        return self.sigma2_ue

    @property
    def Cmax(self):
        return np.diag([self.crb_theta ** 2, self.crb_range ** 2])

    def target_pos(self):
        return self.r0 * np.array([np.cos(self.theta0), np.sin(self.theta0)])

    def ris_element_pos(self, which='all'):
        """Positions of the BD-RIS elements.  Amplifier-equipped ports occupy
        the first Mact slots of a single uniform linear aperture of M elements."""
        d = self.lam / 2.0
        idx = np.arange(self.M)
        base = self.pRIS[None, :] + (idx[:, None] - (self.M - 1) / 2.0) * d \
            * self.ris_axis[None, :]
        if which == 'act':
            return base[:self.Mact]
        if which == 'pas':
            return base[self.Mact:]
        return base


# --------------------------------------------------------------------------
# Steering vectors and path loss
# --------------------------------------------------------------------------
def ula_steer(N, theta):
    """Half-wavelength ULA far-field steering vector."""
    return np.exp(1j * np.pi * np.arange(N) * np.sin(theta))


def ula_steer_deriv(N, theta):
    n = np.arange(N)
    return 1j * np.pi * n * np.cos(theta) * ula_steer(N, theta)


def pathloss(d, exp_, C0_dB=-30.0):
    return 10 ** (C0_dB / 10.0) * np.maximum(d, 1.0) ** (-exp_)


def near_field_steer(elem_pos, p_target, lam):
    """Spherical-wave (near-field) steering vector from array elements to a
    point target, together with its Jacobian w.r.t. the target position."""
    diff = p_target[None, :] - elem_pos              # (M,2)
    dist = np.linalg.norm(diff, axis=1)              # (M,)
    b = np.exp(-1j * 2 * np.pi * dist / lam)
    ddist_dp = diff / dist[:, None]
    db_dp = (-1j * 2 * np.pi / lam) * b[:, None] * ddist_dp
    return b, db_dp


# --------------------------------------------------------------------------
# Channel generation
# --------------------------------------------------------------------------
def rician(nr, nc, kappa, los, rng):
    nlos = (rng.normal(size=(nr, nc)) + 1j * rng.normal(size=(nr, nc))) / np.sqrt(2)
    return np.sqrt(kappa / (1 + kappa)) * los + np.sqrt(1.0 / (1 + kappa)) * nlos


class Channels:
    """Container for one channel realisation."""

    def __init__(self, p: SysParams, rng):
        self.p = p
        Nt, K = p.Nt, p.K
        Ma, Mp = p.Mact, p.Mpas
        lam = p.lam

        # ---- user positions -------------------------------------------------
        rad = rng.uniform(p.user_radius[0], p.user_radius[1], size=K)
        ang = rng.uniform(np.deg2rad(-2), np.deg2rad(14), size=K)
        self.pUE = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)

        # ---- BS -> RIS ------------------------------------------------------
        dBR = np.linalg.norm(p.pRIS - p.pBS)
        aoa_BR = np.arctan2(p.pRIS[1] - p.pBS[1], p.pRIS[0] - p.pBS[0])
        pos_all = p.ris_element_pos('all')
        aBS = ula_steer(Nt, aoa_BR)
        glBR = np.sqrt(pathloss(dBR, p.pl_exp_BR, p.C0_dB))
        losF = np.exp(1j * np.pi * np.arange(p.M))[:, None] * np.conj(aBS)[None, :]
        self.Gfull = glBR * rician(p.M, Nt, p.kappa_BR, losF, rng)
        self.Gact, self.Gpas = self.Gfull[:Ma], self.Gfull[Ma:]

        # ---- RIS -> UE ------------------------------------------------------
        self.hfull = np.zeros((p.M, K), complex)
        self.hd = np.zeros((Nt, K), complex)
        for k in range(K):
            dRU = np.linalg.norm(self.pUE[k] - p.pRIS)
            gl = np.sqrt(pathloss(dRU, p.pl_exp_RU, p.C0_dB))
            angk = np.arctan2(self.pUE[k, 1] - p.pRIS[1], self.pUE[k, 0] - p.pRIS[0])
            self.hfull[:, k] = gl * rician(p.M, 1, p.kappa_RU,
                                           ula_steer(p.M, angk)[:, None], rng).ravel()
            dBU = np.linalg.norm(self.pUE[k] - p.pBS)
            gld = np.sqrt(pathloss(dBU, p.pl_exp_BU, p.C0_dB - p.direct_block_dB))
            self.hd[:, k] = gld * (rng.normal(size=Nt) + 1j * rng.normal(size=Nt)) / np.sqrt(2)
        self.hact, self.hpas = self.hfull[:Ma], self.hfull[Ma:]

        # ---- radar geometry -------------------------------------------------
        pT = p.target_pos()
        # the direct BS-target route carries its absolute propagation phase
        # exp(-j 2 pi r0 / lambda); only the phase *difference* with respect to
        # the BS-RIS-target route is identifiable once alpha is a nuisance.
        ph = np.exp(-1j * 2 * np.pi * p.r0 / lam)
        self.aNt = ph * ula_steer(Nt, p.theta0)
        self.aNt_dth = ph * ula_steer_deriv(Nt, p.theta0)
        self.aNt_dr = (-1j * 2 * np.pi / lam) * self.aNt
        self.aNr = ula_steer(p.Nr, p.theta0)
        self.aNr_dth = ula_steer_deriv(p.Nr, p.theta0)
        b, db_dp = near_field_steer(pos_all, pT, lam)
        dpT_dth = p.r0 * np.array([-np.sin(p.theta0), np.cos(p.theta0)])
        dpT_dr = np.array([np.cos(p.theta0), np.sin(p.theta0)])
        self.b_full = b
        self.b_full_dth = db_dp @ dpT_dth
        self.b_full_dr = db_dp @ dpT_dr
        self.b_act = b[:Ma]
        self.b_act_dth = self.b_full_dth[:Ma]
        self.b_act_dr = self.b_full_dr[:Ma]

        # explicit per-path free-space amplitudes of the two sensing routes
        dBT = np.linalg.norm(pT - p.pBS)
        dRT = np.linalg.norm(pT - p.pRIS)
        self.beta_BT = lam / (4 * np.pi * dBT)      # BS  <-> target one way
        self.beta_RT = lam / (4 * np.pi * dRT)      # RIS  -> target one way
        self.dBT, self.dRT = dBT, dRT
        self.alpha0_sq = 10 ** (p.rcs_dbsm / 10.0)  # target radar cross section


# --------------------------------------------------------------------------
# Effective channels, SINRs, FBL rates
# --------------------------------------------------------------------------
def effective_channels(ch: Channels, Pact, Ppas):
    """h_k(Phi) = h_d,k + Gact^H Phi_act^H h_act,k + Gpas^H Phi_pas^H h_pas,k."""
    return ch.hd + ch.Gact.conj().T @ (Pact.conj().T @ ch.hact) \
                 + ch.Gpas.conj().T @ (Ppas.conj().T @ ch.hpas)


def active_noise_gain(ch: Channels, Pact):
    """||h_act,k^H Phi_act||^2 for every user k -> amplified thermal noise."""
    T = Pact.conj().T @ ch.hact
    return np.sum(np.abs(T) ** 2, axis=0)


def sinrs(H, wc, W, noise_extra, sigma2):
    """Common and private SINRs of the 1-layer RSMA hierarchy."""
    sc = H.conj().T @ wc
    S = H.conj().T @ W
    pj = np.abs(S) ** 2
    tot = pj.sum(axis=1) + noise_extra + sigma2
    g_c = np.abs(sc) ** 2 / tot
    g_p = np.diag(pj) / (tot - np.diag(pj))
    return g_c, g_p


_QINV = {}


def qinv(eps):
    """Q^{-1}(eps), cached: scipy's ppf is expensive and eps never varies
    inside a run."""
    e = float(eps)
    if e not in _QINV:
        _QINV[e] = float(norm.ppf(1 - e))
    return _QINV[e]


def fbl_rate(gamma, Lb, eps):
    """Polyanskiy normal approximation, bit/s/Hz."""
    gamma = np.maximum(np.asarray(gamma, float), 0.0)
    V = 1.0 - (1.0 + gamma) ** (-2)
    return np.maximum(np.log2(1 + gamma)
                      - np.sqrt(np.maximum(V, 0.0) / Lb) * qinv(eps) * np.log2(np.e), 0.0)


def shannon_rate(gamma):
    return np.log2(1.0 + np.maximum(np.asarray(gamma, float), 0.0))


def fbl_kappa(Lb, eps):
    return qinv(eps) * np.log2(np.e) / np.sqrt(Lb)


_FLOOR_CACHE = {}


def sinr_floor(Lb, eps):
    """Smallest SINR at which the finite blocklength rate is still positive.

    Below this point R^FBL is clipped to zero and the derivative of the
    dispersion term diverges, so the minorant must never be anchored there."""
    key = (float(Lb), float(eps))
    if key in _FLOOR_CACHE:
        return _FLOOR_CACHE[key]
    kap = fbl_kappa(Lb, eps)
    lo, hi = 1e-6, 1e4
    f = lambda g: np.log2(1 + g) - kap * np.sqrt(max(1 - (1 + g) ** -2, 0.0))
    if f(hi) <= 0:
        val = hi
    else:
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        val = hi
    _FLOOR_CACHE[key] = val
    return val


def psi(t):
    """psi(t) = -sqrt(1 - (1+t)^-2); convex and decreasing on t >= 0 (Lemma 1)."""
    t = np.maximum(t, 1e-12)
    return -np.sqrt(1.0 - (1.0 + t) ** (-2))


def dpsi(t):
    t = np.maximum(t, 1e-12)
    x = 1.0 + t
    return -1.0 / (x ** 2 * np.sqrt(np.maximum(x ** 2 - 1.0, 1e-16)))


# --------------------------------------------------------------------------
# Per-port reflection gains and the amplifier activation surrogate
# --------------------------------------------------------------------------
def row_gain2(Pa):
    """Squared per-port reflection gain nu_m^2 = ||Phi_a e_m||^2.

    For the complex symmetric scattering matrix of a fully connected active
    sub-surface this is the total power leaving the surface for unit power
    injected into port m, so nu_m > 1 exactly when port m is amplifying."""
    if Pa.size == 0:
        return np.zeros(0)
    return np.sum(np.abs(Pa) ** 2, axis=0)


def excess_gain(Pa):
    """x_m = (nu_m^2 - 1)_+, the amplified excess of port m.  x_m = 0 means the
    reflection amplifier of port m is unbiased and draws no DC power."""
    return np.maximum(row_gain2(Pa) - 1.0, 0.0)


def n_powered(Pa, tol=1e-3):
    """Number of ports that actually draw bias power."""
    return int(np.sum(excess_gain(Pa) > tol))


def gate_phi(x, dl):
    """Smoothed count phi(x) = x/(x+delta): concave on x >= 0, phi(0) = 0."""
    return x / (x + dl)


def gate_weight(x, dl):
    """MM weight w_m = phi'(x_m) = delta/(x_m+delta)^2 of the concave surrogate;
    w_m x_m majorises phi(x_m) up to an additive constant."""
    return dl / (x + dl) ** 2


def gate_retract(Pa, cap, iters=8):
    """Symmetry preserving retraction onto {Phi = Phi^T : ||Phi e_m|| <= cap_m}.

    Phi <- D Phi D with 0 <= d_m <= 1 is complex symmetric and cannot increase
    the spectral norm, because ||D Phi D||_2 <= ||D||_2^2 ||Phi||_2 <= ||Phi||_2,
    so the retraction never leaves M_act (Lemma 6).  The damped fixed point
    d <- min(1, d sqrt(cap/nu(D Phi D))) enlarges d as far as the port
    constraints allow, and the closing one-shot clamp makes feasibility exact:
    ||(D Phi D) e_m|| = d_m ||D Phi e_m|| <= d_m nu_m <= cap_m."""
    if Pa.size == 0:
        return Pa
    cap = np.asarray(cap, float)
    d = np.ones(Pa.shape[0])
    for _ in range(iters):
        nu = np.sqrt(row_gain2((d[:, None] * Pa) * d[None, :]))
        r = np.minimum(cap / np.maximum(nu, 1e-12), 4.0)
        d = np.minimum(1.0, d * np.sqrt(r))
    X = (d[:, None] * Pa) * d[None, :]
    s = np.minimum(1.0, cap / np.maximum(np.sqrt(row_gain2(X)), 1e-12))
    return (s[:, None] * X) * s[None, :]


# --------------------------------------------------------------------------
# Fisher information matrix and matrix CRB
# --------------------------------------------------------------------------
def radar_vectors(ch: Channels, Pact, part=None):
    """Effective transmit-side round-trip radar response

        a(eta) = beta_BT a_Nt(theta0) + beta_RT Gact^H Phi_act^H b_act(eta)

    together with its partial derivatives with respect to (theta0, r0)."""
    if part is None:
        part = ch.p.Mact
    G = ch.Gfull[:part]
    b, bt, br = ch.b_full[:part], ch.b_full_dth[:part], ch.b_full_dr[:part]
    if part == 0 or Pact.shape[0] == 0:
        return (ch.beta_BT * ch.aNt, ch.beta_BT * ch.aNt_dth,
                ch.beta_BT * ch.aNt_dr)
    T = ch.beta_RT * (G.conj().T @ Pact.conj().T)          # (Nt, part)
    a = ch.beta_BT * ch.aNt + T @ b
    a_dth = ch.beta_BT * ch.aNt_dth + T @ bt
    a_dr = ch.beta_BT * ch.aNt_dr + T @ br
    return a, a_dth, a_dr


def _outer_list(ch, Pact, alpha, part=None):
    """The four derivative operators D_i = d(alpha B)/d(xi_i) as lists of
    rank-one outer products (u, v) meaning D = sum_m u_m v_m^H, with
    xi = [theta0, r0, Re{alpha}, Im{alpha}]."""
    a, a_dth, a_dr = radar_vectors(ch, Pact, part)
    u = ch.beta_BT * ch.aNr
    u_dth = ch.beta_BT * ch.aNr_dth
    D_th = [(alpha * u_dth, a), (alpha * u, a_dth)]
    D_r = [(alpha * u, a_dr)]
    D_aR = [(u, a)]
    D_aI = [(1j * u, a)]
    return [D_th, D_r, D_aR, D_aI]


def fim_full(ch: Channels, Pact, Rx, part=None):
    """Exact 4x4 FIM for xi = [theta0, r0, Re alpha, Im alpha] under
       Y = alpha a_Nr(theta) a^H(eta) X + N,  X X^H = Lradar Rx."""
    p = ch.p
    Ds = _outer_list(ch, Pact, np.sqrt(ch.alpha0_sq), part)
    c = 2.0 * p.Lradar / p.sigma2_bs
    J = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            s = 0.0 + 0j
            for (ui, vi) in Ds[i]:
                for (uj, vj) in Ds[j]:
                    s += np.vdot(uj, ui) * (vi.conj() @ Rx @ vj)
            J[i, j] = J[j, i] = c * np.real(s)
    return J


def efim(ch: Channels, Pact, Rx, part=None):
    """2x2 effective FIM of eta = [theta0, r0] after Schur-complementing out the
    nuisance reflection coefficient."""
    J = fim_full(ch, Pact, Rx, part)
    Jee, Jea, Jaa = J[:2, :2], J[:2, 2:], J[2:, 2:]
    return Jee - Jea @ np.linalg.solve(Jaa + 1e-18 * np.eye(2), Jea.T)


def fim(ch: Channels, Pact, Rx, part=None):
    return efim(ch, Pact, Rx, part)


def crb_matrix(J):
    return np.linalg.inv(J + 1e-18 * np.eye(2))


def crb_feasible(J, Cmax):
    """True iff J^-1 <= Cmax, i.e. J - Cmax^-1 is positive semidefinite."""
    Dm = J - np.linalg.inv(Cmax)
    return np.min(np.linalg.eigvalsh((Dm + Dm.conj().T) / 2)) >= -1e-9


def crb_violation(J, Cmax):
    Dm = J - np.linalg.inv(Cmax)
    return max(0.0, -np.min(np.linalg.eigvalsh((Dm + Dm.conj().T) / 2)))


# --------------------------------------------------------------------------
# Sensitivity of the sensing model to the (neglected) passive echo route
# --------------------------------------------------------------------------
def radar_vectors_echo(ch, Pa, Pp, part, w):
    """Transmit side radar response INCLUDING the lossless block's echo route
    weighted by w (w = 0 is the model of the paper, w = 1 the full physical
    passive route):
        a = b_BT a_Nt + b_RT G_a^H Phi_a^H b_a + w b_RT G_p^H Phi_p^H b_p ."""
    a, a_th, a_r = radar_vectors(ch, Pa, part)
    if w == 0.0 or Pp.size == 0:
        return a, a_th, a_r
    G = ch.Gfull[part:]
    T = w * ch.beta_RT * (G.conj().T @ Pp.conj().T)
    return (a + T @ ch.b_full[part:], a_th + T @ ch.b_full_dth[part:],
            a_r + T @ ch.b_full_dr[part:])


def efim_echo(ch, Pa, Pp, Rx, part, w):
    """Nuisance aware 2x2 effective FIM with the passive echo route included."""
    p = ch.p
    a, a_th, a_r = radar_vectors_echo(ch, Pa, Pp, part, w)
    u = ch.beta_BT * ch.aNr
    u_th = ch.beta_BT * ch.aNr_dth
    al = np.sqrt(ch.alpha0_sq)
    Ds = [[(al * u_th, a), (al * u, a_th)], [(al * u, a_r)], [(u, a)], [(1j * u, a)]]
    c = 2.0 * p.Lradar / p.sigma2_bs
    J = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            s = 0.0 + 0j
            for (ui, vi) in Ds[i]:
                for (uj, vj) in Ds[j]:
                    s += np.vdot(uj, ui) * (vi.conj() @ Rx @ vj)
            J[i, j] = J[j, i] = c * np.real(s)
    Jee, Jea, Jaa = J[:2, :2], J[:2, 2:], J[2:, 2:]
    return Jee - Jea @ np.linalg.solve(Jaa + 1e-18 * np.eye(2), Jea.T)
