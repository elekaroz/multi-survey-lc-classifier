from __future__ import annotations

import numpy as np
import sncosmo
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
_c_cgs    = 2.998e10       # speed of light [cm/s]
_sigma_SB = 5.6704e-5      # Stefan-Boltzmann [erg/cm²/s/K⁴]
_h_cgs    = 6.626e-27      # Planck [erg·s]
_kb_cgs   = 1.381e-16      # Boltzmann [erg/K]
_c_Ang    = 2.998e18       # speed of light [Å/s]
_Msun_g   = 1.989e33       # solar mass [g]
_pc_cm    = 3.0857e18      # parsec [cm]

# Neutron star defaults
_I_ns     = 1.0e45         # moment of inertia [g·cm²]
_R_ns     = 1.0e6          # radius [cm] = 10 km


# ---------------------------------------------------------------------------
# Magnetar engine: derived quantities
# ---------------------------------------------------------------------------

def magnetar_Ep(P_i_ms: float) -> float:
    """Rotational energy [erg].  K&B10 Eq. 1."""
    Omega_i = 2.0 * np.pi / (P_i_ms * 1e-3)        # rad/s
    return 0.5 * _I_ns * Omega_i**2


def magnetar_tp(B14: float, P_i_ms: float) -> float:
    """Spin-down timescale [s].  K&B10 Eq. 2."""
    Omega_i = 2.0 * np.pi / (P_i_ms * 1e-3)
    B_gauss = B14 * 1e14
    return 6.0 * _I_ns * _c_cgs**3 / (B_gauss**2 * _R_ns**6 * Omega_i**2)


def ejecta_velocity(E_sn: float, E_p: float, M_ej_g: float) -> float:
    """Characteristic final ejecta velocity [cm/s].  K&B10 below Eq. 12."""
    return np.sqrt((E_p + E_sn) / (2.0 * M_ej_g))


def diffusion_time(M_ej_g: float, kappa: float, v_f: float) -> float:
    """Effective diffusion timescale [s].  K&B10 Eq. 12."""
    return np.sqrt(3.0 / (4.0 * np.pi) * M_ej_g * kappa / (v_f * _c_cgs))


def photospheric_velocity(E_sn: float, E_p: float, M_ej_g: float,
                          delta: float = 1.0) -> float:
    """
    Shell velocity [cm/s].  K&B10 Eq. 6 (Ep < Esn) or Eq. 7 (Ep >= Esn).
    """
    v_t = np.sqrt(2.0 * E_sn / M_ej_g)
    ratio = E_p / E_sn
    if ratio < 1.0:
        # K&B10 Eq. 6
        v_sh = v_t * (7.0 / (16.0 * (3.0 - delta)) * ratio) ** (1.0 / (5.0 - delta))
    else:
        # K&B10 Eq. 7
        v_sh = v_t * np.sqrt(1.0 + ratio)
    return v_sh


# ---------------------------------------------------------------------------
# ODE solver: L_bol(t) 
# ---------------------------------------------------------------------------

def solve_magnetar_lightcurve(
    E_p: float,
    t_p: float,
    t_d: float,
    t_grid_s: np.ndarray,
) -> np.ndarray:
    """
    Solve K&B10 Eq. 10 numerically for L_e(t).

    Defining y(t) = E_int(t) * t, the ODE is:

        dy/dt = t * L_p(t)  -  y * t / t_d^2

    with L_p(t) = (E_p / t_p) / (1 + t/t_p)^2   (magnetic dipole, l=2).


    """
    t_start = max(t_grid_s[0], 1.0)       # avoid t=0 singularity
    t_end   = t_grid_s[-1]

    td2 = t_d**2

    def rhs(t, y):
        Lp = (E_p / t_p) / (1.0 + t / t_p)**2
        return t * Lp - y * t / td2

    sol = solve_ivp(
        rhs,
        t_span=(t_start, t_end),
        y0=[0.0],
        method='RK45',
        t_eval=t_grid_s,
        rtol=1e-8,
        atol=1e-30,
    )

    y = sol.y[0]
    L_e = np.maximum(y / td2, 0.0)     # L_e = y / t_d²  (K&B10 Eq. 12)
    return L_e


# ---------------------------------------------------------------------------
# Blackbody SED
# ---------------------------------------------------------------------------

def blackbody_flux_density(wave_Ang: np.ndarray, T_K: float,
                           R_cm: float) -> np.ndarray:
    """
    Spectral luminosity density of a blackbody at the source.

    """
    wave_cm = wave_Ang * 1e-8                         
    x = _h_cgs * _c_cgs / (wave_cm * _kb_cgs * T_K)  
    x = np.clip(x, 0.0, 500.0)
    B_lam_cm = (2.0 * _h_cgs * _c_cgs**2 / wave_cm**5) / np.expm1(x)
    B_lam = B_lam_cm * 1e-8

    L_lam = 4.0 * np.pi**2 * R_cm**2 * B_lam
    return L_lam


# ---------------------------------------------------------------------------
# sncosmo.Source 
# ---------------------------------------------------------------------------

class MagnetarSource(sncosmo.Source):
    """
    sncosmo Source for SLSN
    
    """

    _param_names = ['amplitude']
    param_names_latex = ['A']

    def __init__(
        self,
        P_i_ms: float,
        B14: float,
        M_ej_Msun: float,
        E_sn: float = 1e51,
        kappa: float = 0.2,
        n_phase: int = 300,
        name: str = 'magnetar_slsn',
        version: str = '1.0',
    ):
        self.name = name
        self.version = version
        self._parameters = np.array([1.0])   


        self._P_i_ms = P_i_ms
        self._B14 = B14
        self._M_ej_g = M_ej_Msun * _Msun_g
        self._E_sn = E_sn
        self._kappa = kappa

        self._E_p = magnetar_Ep(P_i_ms)
        self._t_p = magnetar_tp(B14, P_i_ms)
        self._v_f = ejecta_velocity(E_sn, self._E_p, self._M_ej_g)
        self._t_d = diffusion_time(self._M_ej_g, kappa, self._v_f)
        self._v_ph = photospheric_velocity(E_sn, self._E_p, self._M_ej_g)

        t_min_days = 0.1
        t_max_days = 600.0
        self._phase = np.linspace(t_min_days, t_max_days, n_phase)

        self._wave = np.linspace(100.0, 25000.0, 500)

        t_grid_s = self._phase * 86400.0
        self._L_bol = solve_magnetar_lightcurve(
            self._E_p, self._t_p, self._t_d, t_grid_s
        )

        R_ph = self._v_ph * t_grid_s                  
        T_eff = np.zeros_like(self._L_bol)
        valid = (self._L_bol > 0) & (R_ph > 0)
        T_eff[valid] = (
            self._L_bol[valid] / (4.0 * np.pi * R_ph[valid]**2 * _sigma_SB)
        ) ** 0.25
        T_eff = np.clip(T_eff, 1000.0, None)

        self._T_eff = T_eff
        self._R_ph = R_ph

        d_10pc_cm = 10.0 * _pc_cm
        self._flux_grid = np.zeros((n_phase, len(self._wave)))
        for i in range(n_phase):
            if T_eff[i] > 1000.0 and R_ph[i] > 0:
                L_lam = blackbody_flux_density(self._wave, T_eff[i], R_ph[i])
                # Convert luminosity density to flux at 10 pc
                self._flux_grid[i, :] = L_lam / (4.0 * np.pi * d_10pc_cm**2)

    def _flux(self, phase, wave):
        """
        Return spectral flux density at given phases and wavelengths.

        """
        from scipy.interpolate import RectBivariateSpline

        if not hasattr(self, '_interp'):
            self._interp = RectBivariateSpline(
                self._phase, self._wave, self._flux_grid,
                kx=3, ky=3,
            )

        f = self._parameters[0] * self._interp(phase, wave, grid=True)
        mask = np.atleast_1d(phase) < self._phase[0]
        f[mask, :] = 0.0
        mask_end = np.atleast_1d(phase) > self._phase[-1]
        f[mask_end, :] = 0.0
        return np.maximum(f, 0.0)


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def sample_magnetar_params(rng: np.random.Generator) -> dict:

    log_P = rng.normal(np.log10(2.4), 0.26)
    P_i_ms = float(np.clip(10**log_P, 0.7, 20.0))

    log_Bperp = rng.normal(np.log10(0.8), 0.48)
    B_perp = 10**log_Bperp
    B14 = float(np.clip(np.sqrt(5) * B_perp, 0.1, 100.0))

    log_Mej = rng.normal(np.log10(4.8), 0.38)
    M_ej_Msun = float(np.clip(10**log_Mej, 0.5, 100.0))

    return {
        'P_i_ms':    P_i_ms,
        'B14':       B14,
        'M_ej_Msun': M_ej_Msun,
        'E_sn':      1e51,
        'kappa':     0.2,
    }


# ---------------------------------------------------------------------------
# Cosmology helper
# ---------------------------------------------------------------------------

def _luminosity_distance_cm(z: float, H0: float = 70.0,
                            Omega_m: float = 0.3) -> float:
    """Luminosity distance [cm] in flat LCDM."""
    H0_s = H0 * 1e5 / (1e6 * _pc_cm)  
    Omega_L = 1.0 - Omega_m
    z_arr = np.linspace(0.0, z, 1000)
    dz = z_arr[1] - z_arr[0] if len(z_arr) > 1 else 0.0
    E_z = np.sqrt(Omega_m * (1.0 + z_arr)**3 + Omega_L)
    D_c = (_c_cgs / H0_s) * np.sum(1.0 / E_z) * dz
    return D_c * (1.0 + z)


# ---------------------------------------------------------------------------
# Replacement for _simulate_slsn_magnetar in models.py
# ---------------------------------------------------------------------------

_ZTF_BANDS = {1: 'ztfg', 2: 'ztfr', 3: 'ztfi'}
_LSST_BANDS = {6: 'lsstu', 1: 'lsstg', 2: 'lsstr', 3: 'lssti',
                4: 'lsstz', 5: 'lssty'}


def simulate_slsn_magnetar(
    mjd_obs_ztf: dict,
    mjd_obs_lsst: dict,
    rng: np.random.Generator,
    return_params: bool = False,
):

    import warnings

    #1. Sample physical parameters
    params = sample_magnetar_params(rng)

    #2. Redshift
    if rng.random() < 0.50:
        z = float(rng.uniform(0.05, 0.40))
    else:
        z = float(rng.uniform(0.40, 1.20))

    #3. Peak time placement
    all_ztf = (np.concatenate([v for v in mjd_obs_ztf.values() if len(v) > 0])
               if any(len(v) > 0 for v in mjd_obs_ztf.values())
               else np.array([]))
    all_lsst = (np.concatenate([v for v in mjd_obs_lsst.values() if len(v) > 0])
                if any(len(v) > 0 for v in mjd_obs_lsst.values())
                else np.array([]))

    if len(all_ztf) >= 5:
        all_s = np.sort(all_ztf)
        window = 90.0
        best_start, best_count = all_s[0], 0
        for t in all_s:
            cnt = int(np.sum((all_s >= t) & (all_s <= t + window)))
            if cnt > best_count:
                best_count, best_start = cnt, t
        t0 = float(rng.uniform(best_start + 5.0, best_start + window / 2.0))
    elif len(all_lsst) > 0:
        t_start, t_end = all_lsst.min(), all_lsst.max()
        t0 = float(rng.uniform(t_start + 10.0,
                                t_start + (t_end - t_start) / 3.0))
    else:
        t0 = 59100.0

    #4. Build sncosmo model
    source = MagnetarSource(**params)
    model = sncosmo.Model(source=source)
    model.set(z=z, t0=t0)

    #5. Distance modulus
    D_L_cm = _luminosity_distance_cm(z)
    mu = 5.0 * np.log10(D_L_cm / (10.0 * _pc_cm))

    #6. Evaluate magnitudes
    def _eval(mjd_arr, band_name):
        if len(mjd_arr) == 0:
            return np.array([], dtype=float)
        mags = np.full(len(mjd_arr), 99.0, dtype=float)
        in_range = ((mjd_arr >= model.mintime()) &
                    (mjd_arr <= model.maxtime()))
        if in_range.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    abs_mags = model.bandmag(
                        band_name, 'ab', mjd_arr[in_range]
                    )
                    mags[in_range] = abs_mags + mu
                except Exception:
                    pass
        return mags

    ztf_mags = {fid: _eval(mjd_obs_ztf.get(fid, np.array([])), band)
                for fid, band in _ZTF_BANDS.items()}
    lsst_mags = {fid: _eval(mjd_obs_lsst.get(fid, np.array([])), band)
                 for fid, band in _LSST_BANDS.items()}

    lc_dict = {'ztf': ztf_mags, 'lsst': lsst_mags}

    if return_params:
        full_params = {
            'z': z,
            't0': t0,
            **params,
            'E_p_erg': source._E_p,
            't_p_days': source._t_p / 86400.0,
            't_d_days': source._t_d / 86400.0,
            'v_ph_km_s': source._v_ph / 1e5,
            'L_peak_erg_s': float(np.max(source._L_bol)),
            'M_peak_bol': float(-2.5 * np.log10(
                max(np.max(source._L_bol), 1e30) / 3.828e33) + 4.74),
        }
        return lc_dict, full_params

    return lc_dict


# ---------------------------------------------------------------------------
# Quick diagnostic
# ---------------------------------------------------------------------------

def _diagnostic_summary(P_i_ms=2.0, B14=2.0, M_ej_Msun=5.0):
    """Print derived quantities and optionally plot L(t) and T(t)."""
    params = dict(P_i_ms=P_i_ms, B14=B14, M_ej_Msun=M_ej_Msun)
    src = MagnetarSource(**params)

    E_p = src._E_p
    t_p = src._t_p
    t_d = src._t_d
    v_ph = src._v_ph
    L_peak = np.max(src._L_bol)
    i_peak = np.argmax(src._L_bol)
    t_peak = src._phase[i_peak]

    print(f"=== Magnetar SLSN model: P_i={P_i_ms} ms, B14={B14}, "
          f"M_ej={M_ej_Msun} Msun ===")
    print(f"  E_p          = {E_p:.3e} erg")
    print(f"  t_p          = {t_p/86400:.1f} days")
    print(f"  t_d          = {t_d/86400:.1f} days")
    print(f"  v_ph         = {v_ph/1e5:.0f} km/s")
    print(f"  L_peak       = {L_peak:.3e} erg/s")
    print(f"  t_peak       = {t_peak:.1f} days")
    print(f"  M_bol(peak)  = {-2.5*np.log10(L_peak/3.828e33)+4.74:.1f}")

    return src


if __name__ == '__main__':
    for B14 in [1.0, 5.0, 10.0, 20.0]:
        _diagnostic_summary(P_i_ms=5.0, B14=B14)
        print()

    print("--- SN 2008es fit (K&B10) ---")
    _diagnostic_summary(P_i_ms=2.0, B14=2.0, M_ej_Msun=5.0)
