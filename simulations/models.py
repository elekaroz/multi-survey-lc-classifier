from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import sncosmo
from magnetar_source import simulate_slsn_magnetar as _simulate_slsn_magnetar


#%%

# ---------------------------------------------------------------------------
# Band name mappings
# ---------------------------------------------------------------------------

# Numeric fid, sncosmo bandpass name.
_ZTF_BANDS: Dict[int, str] = {
    1: 'ztfg',
    2: 'ztfr',
    3: 'ztfi',
}
_LSST_BANDS: Dict[int, str] = {
    6: 'lsstu',
    1: 'lsstg',
    2: 'lsstr',
    3: 'lssti',
    4: 'lsstz',
    5: 'lssty',
}

# ---------------------------------------------------------------------------
# Colour offsets  Delta_mag = band_mag − r_mag  (positive -> fainter than r)
# ---------------------------------------------------------------------------

COLOR_OFFSETS: Dict[str, Dict[int, float]] = {
    'QSO':     {6: +0.31, 1: +0.045, 2: +0.00, 3: -0.09, 4: -0.17, 5: -0.47},
    'AGN':     {6: +0.3, 1: -0.103, 2: 0.0, 3: +0.15, 4: +0.35, 5: +0.60},
    'Blazar':  {6: +0.0, 1: -0.319, 2: 0.0, 3: +0.15, 4: +0.30, 5: +0.50},
    'YSO':     {6: +2.5, 1: +1.235, 2: 0.0, 3: -0.40, 4: -0.90, 5: -1.20},
    'CV/Nova': {6: -0.30, 1: +0.116, 2: 0.0, 3: +0.20, 4: +0.45, 5: +0.70},
    'RRL':           {6: +2.14, 1: +0.020, 2: +0.00, 3: -0.21, 4: -0.29, 5: -0.59},
    'CEP':           {6: +2.05, 1: +0.744, 2: +0.00, 3: -0.20, 4: -0.27, 5: -0.57},
    'DSCT':          {6: +0.40, 1: -0.049, 2: 0.0, 3: -0.08, 4: -0.15, 5: -0.20},
    'LPV':           {6: +0.37, 1: +2.211, 2: +0.00, 3: -0.07, 4: -0.19, 5: -0.49},
    'E':             {6: +1.95, 1: +0.539, 2: +0.00, 3: -0.19, 4: -0.24, 5: -0.54},
    'Periodic-Other':{6: +0.45, 1: +0.779, 2: 0.0, 3: -0.10, 4: -0.20, 5: -0.30},
}

# ZTF fids 1, 2, 3 map to the same band offsets as LSST fids 1, 2, 3 (g, r, i).


# ---------------------------------------------------------------------------
# Transients (sncosmo)
# ---------------------------------------------------------------------------

_TRANSIENT_MODELS: Dict[str, Tuple[str, dict]] = {
    'SNIa':  ('salt2',        {}),
    'SNII':  ('nugent-sn2p',  {}),
    'SNIbc': ('nugent-sn1bc', {}),
    #SLSN with magnetar model
}


def simulate_transient(
    class_name: str,
    mjd_obs_ztf:  Dict[int, np.ndarray],
    mjd_obs_lsst: Dict[int, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Generate noiseless magnitudes for a transient object in ZTF and LSST bands.
    Epochs outside the model's time range are set to 99.0.
    """
    if class_name == 'SLSN':
        return _simulate_slsn_magnetar(mjd_obs_ztf, mjd_obs_lsst, rng)

    model_name, _ = _TRANSIENT_MODELS[class_name]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = sncosmo.Model(source=model_name)

    _z_ranges = {

        'SNIa':  (0.005, 0.15, 0.80),
        'SNII':  (0.005, 0.08, 0.40),
        'SNIbc': (0.005, 0.08, 0.40),
        # 'SLSN':  (0.05,  0.50, 1.50),
    }
    z_lo, z_ztf_hi, z_lsst_hi = _z_ranges[class_name]

    if rng.random() < 0.4:
        z = rng.uniform(z_lo, z_ztf_hi)
    else:
        z = rng.uniform(z_ztf_hi, z_lsst_hi)
    model.set(z=z)

    # Place peak time to maximise overlap with ZTF observations.

    all_ztf = np.concatenate([v for v in mjd_obs_ztf.values() if len(v) > 0]) if any(len(v) > 0 for v in mjd_obs_ztf.values()) else np.array([])
    all_lsst = np.concatenate([v for v in mjd_obs_lsst.values() if len(v) > 0]) if any(len(v) > 0 for v in mjd_obs_lsst.values()) else np.array([])
 
    if len(all_ztf) >= 5:
        all_ztf_sorted = np.sort(all_ztf)
        window = 90.0
        best_start = all_ztf_sorted[0]
        best_count = 0
        for t in all_ztf_sorted:
            count = np.sum((all_ztf_sorted >= t) & (all_ztf_sorted <= t + window))
            if count > best_count:
                best_count = count
                best_start = t
        t0 = rng.uniform(best_start + 5.0, best_start + window / 2.0)
    elif len(all_lsst) > 0:
        t_start, t_end = all_lsst.min(), all_lsst.max()
        t0 = rng.uniform(t_start + 10.0, t_start + (t_end - t_start) / 3.0)
    else:
        t0 = 59100.0  # fallback
 
    model.set(t0=t0)
 
    _abs_mags = {'SNIa': -19.3, 'SNII': -17.0, 'SNIbc': -17.5, 'SLSN': -21.0}
    _abs_sig  = {'SNIa':  0.3,  'SNII':  0.5,  'SNIbc':  0.5,  'SLSN':  0.5}
 
    if class_name == 'SNIa':
        model.set(x1=rng.uniform(-2.0, 2.0), c=rng.uniform(-0.2, 0.2))
 
    try:
        peak_abs = _abs_mags[class_name] + rng.normal(0.0, _abs_sig[class_name])
        model.set_source_peakabsmag(peak_abs, 'bessellb', 'ab')
    except Exception:
        _fallback_amp = {'SNIa': 1e-5, 'SNII': 1e-12, 'SNIbc': 1e-12, 'SLSN': 1e-10}
        amp_key = 'x0' if class_name == 'SNIa' else 'amplitude'
        model.set(**{amp_key: _fallback_amp[class_name] * rng.uniform(0.5, 2.0)})
 
    def _eval(mjd_arr: np.ndarray, band_name: str) -> np.ndarray:
        if len(mjd_arr) == 0:
            return np.array([], dtype=float)
        mags = np.full(len(mjd_arr), 99.0, dtype=float)
        in_range = (mjd_arr >= model.mintime()) & (mjd_arr <= model.maxtime())
        if in_range.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    mags[in_range] = model.bandmag(
                        band_name, 'ab', mjd_arr[in_range]
                    )
                except Exception:
                    pass
        return mags
 
    ztf_mags  = {
        fid: _eval(mjd_obs_ztf.get( fid, np.array([])), band)
        for fid, band in _ZTF_BANDS.items()
    }
    lsst_mags = {
        fid: _eval(mjd_obs_lsst.get(fid, np.array([])), band)
        for fid, band in _LSST_BANDS.items()
    }
    return {'ztf': ztf_mags, 'lsst': lsst_mags}


# ---------------------------------------------------------------------------
# Stochastic: Damped Random Walk
# ---------------------------------------------------------------------------
_DRW_PARAMS: Dict[str, Tuple[float, float, Tuple[float, float]]] = {
    
    'QSO':     (250,  0.13, (16.8, 23.5)),  
    'CV/Nova': (6,    0.5,  (14.5, 21.0)), 
    'Blazar':  (30,   0.25, (15.5, 22.0)), 
    'AGN':     (200,  0.12, (15.2, 22.5)),  
    'YSO':     (10,   0.20, (14.5, 21.5)),  
}


def _blazar_lightcurve(
    t_grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    n = len(t_grid)
    t_span = t_grid[-1] - t_grid[0] if n > 1 else 1.0

    #1. Baseline DRW (quiescent) -----------------------------------------
    tau_q   = rng.uniform(20.0, 50.0)
    sigma_q = rng.uniform(0.15, 0.35)
    base    = _drw_process(tau_q, sigma_q, t_grid, rng)

    #2. High-state mask
    n_states   = max(1, int(rng.poisson(t_span / 200.0)))
    high_state = np.zeros(n, dtype=bool)
    for _ in range(n_states):
        t_on     = rng.uniform(t_grid[0], t_grid[-1])
        duration = rng.uniform(30.0, 120.0)
        high_state |= (t_grid >= t_on) & (t_grid <= t_on + duration)

    tau_h   = rng.uniform(10.0, 30.0)
    sigma_h = sigma_q * rng.uniform(2.0, 4.0)
    high_drw = _drw_process(tau_h, sigma_h, t_grid, rng)

    high_offset = rng.uniform(0.3, 1.5)
    lc = np.where(high_state,
                  high_drw - high_offset,
                  base)

    #3. Discrete flares
    n_flares = int(rng.poisson(t_span / 150.0 * rng.uniform(0.5, 2.0)))
    for _ in range(n_flares):
        t_peak   = rng.uniform(t_grid[0], t_grid[-1])
        amp_mag  = rng.uniform(0.3, 2.5)   
        t_rise   = rng.uniform(3.0, 15.0)  
        t_decay  = rng.uniform(10.0, 60.0)  
        dt       = t_grid - t_peak
       
        flare    = np.where(
            dt < 0,
            amp_mag * np.exp(-0.5 * (dt / t_rise) ** 2),
            amp_mag * np.exp(-dt / t_decay),
        )
        lc -= flare

    #4. Micro-variability
    micro_amp = rng.uniform(0.01, 0.08)
    micro     = rng.normal(0.0, micro_amp, n)
    lc += micro

    return lc


def _yso_lightcurve(
    t_grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    n = len(t_grid)
    t_span = t_grid[-1] - t_grid[0] if n > 1 else 1.0

    vtype = rng.choice(['burst', 'dipper', 'qperiodic', 'stochastic'],
                       p=[0.25, 0.25, 0.30, 0.20])

    #Baseline DRW
    tau_base   = rng.uniform(5.0, 25.0)
    sigma_base = rng.uniform(0.08, 0.25)
    base       = _drw_process(tau_base, sigma_base, t_grid, rng)
    lc         = base.copy()

    if vtype == 'burst':
        # Accretion bursts
        n_bursts = max(1, int(rng.poisson(t_span / 30.0)))
        for _ in range(n_bursts):
            t_pk  = rng.uniform(t_grid[0], t_grid[-1])
            amp   = rng.lognormal(mean=-0.5, sigma=0.7)  
            amp   = np.clip(amp, 0.05, 3.0)
            t_r   = rng.uniform(1.0, 8.0)
            t_d   = rng.uniform(5.0, 30.0)
            dt    = t_grid - t_pk
            burst = np.where(dt < 0,
                             amp * np.exp(-0.5 * (dt / t_r) ** 2),
                             amp * np.exp(-dt / t_d))
            lc -= burst

    elif vtype == 'dipper':
        # Disk occultation dips
        n_dips = max(1, int(rng.poisson(t_span / 20.0)))
        for _ in range(n_dips):
            t_dip   = rng.uniform(t_grid[0], t_grid[-1])
            depth   = rng.uniform(0.1, 1.5)
            width_d = rng.uniform(0.5, 8.0)  
        
            dt_dip  = t_grid - t_dip
            dip     = depth * np.exp(-0.5 * (dt_dip / (width_d * 0.6)) ** 2)
            lc += dip  

    elif vtype == 'qperiodic':
        # Quasi-periodic spot rotation
        period_spot = rng.uniform(1.0, 15.0)  
        amp_spot    = rng.uniform(0.05, 0.50)
        phase0      = rng.uniform(0.0, 2.0 * np.pi)
        tau_env  = rng.uniform(50.0, 200.0)
        env_drw  = _drw_process(tau_env, 0.3, t_grid, rng)
        env_fac  = np.clip(1.0 + 0.5 * env_drw / (0.3 + 1e-9), 0.3, 3.0)
        spot_lc  = amp_spot * env_fac * np.sin(2 * np.pi * t_grid / period_spot + phase0)
        lc += spot_lc

        if rng.random() < 0.3:
            t_dip   = rng.uniform(t_grid[0], t_grid[-1])
            depth   = rng.uniform(0.05, 0.8)
            width_d = rng.uniform(1.0, 5.0)
            dt_dip  = t_grid - t_dip
            lc += depth * np.exp(-0.5 * (dt_dip / (width_d * 0.6)) ** 2)

    return lc


def _drw_process(
    tau: float,
    sigma: float,
    t_grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck (DRW) process on an irregular time grid.
    Returns a zero-mean magnitude series.
    """
    n = len(t_grid)
    x = np.zeros(n)
    x[0] = rng.normal(0.0, sigma)
    for k in range(1, n):
        dt = t_grid[k] - t_grid[k - 1]
        decay = np.exp(-dt / tau)
        innov_std = sigma * np.sqrt(max(1.0 - decay ** 2, 0.0))
        x[k] = x[k - 1] * decay + rng.normal(0.0, innov_std)
    return x


def simulate_stochastic(
    class_name: str,
    mjd_obs_ztf:  Dict[int, np.ndarray],
    mjd_obs_lsst: Dict[int, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, Dict[int, np.ndarray]]:

    tau, sigma, (r_lo, r_hi) = _DRW_PARAMS[class_name]
    mean_r = rng.uniform(r_lo, r_hi)
    colors = COLOR_OFFSETS[class_name]

    all_arrays = [v for v in {**mjd_obs_ztf, **mjd_obs_lsst}.values() if len(v) > 0]
    if not all_arrays:
        return {'ztf': {}, 'lsst': {}}
    all_times = np.unique(np.concatenate(all_arrays))

    if class_name == 'Blazar':
        drw_r = _blazar_lightcurve(all_times, rng)
    elif class_name == 'YSO':
        drw_r = _yso_lightcurve(all_times, rng)
    elif class_name in ('QSO', 'AGN'):
        mu_log_tau  = 2.4 if class_name == 'QSO' else 2.3
        tau_draw    = 10 ** rng.normal(mu_log_tau, 0.4)
        tau_draw    = float(np.clip(tau_draw, 20.0, 2000.0))
        sigma_draw  = 10 ** rng.normal(-0.9, 0.3)
        sigma_draw  = float(np.clip(sigma_draw, 0.03, 0.8))
        drw_r = _drw_process(tau_draw, sigma_draw, all_times, rng)
    else:
        drw_r = _drw_process(tau, sigma, all_times, rng)  

    def _interp_drw(t_query: np.ndarray) -> np.ndarray:
        return np.interp(t_query, all_times, drw_r)


    _lc_std = float(np.std(drw_r)) if len(drw_r) > 1 else sigma

    def _band_mag(t_arr: np.ndarray, fid: int) -> np.ndarray:
        if len(t_arr) == 0:
            return np.array([], dtype=float)
        offset = colors.get(fid, 0.0)
        band_scatter = _lc_std * 0.10 * rng.normal(0.0, 1.0, len(t_arr))
        return mean_r + offset + _interp_drw(t_arr) + band_scatter

    if class_name == 'CV/Nova':
        n_outbursts = int(rng.integers(0, 4))
        if n_outbursts > 0:
            t_bursts   = rng.uniform(all_times.min(), all_times.max(), n_outbursts)
            amp_bursts = rng.uniform(2.0, 5.0, n_outbursts)   
            width_bursts = rng.uniform(5.0, 15.0, n_outbursts)  

            def outburst_delta(t_arr: np.ndarray) -> np.ndarray:
                delta = np.zeros(len(t_arr))
                for tb, ab, wb in zip(t_bursts, amp_bursts, width_bursts):
                    delta -= ab * np.exp(-0.5 * ((t_arr - tb) / wb) ** 2)
                return delta   
        else:
            def outburst_delta(t_arr: np.ndarray) -> np.ndarray:
                return np.zeros(len(t_arr))
    else:
        def outburst_delta(t_arr: np.ndarray) -> np.ndarray:
            return np.zeros(len(t_arr))

    ztf_mags = {
        fid: _band_mag(mjd_obs_ztf.get(fid, np.array([])), fid)
             + outburst_delta(mjd_obs_ztf.get(fid, np.array([])))
        for fid in _ZTF_BANDS
    }
    lsst_mags = {
        fid: _band_mag(mjd_obs_lsst.get(fid, np.array([])), fid)
             + outburst_delta(mjd_obs_lsst.get(fid, np.array([])))
        for fid in _LSST_BANDS
    }
    return {'ztf': ztf_mags, 'lsst': lsst_mags}


# ---------------------------------------------------------------------------
# Periodic (Fourier series + special cases)
# ---------------------------------------------------------------------------

_PERIODIC_PARAMS: Dict[str, tuple] = {
    'RRL':            ((0.3,    0.9),   (0.136, 0.835), 3, (14.0, 22.5), True,  False),
    'CEP':            ((1.0,  100.0),   (0.12,  0.84),  3, (14.0, 20.5), True,  True),
    'DSCT':           ((0.02,   0.3),   (0.01,  0.45),  2, (14.0, 19.5), False, False),
    'LPV':            ((100., 1000.),   (0.119, 2.86),  2, (14.0, 21.5), False, True),
    'E':              ((0.1,   10.0),   (0.05,  0.65),  1, (14.0, 21.5), False, False),
    'Periodic-Other': ((0.5,   50.0),   (0.029, 0.526), 2, (14.0, 20.5), False, False),
}


def _eclipsing_binary_model(
    period: float,
    t0: float,
    eclipse_depth: float,
    eclipse_width: float,
    sec_depth: float,
    rng: np.random.Generator,
) -> callable:
    
    morph = rng.choice(['detached', 'semi-detached', 'contact'],
                       p=[0.45, 0.25, 0.30])

    if morph == 'detached':

        ecc_offset = rng.uniform(-0.08, 0.08)   
        sec_phase  = 0.5 + ecc_offset
        sec_width  = eclipse_width * rng.uniform(0.7, 1.3)
        ld_coeff   = rng.uniform(0.3, 0.6)
        ellip_amp  = rng.uniform(0.001, 0.01) * eclipse_depth
        oconnell   = 0.0
        spot_amp   = 0.0

    elif morph == 'semi-detached':
        ecc_offset = rng.uniform(-0.02, 0.02)   
        sec_phase  = 0.5 + ecc_offset
        sec_width  = eclipse_width * rng.uniform(0.4, 0.8)
        ld_coeff   = rng.uniform(0.4, 0.7)
        ellip_amp  = rng.uniform(0.05, 0.20) * eclipse_depth
        oconnell   = rng.uniform(-0.03, 0.03) * eclipse_depth
        spot_amp   = rng.uniform(0.0, 0.10) * eclipse_depth if rng.random() < 0.4 else 0.0
        spot_phase = rng.uniform(0.1, 0.9)
        spot_width = rng.uniform(0.05, 0.20)

    else: 
        ecc_offset = 0.0          
        sec_phase  = 0.5
        eclipse_width = min(eclipse_width * rng.uniform(1.2, 1.8), 0.35)
        sec_width  = eclipse_width * rng.uniform(0.85, 1.15)
        ld_coeff   = rng.uniform(0.5, 0.8)
        ellip_amp  = rng.uniform(0.25, 0.60) * eclipse_depth
        oconnell   = rng.uniform(-0.08, 0.08) * eclipse_depth
        spot_amp   = rng.uniform(0.02, 0.15) * eclipse_depth if rng.random() < 0.65 else 0.0
        spot_phase = rng.uniform(0.1, 0.9)
        spot_width = rng.uniform(0.05, 0.15)

    half_prim = eclipse_width / 2.0
    half_sec  = sec_width / 2.0

    def _eclipse_profile(dphi: np.ndarray, half_w: float, depth: float,
                         ld: float) -> np.ndarray:
    
        x = np.abs(dphi) / half_w          
        in_total  = x < (1.0 - ld * 0.3)
        in_partial = ~in_total
        profile = np.where(in_total, 1.0, np.sqrt(np.maximum(1.0 - x ** 2, 0.0)))
        return depth * profile

    def lc(t_arr: np.ndarray) -> np.ndarray:
        phi = ((t_arr - t0) / period) % 1.0   
        mag = np.zeros(len(t_arr))

        #Primary eclipse
        phi_prim = np.where(phi <= 0.5, phi, phi - 1.0)   
        prim_mask = np.abs(phi_prim) < half_prim
        mag = np.where(
            prim_mask,
            _eclipse_profile(phi_prim, half_prim, eclipse_depth, ld_coeff),
            mag,
        )

        #Secondary eclipse
        phi_sec = phi - sec_phase
        phi_sec = np.where(phi_sec >  0.5, phi_sec - 1.0, phi_sec)
        phi_sec = np.where(phi_sec < -0.5, phi_sec + 1.0, phi_sec)
        sec_mask = np.abs(phi_sec) < half_sec
        sec_profile = _eclipse_profile(phi_sec, half_sec, sec_depth, ld_coeff)
        mag = np.where(sec_mask, np.maximum(mag, sec_profile), mag)

        #Ellipsoidal variation
        if ellip_amp > 0.0:
            mag += ellip_amp * (1.0 - np.cos(4.0 * np.pi * phi)) / 2.0

        #O'Connell effect (asymmetry between maxima)
        if oconnell != 0.0:
            mag += oconnell * np.sin(2.0 * np.pi * phi)

        #Spot modulation
        if spot_amp > 0.0:
            dphi_spot = phi - spot_phase
            dphi_spot = np.where(dphi_spot >  0.5, dphi_spot - 1.0, dphi_spot)
            dphi_spot = np.where(dphi_spot < -0.5, dphi_spot + 1.0, dphi_spot)
            mag += spot_amp * np.exp(-0.5 * (dphi_spot / spot_width) ** 2)

        return mag

    return lc


def simulate_periodic(
    class_name: str,
    mjd_obs_ztf:  Dict[int, np.ndarray],
    mjd_obs_lsst: Dict[int, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, Dict[int, np.ndarray]]:

    p_lo, p_hi   = _PERIODIC_PARAMS[class_name][0]
    a_lo, a_hi   = _PERIODIC_PARAMS[class_name][1]
    n_harm        = _PERIODIC_PARAMS[class_name][2]
    r_lo, r_hi   = _PERIODIC_PARAMS[class_name][3]
    asymmetric    = _PERIODIC_PARAMS[class_name][4]
    log_period    = _PERIODIC_PARAMS[class_name][5]
    colors        = COLOR_OFFSETS[class_name]

    if log_period:
        period = 10 ** rng.uniform(np.log10(p_lo), np.log10(p_hi))
    else:
        period = rng.uniform(p_lo, p_hi)
    mean_r  = rng.uniform(r_lo, r_hi)
    phase0  = rng.uniform(0.0, 2.0 * np.pi)

    if class_name == 'DSCT':
        if rng.random() < 0.20:   # HADS
            a1 = rng.uniform(0.10, 0.50)
        else:                      # LADS
            a1 = rng.uniform(0.01, 0.10)
        a_lo, a_hi = a1, a1
    amps   = rng.uniform(a_lo, a_hi) * np.array([1.0 / k for k in range(1, n_harm + 1)])
    phases = np.zeros(n_harm)
    if n_harm > 1:
        phases[1:] = rng.uniform(0.0, 2.0 * np.pi, n_harm - 1)
    if asymmetric and n_harm >= 2:
        phases[1] = rng.uniform(np.pi / 4.0, 3.0 * np.pi / 4.0)

    if class_name == 'E':
        eclipse_depth = rng.uniform(0.05, 0.85)
        eclipse_width = rng.uniform(0.05, 0.25)
        sec_depth     = eclipse_depth * rng.uniform(0.05, 0.55)
        all_arr = [v for v in {**mjd_obs_ztf, **mjd_obs_lsst}.values() if len(v) > 0]
        t0 = np.concatenate(all_arr).min() if all_arr else 0.0
        _lc = _eclipsing_binary_model(period, t0, eclipse_depth, eclipse_width, sec_depth, rng)

        def lc_model(t_arr: np.ndarray) -> np.ndarray:
            return _lc(t_arr)
    else:
        def lc_model(t_arr: np.ndarray) -> np.ndarray:
            mag = np.zeros(len(t_arr))
            for k, (a, ph) in enumerate(zip(amps, phases), start=1):
                mag += a * np.sin(2.0 * np.pi * k * t_arr / period + ph + phase0)
            return mag

    #LPV
    if class_name == 'LPV':
        all_arrays = [v for v in {**mjd_obs_ztf, **mjd_obs_lsst}.values() if len(v) > 0]
        all_times  = np.unique(np.concatenate(all_arrays)) if all_arrays else np.array([0.0])
        drw_resid  = _drw_process(tau=period * 0.5, sigma=0.3, t_grid=all_times, rng=rng)

        def _residual(t_arr: np.ndarray) -> np.ndarray:
            return np.interp(t_arr, all_times, drw_resid)
    else:
        def _residual(t_arr: np.ndarray) -> np.ndarray:
            return np.zeros(len(t_arr))

    def _band_mag(t_arr: np.ndarray, fid: int) -> np.ndarray:
        if len(t_arr) == 0:
            return np.array([], dtype=float)
        offset    = colors.get(fid, 0.0)
        amp_scale = 1.0 + 0.1 * abs(offset)
        return mean_r + offset + amp_scale * lc_model(t_arr) + _residual(t_arr)

    ztf_mags  = {fid: _band_mag(mjd_obs_ztf.get( fid, np.array([])), fid) for fid in _ZTF_BANDS}
    lsst_mags = {fid: _band_mag(mjd_obs_lsst.get(fid, np.array([])), fid) for fid in _LSST_BANDS}
    return {'ztf': ztf_mags, 'lsst': lsst_mags}


# ---------------------------------------------------------------------------
# Wrappers that return (lc_dict, model_params_dict)
# ---------------------------------------------------------------------------

def simulate_transient_with_params(
    class_name: str,
    mjd_obs_ztf:  Dict[int, np.ndarray],
    mjd_obs_lsst: Dict[int, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict]:

    # SLSN
    if class_name == 'SLSN':
        return _simulate_slsn_magnetar(
            mjd_obs_ztf, mjd_obs_lsst, rng, return_params=True
        )

    import sncosmo as _sncosmo

    model_name, _ = _TRANSIENT_MODELS[class_name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = _sncosmo.Model(source=model_name)

    _z_ranges = {
        'SNIa':  (0.005, 0.15, 0.80),
        'SNII':  (0.005, 0.08, 0.40),
        'SNIbc': (0.005, 0.08, 0.40),
        'SLSN':  (0.05,  0.50, 1.50),
    }
    z_lo, z_ztf_hi, z_lsst_hi = _z_ranges[class_name]
    if rng.random() < 0.6:
        z = rng.uniform(z_lo, z_ztf_hi)
    else:
        z = rng.uniform(z_ztf_hi, z_lsst_hi)
    model.set(z=z)

    all_ztf  = np.concatenate([v for v in mjd_obs_ztf.values()  if len(v) > 0]) \
               if any(len(v) > 0 for v in mjd_obs_ztf.values())  else np.array([])
    all_lsst = np.concatenate([v for v in mjd_obs_lsst.values() if len(v) > 0]) \
               if any(len(v) > 0 for v in mjd_obs_lsst.values()) else np.array([])

    if len(all_ztf) >= 5:
        all_ztf_sorted = np.sort(all_ztf)
        window, best_start, best_count = 90.0, all_ztf_sorted[0], 0
        for t in all_ztf_sorted:
            count = np.sum((all_ztf_sorted >= t) & (all_ztf_sorted <= t + window))
            if count > best_count:
                best_count, best_start = count, t
        t0 = rng.uniform(best_start + 5.0, best_start + window / 2.0)
    elif len(all_lsst) > 0:
        t_start, t_end = all_lsst.min(), all_lsst.max()
        t0 = rng.uniform(t_start + 10.0, t_start + (t_end - t_start) / 3.0)
    else:
        t0 = 59100.0
    model.set(t0=t0)

    x1 = c_val = None
    if class_name == 'SNIa':
        x1    = float(rng.uniform(-2.0, 2.0))
        c_val = float(rng.uniform(-0.2, 0.2))
        model.set(x1=x1, c=c_val)

    _abs_mags = {'SNIa': -19.3, 'SNII': -17.0, 'SNIbc': -17.5, 'SLSN': -21.0}
    _abs_sig  = {'SNIa':  0.3,  'SNII':  0.5,  'SNIbc':  0.5,  'SLSN':  0.5}
    amplitude = None
    try:
        peak_abs = _abs_mags[class_name] + rng.normal(0.0, _abs_sig[class_name])
        model.set_source_peakabsmag(peak_abs, 'bessellb', 'ab')
        amp_key = 'x0' if class_name == 'SNIa' else 'amplitude'
        amplitude = float(model.get(amp_key))
    except Exception:
        _fallback_amp = {'SNIa': 1e-5, 'SNII': 1e-12, 'SNIbc': 1e-12, 'SLSN': 1e-10}
        amp_key = 'x0' if class_name == 'SNIa' else 'amplitude'
        amplitude = float(_fallback_amp[class_name] * rng.uniform(0.5, 2.0))
        model.set(**{amp_key: amplitude})

    def _eval(mjd_arr, band_name):
        if len(mjd_arr) == 0:
            return np.array([], dtype=float)
        mags = np.full(len(mjd_arr), 99.0, dtype=float)
        in_range = (mjd_arr >= model.mintime()) & (mjd_arr <= model.maxtime())
        if in_range.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    mags[in_range] = model.bandmag(band_name, 'ab', mjd_arr[in_range])
                except Exception:
                    pass
        return mags

    ztf_mags  = {fid: _eval(mjd_obs_ztf.get( fid, np.array([])), band) for fid, band in _ZTF_BANDS.items()}
    lsst_mags = {fid: _eval(mjd_obs_lsst.get(fid, np.array([])), band) for fid, band in _LSST_BANDS.items()}

    params = {'z': float(z), 't0': float(t0), 'amplitude': amplitude}
    if x1 is not None:
        params['x1'] = x1
    if c_val is not None:
        params['c'] = c_val

    return {'ztf': ztf_mags, 'lsst': lsst_mags}, params


def simulate_stochastic_with_params(
    class_name: str,
    mjd_obs_ztf:  Dict[int, np.ndarray],
    mjd_obs_lsst: Dict[int, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict]:

    tau_nominal, sigma_nominal, (r_lo, r_hi) = _DRW_PARAMS[class_name]
    colors = COLOR_OFFSETS[class_name]

    mean_r = float(rng.uniform(r_lo, r_hi))   # draw 1 — same order as simulate_stochastic

    all_arrays = [v for v in {**mjd_obs_ztf, **mjd_obs_lsst}.values() if len(v) > 0]
    if not all_arrays:
        params = {'mean_mag_r': mean_r, 'drw_tau': float('nan'), 'drw_sigma': float('nan')}
        return {'ztf': {}, 'lsst': {}}, params
    all_times = np.unique(np.concatenate(all_arrays))

    if class_name == 'Blazar':
        drw_r      = _blazar_lightcurve(all_times, rng)
        tau_draw   = float('nan')
        sigma_draw = sigma_nominal
    elif class_name == 'YSO':
        drw_r      = _yso_lightcurve(all_times, rng)
        tau_draw   = float('nan')
        sigma_draw = sigma_nominal
    elif class_name in ('QSO', 'AGN'):
        mu_log_tau = 2.4 if class_name == 'QSO' else 2.3
        tau_draw   = float(np.clip(10 ** rng.normal(mu_log_tau, 0.4), 20.0, 2000.0))
        sigma_draw = float(np.clip(10 ** rng.normal(-0.9, 0.3), 0.03, 0.8))
        drw_r      = _drw_process(tau_draw, sigma_draw, all_times, rng)
    else:
        tau_draw   = float(tau_nominal)
        sigma_draw = float(sigma_nominal)
        drw_r      = _drw_process(tau_draw, sigma_draw, all_times, rng)

    params = {'mean_mag_r': mean_r, 'drw_tau': tau_draw, 'drw_sigma': sigma_draw}

    def _interp_drw(t_query: np.ndarray) -> np.ndarray:
        return np.interp(t_query, all_times, drw_r)

    _lc_std = float(np.std(drw_r)) if len(drw_r) > 1 else sigma_draw if np.isfinite(sigma_draw) else 0.1

    def _band_mag(t_arr: np.ndarray, fid: int) -> np.ndarray:
        if len(t_arr) == 0:
            return np.array([], dtype=float)
        offset = colors.get(fid, 0.0)
        band_scatter = _lc_std * 0.10 * rng.normal(0.0, 1.0, len(t_arr))
        return mean_r + offset + _interp_drw(t_arr) + band_scatter

    if class_name == 'CV/Nova':
        n_outbursts = int(rng.integers(0, 4))
        if n_outbursts > 0:
            t_bursts     = rng.uniform(all_times.min(), all_times.max(), n_outbursts)
            amp_bursts   = rng.uniform(2.0, 5.0, n_outbursts)
            width_bursts = rng.uniform(5.0, 15.0, n_outbursts)
            def outburst_delta(t_arr: np.ndarray) -> np.ndarray:
                delta = np.zeros(len(t_arr))
                for tb, ab, wb in zip(t_bursts, amp_bursts, width_bursts):
                    delta -= ab * np.exp(-0.5 * ((t_arr - tb) / wb) ** 2)
                return delta
        else:
            def outburst_delta(t_arr: np.ndarray) -> np.ndarray:
                return np.zeros(len(t_arr))
    else:
        def outburst_delta(t_arr: np.ndarray) -> np.ndarray:
            return np.zeros(len(t_arr))

    ztf_mags = {
        fid: _band_mag(mjd_obs_ztf.get(fid, np.array([])), fid)
             + outburst_delta(mjd_obs_ztf.get(fid, np.array([])))
        for fid in _ZTF_BANDS
    }
    lsst_mags = {
        fid: _band_mag(mjd_obs_lsst.get(fid, np.array([])), fid)
             + outburst_delta(mjd_obs_lsst.get(fid, np.array([])))
        for fid in _LSST_BANDS
    }
    return {'ztf': ztf_mags, 'lsst': lsst_mags}, params


def _simulate_periodic_capturing(
    class_name: str,
    mjd_obs_ztf:  Dict[int, np.ndarray],
    mjd_obs_lsst: Dict[int, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict]:
    
    p_lo, p_hi = _PERIODIC_PARAMS[class_name][0]
    a_lo, a_hi = _PERIODIC_PARAMS[class_name][1]
    r_lo, r_hi = _PERIODIC_PARAMS[class_name][3]
    log_period  = _PERIODIC_PARAMS[class_name][5]

    state = rng.bit_generator.state

    if log_period:
        period = float(10 ** rng.uniform(np.log10(p_lo), np.log10(p_hi)))
    else:
        period = float(rng.uniform(p_lo, p_hi))
    mean_r = float(rng.uniform(r_lo, r_hi))
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi))

    if class_name == 'DSCT':
        dsct_roll = rng.random()
        if dsct_roll < 0.20:
            amplitude = float(rng.uniform(0.10, 0.50))
        else:
            amplitude = float(rng.uniform(0.01, 0.10))
    else:
        amplitude = float(rng.uniform(a_lo, a_hi))

    params = {
        'mean_mag_r': mean_r,
        'period':     period,
        'amplitude':  amplitude,
        'phase0':     phase0,
    }

    rng.bit_generator.state = state
    lc = simulate_periodic(class_name, mjd_obs_ztf, mjd_obs_lsst, rng)

    return lc, params