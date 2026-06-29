from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

#%%

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# sgscore1 distributions by class.

_SGSCORE_PARAMS: Dict[str, Tuple[float, float, float, float]] = {
    'SNIa': (0.058, 0.363, 0.000, 0.650),
    'SNII': (0.178, 0.339, 0.000, 0.620),
    'SNIbc': (0.200, 0.337, 0.000, 0.610),
    'SLSN': (0.017, 0.574, 0.000, 1.000),
    'QSO': (0.993, 0.017, 0.510, 1.000),
    'AGN': (0.194, 0.507, 0.000, 1.000),
    'Blazar': (0.985, 0.090, 0.020, 1.000),
    'YSO': (0.978, 0.033, 0.450, 1.000),
    'CV/Nova': (0.984, 0.047, 0.390, 1.000),
    'RRL': (0.993, 0.014, 0.820, 1.000),
    'CEP': (0.983, 0.046, 0.450, 1.000),
    'DSCT': (0.995, 0.010, 0.880, 1.000),
    'LPV': (0.880, 0.359, 0.450, 1.000),
    'E': (0.995, 0.010, 0.890, 1.000),
    'Periodic-Other': (0.994, 0.011, 0.900, 1.000),
}

#rb distribution by class

_RB_PARAMS: Dict[str, Tuple[float, float, float, float]] = {
    'SNIa': (0.870, 0.124, 0.635, 0.970),
    'SNII': (0.889, 0.124, 0.628, 0.971),
    'SNIbc': (0.869, 0.128, 0.640, 0.968),
    'SLSN': (0.835, 0.103, 0.684, 0.947),
    'QSO': (0.881, 0.057, 0.723, 0.957),
    'AGN': (0.887, 0.069, 0.713, 0.967),
    'Blazar': (0.876, 0.062, 0.719, 0.961),
    'YSO': (0.776, 0.086, 0.626, 0.924),
    'CV/Nova': (0.841, 0.070, 0.667, 0.949),
    'RRL': (0.831, 0.066, 0.681, 0.936),
    'CEP': (0.731, 0.078, 0.609, 0.914),
    'DSCT': (0.757, 0.083, 0.613, 0.899),
    'LPV': (0.750, 0.064, 0.627, 0.870),
    'E': (0.780, 0.086, 0.623, 0.916),
    'Periodic-Other': (0.781, 0.081, 0.641, 0.917),
}

_CLASS_CATEGORY: Dict[str, str] = {
    'SNIa': 'transient', 'SNII': 'transient',
    'SNIbc': 'transient', 'SLSN': 'transient',
    'QSO': 'stochastic', 'AGN': 'stochastic',
    'Blazar': 'stochastic', 'YSO': 'stochastic', 'CV/Nova': 'stochastic',
    'RRL': 'periodic', 'CEP': 'periodic', 'DSCT': 'periodic',
    'LPV': 'periodic', 'E': 'periodic', 'Periodic-Other': 'periodic',
}

def sample_sgscore(class_name: str, rng: np.random.Generator) -> float:
    """
    Sample a realistic sgscore1 value for a simulated object.
    
    """
    mean, std, lo, hi = _SGSCORE_PARAMS[class_name]
    for _ in range(50):
        val = rng.normal(mean, std)
        if lo <= val <= hi:
            return float(val)
    return float(np.clip(mean, lo, hi))

def sample_rb(class_name: str, rng: np.random.Generator) -> float:
    """
    Sample a realistic rb value for a simulated object.

    """
    mean, std, lo, hi = _RB_PARAMS[class_name]
    for _ in range(50):
        val = rng.normal(mean, std)
        if lo <= val <= hi:
            return float(val)
    return float(np.clip(mean, lo, hi))


# Upper bound on magnitude uncertainty for a row to count as a detection.
# Epochs with sigmapsf >= this value go to non_detections instead.
_MAX_SIGMAPSF_MAG: float = 1.0

# Detection-table columns
_DET_COLUMNS = [
    'mjd', 'fid', 'magpsf', 'sigmapsf',
    'magpsf_corr', 'sigmapsf_corr', 'sigmapsf_corr_ext',
    'magpsf_ml', 'sigmapsf_ml',
    'ra', 'dec', 'rb', 'sgscore1', 'isdiffpos',
    'tid', 'rbversion', 'step_id_corr',
]
_NDET_COLUMNS = ['mjd', 'fid', 'diffmaglim']


# ---------------------------------------------------------------------------
# Core formatting function
# ---------------------------------------------------------------------------

def _make_detections(
    oid: str,
    obs_per_band: Dict[int, dict],
    mjd_per_band: Dict[int, np.ndarray],
    ra: float,
    dec: float,
    sgscore1: float,
    rb: float,
    tid: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build detections and non-detections DataFrames for one survey from
    the output of survey.observe() / observe_opsim() / observe_ztf().

    """
    det_rows:  list[dict] = []
    ndet_rows: list[dict] = []

    for fid, obs in obs_per_band.items():
        mjds       = mjd_per_band.get(fid, np.array([]))
        magpsf     = obs['magpsf']
        sigmapsf   = obs['sigmapsf']
        detected   = obs['detected']
        diffmaglim = obs['diffmaglim']
        snr_arr    = obs.get('snr', None)

        n = len(mjds)
        if n == 0:
            continue

        for k in range(n):
            mjd = float(mjds[k])
            mag = float(magpsf[k])
            sig = float(sigmapsf[k])
            det = bool(detected[k])
            lim = float(diffmaglim[k])

            is_valid_det = (
                det
                and np.isfinite(mag)
                and np.isfinite(sig)
                and 0.0 < sig < _MAX_SIGMAPSF_MAG
            )

            if is_valid_det:
                if snr_arr is not None:
                    isdiffpos = 1 if float(snr_arr[k]) > 0.0 else -1
                else:
                    isdiffpos = 1

                det_rows.append({
                    'oid':               oid,
                    'mjd':               mjd,
                    'fid':               int(fid),
                    'magpsf':            mag,
                    'sigmapsf':          sig,
                    'magpsf_corr':       mag,
                    'sigmapsf_corr':     sig,
                    'sigmapsf_corr_ext': sig,
                    'magpsf_ml':         mag,
                    'sigmapsf_ml':       sig,
                    'ra':                ra,
                    'dec':               dec,
                    'sgscore1':          sgscore1,
                    'isdiffpos':         isdiffpos,
                    'tid':               tid,
                    'rbversion':         None,
                    'step_id_corr':      None,
                })
            else:

                ndet_rows.append({
                    'oid':       oid,
                    'mjd':       mjd,
                    'fid':       int(fid),
                    'diffmaglim': lim,
                })

    if det_rows:
        det_df = pd.DataFrame(det_rows).set_index('oid')
        det_df = det_df[_DET_COLUMNS]
    else:
        det_df = pd.DataFrame(columns=_DET_COLUMNS)
        det_df.index.name = 'oid'

    if ndet_rows:
        ndet_df = pd.DataFrame(ndet_rows).set_index('oid')
        ndet_df = ndet_df[_NDET_COLUMNS]
    else:
        ndet_df = pd.DataFrame(columns=_NDET_COLUMNS)
        ndet_df.index.name = 'oid'

    return det_df, ndet_df


# ---------------------------------------------------------------------------
# Survey-specific wrappers
# ---------------------------------------------------------------------------

def make_ztf_detections(
    oid: str,
    obs_ztf: Dict[int, dict],
    mjd_ztf: Dict[int, np.ndarray],
    ra: float,
    dec: float,
    class_name: str,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Format ZTF observations for a simulated object.

    """
    if rng is None:
        rng = np.random.default_rng()
    sgscore1 = sample_sgscore(class_name, rng)
    rb = sample_rb(class_name, rng)
    return _make_detections(oid, obs_ztf, mjd_ztf, ra, dec, sgscore1, rb, tid='ztf')


def make_lsst_detections(
    oid: str,
    obs_lsst: Dict[int, dict],
    mjd_lsst: Dict[int, np.ndarray],
    ra: float,
    dec: float,
    class_name: str,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Format LSST observations for a simulated object.

    """
    if rng is None:
        rng = np.random.default_rng()
    sgscore1 = sample_sgscore(class_name, rng)
    rb = sample_rb(class_name, rng)
    return _make_detections(oid, obs_lsst, mjd_lsst, ra, dec, sgscore1, rb, tid='lsst')


# ---------------------------------------------------------------------------
# Object metadata
# ---------------------------------------------------------------------------

def make_object_info(
    oid: str,
    ra: float,
    dec: float,
    seed: Optional[int] = None,
    mjd_start: Optional[float] = None,
    mjd_end: Optional[float] = None,
    model_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Build the object metadata DataFrame (indexed by oid).

    corrected=False tells the preprocessor to use magpsf directly rather
    than magpsf_corr (which would require a host-galaxy reference flux
    not present in simulations).

    """
    row = {'meanra': ra, 'meandec': dec, 'corrected': False}
    if seed is not None:
        row['seed'] = int(seed)
    if mjd_start is not None:
        row['mjd_start'] = float(mjd_start)
    if mjd_end is not None:
        row['mjd_end'] = float(mjd_end)
    if model_params:
        row.update(model_params)
    return pd.DataFrame(
        {k: [v] for k, v in row.items()},
        index=pd.Index([oid], name='oid'),
    )


# ---------------------------------------------------------------------------
# Sky position
# ---------------------------------------------------------------------------

def random_sky_position(
    rng: np.random.Generator,
    dec_lo: float = -30.0,
    dec_hi: float = 30.0,
) -> Tuple[float, float]:
    """
    Draw a uniform random sky position in the ZTF + LSST overlap zone.
    """
    ra  = rng.uniform(0.0, 360.0)
    dec = rng.uniform(dec_lo, dec_hi)
    return float(ra), float(dec)