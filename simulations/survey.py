from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

#%%

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 5-sigma limiting magnitudes (fallback / synthetic mode only)
MAG_LIMITS: Dict[str, Dict[int, float]] = {
    'ztf':  {1: 21.1, 2: 20.9, 3: 20.2},
    'lsst': {6: 23.70, 1: 24.97, 2: 24.52, 3: 24.13, 4: 23.56, 5: 22.55},
}

# Zero-point for AB mag/flux conversion
MAG_ZEROPOINT: float = 31.4

# Scatter in limiting magnitude between visits (synth. mode)
_MAG_LIMIT_SCATTER: float = 0.12

# Minimum SNR required for a detection to be flagged as real.
_DETECTION_SNR_THRESHOLD: float = 3.0

#bands
_ZTF_FILTERCODE_TO_FID: Dict[str, int] = {'zg': 1, 'zr': 2, 'zi': 3}
_LSST_FILTER_TO_FID: Dict[str, int] = {'u': 6, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}


# ---------------------------------------------------------------------------
# Magnitude / flux utilities
# ---------------------------------------------------------------------------

def mag_to_flux(mag: np.ndarray | float) -> np.ndarray:
    """AB magnitude -> flux in nJy."""
    return 10.0 ** (-0.4 * (np.asarray(mag, dtype=float) - MAG_ZEROPOINT))


def flux_to_mag(flux: np.ndarray | float):
    """
    Flux in nJy -> AB magnitude.

    """
    flux = np.asarray(flux, dtype=float)
    valid = flux > 0.0
    mag = np.where(
        valid,
        MAG_ZEROPOINT - 2.5 * np.log10(np.where(valid, flux, 1.0)),
        np.nan,
    )
    return mag, valid


def flux_uncertainty(mag_limit: np.ndarray | float) -> np.ndarray:
    return mag_to_flux(mag_limit) / 5.0


def _apply_noise(
    true_mag: np.ndarray,
    mlim: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Add gaussian noise

    """
    sigma_flux = flux_uncertainty(mlim)
    true_flux  = mag_to_flux(true_mag)

    out_of_model = true_mag >= 98.0
    true_flux = np.where(out_of_model, 0.0, true_flux)

    obs_flux   = true_flux + rng.normal(0.0, sigma_flux)
    obs_mag, valid = flux_to_mag(obs_flux)

    snr = obs_flux / np.maximum(sigma_flux, 1e-30)
    
    sigma_mag = (2.5 / np.log(10.0)) / np.maximum(np.abs(snr), 0.1)
    sigma_mag = np.where(valid, sigma_mag, np.nan)

    detected = valid & (snr > _DETECTION_SNR_THRESHOLD)

    return {
        'magpsf':     np.where(valid, obs_mag, np.nan),
        'sigmapsf':   sigma_mag,
        'detected':   detected,
        'diffmaglim': mlim,
        'snr':        snr,
    }


# ---------------------------------------------------------------------------
# LSST Real mode (OpSim)
# ---------------------------------------------------------------------------

def load_opsim(
    db_path: str | Path,
    ra: float,
    dec: float,
    radius_deg: float = 1.75,
    mjd_start: Optional[float] = None,
    mjd_end: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load LSST OpSim visits near a sky position from an OpSim database
    (e.g. baseline_v5.0.0_10yrs.db).

    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"OpSim database not found: {db_path}")

    ra_lo, ra_hi   = ra  - radius_deg, ra  + radius_deg
    dec_lo, dec_hi = dec - radius_deg, dec + radius_deg

    mjd_clause = ""
    params: dict = {
        'ra_lo': ra_lo, 'ra_hi': ra_hi,
        'dec_lo': dec_lo, 'dec_hi': dec_hi,
    }
    if mjd_start is not None:
        mjd_clause += " AND observationStartMJD >= :mjd_start"
        params['mjd_start'] = mjd_start
    if mjd_end is not None:
        mjd_clause += " AND observationStartMJD <= :mjd_end"
        params['mjd_end'] = mjd_end

    query = f"""
        SELECT
            observationStartMJD,
            filter,
            fiveSigmaDepth,
            seeingFwhmEff,
            skyBrightness,
            airmass,
            fieldRA,
            fieldDec
        FROM observations
        WHERE fieldRA  BETWEEN :ra_lo  AND :ra_hi
          AND fieldDec BETWEEN :dec_lo AND :dec_hi
          {mjd_clause}
        ORDER BY observationStartMJD
    """

    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(query, con, params=params)

    if df.empty:
        warnings.warn(
            f"No OpSim observations found within {radius_deg} deg of "
            f"(RA={ra:.3f}, Dec={dec:.3f}). Check coordinates / DB coverage.",
            RuntimeWarning,
            stacklevel=2,
        )
        return df

    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    pointings = SkyCoord(
        ra=df['fieldRA'].values * u.deg,
        dec=df['fieldDec'].values * u.deg,
    )
    sep = target.separation(pointings).deg
    df = df[sep <= radius_deg].copy()

    df['fid'] = df['filter'].map(_LSST_FILTER_TO_FID)
    df = df.dropna(subset=['fid'])
    df['fid'] = df['fid'].astype(int)

    return df.reset_index(drop=True)


def generate_obs_times_opsim(opsim_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Extract band observation times from an OpSim DataFrame.

    """
    return {
        int(fid): grp['observationStartMJD'].values
        for fid, grp in opsim_df.groupby('fid')
    }


def observe_opsim(
    true_mag_per_band: Dict[int, np.ndarray],
    opsim_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, dict]:
    """
    Apply noise from OpSim to noiseless magnitudes (LSST real mode).

    """
    if rng is None:
        rng = np.random.default_rng()

    result = {}
    for fid, grp in opsim_df.groupby('fid'):
        fid = int(fid)
        if fid not in true_mag_per_band:
            continue

        true_mag = np.asarray(true_mag_per_band[fid], dtype=float)
        mlim     = grp['fiveSigmaDepth'].values

        if len(true_mag) != len(mlim):
            raise ValueError(
                f"fid={fid}: len(true_mag)={len(true_mag)} "
                f"!= len(opsim visits)={len(mlim)}. "
                "Ensure true_mag_per_band was produced with generate_obs_times_opsim()."
            )

        band_result = _apply_noise(true_mag, mlim, rng)

        band_result['seeing'] = grp['seeingFwhmEff'].values
        result[fid] = band_result

    return result


# ---------------------------------------------------------------------------
# ZTF Real mode (IRSA field-epoch log)
# ---------------------------------------------------------------------------

def load_ztf_fieldlog(
    fieldlog_path: str | Path,
    field_id: int,
    mjd_start: Optional[float] = None,
    mjd_end: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load ZTF epoch metadata for a given field from the IRSA field-epoch log.

    """
    fieldlog_path = Path(fieldlog_path)
    if not fieldlog_path.exists():
        raise FileNotFoundError(f"ZTF field log not found: {fieldlog_path}")

    import pyarrow.parquet as pq
    _wanted = ['obsjd', 'field', 'filtercode', 'maglim', 'seeing', 'infobits']
    _available = set(pq.read_schema(fieldlog_path).names)
    _cols = [c for c in _wanted if c in _available]
    df = pd.read_parquet(fieldlog_path, columns=_cols)

    df = df[df['field'] == field_id].copy()

    if df.empty:
        warnings.warn(
            f"ZTF field log contains no rows for field_id={field_id}. "
            "Check that build_ztf_fieldlog.py was run for this field.",
            RuntimeWarning,
            stacklevel=2,
        )
        return df

    # Convert JD to MJD
    df['mjd'] = df['obsjd'] - 2_400_000.5

    if mjd_start is not None:
        df = df[df['mjd'] >= mjd_start]
    if mjd_end is not None:
        df = df[df['mjd'] <= mjd_end]

    # Quality filter
    if 'infobits' in df.columns:
        df = df[(df['infobits'] & 32768) == 0]

    # Map filter code to numeric fid
    df['fid'] = df['filtercode'].map(_ZTF_FILTERCODE_TO_FID)
    df = df.dropna(subset=['fid', 'maglim'])
    df['fid'] = df['fid'].astype(int)

    return df.sort_values('mjd').reset_index(drop=True)


def generate_obs_times_ztf(ztf_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Extract band observation times from a ZTF field-log DataFrame.

    """
    return {
        int(fid): grp['mjd'].values
        for fid, grp in ztf_df.groupby('fid')
    }


def observe_ztf(
    true_mag_per_band: Dict[int, np.ndarray],
    ztf_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, dict]:
    """
    Apply noise from the ZTF field log to noiseless magnitudes.

    """
    if rng is None:
        rng = np.random.default_rng()

    result = {}
    for fid, grp in ztf_df.groupby('fid'):
        fid = int(fid)
        if fid not in true_mag_per_band:
            continue

        true_mag = np.asarray(true_mag_per_band[fid], dtype=float)
        mlim     = grp['maglim'].values

        if len(true_mag) != len(mlim):
            raise ValueError(
                f"fid={fid}: len(true_mag)={len(true_mag)} "
                f"!= len(ztf epochs)={len(mlim)}. "
                "Ensure true_mag_per_band was produced with generate_obs_times_ztf()."
            )

        band_result = _apply_noise(true_mag, mlim, rng)
        if 'seeing' in grp.columns:
            band_result['seeing'] = grp['seeing'].values
        result[fid] = band_result

    return result


# ---------------------------------------------------------------------------
# Synthetic fallback mode
# ---------------------------------------------------------------------------

def generate_obs_times(
    survey: str,
    mjd_start: float,
    mjd_end: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, np.ndarray]:
    """
    Generate synthetic observation times per band (fallback mode).

    """
    if rng is None:
        rng = np.random.default_rng()

    duration = mjd_end - mjd_start

    if survey == 'ztf':
        return _ztf_cadence(mjd_start, duration, rng)
    elif survey == 'lsst':
        return _lsst_cadence(mjd_start, duration, rng)
    else:
        raise ValueError(f"Unknown survey: {survey!r}. Expected 'ztf' or 'lsst'.")


def _ztf_cadence(
    mjd_start: float,
    duration: float,
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    """
    Synthetic ZTF survey cadence.

    """
    cadence_gr = 3.0   
    cadence_i  = 5.0   

    def _night_centres(cadence, jitter):
        n = int(duration / cadence) + 1
        t = mjd_start + np.arange(n) * cadence + rng.uniform(-jitter, jitter, n)
        return t[(t >= mjd_start) & (t <= mjd_start + duration)]

    t_gr = _night_centres(cadence_gr, 0.5)
    t_i  = _night_centres(cadence_i,  0.8)

    def two_visits(t_nights: np.ndarray) -> np.ndarray:
        keep = rng.random(len(t_nights)) > 0.15
        t = t_nights[keep]
        offset = rng.uniform(0.01, 0.04, len(t))
        return np.sort(np.concatenate([t, t + offset]))

    return {
        1: two_visits(t_gr),   # g  
        2: two_visits(t_gr),   # r
        3: two_visits(t_i),    # i
    }


def _lsst_cadence(
    mjd_start: float,
    duration: float,
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    """
    Synthetic LSST WFD cadence (approximate baseline v5 behaviour).
    
    """
    filter_cycle = [
        [1, 2],   
        [3, 4],  
        [2, 5],   
        [1, 3],   
        [4, 2],   
        [5, 3],  
    ]
    u_cadence = 15.0

    n_nights = int(duration) + 1
    obs: Dict[int, list] = {fid: [] for fid in [6, 1, 2, 3, 4, 5]}

    for night in range(n_nights):
        mjd_night = mjd_start + night + rng.uniform(-0.1, 0.1)
        if mjd_night > mjd_start + duration:
            break
        if rng.random() < 0.20:
            continue

        for fid in filter_cycle[night % len(filter_cycle)]:
            t1 = mjd_night + rng.uniform(0.0, 0.02)
            t2 = t1 + rng.uniform(0.010, 0.025)
            obs[fid].extend([t1, t2])

    n_u = int(duration / u_cadence) + 1
    t_u = mjd_start + np.arange(n_u) * u_cadence + rng.uniform(-1.0, 1.0, n_u)
    t_u = t_u[(t_u >= mjd_start) & (t_u <= mjd_start + duration)]
    for t in t_u:
        obs[6].extend([t, t + rng.uniform(0.010, 0.025)])

    return {
        fid: np.sort(np.array(v))
        for fid, v in obs.items()
        if len(v) > 0
    }


def observe(
    true_mag_per_band: Dict[int, np.ndarray],
    survey: str,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, dict]:
    """
    Apply synthetic noise model to noiseless magnitudes (fallback mode).

    """
    if rng is None:
        rng = np.random.default_rng()

    limits = MAG_LIMITS[survey]
    result = {}

    for fid, true_mag in true_mag_per_band.items():
        if fid not in limits:
            continue
        true_mag = np.asarray(true_mag, dtype=float)
        n = len(true_mag)

        # Per-visit depth with scatter (synthetic mode only)
        mlim = limits[fid] + rng.normal(0.0, _MAG_LIMIT_SCATTER, n)
        result[fid] = _apply_noise(true_mag, mlim, rng)

    return result