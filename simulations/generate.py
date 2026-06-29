from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
np.NaN = np.nan

import pandas as pd
from scipy.optimize import OptimizeWarning
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=np.RankWarning)
warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger().setLevel(logging.CRITICAL)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'lc_classifier'))

from lc_classifier.features import LSSTLightcurvePreprocessor, LSSTFeatureExtractor
from lc_classifier.features import ZTFFeatureExtractor3bands

from simulations.features_config import postprocess_features

from simulations.survey    import (
    generate_obs_times, observe,
    load_ztf_fieldlog, generate_obs_times_ztf, observe_ztf,
    load_opsim, generate_obs_times_opsim, observe_opsim,
)
from simulations.models    import (
    simulate_transient,  simulate_stochastic,  simulate_periodic,
    simulate_transient_with_params, simulate_stochastic_with_params,
    _simulate_periodic_capturing,
)
from simulations.formatter import (
    make_ztf_detections, make_lsst_detections,
    make_object_info, random_sky_position,
)

#%%

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLASSES: Dict[str, str] = {
    # Transients
    'SNIa':           'transient',
    'SNII':           'transient',
    'SNIbc':          'transient',
    'SLSN':           'transient',
    # Stochastic
    'QSO':            'stochastic',
    'AGN':            'stochastic',
    'Blazar':         'stochastic',
    'YSO':            'stochastic',
    'CV/Nova':        'stochastic',
    # Periodic
    'RRL':            'periodic',
    'CEP':            'periodic',
    'DSCT':           'periodic',
    'LPV':            'periodic',
    'E':              'periodic',
    'Periodic-Other': 'periodic',
}

# Reference epoch for simulated objects.
T0_SIM:     float = 59000.0   # arbitrary reference MJD
SIM_WINDOW: float = 730.0     # 2-year baseline

MJD_START: float = T0_SIM
MJD_END:   float = T0_SIM + SIM_WINDOW

_ZTF_MJD_MIN: float = 58197.8   # start of ZTF survey (Mar 2018)
_ZTF_MJD_MAX: float = 60675.1   # end of available fieldlog
_OPS_MJD_MIN: float = 60980.0   # start of OpSim (Rubin first light ~2025)
_OPS_MJD_MAX: float = 64632.3   # end of OpSim (10-yr survey)


# ---------------------------------------------------------------------------
# Light-curve truncation parameters
# ---------------------------------------------------------------------------

_TRUNCATION_PARAMS: Dict[str, Tuple[float, float, float]] = {
    'transient':  (0.7, 0.15, 0.95),
    'stochastic': (0.4, 0.30, 0.95),
    'periodic':   (0.4, 0.30, 0.95),
}

_CLASS_CATEGORY_TRUNC: Dict[str, str] = {
    'SNIa': 'transient', 'SNII': 'transient',
    'SNIbc': 'transient', 'SLSN': 'transient',
    'QSO': 'stochastic', 'AGN': 'stochastic',
    'Blazar': 'stochastic', 'YSO': 'stochastic', 'CV/Nova': 'stochastic',
    'RRL': 'periodic', 'CEP': 'periodic', 'DSCT': 'periodic',
    'LPV': 'periodic', 'E': 'periodic', 'Periodic-Other': 'periodic',
}


def _truncate_survey(
    det: pd.DataFrame,
    ndet: pd.DataFrame,
    p_trunc: float,
    frac_lo: float,
    frac_hi: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:

    if det.empty or rng.random() > p_trunc:
        return det, ndet, 1.0

    t_min = det['mjd'].min()
    t_max = det['mjd'].max()
    if t_max <= t_min:
        return det, ndet, 1.0

    frac   = float(rng.uniform(frac_lo, frac_hi))
    cutoff = t_min + frac * (t_max - t_min)

    det_out  = det[det['mjd'] <= cutoff]
    ndet_out = ndet[ndet['mjd'] <= cutoff] if not ndet.empty else ndet

    # Guard: keep at least 6 detections (minimum for preprocessor)
    if len(det_out) < 6:
        return det, ndet, 1.0

    return det_out, ndet_out, frac


def truncate_detections(
    det_ztf:   pd.DataFrame,
    det_lsst:  pd.DataFrame,
    ndet_ztf:  pd.DataFrame,
    ndet_lsst: pd.DataFrame,
    class_name: str,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:

    category = _CLASS_CATEGORY_TRUNC[class_name]
    p_trunc, frac_lo, frac_hi = _TRUNCATION_PARAMS[category]

    det_z_out,  ndet_z_out,  frac_z = _truncate_survey(
        det_ztf,  ndet_ztf,  p_trunc, frac_lo, frac_hi, rng)
    det_l_out,  ndet_l_out,  frac_l = _truncate_survey(
        det_lsst, ndet_lsst, p_trunc, frac_lo, frac_hi, rng)

    return det_z_out, det_l_out, ndet_z_out, ndet_l_out, frac_z, frac_l


# ---------------------------------------------------------------------------
# Detection filter (based on preprocessor filters)
# ---------------------------------------------------------------------------

def _passes_preprocessor_filter(det: pd.DataFrame, min_dets: int = 5) -> bool:

    if det.empty:
        return False
    return any(
        (det['fid'] == fid).sum() > min_dets
        for fid in det['fid'].unique()
    )


# ---------------------------------------------------------------------------
# ZTF cadence
# ---------------------------------------------------------------------------

class ZTFCadenceSelector:
    """
    Selects real ZTF cadence from the IRSA field log for a sky position.

    """

    def __init__(
        self,
        fieldlog_path: str,
        min_visits_per_year: float = 30.0,
    ) -> None:
        from ztfquery.fields import get_fields_containing_target

        self._get_fields = get_fields_containing_target
        self._min_vpyr   = min_visits_per_year

        print(f'Loading ZTF field log: {fieldlog_path}')
        self._log = pd.read_parquet(fieldlog_path)
        self._log['mjd'] = self._log['obsjd'] - 2_400_000.5

        span_years = (self._log['mjd'].max() - self._log['mjd'].min()) / 365.25
        zr = self._log[self._log['filtercode'] == 'zr']
        self._vpyr: Dict[int, float] = (
            zr.groupby('field').size() / span_years
        ).to_dict()

        n_good = sum(1 for v in self._vpyr.values() if v >= min_visits_per_year)
        print(f'Fields in log: {len(self._vpyr)}, '
              f'well-sampled (≥{min_visits_per_year} zr/yr): {n_good}')

    def get(
        self,
        ra: float,
        dec: float,
        mjd_start: float,
        mjd_end: float,
    ) -> Optional[Tuple[Dict[int, np.ndarray], pd.DataFrame]]:

        try:
            fields = self._get_fields(ra, dec)
        except Exception:
            return None

        best_field = None
        best_vpyr  = 0.0
        for fid in fields:
            vpyr = self._vpyr.get(int(fid), 0.0)
            if vpyr >= self._min_vpyr and vpyr > best_vpyr:
                best_field = int(fid)
                best_vpyr  = vpyr

        if best_field is None:
            return None

        mask = (
            (self._log['field'] == best_field) &
            (self._log['mjd']   >= mjd_start)  &
            (self._log['mjd']   <= mjd_end)
        )
        df = self._log[mask].copy()

        if df.empty:
            return None

        _FILTERCODE_TO_FID = {'zg': 1, 'zr': 2, 'zi': 3}
        df['fid'] = df['filtercode'].map(_FILTERCODE_TO_FID)
        df = df.dropna(subset=['fid'])
        df['fid'] = df['fid'].astype(int)

        mjd_ztf = {
            int(fid): grp['mjd'].values
            for fid, grp in df.groupby('fid')
        }
        return mjd_ztf, df


# ---------------------------------------------------------------------------
# LSST cadence
# ---------------------------------------------------------------------------

class LSSTCadenceProvider:
    """
    Provides real LSST cadence and visit depths from an OpSim database
    for a sky position.

    """

    def __init__(
        self,
        opsim_path: str,
        radius_deg: float = 1.75,
    ) -> None:
        if not os.path.exists(opsim_path):
            raise FileNotFoundError(
                f"OpSim database not found: {opsim_path}"
            )
        self._opsim_path = opsim_path
        self._radius_deg = radius_deg
        print(f'LSST OpSim: {opsim_path}  (radius={radius_deg}°)')

    def get(
        self,
        ra: float,
        dec: float,
        mjd_start: float,
        mjd_end: float,
    ) -> Optional[Tuple[Dict[int, np.ndarray], pd.DataFrame]]:

        try:
            df = load_opsim(
                self._opsim_path,
                ra=ra,
                dec=dec,
                radius_deg=self._radius_deg,
                mjd_start=mjd_start,
                mjd_end=mjd_end,
            )
        except Exception:
            return None

        if df.empty:
            return None

        mjd_lsst = generate_obs_times_opsim(df)
        if not mjd_lsst:
            return None

        return mjd_lsst, df


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(ckpt_dir: str, kind: str, class_name: str, idx: int) -> str:
    """Build a deterministic checkpoint file path."""
    safe = class_name.replace('/', '_')
    return os.path.join(ckpt_dir, f'{kind}_{safe}_{idx:06d}.parquet')


def _save_checkpoint(
    ckpt_dir: str,
    class_name: str,
    det_ztf_list:  List[pd.DataFrame],
    ndet_ztf_list: List[pd.DataFrame],
    det_lsst_list:  List[pd.DataFrame],
    ndet_lsst_list: List[pd.DataFrame],
    obj_list:      List[pd.DataFrame],
    idx: int,
) -> None:
    
    def _flush(frames: List[pd.DataFrame], path: str) -> None:
        valid = [f for f in frames if f is not None and not f.empty]
        if valid:
            pd.concat(valid).to_parquet(path)

    _flush(det_ztf_list,   _checkpoint_path(ckpt_dir, 'det_ztf',   class_name, idx))
    _flush(ndet_ztf_list,  _checkpoint_path(ckpt_dir, 'ndet_ztf',  class_name, idx))
    _flush(det_lsst_list,  _checkpoint_path(ckpt_dir, 'det_lsst',  class_name, idx))
    _flush(ndet_lsst_list, _checkpoint_path(ckpt_dir, 'ndet_lsst', class_name, idx))
    _flush(obj_list,       _checkpoint_path(ckpt_dir, 'obj',       class_name, idx))


def _load_checkpoints(
    ckpt_dir: str,
    class_name: str,
) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame],
    Optional[pd.DataFrame], Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
   
    safe = class_name.replace('/', '_')

    def _load(kind: str) -> Optional[pd.DataFrame]:
        files = sorted(glob.glob(
            os.path.join(ckpt_dir, f'{kind}_{safe}_*.parquet')
        ))
        if not files:
            return None
        return pd.concat([pd.read_parquet(f) for f in files])

    return (
        _load('det_ztf'),
        _load('ndet_ztf'),
        _load('det_lsst'),
        _load('ndet_lsst'),
        _load('obj'),
    )


def _done_oids_for_class(ckpt_dir: str, class_name: str) -> Set[str]:
    
    safe  = class_name.replace('/', '_')
    files = sorted(glob.glob(os.path.join(ckpt_dir, f'det_ztf_{safe}_*.parquet')))
    if not files:
        return set()
    oids: Set[str] = set()
    for f in files:
        try:
            oids.update(pd.read_parquet(f, columns=[]).index.unique())
        except Exception:
            pass
    return oids


def _save_features_checkpoint(
    ckpt_dir: str,
    features: Dict[str, pd.DataFrame],
    idx: int,
) -> None:
    for key, df in features.items():
        if df is not None and not df.empty:
            path = os.path.join(ckpt_dir, f'features_{key}_{idx:06d}.parquet')
            df.to_parquet(path)


def _load_features_checkpoints(ckpt_dir: str) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for key in ('ztf', 'lsst', 'combined'):
        files = sorted(glob.glob(os.path.join(ckpt_dir, f'features_{key}_*.parquet')))
        if files:
            result[key] = pd.concat([pd.read_parquet(f) for f in files])
        else:
            result[key] = pd.DataFrame()
    return result


# ---------------------------------------------------------------------------
# Sky position sampling restricted to ZTF and LSST overlap
# ---------------------------------------------------------------------------

def covered_sky_position(
    rng: np.random.Generator,
    ztf_selector:  'ZTFCadenceSelector',
    lsst_provider: 'LSSTCadenceProvider',
    max_tries: int = 30,
) -> Optional[Tuple[float, float, object, object]]:
    """
    Sample (ra, dec) within the ZTF/LSST WFD overlap zone and return
    real cadence data from both surveys remapped onto a common
    timeline [T0_SIM, T0_SIM + SIM_WINDOW].
    """
    DEC_LO, DEC_HI = -30.0, +5.0

    for _ in range(max_tries):
        ra  = float(rng.uniform(0.0, 360.0))
        dec = float(rng.uniform(DEC_LO, DEC_HI))

        ztf_ws  = float(rng.uniform(_ZTF_MJD_MIN, _ZTF_MJD_MAX - SIM_WINDOW))
        ztf_we  = ztf_ws + SIM_WINDOW
        ops_ws  = float(rng.uniform(_OPS_MJD_MIN, _OPS_MJD_MAX - SIM_WINDOW))
        ops_we  = ops_ws + SIM_WINDOW

        ztf_result  = ztf_selector.get(ra, dec, ztf_ws,  ztf_we)
        lsst_result = lsst_provider.get(ra, dec, ops_ws,  ops_we)

        if ztf_result is None or lsst_result is None:
            continue

        ztf_mjd_raw,  ztf_df  = ztf_result
        lsst_mjd_raw, lsst_df = lsst_result

        ztf_mjd_remapped = {
            fid: T0_SIM + (t - ztf_ws)
            for fid, t in ztf_mjd_raw.items()
        }
        lsst_mjd_remapped = {
            fid: T0_SIM + (t - ops_ws)
            for fid, t in lsst_mjd_raw.items()
        }

        ztf_df  = ztf_df.copy()
        lsst_df = lsst_df.copy()
        ztf_df['mjd']                    = ztf_df['mjd']                    - ztf_ws  + T0_SIM
        lsst_df['observationStartMJD']   = lsst_df['observationStartMJD']   - ops_ws  + T0_SIM

        return ra, dec, (ztf_mjd_remapped, ztf_df), (lsst_mjd_remapped, lsst_df)

    return None


# ---------------------------------------------------------------------------
# Object simulation
# ---------------------------------------------------------------------------

def simulate_object(
    oid: str,
    class_name: str,
    category: str,
    rng: np.random.Generator,
    ztf_selector:   Optional['ZTFCadenceSelector']  = None,
    lsst_provider:  Optional['LSSTCadenceProvider'] = None,
) -> Optional[Tuple[
    pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame,
    pd.DataFrame,
]]:
    """
    Simulate one ZTF + LSST object.

    """
    #Sky position + cadence
    if ztf_selector is not None and lsst_provider is not None:
        covered = covered_sky_position(rng, ztf_selector, lsst_provider)
        if covered is None:
            return None
        ra, dec, ztf_real, lsst_real = covered
        mjd_ztf,  ztf_df  = ztf_real;  ztf_mode  = 'real'
        mjd_lsst, lsst_df = lsst_real; lsst_mode = 'real'
    else:
        ra, dec = random_sky_position(rng)

        if ztf_selector is not None:
            ztf_real = ztf_selector.get(ra, dec, MJD_START, MJD_END)
        else:
            ztf_real = None

        if ztf_real is not None:
            mjd_ztf, ztf_df = ztf_real; ztf_mode = 'real'
        else:
            mjd_ztf  = generate_obs_times('ztf', MJD_START, MJD_END, rng)
            ztf_df   = None; ztf_mode = 'synthetic'

        if lsst_provider is not None:
            lsst_real = lsst_provider.get(ra, dec, MJD_START, MJD_END)
        else:
            lsst_real = None

        if lsst_real is not None:
            mjd_lsst, lsst_df = lsst_real; lsst_mode = 'real'
        else:
            mjd_lsst  = generate_obs_times('lsst', MJD_START, MJD_END, rng)
            lsst_df   = None; lsst_mode = 'synthetic'

    #Generate noiseless light curves
    try:
        if category == 'transient':
            lc, model_params = simulate_transient_with_params(  class_name, mjd_ztf, mjd_lsst, rng)
        elif category == 'stochastic':
            lc, model_params = simulate_stochastic_with_params( class_name, mjd_ztf, mjd_lsst, rng)
        else:
            lc, model_params = _simulate_periodic_capturing(    class_name, mjd_ztf, mjd_lsst, rng)
    except Exception:
        return None

    #Apply survey noise
    try:
        if ztf_mode == 'real':
            obs_ztf = observe_ztf(lc['ztf'], ztf_df, rng)
        else:
            obs_ztf = observe(lc['ztf'], 'ztf', rng)

        if lsst_mode == 'real':
            obs_lsst = observe_opsim(lc['lsst'], lsst_df, rng)
        else:
            obs_lsst = observe(lc['lsst'], 'lsst', rng)
    except Exception:
        return None

    #Format into DataFrames
    det_ztf,  ndet_ztf  = make_ztf_detections( oid, obs_ztf,  mjd_ztf,  ra, dec, class_name, rng)
    det_lsst, ndet_lsst = make_lsst_detections(oid, obs_lsst, mjd_lsst, ra, dec, class_name, rng)

    #Truncation
    det_ztf, det_lsst, ndet_ztf, ndet_lsst, frac_ztf, frac_lsst = truncate_detections(
        det_ztf, det_lsst, ndet_ztf, ndet_lsst, class_name, rng
    )

    # Determine mjd_start/end
    all_mjds = [
        t for band_dict in [mjd_ztf, mjd_lsst]
        for t in band_dict.values() if len(t) > 0
    ]
    mjd_start_obj = float(min(v.min() for v in all_mjds)) if all_mjds else T0_SIM
    mjd_end_obj   = float(max(v.max() for v in all_mjds)) if all_mjds else T0_SIM + SIM_WINDOW

    # Extract the integer seed from the rng state for reproducibility.
    obj_seed = abs(hash(oid)) % (2**31)

    obj_info = make_object_info(
        oid, ra, dec,
        seed=obj_seed,
        mjd_start=mjd_start_obj,
        mjd_end=mjd_end_obj,
        model_params=model_params,
    )
    obj_info['ztf_mode']   = ztf_mode
    obj_info['lsst_mode']  = lsst_mode
    obj_info['frac_ztf']   = frac_ztf    
    obj_info['frac_lsst']  = frac_lsst   

    return det_ztf, ndet_ztf, det_lsst, ndet_lsst, obj_info


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    detections:     pd.DataFrame,
    non_detections: pd.DataFrame,
    object_info:    pd.DataFrame,
    preprocessor:   Optional[LSSTLightcurvePreprocessor] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LSSTLightcurvePreprocessor on the combined detections
    """
    if preprocessor is None:
        preprocessor = LSSTLightcurvePreprocessor(stream=False)
    det_pp  = preprocessor.preprocess(detections, objects=object_info)
    ndet_pp = preprocessor.rename_columns_non_detections(non_detections)
    return det_pp, ndet_pp


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_for_object(
    oid:            str,
    det_pp:         pd.DataFrame,
    ndet_ztf_pp:    pd.DataFrame,
    ndet_lsst_pp:   pd.DataFrame,
    extractor_ztf:  ZTFFeatureExtractor3bands,
    extractor_lsst: LSSTFeatureExtractor,
    extractor_comb: LSSTFeatureExtractor,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """
    Extract ZTF, LSST, and combined features for one object.

    """
    det_oid = det_pp[det_pp.index == oid]

    det_ztf_oid  = det_oid[det_oid['tid'] == 'ztf']
    det_lsst_oid = det_oid[det_oid['tid'] == 'lsst']

    ndet_ztf_oid  = ndet_ztf_pp[ndet_ztf_pp.index  == oid]
    ndet_lsst_oid = ndet_lsst_pp[ndet_lsst_pp.index == oid]

    ndet_comb_oid = pd.concat([ndet_ztf_oid, ndet_lsst_oid])

    feat_ztf = feat_lsst = feat_comb = None

    if not det_ztf_oid.empty:
        try:
            feat_ztf = extractor_ztf.compute_features(
                detections=det_ztf_oid, non_detections=ndet_ztf_oid)
            feat_ztf.columns = [f'{c}_ztf' for c in feat_ztf.columns]
        except Exception:
            pass

    if not det_lsst_oid.empty:
        try:
            feat_lsst = extractor_lsst.compute_features(
                detections=det_lsst_oid, non_detections=ndet_lsst_oid)
            feat_lsst.columns = [f'{c}_lsst' for c in feat_lsst.columns]
        except Exception:
            pass

    if not det_oid.empty:
        try:
            feat_comb = extractor_comb.compute_features(
                detections=det_oid, non_detections=ndet_comb_oid)
        except Exception:
            pass

    return feat_ztf, feat_lsst, feat_comb


# ---------------------------------------------------------------------------
# Parallel feature extraction
# ---------------------------------------------------------------------------

def _extract_worker(args: tuple) -> tuple:

    import warnings
    import logging
    import numpy as np
    import pandas as pd
    from scipy.optimize import OptimizeWarning

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=OptimizeWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    try:
        warnings.filterwarnings('ignore', category=np.exceptions.RankWarning)
    except AttributeError:
        pass
    logging.getLogger().setLevel(logging.CRITICAL)

    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, os.path.join(_ROOT, 'lc_classifier'))
    np.NaN = np.nan

    from lc_classifier.features import (
        LSSTFeatureExtractor, ZTFFeatureExtractor3bands,
    )

    (oid,
     det_records,    det_idx,
     ndet_z_records, ndet_z_idx,
     ndet_l_records, ndet_l_idx) = args

    def _rebuild(records, idx, name='oid'):
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(records)
        df.index = pd.Index(idx, name=name)
        return df

    det_pp       = _rebuild(det_records,    det_idx)
    ndet_ztf_pp  = _rebuild(ndet_z_records, ndet_z_idx)
    ndet_lsst_pp = _rebuild(ndet_l_records, ndet_l_idx)

    extractor_ztf  = ZTFFeatureExtractor3bands(bands=(1, 2, 3), stream=False)
    extractor_lsst = LSSTFeatureExtractor(bands=(1, 2, 3, 4, 5, 6), stream=False)
    extractor_comb = LSSTFeatureExtractor(bands=(1, 2, 3, 4, 5, 6), stream=False)

    try:
        fz, fl, fc = extract_features_for_object(
            oid, det_pp, ndet_ztf_pp, ndet_lsst_pp,
            extractor_ztf, extractor_lsst, extractor_comb,
        )
    except Exception:
        fz = fl = fc = None

    return oid, fz, fl, fc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    n_per_class:         int,
    out_dir:             str,
    seed:                int,
    checkpoint_n:        int,
    fieldlog_path:       Optional[str] = None,
    min_visits_per_year: float = 30.0,
    opsim_path:          Optional[str] = None,
    n_jobs:              int = 1,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    #ZTF cadence (optional real mode)
    ztf_selector: Optional[ZTFCadenceSelector] = None
    if fieldlog_path is not None:
        if not os.path.exists(fieldlog_path):
            raise FileNotFoundError(
                f"ZTF field log not found: {fieldlog_path}\n"
                "Run build_ztf_fieldlog.py first."
            )
        ztf_selector = ZTFCadenceSelector(
            fieldlog_path, min_visits_per_year=min_visits_per_year
        )

    #LSST cadence (optional real mode)
    lsst_provider: Optional[LSSTCadenceProvider] = None
    if opsim_path is not None:
        lsst_provider = LSSTCadenceProvider(opsim_path)

    #Phase 1: generate raw detections
    print('Phase 1: Generating simulated observations...')

    all_det_ztf:   List[pd.DataFrame] = []
    all_ndet_ztf:  List[pd.DataFrame] = []
    all_det_lsst:  List[pd.DataFrame] = []
    all_ndet_lsst: List[pd.DataFrame] = []
    all_obj:       List[pd.DataFrame] = []
    labels:        List[dict]          = []

    for class_name, category in CLASSES.items():
        done_oids = _done_oids_for_class(ckpt_dir, class_name)
        n_done    = len(done_oids)

        if n_done >= n_per_class:
            print(f'{class_name}: already completed ({n_done} objects in checkpoint)')
            prev = _load_checkpoints(ckpt_dir, class_name)
            if prev[0] is not None: all_det_ztf.append(prev[0])
            if prev[1] is not None: all_ndet_ztf.append(prev[1])
            if prev[2] is not None: all_det_lsst.append(prev[2])
            if prev[3] is not None: all_ndet_lsst.append(prev[3])
            if prev[4] is not None: all_obj.append(prev[4])
            for oid in sorted(done_oids):
                labels.append({'oid': oid, 'classALeRCE': class_name})
            continue

        print(f'{class_name}: {n_done} in checkpoint, generating '
              f'{n_per_class - n_done} more...')
        

        cls_det_ztf:   List[pd.DataFrame] = []
        cls_ndet_ztf:  List[pd.DataFrame] = []
        cls_det_lsst:  List[pd.DataFrame] = []
        cls_ndet_lsst: List[pd.DataFrame] = []
        cls_obj:       List[pd.DataFrame] = []

        n_ok           = 0   
        n_tried        = 0
        n_fail_coverage = 0  
        n_fail_filter   = 0
        n_total        = n_per_class - n_done

        pbar = tqdm(total=n_total, desc=f'  {class_name}', leave=False)

        while n_ok < n_total:
            oid = f'sim_{class_name}_{n_done + n_ok:05d}'.replace('/', '_')
            result = simulate_object(oid, class_name, category, rng, ztf_selector, lsst_provider)
            n_tried += 1

            if result is None:
                n_fail_coverage += 1
                continue

            det_z, ndet_z, det_l, ndet_l, obj_info = result

            if not _passes_preprocessor_filter(det_z) or \
               not _passes_preprocessor_filter(det_l):
                n_fail_filter += 1
                continue

            cls_det_ztf.append(det_z)
            cls_ndet_ztf.append(ndet_z)
            cls_det_lsst.append(det_l)
            cls_ndet_lsst.append(ndet_l)
            cls_obj.append(obj_info)
            labels.append({'oid': oid, 'classALeRCE': class_name})
            n_ok += 1
            pbar.update(1)

            if n_ok % checkpoint_n == 0:
                _save_checkpoint(
                    ckpt_dir, class_name,
                    cls_det_ztf, cls_ndet_ztf,
                    cls_det_lsst, cls_ndet_lsst,
                    cls_obj, n_done + n_ok,
                )
                cls_det_ztf   = []
                cls_ndet_ztf  = []
                cls_det_lsst  = []
                cls_ndet_lsst = []
                cls_obj       = []

        pbar.close()


        if cls_det_ztf:
            _save_checkpoint(
                ckpt_dir, class_name,
                cls_det_ztf, cls_ndet_ztf,
                cls_det_lsst, cls_ndet_lsst,
                cls_obj, n_done + n_ok,
            )

        prev = _load_checkpoints(ckpt_dir, class_name)
        if prev[0] is not None: all_det_ztf.append(prev[0])
        if prev[1] is not None: all_ndet_ztf.append(prev[1])
        if prev[2] is not None: all_det_lsst.append(prev[2])
        if prev[3] is not None: all_ndet_lsst.append(prev[3])
        if prev[4] is not None: all_obj.append(prev[4])

        if n_tried > n_total * 3:
            print(f' Warning: {n_tried} attempts for {n_ok} successes in {class_name} '
                  f'(success rate: {100*n_ok/n_tried:.0f}%)')
            print(f'No-coverage rejections : {n_fail_coverage} '
                  f'({100*n_fail_coverage/n_tried:.0f}%) '
                  f'-> raise --min-visits-per-year to reduce')
            print(f'Few-detections rejections: {n_fail_filter} '
                  f'({100*n_fail_filter/n_tried:.0f}%) '
                  f'-> adjust _TRUNCATION_PARAMS min_fraction to reduce')
            if n_tried > n_total * 10:
                print(f'⚠  Very low success rate -> Check fieldlog dates '
                      f'and simulation time window')

        if cls_obj:
            obj_df = pd.concat(cls_obj)
            if ztf_selector is not None and 'ztf_mode' in obj_df.columns:
                zm = obj_df['ztf_mode'].value_counts()
                print(f'ZTF  mode: {zm.get("real", 0)} real '
                      f'({100*zm.get("real",0)/n_ok:.0f}%), '
                      f'{zm.get("synthetic", 0)} synthetic')
            if lsst_provider is not None and 'lsst_mode' in obj_df.columns:
                lm = obj_df['lsst_mode'].value_counts()
                print(f'LSST mode: {lm.get("real", 0)} real '
                      f'({100*lm.get("real",0)/n_ok:.0f}%), '
                      f'{lm.get("synthetic", 0)} synthetic')

    #Merge all surveys
    print('Combining ZTF + LSST detections...')
    det_ztf_all  = pd.concat(all_det_ztf)
    ndet_ztf_all = pd.concat([f for f in all_ndet_ztf  if not f.empty]) \
                   if any(not f.empty for f in all_ndet_ztf)  else pd.DataFrame()
    det_lsst_all  = pd.concat(all_det_lsst)
    ndet_lsst_all = pd.concat([f for f in all_ndet_lsst if not f.empty]) \
                    if any(not f.empty for f in all_ndet_lsst) else pd.DataFrame()

    detections_all     = pd.concat([det_ztf_all, det_lsst_all]).sort_index()
    object_info_all    = pd.concat(all_obj)
    object_info_all    = object_info_all.groupby(object_info_all.index).first()

    #Phase 2: preprocess
    print('Phase 2: Preprocessing...')
    preprocessor = LSSTLightcurvePreprocessor(stream=False)

    import io, contextlib
    _devnull = contextlib.redirect_stdout(io.StringIO())

    # Preprocess detections
    with _devnull:
        det_pp, _ = preprocess(detections_all,
                           pd.concat([ndet_ztf_all, ndet_lsst_all])
                           if not (ndet_ztf_all.empty and ndet_lsst_all.empty)
                           else pd.DataFrame(),
                           object_info_all, preprocessor)

    # Preprocess non-detections
    with _devnull:
        _, ndet_ztf_pp  = preprocess(det_ztf_all,  ndet_ztf_all,
                                     object_info_all, preprocessor) \
                          if not ndet_ztf_all.empty  else (None, pd.DataFrame())
        _, ndet_lsst_pp = preprocess(det_lsst_all, ndet_lsst_all,
                                     object_info_all, preprocessor) \
                          if not ndet_lsst_all.empty else (None, pd.DataFrame())

    oids_with_ztf  = set(det_pp[det_pp['tid'] == 'ztf'].index.unique())
    oids_with_lsst = set(det_pp[det_pp['tid'] == 'lsst'].index.unique())
    oids_final     = oids_with_ztf & oids_with_lsst

    print(f'Objects with ZTF:        {len(oids_with_ztf)}')
    print(f'Objects with LSST:       {len(oids_with_lsst)}')
    print(f'Objects with both:       {len(oids_final)}')
    print(f'Discarded (too few det): '
          f'{len(set(lbl["oid"] for lbl in labels)) - len(oids_final)}')

    det_pp       = det_pp[det_pp.index.isin(oids_final)]
    ndet_ztf_pp  = ndet_ztf_pp[ndet_ztf_pp.index.isin(oids_final)]   \
                   if not ndet_ztf_pp.empty  else ndet_ztf_pp
    ndet_lsst_pp = ndet_lsst_pp[ndet_lsst_pp.index.isin(oids_final)] \
                   if not ndet_lsst_pp.empty else ndet_lsst_pp

    #Phase 3: feature extraction
    print('Phase 3: Extracting features...')
    extractor_ztf  = ZTFFeatureExtractor3bands(bands=(1, 2, 3), stream=False)
    extractor_lsst = LSSTFeatureExtractor(bands=(1, 2, 3, 4, 5, 6), stream=False)
    extractor_comb = LSSTFeatureExtractor(bands=(1, 2, 3, 4, 5, 6), stream=False)

    prev_feats = _load_features_checkpoints(ckpt_dir)
    done_feat_oids: Set[str] = set()
    for df in prev_feats.values():
        if not df.empty:
            done_feat_oids.update(df.index.get_level_values(0).unique())

    oids_to_process = [o for o in det_pp.index.unique() if o not in done_feat_oids]
    print(f'  {len(done_feat_oids)} already extracted, '
          f'{len(oids_to_process)} remaining')

    feat_ztf_batch:  List[pd.DataFrame] = []
    feat_lsst_batch: List[pd.DataFrame] = []
    feat_comb_batch: List[pd.DataFrame] = []

    def _flush_feature_batch(idx: int) -> None:
        _save_features_checkpoint(ckpt_dir, {
            'ztf':      pd.concat(feat_ztf_batch)  if feat_ztf_batch  else pd.DataFrame(),
            'lsst':     pd.concat(feat_lsst_batch) if feat_lsst_batch else pd.DataFrame(),
            'combined': pd.concat(feat_comb_batch) if feat_comb_batch else pd.DataFrame(),
        }, idx)
        feat_ztf_batch.clear()
        feat_lsst_batch.clear()
        feat_comb_batch.clear()

    def _make_worker_args(oid: str) -> tuple:

        det_oid  = det_pp[det_pp.index == oid]
        ndet_z   = ndet_ztf_pp[ndet_ztf_pp.index   == oid] if not ndet_ztf_pp.empty  else pd.DataFrame()
        ndet_l   = ndet_lsst_pp[ndet_lsst_pp.index == oid] if not ndet_lsst_pp.empty else pd.DataFrame()
        return (
            oid,
            det_oid.to_dict('records'),  det_oid.index.tolist(),
            ndet_z.to_dict('records'),   ndet_z.index.tolist(),
            ndet_l.to_dict('records'),   ndet_l.index.tolist(),
        )


    _existing_feat_files = glob.glob(os.path.join(ckpt_dir, 'features_ztf_*.parquet'))
    if _existing_feat_files:
        offset_feat = max(
            int(os.path.basename(f).split('_')[-1].replace('.parquet', ''))
            for f in _existing_feat_files
        )
    else:
        offset_feat = 0

    if n_jobs == 1:
        n_ok_feat   = 0
        for oid in tqdm(oids_to_process, desc='Extracting features'):
            fz, fl, fc = extract_features_for_object(
                oid, det_pp, ndet_ztf_pp, ndet_lsst_pp,
                extractor_ztf, extractor_lsst, extractor_comb,
            )
            if fz is not None: feat_ztf_batch.append(fz)
            if fl is not None: feat_lsst_batch.append(fl)
            if fc is not None: feat_comb_batch.append(fc)

            if fz is not None or fl is not None or fc is not None:
                n_ok_feat += 1

            if n_ok_feat > 0 and n_ok_feat % checkpoint_n == 0:
                _flush_feature_batch(offset_feat + n_ok_feat)

    else:
        import time
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from collections import deque

        ctx          = mp.get_context('spawn')
        window       = n_jobs * 4  
        _print_every = max(50, checkpoint_n)
        n_done       = 0            
        n_ok_feat    = 0            
        n_total_proc = len(oids_to_process)

        print(f'Parallel extraction: {n_jobs} workers, '
              f'window={window}, {n_total_proc} objects')

        pbar = tqdm(
            total=n_total_proc,
            desc='Extracting features',
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=2.0,
        )
        _t0 = time.time()

        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
            pending: deque = deque()  
            oid_iter = iter(oids_to_process)
            exhausted = False

            while len(pending) < window and not exhausted:
                try:
                    oid = next(oid_iter)
                    fut = pool.submit(_extract_worker, _make_worker_args(oid))
                    pending.append((fut, oid))
                except StopIteration:
                    exhausted = True

            while pending:
                fut, oid_submitted = pending.popleft()
                try:
                    oid_result, fz, fl, fc = fut.result()
                except Exception as exc:
                    oid_result = oid_submitted
                    tqdm.write(f'Worker error for {oid_result}: {exc}',
                               file=sys.stderr)
                    fz = fl = fc = None

                if fz is not None: feat_ztf_batch.append(fz)
                if fl is not None: feat_lsst_batch.append(fl)
                if fc is not None: feat_comb_batch.append(fc)

                if fz is not None or fl is not None or fc is not None:
                    n_ok_feat += 1

                n_done += 1
                pbar.update(1)

                if n_done % _print_every == 0:
                    elapsed = time.time() - _t0
                    rate    = n_done / elapsed if elapsed > 0 else 0
                    eta_min = (n_total_proc - n_done) / rate / 60 if rate > 0 else 0
                    tqdm.write(
                        f'  [{n_done}/{n_total_proc}] '
                        f'{rate:.1f} obj/s  ETA {eta_min:.0f} min',
                        file=sys.stderr,
                    )

                if n_ok_feat > 0 and n_ok_feat % checkpoint_n == 0:
                    _flush_feature_batch(offset_feat + n_ok_feat)

                if not exhausted:
                    try:
                        oid = next(oid_iter)
                        fut = pool.submit(_extract_worker, _make_worker_args(oid))
                        pending.append((fut, oid))
                    except StopIteration:
                        exhausted = True

        pbar.close()


    def _safe_concat(*frames) -> pd.DataFrame:
        valid = [f for f in frames if f is not None and not f.empty]
        return pd.concat(valid) if valid else pd.DataFrame()

    remaining = {
        'ztf':      pd.concat(feat_ztf_batch)  if feat_ztf_batch  else pd.DataFrame(),
        'lsst':     pd.concat(feat_lsst_batch) if feat_lsst_batch else pd.DataFrame(),
        'combined': pd.concat(feat_comb_batch) if feat_comb_batch else pd.DataFrame(),
    }

    if feat_ztf_batch or feat_lsst_batch or feat_comb_batch:
        _flush_feature_batch(offset_feat + n_ok_feat)


    all_feats_from_ckpt = _load_features_checkpoints(ckpt_dir)

    all_feats = {
        key: _safe_concat(all_feats_from_ckpt[key], remaining[key])
        for key in ('ztf', 'lsst', 'combined')
    }

    #Postprocess: filter, reorder and add 'survey' column ─────────────────
    print('Postprocessing features...')
    all_feats = postprocess_features(all_feats)
    for key, df in all_feats.items():
        if not df.empty:
            print(f'{key}: {len(df)} objects, {len(df.columns)} features')

    #Save labels
    feat_oids: Set[str] = set()
    for df in all_feats.values():
        if not df.empty:
            feat_oids.update(df.index.get_level_values(0).unique())

    labels_df = pd.DataFrame(labels)
    labels_df = labels_df[labels_df['oid'].isin(oids_final)]
    labels_df.to_csv(os.path.join(out_dir, 'labels.csv'), index=False)
    print(f'Saved {len(labels_df)} labels to {out_dir}/labels.csv')

    #Save final parquet files
    for key, name in [
        ('ztf',      'features_ztf'),
        ('lsst',     'features_lsst'),
        ('combined', 'features_comb'),
    ]:
        df = all_feats[key]
        if df is not None and not df.empty:
            path = os.path.join(out_dir, f'{name}.parquet')
            df.to_parquet(path)
            print(f'{name}.parquet: {len(df)} rows, {len(df.columns)} features')

    print('Done.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate ZTF + LSST training set for the classifier'
    )
    parser.add_argument(
        '--n-per-class', type=int, default=1000,
        help='Number of objects per class (default: 1000)',
    )
    parser.add_argument(
        '--out-dir', type=str,
        default='../data/simulated/',
        help=(
            'Output directory for parquet files and labels '
            '(default: ../data/simulated/)'
        ),
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    parser.add_argument(
        '--checkpoint-n', type=int, default=100,
        help='Save checkpoint every N objects (default: 100)',
    )
    parser.add_argument(
        '--ztf-fieldlog', type=str,
        default='../data/simlibs/ztf/ztf_fieldlog.parquet',
        help=(
            'Path to ztf_fieldlog.parquet built by build_ztf_fieldlog.py. '
            'If provided, well-sampled fields use real per-epoch cadence and '
            'depths from IRSA; other positions fall back to synthetic cadence. '
            'If the file is not found, falls back to synthetic ZTF cadence.'
        ),
    )
    
    parser.add_argument(
        '--min-visits-per-year', type=float, default=30.0,
        help=(
            'Minimum visits/year for a ZTF field to be considered '
            'well-sampled and eligible for real-cadence mode (default: 30).'),
    )
    parser.add_argument(
        '--opsim-db', type=str, default=None,
        help=(
            'Path to the Rubin OpSim SQLite database '),
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help=(
            'Number of parallel workers for feature extraction. '
            'Default: 1 (serial). Recommended: leave 1-2 cores free for the OS, '
            'Use --n-jobs 1 for debugging or if you see worker crashes.'
        ),
    )
    args = parser.parse_args()

    main(
        n_per_class=args.n_per_class,
        out_dir=args.out_dir,
        seed=args.seed,
        checkpoint_n=args.checkpoint_n,
        fieldlog_path=args.ztf_fieldlog,
        min_visits_per_year=args.min_visits_per_year,
        opsim_path=args.opsim_db,
        n_jobs=args.n_jobs,
    )