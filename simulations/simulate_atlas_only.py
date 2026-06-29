from __future__ import annotations
 
import argparse
import os
import sys
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
from tqdm import tqdm
 
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.CRITICAL)
 
_TFM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TFM_ROOT)
sys.path.insert(0, os.path.join(_TFM_ROOT, 'lc_classifier'))
 
import numpy as np
np.NaN = np.nan
 
from simulate_bandpasses import register_atlas_bandpasses
from magnetar_source import MagnetarSource, _luminosity_distance_cm, _pc_cm
 
ATLAS_BANDS = {'c': 0, 'o': 1}
ATLAS_FID   = {0: 'c', 1: 'o'}
 
TRANSIENT_CLASSES  = {'SNIa', 'SNII', 'SNIbc', 'SLSN'}
STOCHASTIC_CLASSES = {'QSO', 'AGN', 'Blazar', 'YSO', 'CV/Nova'}
PERIODIC_CLASSES   = {'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other'}
 
# Profundidad límite a 5σ de ATLAS
ATLAS_M5 = {'c': 19.7, 'o': 19.5}

 
@dataclass
class ATLASSimConfig:
    input_dir:    str
    output_dir:   str
    simlib_dir:   str
    bandpass_dir: str  = '../data/filter_profiles/'
    n_workers:    int  = 4
    chunk_size:   int  = 500
    n_objects:    int  = 1000   
    
#SIMLIB loader
 
class ATLASSimlib:
    """
    Carga las distribuciones empíricas del SIMLIB de ATLAS y genera tiempos de observación.

    """
 
    def __init__(self, simlib_dir: str):
        self._dir = Path(simlib_dir)
        self._dists = {}
        for band in ['c', 'o']:
            self._dists[band] = {
                'cadence':  np.load(self._dir / f'atlas_{band}_cadence.npy'),
                'skynoise': np.load(self._dir / f'atlas_{band}_skynoise.npy'),
            }
 
    def generate_obs_times(
        self,
        mjd_start: float,
        mjd_end:   float,
        band:      str,
        rng:       np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:

        cadence_dist  = self._dists[band]['cadence']
        skynoise_dist = self._dists[band]['skynoise']
 
        mjds = [mjd_start]
        while mjds[-1] < mjd_end:
            gap = float(rng.choice(cadence_dist))
            mjds.append(mjds[-1] + gap)
        mjds = np.array(mjds[:-1])
 
        if len(mjds) == 0:
            return np.array([]), np.array([])
 
        sky = rng.choice(skynoise_dist, size=len(mjds), replace=True).astype(float)
        return mjds, sky
 
 
#Filtro de preprocesador
 
def _passes_preprocessor_filter(det: pd.DataFrame, min_dets: int = 5) -> bool:
    """
    Replica el filtro enough_alerts() del ATLASLightcurvePreprocessor.
 

    """
    if det.empty:
        return False
    return any(
        (det['fid'] == fid).sum() > min_dets
        for fid in det['fid'].unique()
    )
 
 
#Función observe para ATLAS
 
def observe_atlas(
    flux_uJy:   np.ndarray,
    skynoises:  np.ndarray,
    rng:        np.random.Generator,
) -> dict:
    """
    Aplica ruido fotométrico de ATLAS a un array de flujos en mJy.
 
    """
    noise_uJy = np.sqrt(np.abs(flux_uJy) + skynoises**2)
    flux_obs  = flux_uJy + rng.normal(0.0, noise_uJy)
    snr       = np.where(noise_uJy > 0, flux_obs / noise_uJy, 0.0)
    detected  = snr > 3.0
 
    magpsf   = np.full(len(flux_uJy), 99.0)
    sigmapsf = np.full(len(flux_uJy), 99.0)
    valid    = detected & (flux_obs > 0)
    if valid.any():
        magpsf[valid]   = -2.5 * np.log10(flux_obs[valid]) + 23.9
        sigmapsf[valid] = 2.5 / np.log(10) / np.abs(snr[valid])

    diffmaglim = np.where(
        skynoises > 0,
        -2.5 * np.log10(5.0 * skynoises) + 23.9,
        99.0,
    )
 
    return {
        'magpsf':       magpsf,
        'sigmapsf':     sigmapsf,
        'detected':     detected,
        'diffmaglim':   diffmaglim,
        'snr':          snr,
        'flux_obs_ujy': flux_obs,    
        'noise_ujy':    noise_uJy,
    }
 
 
#Reconstitución del modelo físico
 
def reconstruct_model_and_evaluate(
    obj_info:    pd.Series,
    mjds_atlas:  dict[str, np.ndarray],   # {'c': array, 'o': array}
) -> dict[str, np.ndarray] | None:
    """
    Reconstituye el modelo físico del objeto desde obj_info y evalúa el
    flujo en mJy en los tiempos de observación ATLAS.
 
    """
    import sncosmo
 
    class_name = obj_info['classALeRCE'].replace('CV_Nova', 'CV/Nova')
    seed       = int(obj_info.get('seed', 0))
    rng        = np.random.default_rng(seed)
 
    try:
        if class_name in TRANSIENT_CLASSES:
            return _eval_transient(obj_info, mjds_atlas, rng)
        elif class_name in STOCHASTIC_CLASSES:
            return _eval_stochastic(obj_info, mjds_atlas, rng)
        elif class_name in PERIODIC_CLASSES:
            return _eval_periodic(obj_info, mjds_atlas, rng)
        else:
            return None
    except Exception as e:
        return None
 
 
def _eval_transient(
    obj_info: pd.Series,
    mjds_atlas: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    import sncosmo

    class_name = obj_info['classALeRCE']

    #Rama SLSN: modelo de magnetar
    if class_name == 'SLSN':
        z          = obj_info.get('z')
        t0         = obj_info.get('t0')
        P_i_ms     = obj_info.get('P_i_ms')
        B14        = obj_info.get('B14')       
        M_ej_Msun  = obj_info.get('M_ej_Msun')
        E_sn       = obj_info.get('E_sn',   1e51)
        kappa      = obj_info.get('kappa',  0.2)

        if any(v is None or not np.isfinite(float(v))
               for v in [z, t0, P_i_ms, B14, M_ej_Msun]):
            return None

        z, t0 = float(z), float(t0)

        source = MagnetarSource(
            P_i_ms    = float(P_i_ms),
            B14       = float(B14),
            M_ej_Msun = float(M_ej_Msun),
            E_sn      = float(E_sn),
            kappa     = float(kappa),
        )
        model = sncosmo.Model(source=source)
        model.set(z=z, t0=t0)

        D_L_cm = _luminosity_distance_cm(z)
        mu = 5.0 * np.log10(D_L_cm / (10.0 * _pc_cm))

        t_min = model.mintime()
        t_max = model.maxtime()

        result = {}
        for band_name in ATLAS_BANDS:
            atlas_band = f'atlas{band_name}'
            mjds = mjds_atlas.get(band_name, np.array([]))
            if len(mjds) == 0:
                result[band_name] = np.array([])
                continue

            flux_uJy = np.zeros(len(mjds))
            in_range = (mjds >= t_min) & (mjds <= t_max)
            if in_range.any():
                try:

                    abs_mags = model.bandmag(atlas_band, 'ab', mjds[in_range])
                    app_mags = abs_mags + mu
                    valid = np.isfinite(app_mags) & (app_mags < 90.0)
                    flux_uJy[in_range] = np.where(
                        valid,
                        10.0 ** ((23.9 - app_mags) / 2.5),
                        0.0,
                    )
                except Exception:
                    pass

            result[band_name] = flux_uJy

        return result

    #Rama sncosmo: SNIa / SNII / SNIbc
    template_map = {
        'SNIa':  'salt2',
        'SNII':  'nugent-sn2p',
        'SNIbc': 'nugent-sn1bc',
    }
    template = template_map.get(class_name, 'hsiao')

    try:
        model = sncosmo.Model(source=template)
    except Exception:
        return None

    params = {}
    for key in ['z', 't0', 'amplitude', 'x1', 'c']:
        val = obj_info.get(key)
        if val is not None and np.isfinite(float(val)):
            params[key] = float(val)

    if class_name == 'SNIa' and 'amplitude' in params:
        params['x0'] = params.pop('amplitude')

    if not params.get('z') or not params.get('t0'):
        return None  # sin z y t0 no podemos evaluar el modelo

    try:
        model.set(**{k: v for k, v in params.items() if k in model.param_names})
    except Exception:
        return None


    t_max = model.maxtime()

    result = {}
    for band_name in ATLAS_BANDS:
        atlas_band = f'atlas{band_name}'
        mjds = mjds_atlas.get(band_name, np.array([]))
        if len(mjds) == 0:
            result[band_name] = np.array([])
            continue

        flux_uJy = np.zeros(len(mjds))
        in_range  = (mjds >= t_min) & (mjds <= t_max)
        if in_range.any():
            try:
                flux = model.bandflux(
                    atlas_band, mjds[in_range], zp=25.0, zpsys='ab'
                )
               
                flux_uJy[in_range] = np.where(
                    np.isfinite(flux), flux * 0.363, 0.0
                )
            except Exception:
                pass

        result[band_name] = flux_uJy

    return result
 
 
def _eval_stochastic(
    obj_info: pd.Series,
    mjds_atlas: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:

    ATLAS_COLOR_OFFSETS = {
        'QSO':            {'c':  0.100, 'o': -0.032},
        'AGN':            {'c':  0.060, 'o': -0.019},
        'Blazar':         {'c': -0.061, 'o':  0.018},
        'YSO':            {'c':  0.612, 'o': -0.180},
        'CV/Nova':        {'c': -0.068, 'o':  0.020},
        'LPV':            {'c':  0.828, 'o': -0.255},
        'E':              {'c':  0.277, 'o': -0.044},
        'DSCT':           {'c':  0.104, 'o':  0.003},
        'RRL':            {'c':  0.196, 'o': -0.021},
        'CEP':            {'c':  0.277, 'o': -0.044},
        'Periodic-Other': {'c':  0.317, 'o': -0.056},
    }
 
    class_name   = obj_info['classALeRCE']
    mean_mag_r   = float(obj_info.get('mean_mag_r', 18.0))
    offsets      = ATLAS_COLOR_OFFSETS.get(class_name, {'c': 0.0, 'o': 0.0})
 
    tau_raw   = obj_info.get('drw_tau')
    sigma_raw = obj_info.get('drw_sigma')
 
    _DEFAULT_TAU   = {'QSO': 250., 'AGN': 200., 'Blazar': 30., 'YSO': 10., 'CV/Nova': 6.}
    _DEFAULT_SIGMA = {'QSO': 0.13, 'AGN': 0.12, 'Blazar': 0.25, 'YSO': 0.20, 'CV/Nova': 0.5}
    class_name_s   = obj_info['classALeRCE']
 
    tau   = float(tau_raw)   if (tau_raw   is not None and np.isfinite(float(tau_raw)))   else _DEFAULT_TAU.get(class_name_s,   200.)
    sigma = float(sigma_raw) if (sigma_raw is not None and np.isfinite(float(sigma_raw))) else _DEFAULT_SIGMA.get(class_name_s, 0.15)
 
    result = {}
    for band_name in ['c', 'o']:
        mjds = mjds_atlas.get(band_name, np.array([]))
        if len(mjds) == 0:
            result[band_name] = np.array([])
            continue
 
        lc_mag = _drw_at_times(mjds, mean_mag_r + offsets[band_name], tau, sigma, rng)
 
        flux_uJy = 10.0**((23.9 - lc_mag) / 2.5)
        result[band_name] = flux_uJy
 
    return result
 
 
def _eval_periodic(
    obj_info: pd.Series,
    mjds_atlas: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:

    ATLAS_COLOR_OFFSETS = {
        'LPV':            {'c':  0.828, 'o': -0.255},
        'E':              {'c':  0.277, 'o': -0.044},
        'DSCT':           {'c':  0.104, 'o':  0.003},
        'RRL':            {'c':  0.196, 'o': -0.021},
        'CEP':            {'c':  0.277, 'o': -0.044},
        'Periodic-Other': {'c':  0.317, 'o': -0.056},
    }
 
    class_name = obj_info['classALeRCE']
    period     = float(obj_info.get('period',   10.0))
    mean_mag   = float(obj_info.get('mean_mag_r', 17.0))
    amplitude  = float(obj_info.get('amplitude',  0.3))
    phase0     = float(obj_info.get('phase0',     0.0))
    offsets    = ATLAS_COLOR_OFFSETS.get(class_name, {'c': 0.0, 'o': 0.0})
 
    result = {}
    for band_name in ['c', 'o']:
        mjds = mjds_atlas.get(band_name, np.array([]))
        if len(mjds) == 0:
            result[band_name] = np.array([])
            continue
 
        phase    = ((mjds - phase0) % period) / period
        lc_mag   = mean_mag + offsets[band_name] + amplitude * np.sin(2 * np.pi * phase)
        flux_uJy = 10.0**((23.9 - lc_mag) / 2.5)
        result[band_name] = flux_uJy
 
    return result
 
 
def _drw_at_times(
    mjds: np.ndarray,
    mean_mag: float,
    tau: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:

    n   = len(mjds)
    lc  = np.zeros(n)
    lc[0] = rng.normal(0, sigma)
    for i in range(1, n):
        dt    = mjds[i] - mjds[i-1]
        decay = np.exp(-dt / tau)
        noise = np.sqrt(sigma**2 * (1 - decay**2))
        lc[i] = decay * lc[i-1] + rng.normal(0, noise)
    return mean_mag + lc
 
 
 
def make_atlas_detections(
    oid:       str,
    obs:       dict,
    mjds:      np.ndarray,
    fid:       int,
    ra:        float,
    dec:       float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    det_rows  = []
    ndet_rows = []
 
    for k in range(len(mjds)):
        mjd      = mjds[k]
        mag      = obs['magpsf'][k]
        sig      = obs['sigmapsf'][k]
        detected = obs['detected'][k]
        lim      = obs['diffmaglim'][k]
        snr      = obs['snr'][k]
 
        if detected and np.isfinite(mag) and 0 < sig < MAX_SIGMA_MAG:
          
            ujy  = float(obs['flux_obs_ujy'][k])
            dujy = float(obs['noise_ujy'][k])
 
            det_rows.append({
                'oid':               oid,
                'mjd':               mjd,
                'fid':               fid,
                'magpsf':            mag,
                'sigmapsf':          sig,
                'magpsf_corr':       mag,
                'sigmapsf_corr':     sig,
                'sigmapsf_corr_ext': sig,
                'magpsf_ml':         mag,
                'sigmapsf_ml':       sig,
                'uJy':               ujy,   
                'duJy':              dujy,  
                'ra':                ra,
                'dec':               dec,
                'rb':                1.0,
                'sgscore1':          np.nan,
                'isdiffpos':         1 if snr >= 0 else -1,
                'tid':               'ATLAS',
                'rbversion':         None,
                'step_id_corr':      None,
            })
        else:
            ndet_rows.append({'oid': oid, 'mjd': mjd, 'fid': fid, 'diffmaglim': lim})
 
    if det_rows:
        det_df = pd.DataFrame(det_rows).set_index('oid')
    else:
        det_df = _empty_det_df()
 
    if ndet_rows:
        ndet_df = pd.DataFrame(ndet_rows).set_index('oid')
    else:
        ndet_df = _empty_ndet_df()
 
    return det_df, ndet_df
 
 
MAX_SIGMA_MAG = 0.5
 
def _empty_det_df():
    
    cols = ['mjd','fid','magpsf','sigmapsf','magpsf_corr','sigmapsf_corr',
            'sigmapsf_corr_ext','magpsf_ml','sigmapsf_ml',
            'uJy','duJy',
            'ra','dec','rb','sgscore1','isdiffpos','tid','rbversion','step_id_corr']
    df = pd.DataFrame(columns=cols); df.index.name = 'oid'; return df
 
def _empty_ndet_df():
    df = pd.DataFrame(columns=['mjd','fid','diffmaglim']); df.index.name = 'oid'; return df
 
 
 
def _class_worker(
    class_name:    str,
    class_df_data: list,   
    class_df_idx:  list,  
    simlib_dir:    str,
    cfg:           'ATLASSimConfig',
    out_dir_str:   str,
) -> dict:

    import pandas as pd
    from pathlib import Path
    class_df = pd.DataFrame(class_df_data, index=class_df_idx)
    simlib   = ATLASSimlib(simlib_dir)
    return simulate_atlas_for_class(class_name, class_df, simlib, cfg, Path(out_dir_str))


def simulate_atlas_for_class(
    class_name:  str,
    obj_info_df: pd.DataFrame,
    simlib:      ATLASSimlib,
    cfg:         ATLASSimConfig,
    out_dir:     Path,
) -> dict:
    """
    Simula observaciones ATLAS para todos los objetos de una clase.
   
    """
    register_atlas_bandpasses(bandpass_dir=cfg.bandpass_dir, force=True)
 
    det_chunks  = []
    ndet_chunks = []
    n_ok   = 0
    n_skip = 0
    chunk_id = 0
 
    with tqdm(total=len(obj_info_df), desc=f'{class_name}', leave=False) as pbar:
        for _, obj_info in obj_info_df.iterrows():
            oid        = str(obj_info.name)
            atlas_seed = abs(hash(f'{oid}_atlas')) % (2**31)
            rng        = np.random.default_rng(atlas_seed)
 
            ra        = float(obj_info.get('ra',  obj_info.get('meanra',  0.0)))
            dec       = float(obj_info.get('dec', obj_info.get('meandec', 0.0)))
            mjd_start = float(obj_info.get('mjd_start', 59000.0))
            mjd_end   = float(obj_info.get('mjd_end',   59730.0))
 
            mjds_atlas = {}
            sky_atlas  = {}
            for band in ['c', 'o']:
                mjds, sky = simlib.generate_obs_times(mjd_start, mjd_end, band, rng)
                mjds_atlas[band] = mjds
                sky_atlas[band]  = sky
 
            flux_dict = reconstruct_model_and_evaluate(obj_info, mjds_atlas)
            if flux_dict is None:
                n_skip += 1
                pbar.update(1)
                continue

            obj_det_dfs  = []
            obj_ndet_dfs = []
 
            for band_name, fid in ATLAS_BANDS.items():
                mjds = mjds_atlas[band_name]
                if len(mjds) == 0:
                    continue
                flux_uJy = flux_dict.get(band_name, np.zeros(len(mjds)))
                obs = observe_atlas(flux_uJy, sky_atlas[band_name], rng)
                det_df, ndet_df = make_atlas_detections(oid, obs, mjds, fid, ra, dec)
                obj_det_dfs.append(det_df)
                obj_ndet_dfs.append(ndet_df)
 
            obj_det_all = pd.concat(obj_det_dfs) if obj_det_dfs else pd.DataFrame()
            if not _passes_preprocessor_filter(obj_det_all):
                n_skip += 1
                pbar.update(1)
                continue
 
            if obj_det_dfs:
                det_chunks.append(obj_det_all)
            if obj_ndet_dfs:
                ndet_chunks.append(pd.concat(obj_ndet_dfs))
            n_ok += 1
            pbar.update(1)
 
            if n_ok % cfg.chunk_size == 0:
                _flush_chunk(det_chunks, ndet_chunks, class_name, n_ok, out_dir)
                det_chunks  = []
                ndet_chunks = []
                chunk_id += 1

    if det_chunks:
        _flush_chunk(det_chunks, ndet_chunks, class_name, chunk_id, out_dir)
 
    return {'class': class_name, 'n_ok': n_ok, 'n_skip': n_skip}
 
 
def _flush_chunk(det_list, ndet_list, class_name, chunk_id, out_dir):
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    if det_list:
        pd.concat(det_list).to_parquet(
            ckpt_dir / f'det_atlas_{class_name}_{chunk_id:04d}.parquet'
        )
    if ndet_list:
        valid = [df for df in ndet_list if not df.empty]
        if valid:
            pd.concat(valid).to_parquet(
                ckpt_dir / f'ndet_atlas_{class_name}_{chunk_id:04d}.parquet'
            )
 
 
#Postprocesado de features ATLAS 

_ATLAS_RM_COLS = {
    'MHPS_non_zero_0', 'MHPS_non_zero_1',
    'MHPS_PN_flag_0',  'MHPS_PN_flag_1',
    'c-o_max_corr',    'c-o_mean_corr',
}
 
def postprocess_atlas_features_simulated(features: pd.DataFrame) -> pd.DataFrame:

    if features.empty:
        return features
    cols = [c for c in features.columns if c not in _ATLAS_RM_COLS]
    out  = features[cols].copy()
    out['survey'] = 3
    out.columns = [f'{c}_atlas' if c != 'survey' else c for c in out.columns]
    return out
 
 

def _atlas_extract_worker(args: tuple) -> tuple:

    import warnings, logging
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
    _TFM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _TFM_ROOT)
    sys.path.insert(0, os.path.join(_TFM_ROOT, 'lc_classifier'))
    np.NaN = np.nan

    from lc_classifier.features.custom.ztf_feature_extractor import ATLASFeatureExtractor

    oid, det_records, det_idx = args

    if not det_records:
        return oid, None

    det_oid = pd.DataFrame.from_records(det_records)
    det_oid.index = pd.Index(det_idx, name='oid')

    extractor = ATLASFeatureExtractor()
    try:
        feat = extractor.compute_features(detections=det_oid)
        if feat is not None and not feat.empty:
            return oid, feat
    except Exception:
        pass
    return oid, None


#Extracción de features ATLAS
 
def extract_atlas_features_simulated(
    out_dir:      Path,
    checkpoint_n: int = 500,
    n_workers:    int = 1,
) -> pd.DataFrame:

    from lc_classifier.features.preprocess.preprocess_ztf import ATLASLightcurvePreprocessor
    from lc_classifier.features.custom.ztf_feature_extractor import ATLASFeatureExtractor

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    preprocessor = ATLASLightcurvePreprocessor()

    det_files = sorted((out_dir / "checkpoints").glob("det_atlas_*.parquet"))
    if not det_files:
        print("[WARN] det_atlas_*.parquet not found in checkpoints/. No features to extract.")
        return pd.DataFrame()

    print(f"Loading {len(det_files)} files with ATLAS sims detections...")
    det_all = pd.concat([pd.read_parquet(f) for f in det_files])
    n_simulated = det_all.index.get_level_values(0).nunique()
    print(f" -> {n_simulated} simulated objects with ATLAS detections")

    #Preprocesado
    print("Preprocessing detections...")
    det_pp_all = preprocessor.preprocess(det_all)
    oids_final = set(det_pp_all.index.get_level_values(0).unique())
    n_discarded = n_simulated - len(oids_final)
    print(f" -> {len(oids_final)} objects after preprocessing, "
          f"({n_discarded} discarded (too few detections))")

    #checkpoint
    ckpt_files = sorted(ckpt_dir.glob("features_atlas_sim_*.parquet"))
    if ckpt_files:
        results_df = pd.concat([pd.read_parquet(f) for f in ckpt_files])
        done_oids  = set(results_df.index.get_level_values(0).unique())
        print(f"Resuming: {len(done_oids)} objects already done")
    else:
        results_df = pd.DataFrame()
        done_oids  = set()

    oids_pending = [
        oid for oid in oids_final
        if oid not in done_oids
    ]
    print(f"Extraycting features for {len(oids_pending)} objects...")

    if ckpt_files:
        offset = max(int(f.stem.split('_')[-1]) for f in ckpt_files)
    else:
        offset = 0
    n_ok   = 0
    batch  = []

    def _flush_batch(idx: int) -> None:
        nonlocal results_df
        if batch:
            chunk = pd.concat(batch)
            chunk.to_parquet(ckpt_dir / f"features_atlas_sim_{idx:06d}.parquet")
            print(f"Checkpoint: features_atlas_sim_{idx:06d} ({len(chunk)} objects)")
            results_df = pd.concat([results_df, chunk]) if not results_df.empty else chunk
            batch.clear()

    def _make_worker_args(oid: str) -> tuple:
        det_oid = det_pp_all[det_pp_all.index == oid]
        return (oid, det_oid.to_dict('records'), det_oid.index.tolist())

    if n_workers == 1:
        extractor = ATLASFeatureExtractor()
        for oid in tqdm(oids_pending, desc="Simulated ATLAS features"):
            det_oid = det_pp_all[det_pp_all.index == oid]
            if det_oid.empty:
                continue
            try:
                feat = extractor.compute_features(detections=det_oid)
                if feat is not None and not feat.empty:
                    batch.append(feat)
            except Exception as e:
                print(f"[ATLAS] Error with {oid}: {e}")
            n_ok += 1
            if n_ok % checkpoint_n == 0 and batch:
                _flush_batch(offset + n_ok)

    else:
        import sys, time
        import multiprocessing as mp
        from collections import deque

        ctx          = mp.get_context('spawn')
        window       = n_workers * 4
        _print_every = max(50, checkpoint_n)
        n_done       = 0
        n_total      = len(oids_pending)

        print(f"Parallel extraction: {n_workers} workers, "
              f"window={window}, {n_total} objects")

        pbar = tqdm(total=n_total, desc="Simulated ATLAS features",
                    file=sys.stderr, dynamic_ncols=True, mininterval=2.0)
        _t0 = time.time()

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            pending_futs: deque = deque()
            oid_iter  = iter(oids_pending)
            exhausted = False

            while len(pending_futs) < window and not exhausted:
                try:
                    oid = next(oid_iter)
                    fut = pool.submit(_atlas_extract_worker, _make_worker_args(oid))
                    pending_futs.append((fut, oid))
                except StopIteration:
                    exhausted = True

            while pending_futs:
                fut, oid_submitted = pending_futs.popleft()
                try:
                    oid_result, feat = fut.result()
                except Exception as exc:
                    tqdm.write(f"Worker error for {oid_submitted}: {exc}",
                               file=sys.stderr)
                    feat = None

                if feat is not None:
                    batch.append(feat)
                    n_ok += 1

                n_done += 1
                pbar.update(1)

                if n_done % _print_every == 0:
                    elapsed = time.time() - _t0
                    rate    = n_done / elapsed if elapsed > 0 else 0
                    eta_min = (n_total - n_done) / rate / 60 if rate > 0 else 0
                    tqdm.write(
                        f"[{n_done}/{n_total}] "
                        f"{rate:.1f} obj/s  ETA {eta_min:.0f} min",
                        file=sys.stderr,
                    )

                if n_ok > 0 and n_ok % checkpoint_n == 0:
                    _flush_batch(offset + n_ok)

                if not exhausted:
                    try:
                        oid = next(oid_iter)
                        fut = pool.submit(_atlas_extract_worker, _make_worker_args(oid))
                        pending_futs.append((fut, oid))
                    except StopIteration:
                        exhausted = True

        pbar.close()

    if batch:
        _flush_batch(offset + n_ok)
 
 
    #Guardar labels
    if not results_df.empty:
        def _class_from_oid(oid: str) -> str:
            parts = oid.split("_")
            if len(parts) >= 3 and parts[0] == "sim":
                return "_".join(parts[1:-1])
            return "Unknown"
 
        feat_oids = set(results_df.index.get_level_values(0).unique())
        labels_df = pd.DataFrame([
            {"oid": oid, "classALeRCE": _class_from_oid(str(oid))}
            for oid in feat_oids
        ])
        labels_path = out_dir / "labels_atlas.csv"
        labels_df.to_csv(labels_path, index=False)
        print(f"Labels saved: {len(labels_df)} objects in {labels_path}")
 
    return results_df
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description='Simulate ATLAS lightcurves for existing objects')
    parser.add_argument('--input-dir',   required=False,
                    default='../data/simulated/checkpoints/',
                    help='Directory with existing ZTF+LSST simulations')
    parser.add_argument('--output-dir',  required=False,
                    default='../data/simulated/atlas/',
                    help='Output directory for ATLAS simulations')
    parser.add_argument('--simlib-dir',  required=False,
                    default='../data/simlibs/atlas/',
                    help='Directory with ATLAS SIMLIB')
    parser.add_argument('--bandpass-dir', required=False,
                    default='../data/filter_profiles/',
                    help='Directory with ATLAS transmission bands files (SVO Filter Profile Service)')
    parser.add_argument('--n-workers',   type=int, default=4)
    parser.add_argument('--chunk-size',  type=int, default=500,
                        help='Save simulation checkpoint after N objects (default: 500)')
    parser.add_argument('--n-objects',   type=int, default=1000,
                        help='Number of objects to simulate in each class (default: 1000)')
    parser.add_argument('--classes',     nargs='+', default=None,
                        help='Classes to simulate (default: all)')
    parser.add_argument('--extract-features', action='store_true',
                        help='Extract features after simulation'
                             '(or extract for existing sims with --skip-sim)')
    parser.add_argument('--skip-sim',    action='store_true',
                        help='Skip simulation, extract features only (existing sims required)')
    parser.add_argument('--checkpoint-n', type=int, default=500,
                        help='Save features checkpoint after N objects (default: 500)')
    args = parser.parse_args()
 
    cfg = ATLASSimConfig(
        input_dir    = args.input_dir,
        output_dir   = args.output_dir,
        simlib_dir   = args.simlib_dir,
        bandpass_dir = args.bandpass_dir,
        n_workers    = args.n_workers,
        chunk_size   = args.chunk_size,
        n_objects    = args.n_objects,
    )
 
    in_dir  = Path(cfg.input_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    #Modo --skip-sim: solo extracción de features
    if args.skip_sim:
        print("\n--skip-sim mode: sskipping simulation, extracting features directly.")
        print("\n=== ATLAS feature extraction ===")
        features = extract_atlas_features_simulated(out_dir, args.checkpoint_n, args.n_workers)
        if not features.empty:
            out_path = out_dir / 'features_atlas.parquet'
            features = postprocess_atlas_features_simulated(features)  
            features.to_parquet(out_path)
            print(f"\nATLAS features saved in {out_path}")
            print(f"{len(features)} objetcts, {len(features.columns)} features")
        return
 
    #1. Cargar object_info de las simulaciones existentes
    obj_files = sorted(in_dir.glob('obj_*.parquet'))
    if not obj_files:
        print(f"ERROR: obj_*.parquet files not found in {in_dir}")
        return
 
    print(f"Loading object_info from {len(obj_files)} files...")
    obj_df = pd.concat([pd.read_parquet(f) for f in obj_files])
    print(f"Total objects: {len(obj_df)}")
 
    if 'classALeRCE' not in obj_df.columns:
        def _class_from_oid(oid: str) -> str:
            parts = oid.split('_')
            if len(parts) >= 3 and parts[0] == 'sim':
                return '_'.join(parts[1:-1])
            return 'Unknown'
        obj_df['classALeRCE'] = [_class_from_oid(str(oid)) for oid in obj_df.index]
        print("[INFO] class extracted from oid")
 

    if 'ra' not in obj_df.columns and 'meanra' in obj_df.columns:
        obj_df['ra']  = obj_df['meanra']
        obj_df['dec'] = obj_df['meandec']
 

    required_cols = ['classALeRCE', 'seed', 'ra', 'dec', 'mjd_start', 'mjd_end']
    missing = [c for c in required_cols if c not in obj_df.columns]
    if missing:
        print(f"ERROR: missing columns in object_info -> {missing}")
        print("Simulation pipeline must save this fields in obj_info.")
        print("Check make_object_info() in simulate/formatter.py")
        return
 
    #2. Filtrar clases a procesar 
    all_classes = sorted(obj_df['classALeRCE'].unique())
    if args.classes:
        classes = [c for c in args.classes if c in all_classes]
        unknown = [c for c in args.classes if c not in all_classes]
        if unknown:
            print(f"[WARN] Clasees not found in input: {unknown}")
    else:
        classes = all_classes
 
    print(f"\nClasses to process: {classes}")
    print(f"Workers: {cfg.n_workers}")
 
    #3. Cargar SIMLIB
    try:
        simlib = ATLASSimlib(cfg.simlib_dir)
        print(f"\nSIMLIB ATLAS loaded from {cfg.simlib_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: SIMLIB not found: {e}")
        print("Run build_atlas_simlib.py first (or download SIMLIB from GitHub)")
        return
 
    #4. Registrar bandas ATLAS en sncosmo
    register_atlas_bandpasses(bandpass_dir=cfg.bandpass_dir, force=True)
 
    import sncosmo as _snc
    for bname in ['atlasc', 'atlaso']:
        try:
            bp = _snc.get_bandpass(bname)
            print(f"{bname}: {bp.wave.min():.0f}–{bp.wave.max():.0f} Å  "
                  f"(N={len(bp.wave)})")
        except Exception as e:
            print(f"{bname} not available: {e}")
        
    #5. Simular por clase
    results = []
    for class_name in classes:
        class_df = obj_df[obj_df['classALeRCE'] == class_name]
        print(f"\n[{class_name}] {len(class_df)} objects")

        existing = list((out_dir / 'checkpoints').glob(f'det_atlas_{class_name}_*.parquet'))
        if existing:
            print(f"{len(existing)} chunks done, skipping (usa --classes to force resimulation of specific class)")
            continue

        result = simulate_atlas_for_class(
            class_name, class_df, simlib, cfg, out_dir
        )
        results.append(result)
        print(f"OK: {result['n_ok']}, Skipped: {result['n_skip']}")
 
    #6. Resumen

    print("\n==== Summary ====")
    ckpt_dir = out_dir / 'checkpoints'

    all_det_files = sorted(ckpt_dir.glob('det_atlas_*.parquet'))
    clase_files = {}
    for f in all_det_files:
        parts = f.stem.split('_')
        clase = '_'.join(parts[2:-1])
        clase_files.setdefault(clase, []).append(f)
    total = 0
    for clase in sorted(clase_files):
        all_oids = set()
        for f in clase_files[clase]:
            all_oids.update(
                pd.read_parquet(f, columns=[]).index.get_level_values(0).unique()
            )
        print(f"{clase:15s}: {len(all_oids):5d} unique objects")
        total += len(all_oids)
    print(f"{'TOTAL':15s}: {total:5d} unique objects")
    print(f"\nATLAS detections saved in {ckpt_dir}/")
 
    #7. Extracción de features (opcional)
    if args.extract_features:
        print("\n=== ATLAS feature extraction ===")
        features = extract_atlas_features_simulated(out_dir, args.checkpoint_n, args.n_workers)
        if not features.empty:
            out_path = out_dir / 'features_atlas.parquet'
            features = postprocess_atlas_features_simulated(features)
            features.to_parquet(out_path)
            print(f"\nATLAS features saved in {out_path}")
            print(f"{len(features)} objects, {len(features.columns)} features")
 
 
if __name__ == '__main__':
    main()