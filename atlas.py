from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
import warnings
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
np.NaN = np.nan   # compatibilidad con lc_classifier
import pandas as pd
import requests
from scipy.optimize import OptimizeWarning
from tqdm import tqdm

from lc_classifier.features import ATLASLightcurvePreprocessor, ATLASFeatureExtractor


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=np.RankWarning)
warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger().setLevel(logging.CRITICAL)


ATLAS_BASE_URL  = 'https://fallingstar-data.com/forcedphot'
ATLAS_TOKEN_URL = f'{ATLAS_BASE_URL}/api-token-auth/'
ATLAS_QUEUE_URL = f'{ATLAS_BASE_URL}/queue/'

SUBMIT_DELAY  = 1.5     # segundos entre envíos
POLL_INTERVAL = 30.0   # segundos entre rondas de polling
POLL_MAX_WAIT = 86400  # segundos máximos esperando tareas pendientes (24h)

_EXPIRED = object()  #tarea 404 en el servidor

# Nombres de columna oficiales de ATLAS forced photometry
ATLAS_COLUMNS = [
    'MJD', 'm', 'dm', 'uJy', 'duJy', 'F', 'err', 'chi/N',
    'RA', 'Dec', 'x', 'y', 'maj', 'min', 'phi',
    'apfit', 'Sky', 'ZP', 'Obs',
]

# Features ruidosas/redundantes a eliminar
ATLAS_RM_COLS = {
    'MHPS_non_zero_0', 'MHPS_non_zero_1',
    'MHPS_PN_flag_0',  'MHPS_PN_flag_1',
    'c-o_max_corr',    'c-o_mean_corr',  
}

SURVEY_ID_ATLAS = 3   # 0=ZTF, 1=LSST, 2=combined, 3=ATLAS


#Autenticación

def get_atlas_token(username: str, password: str) -> str:
    resp = requests.post(
        ATLAS_TOKEN_URL,
        data={'username': username, 'password': password},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()['token']


# Formato ATLAS

def _parse_atlas_text(text: str) -> pd.DataFrame:
    header     = None
    data_lines = []

    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith('###'):
            header = line.lstrip('#').split()
        elif line.startswith('#'):
            continue
        else:
            data_lines.append(line)

    if not data_lines:
        return pd.DataFrame()

    col_names = header if header else ATLAS_COLUMNS
    try:
        df = pd.read_csv(
            StringIO('\n'.join(data_lines)),
            sep=r'\s+',
            header=None,
            names=col_names,
        )
    except Exception:
        df = pd.read_csv(StringIO('\n'.join(data_lines)), sep=r'\s+', header=None)
        df.columns = ATLAS_COLUMNS[:len(df.columns)]

    return df


#Filtros de calidad

def apply_quality_cuts(df: pd.DataFrame, sigma_clip_flux: bool = True) -> pd.DataFrame:
    
    from astropy.stats import sigma_clip as astropy_sigma_clip

    if df.empty:
        return df

    col_map  = {c.lower(): c for c in df.columns}
    chi_col  = next((c for c in df.columns if c.lower() in ('chi/n', 'chi_n')), None)
    dujy_col = col_map.get('dujy')
    ujy_col  = col_map.get('ujy')
    f_col    = col_map.get('f')

    #1. Cortes fijos
    mask = pd.Series(True, index=df.index)
    if dujy_col:
        mask &= pd.to_numeric(df[dujy_col], errors='coerce') <= 4000
    if chi_col:
        mask &= pd.to_numeric(df[chi_col],  errors='coerce') <= 100
    df = df[mask].copy()

    if df.empty or not sigma_clip_flux or f_col is None or ujy_col is None:
        return df

    #2. Rolling sigma clip por banda
    keep = pd.Series(True, index=df.index)
    for band in ['c', 'o']:
        band_mask = df[f_col].str.lower() == band
        band_idx  = df.index[band_mask]
        if len(band_idx) < 3:
            continue

        flux   = pd.to_numeric(df.loc[band_idx, ujy_col], errors='coerce').values
        window = 11
        clipped_mask = np.zeros(len(flux), dtype=bool)

        for i in range(len(flux)):
            lo = max(0, i - window // 2)
            hi = min(len(flux), i + window // 2 + 1)
            window_flux = flux[lo:hi]
            finite = window_flux[np.isfinite(window_flux)]
            if len(finite) < 3:
                continue
            clipped = astropy_sigma_clip(finite, sigma=3.0, maxiters=3)
            mean = np.mean(clipped.data[~clipped.mask])
            std  = np.std(clipped.data[~clipped.mask])
            if np.isfinite(flux[i]) and std > 0 and abs(flux[i] - mean) > 3.0 * std:
                clipped_mask[i] = True

        keep.loc[band_idx] = ~clipped_mask

    return df[keep].copy()


#Checkpoint

def _raw_dir(out_dir: Path) -> Path:
    p = out_dir / 'raw'
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ckpt_dir(out_dir: Path) -> Path:
    p = out_dir / 'checkpoints'
    p.mkdir(parents=True, exist_ok=True)
    return p

def _pending_path(out_dir: Path) -> Path:
    return _raw_dir(out_dir) / 'pending.json'

def _oid_path(out_dir: Path, oid: str) -> Path:
    safe = oid.replace('/', '_').replace(' ', '_')
    return _raw_dir(out_dir) / f'oid_{safe}.parquet'

def _load_pending(out_dir: Path) -> Dict[str, str]:
    p = _pending_path(out_dir)
    return json.loads(p.read_text()) if p.exists() else {}

def _save_pending(out_dir: Path, pending: Dict[str, str]) -> None:
    _pending_path(out_dir).write_text(json.dumps(pending, indent=2))

def _already_done(out_dir: Path) -> Set[str]:
    done = set()
    for p in _raw_dir(out_dir).glob('oid_*.parquet'):
        try:
            df = pd.read_parquet(p, columns=[])   # solo índice
            done.update(df.index.get_level_values(0).unique())
        except Exception:
            pass
    return done


def _save_empty_marker(out_dir: Path, oid: str) -> None:
    """
    Marca un objeto como ya procesado sin detecciones ATLAS válidas.

    """
    marker = pd.DataFrame(index=pd.Index([oid], name='oid'))
    marker.to_parquet(_oid_path(out_dir, oid))


#Carga de objetos desde obj_ztf_*.parquet + det_{ztf,lsst}_*.parquet

MJD_PRE_BUFFER = 30.0   # días de buffer antes de la primera detección ZTF/LSST
MJD_POST_BUFFER = 30.0  # días de buffer después de la última detección ZTF/LSST


def _utc_now_mjd() -> float:
    """MJD del momento actual en UTC."""
    import datetime
    t0 = datetime.datetime(1858, 11, 17)
    return (datetime.datetime.utcnow() - t0).total_seconds() / 86400.0


def load_object_coords(
    obj_parquet_glob: str,
    pre_buffer:  float = MJD_PRE_BUFFER,
    post_buffer: float = MJD_POST_BUFFER,
    feat_dir: Optional[Path] = None,
) -> pd.DataFrame:
    
    files = sorted(glob.glob(obj_parquet_glob))
    if not files:
        raise FileNotFoundError(
            f"obj_ztf_*.parquet not found in: {obj_parquet_glob}"
        )

    #Coordenadas desde obj_ztf_*.parquet
    
    coord_dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if 'meanra' in df.columns and 'ra' not in df.columns:
            df = df.rename(columns={'meanra': 'ra', 'meandec': 'dec'})
        coord_dfs.append(df[['ra', 'dec']])

    coords = pd.concat(coord_dfs)
    coords = coords[~coords.index.duplicated(keep='first')]
    coords = coords.dropna(subset=['ra', 'dec'])

    #Filtrar a OIDs presentes en alguna rama (strict o relaxed)
    if feat_dir is not None:
        feat_dir = Path(feat_dir)
        oids_rama = set()
        for fname in ('features_comb_strict.parquet', 'features_comb_relaxed.parquet'):
            p = feat_dir / fname
            if p.exists():
                oids_rama.update(
                    pd.read_parquet(p, columns=[])
                    .index.get_level_values(0).unique()
                )
        n_before_filter = len(coords)
        coords = coords[coords.index.isin(oids_rama)]
        n_filtered = n_before_filter - len(coords)
        print(f"{n_filtered} objects discarded (they don't belong to any branch')")
        print(f"{len(coords)} objects after branch filter")

    #Rango MJD desde det_ztf_*.parquet + det_lsst_*.parquet
    base_dir  = str(Path(obj_parquet_glob).parent) + os.sep
    det_files = sorted(
        glob.glob(base_dir + 'det_ztf_*.parquet') +
        glob.glob(base_dir + 'det_lsst_*.parquet')
    )
    mjd_now   = _utc_now_mjd()

    if det_files:
        mjd_dfs = []
        for f in det_files:
            try:
                df = pd.read_parquet(f, columns=['mjd'])
                mjd_dfs.append(df[['mjd']])
            except Exception:
                pass

        if mjd_dfs:
            det_all   = pd.concat(mjd_dfs)
            mjd_range = det_all.groupby(level=0)['mjd'].agg(
                mjd_min='min', mjd_max='max'
            )
            mjd_range['mjd_min'] -= pre_buffer

            mjd_range['mjd_max'] = np.minimum(
                mjd_range['mjd_max'] + post_buffer, mjd_now
            )
            coords = coords.join(mjd_range, how='left')
            n_with_mjd = coords['mjd_min'].notna().sum()
            print(f"MJD range obtained for {n_with_mjd}/{len(coords)} objects "
                  f"(−{pre_buffer:.0f}d / +{post_buffer:.0f}d, cap={mjd_now:.1f})")
        else:
            coords['mjd_min'] = np.nan
            coords['mjd_max'] = np.nan
            print(f"[WARN] Could not read det_*.parquet")
    else:
        coords['mjd_min'] = np.nan
        coords['mjd_max'] = np.nan
        print(f"[WARN] Could not find det_ztf_/*.parquet or det_lsst_/*.parquet in {base_dir}")

    #Descartar objetos sin rango MJD (no deberían existir)
    n_before = len(coords)
    coords = coords.dropna(subset=['mjd_min', 'mjd_max'])
    n_dropped = n_before - len(coords)
    if n_dropped > 0:
        print(f"[WARN] {n_dropped} object discarded: mjd_min/max NaN")

    print(f"{len(coords)} objects ready for ATLAS query")
    return coords


#Fase 1: envío

def submit_objects(
    coords:  pd.DataFrame,
    token:   str,
    out_dir: Path,
) -> Dict[str, str]:
    """
    Envía peticiones al servidor para todos los objetos no descargados ni
    pendientes. Devuelve el dict {oid: task_url} actualizado.

    """
    headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}
    done    = _already_done(out_dir)
    pending = _load_pending(out_dir)

    to_submit = [
        oid for oid in coords.index
        if oid not in done and oid not in pending
    ]

    if not to_submit:
        print(f"All objects already sent or downloaded.")
        return pending

    print(f"Submitting {len(to_submit)} queries "
          f"({len(done)} done, {len(pending)} pending)...")

    for oid in tqdm(to_submit, desc='Sent'):
        row = coords.loc[oid]
        payload = {
            'ra':         float(row['ra']),
            'dec':        float(row['dec']),
            'send_email': False,
        }
        if pd.notna(row.get('mjd_min')):
            payload['mjd_min'] = float(row['mjd_min'])
        if pd.notna(row.get('mjd_max')):
            payload['mjd_max'] = float(row['mjd_max'])

        try:
            resp = requests.post(
                ATLAS_QUEUE_URL, json=payload, headers=headers, timeout=30
            )
            resp.raise_for_status()
            task_url = resp.json()['url']
            pending[oid] = task_url
        except Exception as e:
            print(f"\n[WARN] Error with {oid} "
                  f"({row['ra']:.4f}, {row['dec']:.4f}): {e}")

        time.sleep(SUBMIT_DELAY)

    _save_pending(out_dir, pending)
    print(f"{len(pending)} tasks in queue. Saved in pending.json")
    return pending


#Fase 2: polling y descarga

def _fetch_result(task_url: str, headers: dict):
    """
    Comprueba si una tarea está lista y descarga el resultado.

    """
    url = task_url.rstrip('/') + '/?format=json'
    try:
        task_resp = requests.get(url, headers=headers, timeout=30)
        if task_resp.status_code == 404:
            return _EXPIRED
        task_data = task_resp.json()
    except Exception:
        return None

    if not task_data.get('finishtimestamp'):
        return None

    result_url = task_data.get('result_url')
    if not result_url:
        return pd.DataFrame()

    try:
        data_resp = requests.get(result_url, headers=headers, timeout=60)
        return _parse_atlas_text(data_resp.text)
    except Exception as e:
        print(f"\n[WARN] Error downloading results: {e}")
        return None


def _preprocess_and_save(
    oid:          str,
    raw_df:       pd.DataFrame,
    coords:       pd.DataFrame,
    preprocessor: ATLASLightcurvePreprocessor,
    out_dir:      Path,
) -> Optional[pd.DataFrame]:
    """
    Preprocesa la curva de un objeto y la guarda en raw/oid_<oid>.parquet.
    Devuelve el DataFrame preprocesado o None si queda vacío.
    """
    raw_df = raw_df.copy()
    raw_df['oid'] = oid
    raw_df = raw_df.set_index('oid')

    if oid in coords.index:
        raw_df['ra']  = coords.loc[oid, 'ra']
        raw_df['dec'] = coords.loc[oid, 'dec']

    try:
        det_pp = preprocessor.preprocess(raw_df)
    except Exception as e:
        warnings.warn(f"Preprocessing failed for {oid}: {e}")
        return None

    if det_pp is None or det_pp.empty:
        return None

    det_pp.to_parquet(_oid_path(out_dir, oid))
    return det_pp


def poll_and_download(
    token:        str,
    out_dir:      Path,
    coords:       pd.DataFrame,
    preprocessor: ATLASLightcurvePreprocessor,
    timeout:      int = POLL_MAX_WAIT,
) -> Tuple[int, int]:
    """
    Hace polling sobre todas las tareas pendientes hasta que estén resueltas
    o se alcance el timeout. Cada resultado se preprocesa y guarda en raw/.

    Las tareas que devuelven 404 (expiradas en el servidor) se reenvían
    automáticamente.
    
    """
    headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}
    pending = _load_pending(out_dir)
    done    = _already_done(out_dir)

    pending = {oid: url for oid, url in pending.items() if oid not in done}

    if not pending:
        print("There are no pending downloads.")
        return 0, 0

    print(f"Polling {len(pending)} pending tasks...")

    n_ok    = 0
    n_empty = 0
    t_start = time.time()

    with tqdm(total=len(pending), desc='Dowloading') as pbar:
        round_n = 0
        while pending:
            if time.time() - t_start > timeout:
                print(f"\n[WARN] Timeout with {len(pending)} tasks still pending.")
                print(f"Rerun with --poll-only to continue.")
                break

            round_n += 1
            elapsed = int(time.time() - t_start)
            tqdm.write(
                f"  [round {round_n}  elapsed={elapsed}s  pending={len(pending)}]"
                f" asking server..."
            )

            resolved = {}
            expired  = []
            for oid, task_url in list(pending.items()):
                result = _fetch_result(task_url, headers)
                if result is _EXPIRED:
                    expired.append(oid)
                elif result is not None:
                    resolved[oid] = result

            #Reenvío automático de tareas expiradas
            if expired:
                tqdm.write(f"[WARN] {len(expired)} tasks expired (404), resubmitting...")
                for oid in expired:
                    if oid not in coords.index:
                        tqdm.write(f"[WARN] {oid} not found in coords, skipping")
                        del pending[oid]
                        continue
                    row = coords.loc[oid]
                    payload = {
                        'ra':         float(row['ra']),
                        'dec':        float(row['dec']),
                        'send_email': False,
                    }
                    if pd.notna(row.get('mjd_min')):
                        payload['mjd_min'] = float(row['mjd_min'])
                    if pd.notna(row.get('mjd_max')):
                        payload['mjd_max'] = float(row['mjd_max'])
                    try:
                        resp = requests.post(
                            ATLAS_QUEUE_URL,
                            json=payload,
                            headers=headers,
                            timeout=30,
                        )
                        resp.raise_for_status()
                        new_url = resp.json()['url']
                        pending[oid] = new_url
                        tqdm.write(f"{oid} resubmitted -> {new_url}")
                    except Exception as e:
                        tqdm.write(f"[ERROR] Could not resubmit {oid}: {e}")
                    time.sleep(SUBMIT_DELAY)

            tqdm.write(f"{len(resolved)} tasks solved in this round")

            for oid, raw_df in resolved.items():
                if raw_df is not None and not raw_df.empty:
                    clean  = apply_quality_cuts(raw_df)
                    det_pp = _preprocess_and_save(
                        oid, clean, coords, preprocessor, out_dir
                    )
                    if det_pp is not None:
                        n_ok += 1
                    else:
                        _save_empty_marker(out_dir, oid)
                        n_empty += 1
                else:
                    _save_empty_marker(out_dir, oid)
                    n_empty += 1

                del pending[oid]
                pbar.update(1)

            if pending:
                _save_pending(out_dir, pending)
                if not resolved and not expired:
                    time.sleep(POLL_INTERVAL)

    if not pending:
        _save_pending(out_dir, {})

    return n_ok, n_empty


#Carga de curvas descargadas

def load_all_detections(out_dir: Path) -> Optional[pd.DataFrame]:
    """
    Concatena todos los parquets no vacíos de raw/oid_*.parquet.
    """
    files = sorted(_raw_dir(out_dir).glob('oid_*.parquet'))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return None

    det_all = pd.concat(dfs)
    n_obj   = det_all.index.get_level_values(0).nunique()
    print(f"{n_obj} objects with valid ATLAS detections")
    return det_all


#Extracción de features

def extract_atlas_features(
    det_all:      pd.DataFrame,
    out_dir:      Path,
    checkpoint_n: int = 100,
) -> pd.DataFrame:
    """
    Extrae features ATLAS para todos los objetos en det_all.

    """
    extractor = ATLASFeatureExtractor()
    ckpt_dir  = _ckpt_dir(out_dir)

    #Reanudar desde checkpoint
    ckpt_files = sorted(ckpt_dir.glob('features_atlas_*.parquet'))
    if ckpt_files:
        results_df = pd.concat([pd.read_parquet(f) for f in ckpt_files])
        done_oids  = set(results_df.index.get_level_values(0).unique())
        print(f"Restarting extraction: {len(done_oids)} already done")
    else:
        results_df = pd.DataFrame()
        done_oids  = set()

    oids_all = [
        oid for oid in det_all.index.get_level_values(0).unique()
        if oid not in done_oids
    ]
    print(f"Extracting features for {len(oids_all)} objects...")

    offset = len(done_oids)
    n_ok   = 0
    batch: List[pd.DataFrame] = []

    for oid in tqdm(oids_all, desc='Extracting ATLAS features...'):
        det_oid = det_all[det_all.index == oid]
        if det_oid.empty:
            continue

        try:
            feat = extractor.compute_features(detections=det_oid)
            if feat is not None and not feat.empty:
                batch.append(feat)
        except Exception as e:
            print(f"[atlas] Error with {oid}: {e}")

        n_ok += 1

        if n_ok % checkpoint_n == 0 and batch:
            chunk    = pd.concat(batch)
            ckpt_idx = offset + n_ok
            chunk.to_parquet(ckpt_dir / f'features_atlas_{ckpt_idx:06d}.parquet')
            print(f"Checkpoint: features_atlas_{ckpt_idx:06d} ({len(chunk)} objects)")
            results_df = pd.concat([results_df, chunk]) \
                         if not results_df.empty else chunk
            batch = []

    if batch:
        chunk    = pd.concat(batch)
        ckpt_idx = offset + n_ok
        chunk.to_parquet(ckpt_dir / f'features_atlas_{ckpt_idx:06d}.parquet')
        print(f"Final checkpoint: features_atlas_{ckpt_idx:06d}")
        results_df = pd.concat([results_df, chunk]) \
                     if not results_df.empty else chunk

    return results_df


#Postprocesado

def postprocess_atlas_features(features: pd.DataFrame) -> pd.DataFrame:

    if features.empty:
        return features
    cols = [c for c in features.columns if c not in ATLAS_RM_COLS]
    out  = features[cols].copy()
    out['survey'] = SURVEY_ID_ATLAS
    out.columns = [
        f'{c}_atlas' if c != 'survey' else c
        for c in out.columns
    ]
    return out


def _save_branch_safely(df_branch: pd.DataFrame, branch_path: Path, branch: str) -> None:
    if branch_path.exists():
        try:
            prev = pd.read_parquet(branch_path, columns=[])
            if len(df_branch) < len(prev):
                print(f"{branch_path.name}: ({len(df_branch)} objects) "
                      f"has fewer objects than previous ({len(prev)} objects) "
                      f"It won't overwrite. Delete manually to force.")
                return
        except Exception as e:
            print(f"Could not read {branch_path.name} to compare ({e}), it will overwrite anyway.")

    df_branch.to_parquet(branch_path)
    print(f"{branch_path.name}  ({len(df_branch)} objects)")


#Resumen

def print_coverage_summary(coords: pd.DataFrame, features: pd.DataFrame) -> None:
    n_total = len(coords)
    n_atlas = len(features)
    print(f"\n== Coverage summary ==")
    print(f"ZTF/LSST objects:    {n_total}")
    print(f"With ATLAS:  {n_atlas}  ({100*n_atlas/n_total:.1f}%)")
    print(f"Without ATLAS: {n_total - n_atlas}")


#Main

def main():
    parser = argparse.ArgumentParser(
        description='Query and extraction of ATLAS features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Use cases:
  # Download + extraction:
  python atlas_pipeline.py --username U --password P

  # If interrupted, rerun wih same parameters.

  # Polling only (submit already done):
  python atlas_pipeline.py --username U --password P --poll-only

  # Extraction only (raw/ already dowloaded):
  python atlas_pipeline.py --extract-only
        """,
    )
    parser.add_argument('--username',     default=None)
    parser.add_argument('--password',     default=None)
    parser.add_argument(
        '--obj-glob',
        default='./data/raw_features/checkpoints_both/obj_*.parquet',
        help='Glob for obj_*.parquet from ztf_lsst.py (adjust suffix '
             'checkpoints_{both,ztf_only,lsst_only}).',
    )
    parser.add_argument(
        '--output-dir',
        default='./data/raw_features/atlas/',
    )
    parser.add_argument(
        '--feat-dir',
        default='./data/raw_features/',
        help='Directorio con features_{comb}_{strict,relaxed}.parquet de ztf_lsst.py '
             '(mismo --output-base usado allí). Si se especifica, solo se consultan '
             'objetos de esas ramas.'
    )
    parser.add_argument('--pre-buffer',  type=float, default=MJD_PRE_BUFFER,
                        help=f'Days before first detection (default: {MJD_PRE_BUFFER:.0f}d)')
    parser.add_argument('--post-buffer', type=float, default=MJD_POST_BUFFER,
                        help=f'Days after last detection (default: {MJD_POST_BUFFER:.0f}d)')
    parser.add_argument('--checkpoint-n', type=int,   default=100,
                        help='Features checkpoint every N obkects')
    parser.add_argument('--submit-only',  action='store_true',
                        help='Query submition only, no polling or extracción')
    parser.add_argument('--poll-only',    action='store_true',
                        help='Polling only, without new queries')
    parser.add_argument('--extract-only', action='store_true',
                        help='Extraction of features from downloaded raw/ only')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar coordenadas
    print("\nLoading objects from obj_ztf_*.parquet")
    feat_dir_arg = Path(args.feat_dir) if args.feat_dir else None
    coords = load_object_coords(
        args.obj_glob,
        pre_buffer=args.pre_buffer,
        post_buffer=args.post_buffer,
        feat_dir=feat_dir_arg,
    )

    # Modo --extract-only
    if args.extract_only:
        det_all = load_all_detections(out_dir)
        if det_all is None:
            print("No ATLAS lightcurves in raw/. Run without --extract-only first.")
            return
        features_raw   = extract_atlas_features(det_all, out_dir, args.checkpoint_n)
        features_atlas = postprocess_atlas_features(features_raw)
        print_coverage_summary(coords, features_atlas)
        feat_dir_arg = Path(args.feat_dir) if args.feat_dir else None
        if feat_dir_arg is not None:
            atlas_oids = set(features_atlas.index.get_level_values(0).unique())
            for branch in ('strict', 'relaxed'):
                p = feat_dir_arg / f'features_comb_{branch}.parquet'
                if not p.exists():
                    print(f"Not found: {p.name}. Skipping {branch} branch")
                    continue
                oids_branch = set(
                    pd.read_parquet(p, columns=[])
                    .index.get_level_values(0).unique()
                )
                df_branch = features_atlas[
                    features_atlas.index.get_level_values(0)
                    .isin(oids_branch & atlas_oids)
                ]
                branch_path = feat_dir_arg / f'features_atlas_{branch}.parquet'
                _save_branch_safely(df_branch, branch_path, branch)
        return

    #Resto de modos: necesitan autenticación
    if not args.username or not args.password:
        parser.error('--username and --password are required, except in --extract-only mode')

    print("\nAuthenticating with ATLAS server...")
    try:
        token = get_atlas_token(args.username, args.password)
        print("Token OK")
    except Exception as e:
        print(f"Authentication ERROR: {e}")
        return

    preprocessor = ATLASLightcurvePreprocessor()
    done = _already_done(out_dir)
    print(f"\nStatus: {len(done)}/{len(coords)} objects already done")

    #2. Fase SUBMIT
    if not args.poll_only:
        print("\n== SUBMIT phase ==")
        submit_objects(coords, token, out_dir)

    if args.submit_only:
        print("\n--submit-only mode completed.")
        print("Rerun with --poll-only when the server completes the tasks.")
        return

    # 3. Fase POLL
    print("\n== POLL phase ==")
    n_ok, n_empty = poll_and_download(token, out_dir, coords, preprocessor)

    done_after = _already_done(out_dir)
    pending    = _load_pending(out_dir)
    n_pending  = sum(1 for v in pending.values() if v)

    print(f"\nDowloading completed:")
    print(f"Valid data: {n_ok}")
    print(f"Without valid data: {n_empty}")
    print(f"Total downloaded: {len(done_after)}/{len(coords)}")
    print(f"Pending in server: {n_pending}")

    if n_pending > 0:
        print(f"\nRerun with --poll-only to continue.")
        return

    # 4. Extracción de features
    print("\n== Feature extraction ==")
    det_all = load_all_detections(out_dir)
    if det_all is None:
        print("No valid data after preprocessing.")
        return

    features_raw   = extract_atlas_features(det_all, out_dir, args.checkpoint_n)
    features_atlas = postprocess_atlas_features(features_raw)

    #5. Split strict / relaxed y guardar
    print_coverage_summary(coords, features_atlas)

    if feat_dir_arg is not None:
        atlas_oids = set(features_atlas.index.get_level_values(0).unique())
        for branch in ('strict', 'relaxed'):
            p = feat_dir_arg / f'features_comb_{branch}.parquet'
            if not p.exists():
                print(f"Not found: {p.name}. Skipping {branch} branch")
                continue
            oids_branch = set(
                pd.read_parquet(p, columns=[])
                .index.get_level_values(0).unique()
            )
            df_branch = features_atlas[
                features_atlas.index.get_level_values(0)
                .isin(oids_branch & atlas_oids)
            ]

            branch_path = feat_dir_arg / f'features_atlas_{branch}.parquet'
            _save_branch_safely(df_branch, branch_path, branch)


if __name__ == '__main__':
    main()