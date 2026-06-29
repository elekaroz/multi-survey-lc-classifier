import numpy as np
np.NaN = np.nan
import pandas as pd
from lc_classifier.features import LSSTLightcurvePreprocessor, ZTFLightcurvePreprocessor, ZTFFeatureExtractor3bands
from lc_classifier.features import ZTFFeatureExtractor, LSSTFeatureExtractor
from lc_classifier.features import FeatureExtractorComposer, HarmonicsExtractor, PeriodExtractor
from tqdm import tqdm
from alerce.core import Alerce
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import os
import time
import argparse
from simulations.features_config import postprocess_features
 
#%%
 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.RankWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger().setLevel(logging.CRITICAL)
 
#%%

####CONFIGURACIÓN######
 
#Modo de survey

# 'both'      -> pipeline completo ZTF + LSST
#                Lee ztf_list.txt + lsst_list.txt, columnas con ambos oids o una tupla. oid_combined = {lsst_id}_{ztf_id}
# 'ztf_only'  -> solo ZTF, sin contrapartida LSST
#                Lee ztf_list.txt, columna con oid ZTF o una lista ZTF. oid_combined = ztf_id
# 'lsst_only' -> solo LSST, sin contrapartida ZTF
#                Lee lsst_only_list.txt, columna con oid LSST o una lista LSST. oid_combined = lsst_id

SURVEY_MODE = 'lsst_only'

#Rama B (relaxed)
# Solo aplica en SURVEY_MODE = 'both'. Si False, solo se ejecuta Rama A (strict).

# Strict  -> para que un objeto no de descarte, cada survey tiene que pasar individualmente el preprocesador
#            (>5 det en al menos una banda)
#Relaxed  -> basta con que la curva de luz combinada pase el preprocesador, aunque un survey individual no lo haga.
#            Útil para clasificación temprana con Modelo D (combinado), cuando hay pocas detecciones todavía.
#            Las features no combinadas y la clasifiación del resto de modelos puede empeorar considerablemente.

ENABLE_BRANCH_B = True

#Fuente de OIDs

# Puede ser:
#   - lista con oids de un survey, o tupla con oids  ([ZTF_oid1, ZTF_oid2...], [LSST_oid1, LSST_oid2...])
#   - un directorio con ztf_list.txt y/o lsst_list.txt
#   - un fichero .csv con columnas que contengan 'ZTF' y/o 'LSST' en el nombre (insensible a mayúsculas)
#
# OIDS_SOURCE_DEFAULT solo se usa en ejecución directa (Spyder/IPython) sin
# --oids-source, aquí se deja un par de OIDs de ejemplo.
# Por CLI, --oids-source acepta una ruta a .csv, una ruta a un directorio con
# ztf_list.txt/lsst_list.txt, o (en modos ztf_only/lsst_only) una lista plana
# de OIDs separados por comas.

OIDS_SOURCE_DEFAULT = ['313994139532263447', '170411112000913487']

#Chekpoint por si falla el fetch o extracción

CHECKPOINT_FETCH   = 100  # guardar cada N objetos (fetch)
CHECKPOINT_EXTRACT = 100  # guardar cada N objetos (extracción)

#Directorio de salida

# Las features finales (features_{ztf,lsst,comb}_{strict,relaxed}.parquet) se
# escriben directamente aquí, sin subcarpeta por modo — coincide con
# --features-dir de xmatch.py. Dentro de OUTPUT_BASE se crea además
# checkpoints_{SURVEY_MODE}/ para los checkpoints intermedios de fetch/
# extracción (obj_*.parquet, det_*.parquet), que xmatch.py lee vía --obj-dir
OUTPUT_BASE = './data/raw_features/'

#Modo de ejecución

# SKIP_FETCH = True  -> salta la descarga y va directo a preprocesamiento
#                       y extracción de features usando los checkpoints existentes.
# SKIP_FETCH = False -> ejecuta el pipeline completo (descarga + extracción).
# En modos lsst_only/ztf_only se omite automáticamente el fetch del survey ausente.
SKIP_FETCH = False

#API
#Parámetros de robustez frente a errores de API

FETCH_DELAY   = 0.5   # segundos entre llamadas exitosas
RETRY_403_WAIT = 60   # segundos de espera tras un error 403 antes de reintentar

#%%

def parse_args():
    parser = argparse.ArgumentParser(
        description="ZTF/LSST query via ALeRCE and feature extraction.")
    parser.add_argument('--survey-mode', choices=['both', 'ztf_only', 'lsst_only'],
                         default=SURVEY_MODE,
                         help="Survey mode: both / ztf_only / lsst_only.")
    parser.add_argument('--no-branch-b', dest='enable_branch_b', action='store_false',
                         default=ENABLE_BRANCH_B,
                         help="Disable B branch (relaxed). Only applies to --survey-mode both.")
    parser.add_argument('--oids-source', default=None,
                         help="Path to CSV, a directory with ztf_list.txt/lsst_list.txt, "
                              "or list of OIDs separated with commas (ztf_only/lsst_only modes). "
                              "If not given, example OIDS_SOURCE_DEFAULT will be used.")
    parser.add_argument('--output-base', default=OUTPUT_BASE,
                         help="Output directory (features and checkpoints).")
    parser.add_argument('--checkpoint-fetch', type=int, default=CHECKPOINT_FETCH,
                         help="Checkpoint for fetch every N objects.")
    parser.add_argument('--checkpoint-extract', type=int, default=CHECKPOINT_EXTRACT,
                         help="Checkpoint for feature extraction every N objects.")
    parser.add_argument('--skip-fetch', action='store_true', default=SKIP_FETCH,
                         help="Skip query and extract features with existing checkpoints.")
    parser.add_argument('--fetch-delay', type=float, default=FETCH_DELAY,
                         help="Delay time between successful queries.")
    parser.add_argument('--retry-403-wait', type=float, default=RETRY_403_WAIT,
                         help="Delay time after 403 error before retrying.")
    return parser.parse_args()


args = parse_args()

SURVEY_MODE        = args.survey_mode
ENABLE_BRANCH_B    = args.enable_branch_b
OUTPUT_BASE        = args.output_base
checkpoint_fetch   = args.checkpoint_fetch
checkpoint_extract = args.checkpoint_extract
SKIP_FETCH         = args.skip_fetch
FETCH_DELAY        = args.fetch_delay
RETRY_403_WAIT     = args.retry_403_wait

# fuente de OIDs: CLI tiene prioridad; si no se da, se usa el default de Spyder
_oids_source = args.oids_source if args.oids_source is not None else OIDS_SOURCE_DEFAULT

checkpoint_dir = os.path.join(OUTPUT_BASE, f'checkpoints_{SURVEY_MODE}', '')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

#%%

def _load_oids(source, mode):
    
    """
    Carga listas de OIDs desde:
      - una lista de Python (OIDs directos)
      - un fichero .csv
      - un directorio con ztf_list.txt / lsst_list.txt
    """
    #Lista directa
    
    if isinstance(source, (list, tuple)):
        if mode == 'both':
            if (len(source) == 2
                    and isinstance(source[0], (list, tuple))
                    and isinstance(source[1], (list, tuple))):
                return list(source[0]), list(source[1])
            raise ValueError(
                "mode='both' with list requires "
                "([ztf_oids], [lsst_oids])")
        oids = [str(o).strip() for o in source if str(o).strip()]
        if mode == 'ztf_only':
            return oids, []
        else:
            return [], oids
        
     #CSV
     
    if isinstance(source, str) and source.lower().endswith('.csv'):
        df = pd.read_csv(source, dtype=str)
        ztf_cols  = [c for c in df.columns if 'ztf'  in c.lower()]
        lsst_cols = [c for c in df.columns if 'lsst' in c.lower()]

        if mode == 'ztf_only':
            if not ztf_cols:
                raise ValueError(
                    f"No columns with 'ZTF' were found in {source}. "
                    f"Columns: {list(df.columns)}")
            oids = df[ztf_cols[0]].dropna().str.strip()
            return oids[oids != ''].tolist(), []

        elif mode == 'lsst_only':
            if not lsst_cols:
                raise ValueError(
                    f"No columns with 'LSST' were found in {source}. "
                    f"Columns: {list(df.columns)}")
            oids = df[lsst_cols[0]].dropna().str.strip()
            return [], oids[oids != ''].tolist()

        else:  # both
            if not ztf_cols or not lsst_cols:
                raise ValueError(
                    f"mode='both' requirea at least one ZTF and one LSST column. "
                    f"ZTF={ztf_cols}, LSST={lsst_cols} were found in {source}")
            zcol, lcol = ztf_cols[0], lsst_cols[0]
            paired = df[[zcol, lcol]].dropna().apply(lambda s: s.str.strip())
            paired = paired[(paired[zcol] != '') & (paired[lcol] != '')]
            print(f"  CSV mode='both': {len(paired)} pairs in "
                  f"'{zcol}' y '{lcol}' (of {len(df)} rows)")
            return paired[zcol].tolist(), paired[lcol].tolist()
        
    #Lista de OIDs separados por comas
    if isinstance(source, str) and not source.lower().endswith('.csv') \
            and not os.path.isdir(source) and ',' in source:
        if mode == 'both':
            raise ValueError(
                "A list of OIDs separated with commas is not valid in"
                "mode='both', use a .csv or directory with"
                "ztf_list.txt/lsst_list.txt instead.")
        oids = [o.strip() for o in source.split(',') if o.strip()]
        return (oids, []) if mode == 'ztf_only' else ([], oids)

    #TXT (directorio con ztf_list.txt / lsst_list.txt)

    else:
        if mode == 'both':
            with open(os.path.join(source, 'ztf_list.txt')) as f:
                ztf = f.read().splitlines()
            with open(os.path.join(source, 'lsst_list.txt')) as f:
                lsst = f.read().splitlines()
            return ztf, lsst
        elif mode == 'lsst_only':
            with open(os.path.join(source, 'lsst_list.txt')) as f:
                lsst = f.read().splitlines()
            return [], lsst
        elif mode == 'ztf_only':
            with open(os.path.join(source, 'ztf_list.txt')) as f:
                ztf = f.read().splitlines()
            return ztf, []
        else:
            raise ValueError(f"SURVEY_MODE not valid: '{mode}'")
            

ztf_oid, lsst_oid = _load_oids(_oids_source, SURVEY_MODE)

print(f"SURVEY_MODE={SURVEY_MODE!r}  ZTF={len(ztf_oid)}  LSST={len(lsst_oid)}")

#%%
 
# Lista persistente de OIDs con 404
# Se carga al inicio y se actualiza en disco tras cada nuevo 404.
# Evita hacer requests innecesarias a objetos inexistentes en ALeRCE.
_404_path = checkpoint_dir + 'skipped_404.txt'
 
def load_404_list():
    if os.path.exists(_404_path):
        with open(_404_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()
 
def save_404(oid):
    with open(_404_path, 'a') as f:
        f.write(oid + '\n')
 
skipped_404 = load_404_list()
print(f"{len(skipped_404)} OIDs in 404s list")
# ─────────────────────────────────────────────────────────────────────────────
 
def _flush_checkpoint(det_list, ndet_list, obj_list, name, n_ok):
    valid_det  = [x for x in det_list  if x is not None]
    valid_ndet = [x for x in ndet_list if x is not None]
    valid_obj  = [x for x in obj_list  if x is not None]
    if valid_det:
        suffix = f"{name}_{n_ok:06d}"
        pd.concat(valid_det ).to_parquet(f"{checkpoint_dir}det_{suffix}.parquet")
        pd.concat(valid_ndet).to_parquet(f"{checkpoint_dir}ndet_{suffix}.parquet")
        pd.concat(valid_obj ).to_parquet(f"{checkpoint_dir}obj_{suffix}.parquet")
        print(f"Checkpoint saved: {suffix} ({len(valid_det)} objects)")
 
def load_checkpoints(name):
    det_files = sorted(glob.glob(f"{checkpoint_dir}det_{name}_*.parquet"))
    if not det_files:
        return None, None, None
    det  = pd.concat([pd.read_parquet(f) for f in det_files])
    ndet = pd.concat([pd.read_parquet(f.replace('det_', 'ndet_')) for f in det_files])
    obj  = pd.concat([pd.read_parquet(f.replace('det_', 'obj_' )) for f in det_files])
    # OIDs ya obtenidos
    done_oids = set(det.index.get_level_values(0).unique())
    print(f"{len(done_oids)} objects already done in '{name}' checkpoints")
    return det, ndet, obj
 
def save_features_checkpoint(features_dict, idx, namespace='strict'):
    suffix = f"{idx:06d}"
    for key, df in features_dict.items():
        if df is not None and not df.empty:
            df.to_parquet(f"{checkpoint_dir}features_{namespace}_{key}_{suffix}.parquet")
    print(f"Features checkpoint saved: features_{namespace} {suffix}")
 
def load_features_checkpoints(namespace='strict'):
    result = {}
    for key in ('ztf', 'lsst', 'combined'):
        files = sorted(glob.glob(f"{checkpoint_dir}features_{namespace}_{key}_*.parquet"))
        if files:
            result[key] = pd.concat([pd.read_parquet(f) for f in files])
            print(f"{len(result[key])} rows already extracted for '{namespace}/{key}'")
        else:
            result[key] = pd.DataFrame()
    return result
 
#%%
 
#query paralelo
 
def fetch_ztf_data(ztf_id):
    alerce = Alerce()
    with ThreadPoolExecutor(max_workers=5) as executor:
        f_det    = executor.submit(alerce.query_detections,        ztf_id, format='pandas', survey='ztf')
        f_ndet   = executor.submit(alerce.query_non_detections,    ztf_id, format='pandas', survey='ztf')
        f_forced = executor.submit(alerce.query_forced_photometry, ztf_id, format='pandas', survey='ztf')
        f_obj    = executor.submit(alerce.query_object,            ztf_id, format='pandas', survey='ztf')
        f_sg     = executor.submit(alerce.query_feature,           ztf_id, 'sgscore1', format='pandas', survey='ztf')
 
    return f_det.result(), f_ndet.result(), f_forced.result(), f_obj.result(), f_sg.result()
 
 
def fetch_lsst_data(lsst_id):
    alerce = Alerce()
    with ThreadPoolExecutor(max_workers=4) as executor:
        f_det    = executor.submit(alerce.query_detections,        lsst_id, format='pandas', survey='lsst')
        f_ndet   = executor.submit(alerce.query_non_detections,    lsst_id, format='pandas', survey='lsst')
        f_forced = executor.submit(alerce.query_forced_photometry, lsst_id, format='pandas', survey='lsst')
        f_obj    = executor.submit(alerce.query_object,            lsst_id, format='pandas', survey='lsst')
 
    return f_det.result(), f_ndet.result(), f_forced.result(), f_obj.result()
 
#%%
 
#procesar datos ztf
 
def process_ztf_object(i, ztf_id, lsst_id):
    """Process a single ZTF object."""
    oid_combined = ztf_id if SURVEY_MODE == 'ztf_only' else f"{lsst_id}_{ztf_id}"
 
    detections_oid, non_detections_oid, forced_phot_oid, object_info_oid, sgscore = fetch_ztf_data(ztf_id)
 
    # add oid column
    detections_oid['oid']     = oid_combined
    non_detections_oid['oid'] = oid_combined
    object_info_oid['oid']    = oid_combined
 
    # sgscore
    if sgscore.empty or 'value' not in sgscore.columns:
        sgscore_oid = 0.5
    else:
        valid_values = sgscore['value'].dropna()
        sgscore_oid = float(valid_values.iloc[0]) if len(valid_values) > 0 else 0.5
 
    detections_oid['sgscore1'] = sgscore_oid
 
    # set index
    detections_oid     = detections_oid.set_index('oid')
    non_detections_oid = non_detections_oid.set_index('oid')
    object_info_oid    = object_info_oid.set_index('oid')
 
    # forced photometry
    if not forced_phot_oid.empty:
        # forced_phot_oid = forced_phot_oid[forced_phot_oid['fid'].isin([1, 2])].copy()
        forced_phot_oid['oid'] = oid_combined
        forced_phot_oid = forced_phot_oid.set_index('oid')
 
        forced_phot_oid = forced_phot_oid.drop(columns=[
            'e_ra', 'e_dec', 'field', 'procstatus', 'magzpscirms', 'ranr', 'clrcounc',
            'parent_candid', 'scibckgnd', 'exptime', 'magnr', 'scisigpix', 'adpctdif1',
            'adpctdif2', 'sigmagnr', 'magzpsci', 'chinr', 'programid', 'sharpnr',
            'magzpsciunc', 'rcid', 'distnr', 'clrcoeff', 'sciinpseeing', 'decnr'
        ], errors='ignore')
        forced_phot_oid = forced_phot_oid.rename(columns={
            'mag': 'magpsf', 'e_mag': 'sigmapsf',
            'mag_corr': 'magpsf_corr', 'e_mag_corr': 'sigmapsf_corr',
            'e_mag_corr_ext': 'sigmapsf_corr_ext'
        })
 
        forced_phot_oid['nid']         = np.floor(forced_phot_oid['mjd']) - 57754
        forced_phot_oid['magap']       = np.nan
        forced_phot_oid['sigmagap']    = np.nan
        forced_phot_oid['magapbig']    = np.nan
        forced_phot_oid['sigmagapbig'] = np.nan
        forced_phot_oid['phase']       = 0.0
        forced_phot_oid['rb']          = 1.0
        forced_phot_oid['drb']         = 1.0
 
        if not detections_oid.empty:
            first_det    = detections_oid.iloc[0]
            rbversion    = first_det.get('rbversion', None)
            step_id_corr = first_det.get('step_id_corr', None)
        else:
            rbversion, step_id_corr = None, None
 
        forced_phot_oid['rbversion']    = rbversion
        forced_phot_oid['step_id_corr'] = step_id_corr
        forced_phot_oid['sgscore1']     = sgscore_oid
 
        detections_all = pd.concat([detections_oid, forced_phot_oid])
    else:
        detections_all = detections_oid
 
    return detections_all, non_detections_oid, object_info_oid
 
 
#%%
 
#procesar datos lsst
 
def process_lsst_object(i, ztf_id, lsst_id):
    """Process a single LSST object."""
    oid_combined = lsst_id if SURVEY_MODE == 'lsst_only' else f"{lsst_id}_{ztf_id}"
 
    detections_oid, non_detections_oid, forced_phot_oid, object_info_oid = fetch_lsst_data(lsst_id)
 
    detections_oid['oid']     = oid_combined
    non_detections_oid['oid'] = oid_combined
    object_info_oid['oid']    = oid_combined
 
    if not forced_phot_oid.empty:
        forced_phot_oid['oid'] = oid_combined
        detections_oid  = pd.concat([detections_oid, forced_phot_oid])
 
    # epoch averaging (weighted mean per mjd day)
    detections_oid['mjd']    = np.floor(detections_oid['mjd'])
    detections_oid['w']      = 1 / detections_oid['psfFluxErr'] ** 2
    detections_oid['w_flux'] = detections_oid['w'] * detections_oid['psfFlux']
 
    detections_mean = detections_oid.groupby(['band', 'mjd'], as_index=False).agg(
        sum_w        = ('w',            'sum'),
        sum_w_flux   = ('w_flux',       'sum'),
        reliability  = ('reliability',  'mean'),
        extendedness = ('extendedness', 'mean')
    )
 
    detections_mean['flux_mean'] = detections_mean['sum_w_flux'] / detections_mean['sum_w']
    detections_mean['flux_err']  = np.sqrt(1 / detections_mean['sum_w'])
    detections_mean['magpsf']    = 31.4 - 2.5 * np.log10(np.abs(detections_mean['flux_mean']))
    detections_mean['sigmapsf']  = (2.5 / np.log(10)) * detections_mean['flux_err'] / np.abs(detections_mean['flux_mean'])
 
    detections_mean.rename(columns={'band': 'fid', 'reliability': 'rb'}, inplace=True)
 
    detections_mean['magpsf_ml']   = detections_mean['magpsf']
    detections_mean['sigmapsf_ml'] = detections_mean['sigmapsf']
    detections_mean['extendedness'] = detections_mean['extendedness'].fillna(0.5)
    detections_mean['sgscore1']    = 1 - detections_mean['extendedness']
    detections_mean['isdiffpos']   = np.where(detections_mean['flux_mean'] > 0, 1, -1)
    detections_mean['rb'] = detections_mean['rb'].fillna(0.0)
 
    detections_mean['magpsf_corr']       = detections_mean['magpsf']
    detections_mean['sigmapsf_corr']     = detections_mean['sigmapsf']
    detections_mean['sigmapsf_corr_ext'] = detections_mean['sigmapsf']
 
    detections_mean['tid']         = 'lsst'
    detections_mean['ra']          = object_info_oid['meanra'].iloc[0]
    detections_mean['dec']         = object_info_oid['meandec'].iloc[0]
 
    detections_mean['oid'] = oid_combined
    detections_mean        = detections_mean.set_index('oid')
 
    object_info_oid['corrected'] = False
    object_info_oid['oid']       = oid_combined
    object_info_oid              = object_info_oid.set_index('oid')
 
    return detections_mean, non_detections_oid, object_info_oid
 
#%%
 
def preprocess(preprocessor, detections, non_detections, object_info):
    '''
    Aplica el preprocesador a los objetos. Entre otras cosas, aquí se
    descartan los objetos que no llegan al mínimo de detecciones.
    '''
 
    detections_pp = preprocessor.preprocess(detections, objects=object_info)
    non_detections_pp = preprocessor.rename_columns_non_detections(non_detections)
 
    return detections_pp, non_detections_pp
 
#%%
 
#extracción de fatures combinadas y separadas por survey
 
def extract_features(detections_pp, non_detections_pp,
                                 object_info, bands_ztf=(1,2,3),
                                 bands_lsst=(1,2,3,4,5,6), checkpoint=None,
                                 namespace='strict'):
    """
    Extrae features separadas para ZTF y LSST del mismo objeto,
    además de las features de la curva de luz combinada.

    """

    if checkpoint is None:
        checkpoint = checkpoint_extract

    #features combinadas
    extractor_combined = LSSTFeatureExtractor(bands=bands_lsst, stream=False)
 
    #features ZTF
    extractor_ztf  = ZTFFeatureExtractor3bands(bands=bands_ztf, stream=False)
 
    #features LSST
    extractor_lsst = LSSTFeatureExtractor(bands=bands_lsst, stream=False)
 
    #cargar checkpoint (si lo hay)
    results = load_features_checkpoints(namespace)
    done_oids = set()
 
    for df in results.values():
        if not df.empty:
            done_oids.update(df.index.get_level_values(0).unique())
    if done_oids:
        print(f"Restarting extraction [{namespace}]: {len(done_oids)} objects already done")
 
    #reanudar desde checkpoint (si lo hay)
    oids = [oid for oid in detections_pp.index.get_level_values(0).unique()
            if oid not in done_oids]
 
    offset_feat = len(done_oids)
    n_ok_feat   = 0
 
    features_combined_batch = []
    features_ztf_batch      = []
    features_lsst_batch     = []
 
    for oid in tqdm(oids, desc="Extracting features..."):
        det_oid  = detections_pp[detections_pp.index == oid]
        ndet_oid = non_detections_pp[non_detections_pp.index == oid]
 
        # separar por survey
        det_ztf  = det_oid[det_oid['tid'] == 'ztf']
        det_lsst = det_oid[det_oid['tid'] == 'lsst']
 
        # print(f"{oid}: {len(det_ztf)} det ZTF, {len(det_lsst)} det LSST, tids={sorted(det_oid['tid'].unique().tolist())}")
 
        # combined, siempre añade una fila (NaN si falla)
        try:
            feat_combined = extractor_combined.compute_features(
                detections=det_oid, non_detections=ndet_oid)
        except Exception as e:
            print(f"[combined] Error in {oid}: {e}")
            feat_combined = pd.DataFrame(index=[oid])
        features_combined_batch.append(feat_combined)
 
        # ZTF, siempre añade una fila (NaN si no hay datos o si falla)
        try:
            feat_ztf = extractor_ztf.compute_features(
                detections=det_ztf, non_detections=ndet_oid)
            feat_ztf.columns = [f'{c}_ztf' if c not in ['oid']
                                else c for c in feat_ztf.columns]
        except Exception as e:
            print(f"[ztf] Error in {oid}: {e}")
            feat_ztf = pd.DataFrame(index=[oid])
        features_ztf_batch.append(feat_ztf)
 
        # LSST, siempre añade una fila (NaN si no hay datos o si falla)
        try:
            feat_lsst = extractor_lsst.compute_features(
                detections=det_lsst, non_detections=ndet_oid)
            feat_lsst.columns = [f'{c}_lsst' if c not in ['oid']
                                 else c for c in feat_lsst.columns]
        except Exception as e:
            print(f"[lsst] Error in {oid}: {e}")
            feat_lsst = pd.DataFrame(index=[oid])
        features_lsst_batch.append(feat_lsst)
 
        n_ok_feat += 1
 
        if n_ok_feat % checkpoint == 0:
            batch_results = {
                'combined': pd.concat(features_combined_batch) if features_combined_batch else pd.DataFrame(),
                'ztf':      pd.concat(features_ztf_batch)      if features_ztf_batch      else pd.DataFrame(),
                'lsst':     pd.concat(features_lsst_batch)     if features_lsst_batch     else pd.DataFrame(),
            }
            save_features_checkpoint(batch_results, offset_feat + n_ok_feat, namespace)
            for key in results:
                if not batch_results[key].empty:
                    results[key] = pd.concat([results[key], batch_results[key]]) \
                                   if not results[key].empty else batch_results[key]
            features_combined_batch = []
            features_ztf_batch      = []
            features_lsst_batch     = []
 
    # último batch
    if features_combined_batch or features_ztf_batch or features_lsst_batch:
        batch_results = {
            'combined': pd.concat(features_combined_batch) if features_combined_batch else pd.DataFrame(),
            'ztf':      pd.concat(features_ztf_batch)      if features_ztf_batch      else pd.DataFrame(),
            'lsst':     pd.concat(features_lsst_batch)     if features_lsst_batch     else pd.DataFrame(),
        }
        save_features_checkpoint(batch_results, offset_feat + n_ok_feat, namespace)
        for key in results:
            if not batch_results[key].empty:
                results[key] = pd.concat([results[key], batch_results[key]]) \
                               if not results[key].empty else batch_results[key]
 
    return results
 
#%%
 
if not SKIP_FETCH:
    print("\ncheckpoint_dir:", checkpoint_dir)
    print("checkpoints:", len(glob.glob(f"{checkpoint_dir}det_ztf_*.parquet")))

    #Fetch ZTF
    if SURVEY_MODE != 'lsst_only':
        det_ztf_prev, ndet_ztf_prev, obj_ztf_prev = load_checkpoints('ztf')
        done_ztf = set(det_ztf_prev.index.get_level_values(0).unique()) if det_ztf_prev is not None else set()
        print(f"done_ztf: {len(done_ztf)} OIDs")

        print("\nObtaining ZTF")
        if SURVEY_MODE == 'ztf_only':
            pairs_ztf = [(z, '') for z in ztf_oid
                         if z not in done_ztf and z not in skipped_404]
        else:
            pairs_ztf = [(z, l) for z, l in zip(ztf_oid, lsst_oid)
                         if f"{l}_{z}" not in done_ztf and z not in skipped_404]
        print(f"{len(done_ztf)} ZTF objects already done, ({len(skipped_404)} known 404s), {len(pairs_ztf)} remaining")

        det_batch_ztf  = []
        ndet_batch_ztf = []
        obj_batch_ztf  = []
        offset_ztf = len(done_ztf)
        n_ok_ztf   = 0

        for i, (ztf_id, lsst_id) in enumerate(tqdm(pairs_ztf, desc="Obtaining ZTF objects...")):
            if ztf_id in skipped_404:
                continue
            try:
                det, ndet, obj = process_ztf_object(i, ztf_id, lsst_id)
                det_batch_ztf.append(det)
                ndet_batch_ztf.append(ndet)
                obj_batch_ztf.append(obj)
                n_ok_ztf += 1
                time.sleep(FETCH_DELAY)
            except Exception as e:
                err_str = str(e)
                if '404' in err_str:
                    skipped_404.add(ztf_id)
                    save_404(ztf_id)
                elif '403' in err_str:
                    print(f"Error 403 for {ztf_id}. Witing {RETRY_403_WAIT}s before retrying...")
                    time.sleep(RETRY_403_WAIT)
                    try:
                        det, ndet, obj = process_ztf_object(i, ztf_id, lsst_id)
                        det_batch_ztf.append(det)
                        ndet_batch_ztf.append(ndet)
                        obj_batch_ztf.append(obj)
                        n_ok_ztf += 1
                        time.sleep(FETCH_DELAY)
                    except Exception as e2:
                        print(f"Error obtaining ZTF object {ztf_id} (retry): {e2}")
                else:
                    print(f"Error obtaining ZTF object {ztf_id}: {e}")

            if n_ok_ztf > 0 and n_ok_ztf % checkpoint_fetch == 0:
                _flush_checkpoint(det_batch_ztf, ndet_batch_ztf, obj_batch_ztf,
                                  'ztf', offset_ztf + n_ok_ztf)
                det_batch_ztf  = []
                ndet_batch_ztf = []
                obj_batch_ztf  = []

        if det_batch_ztf:
            _flush_checkpoint(det_batch_ztf, ndet_batch_ztf, obj_batch_ztf,
                              'ztf', offset_ztf + n_ok_ztf)
    # end SURVEY_MODE != 'lsst_only'
 
    #Fetch LSST
    #%%
    if SURVEY_MODE != 'ztf_only':
        print("Obtaining LSST")

        det_lsst_prev, ndet_lsst_prev, obj_lsst_prev = load_checkpoints('lsst')
        done_lsst = set(det_lsst_prev.index.get_level_values(0).unique()) if det_lsst_prev is not None else set()
        if SURVEY_MODE == 'lsst_only':
            pairs_lsst = [('', l) for l in lsst_oid
                          if l not in done_lsst and l not in skipped_404]
        else:
            pairs_lsst = [(z, l) for z, l in zip(ztf_oid, lsst_oid)
                          if f"{l}_{z}" not in done_lsst and l not in skipped_404]
        print(f"{len(done_lsst)} LSST objects already done, ({len(skipped_404)} known 404s), {len(pairs_lsst)} remaining")

        det_batch_lsst  = []
        ndet_batch_lsst = []
        obj_batch_lsst  = []

        offset_lsst = len(done_lsst)
        n_ok_lsst   = 0

        for i, (ztf_id, lsst_id) in enumerate(tqdm(pairs_lsst, desc="Obtaining LSST objects...")):
            if lsst_id in skipped_404:
                continue
            try:
                det, ndet, obj = process_lsst_object(i, ztf_id, lsst_id)
                det_batch_lsst.append(det)
                ndet_batch_lsst.append(ndet)
                obj_batch_lsst.append(obj)
                n_ok_lsst += 1
                time.sleep(FETCH_DELAY)
            except Exception as e:
                err_str = str(e)
                if '404' in err_str:
                    skipped_404.add(lsst_id)
                    save_404(lsst_id)
                elif '403' in err_str:
                    print(f"Error 403 for {lsst_id}. Waiting {RETRY_403_WAIT}s before retrying...")
                    time.sleep(RETRY_403_WAIT)
                    try:
                        det, ndet, obj = process_lsst_object(i, ztf_id, lsst_id)
                        det_batch_lsst.append(det)
                        ndet_batch_lsst.append(ndet)
                        obj_batch_lsst.append(obj)
                        n_ok_lsst += 1
                        time.sleep(FETCH_DELAY)
                    except Exception as e2:
                        print(f"Error obtaining LSST object {lsst_id} (retry): {e2}")
                else:
                    print(f"Error obtaining LSST object {lsst_id}: {e}")

            if n_ok_lsst > 0 and n_ok_lsst % checkpoint_fetch == 0:
                _flush_checkpoint(det_batch_lsst, ndet_batch_lsst, obj_batch_lsst,
                                  'lsst', offset_lsst + n_ok_lsst)
                det_batch_lsst  = []
                ndet_batch_lsst = []
                obj_batch_lsst  = []

        if det_batch_lsst:
            _flush_checkpoint(det_batch_lsst, ndet_batch_lsst, obj_batch_lsst,
                              'lsst', offset_lsst + n_ok_lsst)
    # end if SURVEY_MODE != 'ztf_only'
 
else:
    print("SKIP_FETCH=True: skipping fetch, loading existing checkpoints...")
 
#Checkpoints según SURVEY_MODE
_empty_det  = pd.DataFrame()
_empty_ndet = pd.DataFrame()
_empty_obj  = pd.DataFrame()

if SURVEY_MODE != 'lsst_only':
    det_ztf_all, ndet_ztf_all, obj_ztf_all = load_checkpoints('ztf')
    if det_ztf_all is None:
        det_ztf_all, ndet_ztf_all, obj_ztf_all = _empty_det, _empty_ndet, _empty_obj
else:
    det_ztf_all, ndet_ztf_all, obj_ztf_all = _empty_det, _empty_ndet, _empty_obj

if SURVEY_MODE != 'ztf_only':
    det_lsst_all, ndet_lsst_all, obj_lsst_all = load_checkpoints('lsst')
    if det_lsst_all is None:
        det_lsst_all, ndet_lsst_all, obj_lsst_all = _empty_det, _empty_ndet, _empty_obj
else:
    det_lsst_all, ndet_lsst_all, obj_lsst_all = _empty_det, _empty_ndet, _empty_obj
 
#%%
 
#Unir datos de surveys disponibles
_to_concat_det  = [d for d in [det_ztf_all,  det_lsst_all]  if not d.empty]
_to_concat_ndet = [d for d in [ndet_ztf_all, ndet_lsst_all] if not d.empty]
_to_concat_obj  = [d for d in [obj_ztf_all,  obj_lsst_all]  if not d.empty]

detections     = pd.concat(_to_concat_det).sort_index()  if _to_concat_det  else pd.DataFrame()
non_detections = pd.concat(_to_concat_ndet)              if _to_concat_ndet else pd.DataFrame()
object_info    = pd.concat(_to_concat_obj)               if _to_concat_obj  else pd.DataFrame()
if not object_info.empty:
    object_info = object_info.groupby(object_info.index).first()
 
#%%
 
#Preprocesamiento bifurcado
#
# RAMA A "strict":
#   Objetos que pasan el preprocesador estricto (≥5 det en ≥1 banda) de
#   cada survey por separado.
#   Alimenta Modelos A (ZTF), B (LSST), C (features split) y D (combined).
#
# RAMA B "relaxed":
#   Objetos que fallan el preprocesador de al menos uno de los surveys, pero
#   cuya curva combinada ZTF+LSST pasa el preprocesador LSST.
#   Permite clasificación temprana cuando ningún survey tiene datos suficientes
#   por sí solo. Se extraen features ZTF, LSST y combined (las dos primeras
#   probablemente malas, pero útiles para cuantificar la mejora del Modelo D).
#

 
print('Preprocessing objects...')

#Preprocesamiento según SURVEY_MODE
if SURVEY_MODE == 'lsst_only':
    # Solo LSST: preprocesador LSST sobre todos los datos.
    # Rama A = pasa el preprocesador LSST. No hay Rama B (no hay ZTF).
    det_lsst_pp, ndet_lsst_pp = preprocess(
        LSSTLightcurvePreprocessor(stream=False),
        det_lsst_all, ndet_lsst_all, obj_lsst_all)
    det_ztf_pp  = pd.DataFrame()
    ndet_ztf_pp = pd.DataFrame()
    oids_con_ztf  = set()
    oids_con_lsst = set(det_lsst_pp.index.unique())
    oids_strict   = oids_con_lsst
    oids_relaxed  = set()
    det_cand_pp   = pd.DataFrame()
    ndet_cand_pp  = pd.DataFrame()

elif SURVEY_MODE == 'ztf_only':
    # Solo ZTF: preprocesador ZTF sobre todos los datos.
    # Rama A = pasa el preprocesador ZTF. No hay Rama B.
    det_ztf_pp, ndet_ztf_pp = preprocess(
        ZTFLightcurvePreprocessor(stream=False),
        det_ztf_all, ndet_ztf_all, obj_ztf_all)
    det_lsst_pp  = pd.DataFrame()
    ndet_lsst_pp = pd.DataFrame()
    oids_con_ztf  = set(det_ztf_pp.index.unique())
    oids_con_lsst = set()
    oids_strict   = oids_con_ztf
    oids_relaxed  = set()
    det_cand_pp   = pd.DataFrame()
    ndet_cand_pp  = pd.DataFrame()

else:
    # Modo 'both': preprocesamiento bifurcado original
    det_ztf_pp,  ndet_ztf_pp  = preprocess(
        ZTFLightcurvePreprocessor(stream=False),
        det_ztf_all, ndet_ztf_all, obj_ztf_all)
 
    det_lsst_pp, ndet_lsst_pp = preprocess(
        LSSTLightcurvePreprocessor(stream=False),
        det_lsst_all, ndet_lsst_all, obj_lsst_all)
 
    oids_con_ztf  = set(det_ztf_pp.index.unique())
    oids_con_lsst = set(det_lsst_pp.index.unique())
 
    # Rama A: pasan ambos preprocesadores estrictos
    oids_strict = oids_con_ztf & oids_con_lsst
    
    if  ENABLE_BRANCH_B:
        # Rama B: fallan al menos uno → aplicar preprocesador LSST a datos combinados
        oids_candidatos = (oids_con_ztf | oids_con_lsst) - oids_strict
     
        det_cand_raw  = pd.concat([
            det_ztf_all[det_ztf_all.index.isin(oids_candidatos)],
            det_lsst_all[det_lsst_all.index.isin(oids_candidatos)]
        ]).sort_index()
        ndet_cand_raw = pd.concat([
            ndet_ztf_all[ndet_ztf_all.index.isin(oids_candidatos)],
            ndet_lsst_all[ndet_lsst_all.index.isin(oids_candidatos)]
        ])
        obj_cand = object_info[object_info.index.isin(oids_candidatos)]
     
        det_cand_pp, ndet_cand_pp = preprocess(
            LSSTLightcurvePreprocessor(stream=False),
            det_cand_raw, ndet_cand_raw, obj_cand)
     
        oids_con_det_ztf  = set(det_ztf_all.index.unique())
        oids_con_det_lsst = set(det_lsst_all.index.unique())
     
        oids_relaxed = set()
        for oid in det_cand_pp.index.unique():
            if oid not in oids_con_det_ztf or oid not in oids_con_det_lsst:
                survey = 'ZTF' if oid not in oids_con_det_ztf else 'LSST'
                print(f"[WARN] {oid}: no detections in {survey}. Skipping B branch")
                continue
            oids_relaxed.add(oid)
    else:
        oids_relaxed    = set()
        oids_candidatos = set()
        det_cand_pp     = pd.DataFrame()
        ndet_cand_pp    = pd.DataFrame()
    
print('\n== Preprocessing ==')
print(f"SURVEY_MODE: {SURVEY_MODE}")
if SURVEY_MODE != 'lsst_only':
    print(f"Objects after ZTF-pp: {len(oids_con_ztf)}")
if SURVEY_MODE != 'ztf_only':
    print(f"Objects after LSST-pp: {len(oids_con_lsst)}")
print(f"[A branch] Strict objects: {len(oids_strict)}")
if SURVEY_MODE == 'both' and ENABLE_BRANCH_B:
    print(f"B branch candidates (fail ≥1 pp individually): {len(oids_candidatos)}")
    print(f"[B branch] Combined data passes filter: {len(oids_relaxed)}")
    print(f"Discarded:                        "
          f"{len(oids_candidatos) - len(oids_relaxed)}")
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n')
 
#Diagnóstico Rama B: solo en modo 'both'
if SURVEY_MODE == 'both' and ENABLE_BRANCH_B and oids_relaxed:
    max_det_ztf = (det_ztf_all[det_ztf_all.index.isin(oids_relaxed)]
                   .groupby([det_ztf_all.index.name or 'oid', 'fid']).size()
                   .groupby(level=0).max()
                   .rename('max_det_ztf'))

    max_det_lsst = (det_lsst_all[det_lsst_all.index.isin(oids_relaxed)]
                    .groupby([det_lsst_all.index.name or 'oid', 'fid']).size()
                    .groupby(level=0).max()
                    .rename('max_det_lsst'))

    diag_relaxed = pd.DataFrame(index=pd.Index(sorted(oids_relaxed), name='oid'))
    diag_relaxed = diag_relaxed.join(max_det_ztf).join(max_det_lsst).fillna(0).astype(int)
    diag_relaxed['pass_ztf_pp']  = diag_relaxed.index.isin(oids_con_ztf)
    diag_relaxed['pass_lsst_pp'] = diag_relaxed.index.isin(oids_con_lsst)

    diag_relaxed['motivo'] = 'lsst_falla'
    diag_relaxed.loc[~diag_relaxed['pass_ztf_pp'] & ~diag_relaxed['pass_lsst_pp'], 'motivo'] = 'ambos_fallan'
    diag_relaxed.loc[~diag_relaxed['pass_ztf_pp'] &  diag_relaxed['pass_lsst_pp'], 'motivo'] = 'ztf_falla'
else:
    diag_relaxed = pd.DataFrame()
 
if SURVEY_MODE == 'both' and ENABLE_BRANCH_B and oids_relaxed:
    print('== B branch summary ==')
    print(diag_relaxed['motivo'].value_counts().to_string())
    print(f"\nmax det by banda (median):")
    print(f"ZTF:  {diag_relaxed['max_det_ztf'].median():.0f}  "
          f"(min={diag_relaxed['max_det_ztf'].min()}, max={diag_relaxed['max_det_ztf'].max()})")
    print(f"LSST: {diag_relaxed['max_det_lsst'].median():.0f}  "
          f"(min={diag_relaxed['max_det_lsst'].min()}, max={diag_relaxed['max_det_lsst'].max()})")
    print()
 
#Preparar datos de Rama A 
if SURVEY_MODE == 'lsst_only':
    # Solo LSST: usar el preprocesador LSST directamente
    _det_strict  = [det_lsst_pp[det_lsst_pp.index.isin(oids_strict)]]
    _ndet_strict = [ndet_lsst_pp[ndet_lsst_pp.index.isin(oids_strict)]]
elif SURVEY_MODE == 'ztf_only':
    _det_strict  = [det_ztf_pp[det_ztf_pp.index.isin(oids_strict)]]
    _ndet_strict = [ndet_ztf_pp[ndet_ztf_pp.index.isin(oids_strict)]]
else:
    det_ztf_strict  = det_ztf_pp[det_ztf_pp.index.isin(oids_strict)]
    det_lsst_strict = det_lsst_pp[det_lsst_pp.index.isin(oids_strict)]
    _det_strict  = [det_ztf_strict, det_lsst_strict]
    _ndet_strict = [ndet_ztf_pp[ndet_ztf_pp.index.isin(oids_strict)],
                    ndet_lsst_pp[ndet_lsst_pp.index.isin(oids_strict)]]

detections_strict     = pd.concat(_det_strict).sort_index()
non_detections_strict = pd.concat(_ndet_strict)
object_info_strict    = object_info[object_info.index.isin(oids_strict)]
 
#Preparar datos de Rama B (solo en modo 'both')
if SURVEY_MODE == 'both' and ENABLE_BRANCH_B and oids_relaxed:
    detections_relaxed     = det_cand_pp[det_cand_pp.index.isin(oids_relaxed)]
    non_detections_relaxed = ndet_cand_pp[ndet_cand_pp.index.isin(oids_relaxed)]
    object_info_relaxed    = object_info[object_info.index.isin(oids_relaxed)]
else:
    detections_relaxed     = pd.DataFrame()
    non_detections_relaxed = pd.DataFrame()
    object_info_relaxed    = pd.DataFrame()
 
#%%
 
#Extracción de features por rama
 
# Rama A: extracción completa (ZTF, LSST y combined)
print('Extracting features (A branch)...')
features_strict = extract_features(detections_strict, non_detections_strict,
                                   object_info_strict, namespace='strict')
 
# Rama B: solo en modo 'both' con objetos disponibles
if SURVEY_MODE == 'both' and ENABLE_BRANCH_B and not detections_relaxed.empty:
    print('\nExtracting features (B branch)...')
    features_relaxed = extract_features(detections_relaxed, non_detections_relaxed,
                                        object_info_relaxed, namespace='relaxed')
else:
    print("\nB branch skipped (SURVEY_MODE != both or no 'relaxed' objects).")
    features_relaxed = {'ztf': pd.DataFrame(), 'lsst': pd.DataFrame()}
                        # 'combined': pd.DataFrame()}
 
#%%
 
#Postprocesamiento y guardado
 
print('\nPostprocessing features (A branch)...')
pp_strict     = postprocess_features(features_strict)
features_ztf  = pp_strict['ztf']
features_lsst = pp_strict['lsst']
features_comb = pp_strict['combined']
 
if ENABLE_BRANCH_B:
    print('Postprocessing features (B branch)...')
    pp_relaxed            = postprocess_features(features_relaxed)
    features_ztf_relaxed  = pp_relaxed['ztf']
    features_lsst_relaxed = pp_relaxed['lsst']
    features_comb_relaxed = pp_relaxed['combined']
else:
    features_ztf_relaxed  = pd.DataFrame()
    features_lsst_relaxed = pd.DataFrame()
    features_comb_relaxed = pd.DataFrame()
 
#%%
 
print('Saving...')
 
#Directorio de salida según SURVEY_MODE

# Se escribe directamente en OUTPUT_BASE (sin subcarpeta por modo) para que
# coincida con --features-dir de xmatch.py, que lee features_comb_*.parquet
# directamente de ese directorio. Si se ejecutan varios SURVEY_MODE contra el
# mismo OUTPUT_BASE, los resultados se mezclan — usar --output-base distintos
# por modo si se necesita conservarlos por separado.

_mode_suffix = {'both': 'Final', 'lsst_only': 'Final_lsst_only',
                'ztf_only': 'Final_ztf_only'}
path = OUTPUT_BASE if OUTPUT_BASE.endswith(os.sep) else OUTPUT_BASE + os.sep
os.makedirs(path, exist_ok=True)
print(f"Saved in: {path}")
 
#Rama A (strict)
if not features_ztf.empty:
    features_ztf.to_parquet(path + 'features_ztf_strict.parquet')
    print(f"features_ztf_strict.parquet    ({len(features_ztf)} objects)")
 
if not features_lsst.empty:
    features_lsst.to_parquet(path + 'features_lsst_strict.parquet')
    print(f"features_lsst_strict.parquet   ({len(features_lsst)} objects)")
 
if not features_comb.empty:
    features_comb.to_parquet(path + 'features_comb_strict.parquet')
    print(f"features_comb_strict.parquet   ({len(features_comb)} objects)")
 
#Diagnóstico Rama B (solo modo 'both')
if SURVEY_MODE == 'both' and ENABLE_BRANCH_B and oids_relaxed:
    diag_relaxed.to_parquet(path + 'diag_rama_b.parquet')
    print(f"diag_rama_b.parquet            ({len(diag_relaxed)} objects)")
 
#Rama B (relaxed, solo modo 'both') 
if SURVEY_MODE == 'both' and ENABLE_BRANCH_B:
    if not features_comb_relaxed.empty:
        features_ztf_relaxed.to_parquet(path + 'features_ztf_relaxed.parquet')
        print(f"features_ztf_relaxed.parquet   ({len(features_ztf_relaxed)} objects)")
 
        features_lsst_relaxed.to_parquet(path + 'features_lsst_relaxed.parquet')
        print(f"features_lsst_relaxed.parquet  ({len(features_lsst_relaxed)} objects)")
 
        features_comb_relaxed.to_parquet(path + 'features_comb_relaxed.parquet')
        print(f"features_comb_relaxed.parquet  ({len(features_comb_relaxed)} objects)")
    else:
        print("B branch empty: no files saved.")
 
print('\nSummary:')
print(f"SURVEY_MODE: {SURVEY_MODE}")
print(f"A branch (ZTF):      {len(features_ztf)} objects")
print(f"A branch (LSST):     {len(features_lsst)} objects")
print(f"A branch (combined): {len(features_comb)} objects")
if SURVEY_MODE == 'both' and ENABLE_BRANCH_B:
    if not features_comb_relaxed.empty:
        print(f"B branch (ZTF):      {len(features_ztf_relaxed)} objects")
        print(f"B branch (LSST):     {len(features_lsst_relaxed)} objects")
        print(f"B branch (combined): {len(features_comb_relaxed)} objects")
    else:
        print("B branch: 0 objects")