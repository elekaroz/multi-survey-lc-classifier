from __future__ import annotations

import os
import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# ── Importar todo del pipeline original ──────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_training_functions as mtf

warnings.filterwarnings("ignore")

#%%

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Estos valores son los DEFAULTS usados si el script se lanza sin argumentos
# (p. ej. con runfile() desde Spyder). Si se lanza desde terminal, cualquier
# --argumento sobrescribe el default correspondiente (ver parse_args() al
# final del fichero). Las rutas por defecto asumen la estructura de carpetas
# documentada en el README del repositorio.

# ── Inputs ────────────────────────────────────────────────────────────────────
# Consensus CSV generado por feature_selection.py
CONSENSUS_CSV = "./data/feature_selection/consensus_features.csv"
# Features simuladas
FEATURES_ZTF_FILE      = "./data/simulated/features_ztf.parquet"
FEATURES_LSST_FILE     = "./data/simulated/features_lsst.parquet"
FEATURES_ATLAS_FILE    = "./data/simulated/features_atlas.parquet"
FEATURES_COMBINED_FILE = "./data/simulated/features_comb.parquet"
LABELS_FILE            = "./data/simulated/labels.csv"
# Features reales strict para CORAL (target domain)
REAL_ZTF_STRICT   = "./data/real/features_ztf_strict.parquet"
REAL_LSST_STRICT  = "./data/real/features_lsst_strict.parquet"
REAL_COMB_STRICT  = "./data/real/features_comb_strict.parquet"
REAL_ATLAS_STRICT = "./data/real/features_atlas_strict.parquet"
# Labels del test set real (strict). Debe contener columna 'class_original'
# (o 'classALeRCE') e índice oid.
REAL_LABELS_FILE = "./data/real/labels_testset.csv"

# ── Outputs ───────────────────────────────────────────────────────────────────
# Los outputs se guardan en subdirectorios dentro de OUTPUT_DIR para no
# mezclarlos con otros experimentos:
#   models/  -> modelos .pkl entrenados
#   plots/   -> plots de evaluación
#   oof/     -> predicciones OOF
#   coral/   -> transformaciones CORAL guardadas para inferencia
OUTPUT_DIR     = "./output/"
MODELS_DIR_ADP = os.path.join(OUTPUT_DIR, "models") + "/"
PLOTS_DIR_ADP  = os.path.join(OUTPUT_DIR, "plots")  + "/"
OOF_DIR_ADP    = os.path.join(OUTPUT_DIR, "oof")    + "/"
CORAL_DIR      = os.path.join(OUTPUT_DIR, "coral")  + "/"

# ── CORAL hyperparameters ────────────────────────────────────────────────────

CORAL_LAMBDA       = 0.1   
CORAL_KS_THRESHOLD = 0.4   

# ── Selección de features ─────────────────────────────────────────────────────
# Conservar features UNKNOWN (aquellas sin gap calculado por falta de datos reales).
# True: incluye UNKNOWN con SHAP > umbral
# False : excluye todas las UNKNOWN (solo KEEP + CORAL del consenso)
INCLUDE_UNKNOWN = False

# Umbral de SHAP para conservar features UNKNOWN (relativo a la mediana KEEP)
UNKNOWN_SHAP_PERCENTILE = 75   # conservar UNKNOWN con shap > UNKNOWN_SHAP_PERCENTILE de KEEP

# Excluir coordenadas galácticas independientemente del cuadrante
_ALWAYS_EXCLUDE_SUFFIXES = ["gal_b_", "gal_l_"]

# ── Metamodelo aumentado con datos reales ────────────────────────────────────

# Activa el entrenamiento del metamodelo incluyendo los objetos reales
# etiquetados como filas adicionales de entrenamiento.

USE_REAL_META_AUG = True

# Peso para las filas reales en el metamodelo.
# Las filas reales reciben peso REAL_META_UPWEIGHT, y las simuladas 1.0.

REAL_META_UPWEIGHT = 10


# ── Model dropout para robustez del metamodelo ──────────────────────────────

# Durante entrenamiento del metamodelo, con probabilidad META_DROPOUT_PROB se
# dropea cada SURVEY (ZTF, LSST, ATLAS) y se propaga la ausencia a los
# modelos dependientes.  Esto enseña al metamodelo que "modelo sin datos" es
# una señal válida de entrada y que debe funcionar con cualquier combinación
# de surveys (solo ZTF, solo LSST, ZTF+ATLAS, etc.).

# Dependencias: A-ZTF, B-LSST, C-ZTF+LSST, D-ZTF+LSST, E-ATLAS.
META_DROPOUT_PROB = 0.3   # 0.0 para desactivar

# =============================================================================
# COLORES POR MODELO PARA PLOTS (misma paleta que bootstrap_confmat.py)
# =============================================================================
from matplotlib.colors import LinearSegmentedColormap

MODEL_CMAPS = {
    'A': LinearSegmentedColormap.from_list('teal_a',
         ['#E1F5EE', '#9FE1CB', '#5DCAA5', '#1D9E75', '#0F6E56', '#085041', '#04342C']),
    'B': LinearSegmentedColormap.from_list('amber_b',
         ['#FAEEDA', '#FAC775', '#EF9F27', '#BA7517', '#854F0B', '#633806', '#412402']),
    'C': LinearSegmentedColormap.from_list('purples_c',
         ['#EDE8F5', '#D9C9E8', '#C4AADA', '#9067B2', '#6A3495', '#4E2275', '#3F0070']),
    'D': LinearSegmentedColormap.from_list('green_d',
         ['#EBF3E0', '#C4DD9E', '#97C459', '#639922', '#3B6D11', '#27500A', '#173404']),
    'E': LinearSegmentedColormap.from_list('pink_e',
         ['#FBEAF0', '#F4C0D1', '#ED93B1', '#D4537E', '#993556', '#72243E', '#4B1528']),
    'Metamodel': plt.cm.Blues,
}
# Color representativo (punto medio de cada cmap) para bar plots
MODEL_COLORS = {
    'A': '#1D9E75',   # teal
    'B': '#BA7517',   # amber
    'C': '#9067B2',   # purple
    'D': '#639922',   # green
    'E': '#D4537E',   # pink
    'Metamodel': '#3171B0',  # blue
}

# =============================================================================
# CORAL
# =============================================================================

def fit_coral(
    X_source: np.ndarray,
    X_target: np.ndarray,
    lam: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ajusta la transformación CORAL: alinea la distribución de X_source
    con la de X_target igualando medias y covarianzas.

    La transformación se guarda en disco en CORAL_DIR para usarla en
    inferencia sobre nuevos datos reales.
    """
    if lam is None:
        lam = CORAL_LAMBDA

    mu_s = X_source.mean(axis=0)
    mu_t = X_target.mean(axis=0)

    Xs_c = X_source - mu_s
    Xt_c = X_target - mu_t

    Cs = (Xs_c.T @ Xs_c) / (len(Xs_c) - 1) + lam * np.eye(Xs_c.shape[1])
    Ct = (Xt_c.T @ Xt_c) / (len(Xt_c) - 1) + lam * np.eye(Xt_c.shape[1])

    def _mat_sqrt(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def _mat_inv_sqrt(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    Cs_inv_sqrt = _mat_inv_sqrt(Cs)
    Ct_sqrt     = _mat_sqrt(Ct)
    W = Cs_inv_sqrt @ Ct_sqrt

    return mu_s, mu_t, W


def apply_coral(
    X: np.ndarray,
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    return (X - mu_s) @ W + mu_t


def fit_and_apply_coral_for_model(
    X_sim: pd.DataFrame,
    df_real: pd.DataFrame,
    coral_features: List[str],
    model_id: str,
    coral_dir: str,
    lam: Optional[float] = None,
) -> pd.DataFrame:
    """
    Aplica CORAL sobre las columnas `coral_features` de X_sim usando
    df_real como target domain.

    La transformación se ajusta sobre todos los objetos del real strict
    disponibles para esas features (sin etiqueta), y se aplica al
    conjunto simulado completo.

    Guarda la transformación en {coral_dir}/coral_model{model_id}.pkl
    para poder aplicarla en inferencia.
    """
    if lam is None:
        lam = CORAL_LAMBDA

    os.makedirs(coral_dir, exist_ok=True)

    available = [
        f for f in coral_features
        if f in X_sim.columns
        and f in df_real.columns
        and df_real[f].notna().sum() >= 10
    ]

    if not available:
        print(f"  [CORAL model {model_id}] Ninguna feature disponible en el real — sin adaptar")
        return X_sim.copy()

    skipped = [f for f in coral_features if f not in available]
    if skipped:
        print(f"  [CORAL model {model_id}] {len(skipped)} features sin datos reales suficientes"
              f" — omitidas: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # Extraer arrays limpios
    # Source: simulado — reemplazar -999 (fillna de build_X) por NaN y
    # luego usar la mediana del simulado para no contaminar la covarianza.
    X_s = X_sim[available].copy().replace(-999, np.nan)
    for col in available:
        med = X_s[col].median()
        X_s[col] = X_s[col].fillna(med if np.isfinite(med) else 0.0)

    # Target: real strict — usar mediana por columna para NaN
    X_t = df_real[available].copy()
    for col in available:
        med = X_t[col].median()
        X_t[col] = X_t[col].fillna(med if np.isfinite(med) else 0.0)

    X_s_arr = X_s.values.astype(float)
    X_t_arr = X_t.values.astype(float)

    print(f"  [CORAL model {model_id}] Ajustando sobre {len(available)} features "
          f"(n_source={len(X_s_arr)}, n_target={len(X_t_arr)})")

    mu_s, mu_t, W = fit_coral(X_s_arr, X_t_arr, lam)

    X_adapted_arr = apply_coral(X_s_arr, mu_s, mu_t, W)

    # Guardar transformación
    coral_path = os.path.join(coral_dir, f"coral_model{model_id}.pkl")
    with open(coral_path, "wb") as f:
        pickle.dump({
            "model_id":       model_id,
            "features":       available,
            "mu_s":           mu_s,
            "mu_t":           mu_t,
            "W":              W,
            "coral_lambda":   lam,
        }, f, pickle.HIGHEST_PROTOCOL)
    print(f"  [CORAL model {model_id}] Transformation saved: {coral_path}")

    # Reconstruir DataFrame con columnas adaptadas
    X_out = X_sim.copy()

    X_adapted_df = pd.DataFrame(X_adapted_arr, index=X_sim.index, columns=available)

    original_nan_mask = X_sim[available].eq(-999)
    X_adapted_df[original_nan_mask] = -999
    X_out[available] = X_adapted_df

    return X_out


# =============================================================================
# SELECCIÓN DE FEATURES
# =============================================================================

def load_consensus(consensus_csv: str) -> pd.DataFrame:
    """Carga y valida el CSV de consenso de feature_selection.py."""
    df = pd.read_csv(consensus_csv)
    required = {"feature_base", "recommendation", "quadrants", "mean_shap", "mean_ks"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"consensus_features.csv: {missing} columns missing")
    return df


def build_feature_list_for_model(
    consensus: pd.DataFrame,
    model_id: str,
    all_sim_cols: List[str],
    suffix: str,
    unknown_shap_percentile: Optional[int] = None,
    include_unknown: Optional[bool] = None,
) -> Tuple[List[str], List[str]]:
    """
    Construye las listas de features a usar para un modelo dado.

    Lógica:
    - KEEP     -> incluir siempre
    - CORAL    -> incluir (con adaptación CORAL)
    - OPTIONAL -> excluir (poco impacto)
    - DROP     -> excluir
    - UNKNOWN  -> según configuración más arriba
    """
    if unknown_shap_percentile is None:
        unknown_shap_percentile = UNKNOWN_SHAP_PERCENTILE
    if include_unknown is None:
        include_unknown = INCLUDE_UNKNOWN

    import re

    # Filtrar el consenso al modelo actual
    model_rows = consensus[consensus["models"].str.contains(model_id, na=False)]


    keep_rows  = model_rows[model_rows["recommendation"].str.startswith("KEEP")]
    shap_threshold = np.percentile(keep_rows["mean_shap"].values,
                                   unknown_shap_percentile) if len(keep_rows) else 0.0


    def with_suffix(base: str) -> List[str]:
        """Devuelve las columnas del simulado que corresponden a feature_base."""

        candidates = [
            c for c in all_sim_cols
            if re.sub(r"_(" + suffix.lstrip("_") + r")$", "", c) == base
            or c == base + suffix
        ]
        if not candidates:
            direct = base + suffix
            if direct in all_sim_cols:
                return [direct]
        return candidates

    selected = []
    coral    = []

    # KEEP y CORAL
    for _, row in model_rows.iterrows():
        base  = row["feature_base"]
        rec   = row["recommendation"]
        cols  = with_suffix(base)
        if not cols:
            continue

        # Excluir coordenadas galácticas
        cols = [c for c in cols
                if not any(c.startswith(exc) for exc in _ALWAYS_EXCLUDE_SUFFIXES)]
        if not cols:
            continue

        if rec.startswith("KEEP") or rec == "CORAL prioritario":
            selected.extend(cols)
            if rec == "CORAL prioritario":
                coral.extend(cols)

    # UNKNOWN
    if include_unknown:
        consensus_bases = set(consensus["feature_base"].tolist())
        for col in all_sim_cols:
            if col in selected:
                continue
            if any(col.startswith(exc) for exc in _ALWAYS_EXCLUDE_SUFFIXES):
                continue
            # Obtener base de esta columna
            base = re.sub(r"_\d+_(" + suffix.lstrip("_") + r")$", "", col)
            base = re.sub(r"_(" + suffix.lstrip("_") + r")$", "", base)
            if base in consensus_bases:
                continue  
            selected.append(col)
    
    seen = set()
    selected_dedup = []
    for c in selected:
        if c not in seen:
            seen.add(c)
            selected_dedup.append(c)

    coral_dedup = list(dict.fromkeys(c for c in coral if c in seen))

    print(f"  Model {model_id}: {len(selected_dedup)} features selected "
          f"({len(coral_dedup)} CORAL) of {len(all_sim_cols)} available")

    return selected_dedup, coral_dedup


# =============================================================================
# UTILIDADES
# =============================================================================

def _ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _filter_by_ks(coral_features, ranking_csv, ks_threshold=None, suffix_hint=""):
    """
    Filtra features CORAL manteniendo solo las con KS > ks_threshold
    segun el CSV de ranking de feature_selection.py.
    Features no encontradas en el CSV se conservan por precaución.
    """
    if ks_threshold is None:
        ks_threshold = CORAL_KS_THRESHOLD
    if not ranking_csv or not os.path.exists(ranking_csv):
        return coral_features
    try:
        df_rank = pd.read_csv(ranking_csv, index_col=0)
    except Exception:
        return coral_features

    filtered = []
    skipped  = []
    for feat in coral_features:
        if feat in df_rank.index:
            ks = df_rank.loc[feat, "ks_statistic"]
            if pd.notna(ks) and float(ks) >= ks_threshold:
                filtered.append(feat)
            else:
                skipped.append(feat)
        else:
            filtered.append(feat)

    tag = f" {suffix_hint}" if suffix_hint else ""
    if skipped:
        n_show = min(5, len(skipped))
        print(f"  [CORAL{tag}] {len(skipped)} features skipped: KS < {ks_threshold}: "
              f"{skipped[:n_show]}{'...' if len(skipped) > 5 else ''}")
    print(f"  [CORAL{tag}] {len(filtered)}/{len(coral_features)} features with KS >= {ks_threshold}")
    return filtered



def _load_real_strict(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"  [CORAL] Real strict not found: {path}")
        return None
    df = pd.read_parquet(path).replace([np.inf, -np.inf], np.nan)
    return df


def _compare_results(
    summary_original: Optional[str],
    summary_adapted:  pd.DataFrame,
    models_dir_orig:  str,
):
    """Imprime comparativa baseline vs adaptado si el CSV original existe."""
    orig_path = os.path.join(models_dir_orig, "results_summary.xlsx")
    if not os.path.exists(orig_path):
        print("  [Comparativa] Original results_summary.xlsx not found.")
        return

    try:
        df_orig = pd.read_excel(orig_path, sheet_name="Summary")
    except Exception:
        print("  [Comparativa] Couldn't read results_summary.xlsx.")
        return

    print("\n" + "=" * 70)
    print("Baseline  vs  Adapted (CORAL + feature selection)")
    print("=" * 70)

    # Normalizar nombres de columnas
    col_map = {c.lower(): c for c in df_orig.columns}
    orig_bacc = col_map.get("balanced_accuracy", col_map.get("balanced_accuracy", None))
    orig_f1   = col_map.get("macro_f1", None)


    df_orig_norm = df_orig.rename(columns={v: k for k, v in col_map.items()})

    metrics_cols = ["model", "balanced_accuracy", "macro_f1"]
    summary_adapted_norm = summary_adapted.rename(
        columns={c: c.lower() for c in summary_adapted.columns})

    available = [c for c in metrics_cols
                 if c in df_orig_norm.columns and c in summary_adapted_norm.columns]
    if not available:
        print("Couldn't find equivalent columns in baseline and adapted.")
        print(f"  Baseline cols: {list(df_orig.columns)}")
        print(f"  Adapted cols: {list(summary_adapted.columns)}")
        return

    df_orig_norm["version"]        = "baseline"
    summary_adapted_norm["version"] = "adapted"

    compare = pd.concat(
        [df_orig_norm[available + ["version"]],
         summary_adapted_norm[available + ["version"]]],
        ignore_index=True,
    ).sort_values(["model", "version"])

    print(compare.to_string(index=False, float_format="{:.4f}".format))


# =============================================================================
# METAMODELO — AUMENTADO CON DATOS REALES
# =============================================================================

def _build_real_X_for_meta(
    df_real_ztf, df_real_lsst, df_real_comb, df_real_atlas,
    sel_B, sel_C_ztf, sel_C_lsst, sel_C_atlas,
    sel_D, ztf_cols, atlas_cols,
    coral_dir,
    real_labels_file,
):
    """
    Construye las matrices de features reales (strict) para los modelos base,
    aplica las transformaciones CORAL guardadas, y devuelve un dict
    {model_name: X_real_df} junto con las labels reales.
    """
    # ── Cargar labels reales ──────────────────────────────────────────────────
    try:
        _lbl_path = Path(real_labels_file)
        if _lbl_path.suffix == '.parquet':
            df_real_labels = pd.read_parquet(_lbl_path)
        else:
            df_real_labels = pd.read_csv(_lbl_path)
        if 'oid_combined' in df_real_labels.columns:
            df_real_labels = df_real_labels.set_index('oid_combined')
        elif df_real_labels.index.name != 'oid_combined':
            df_real_labels = df_real_labels.set_index(df_real_labels.columns[0])
        if 'class_original' not in df_real_labels.columns:
            if 'classALeRCE' in df_real_labels.columns:
                df_real_labels['class_original'] = df_real_labels['classALeRCE']
            else:
                raise ValueError("Labels file: column 'class_original' or "
                                 "'classALeRCE' not found.")

        mask = df_real_labels['class_original'].isin(mtf.LABEL_ORDER)
        if 'label_conflict' in df_real_labels.columns:
            mask = mask & (~df_real_labels['label_conflict'].astype(bool))
        real_labels_clean = df_real_labels.loc[mask, ['class_original']].copy()
        print(f'  [Meta aug] Loaded real labels: {len(real_labels_clean)} objects')
    except Exception as e:
        print(f'  [Meta aug] ERROR loading labels: {e}. Skipping augmentation.')
        return None, None

    # ── Unir features reales ───────────────────────

    KNOWN_SUFFIXES = ('_ztf', '_lsst', '_atlas', '_combined')

    def _drop_unsuffixed(df):
        """Elimina columnas sin sufijo de survey (p.ej. 'survey')."""
        if df is None:
            return None
        cols_drop = [c for c in df.columns
                     if not any(c.endswith(s) for s in KNOWN_SUFFIXES)]
        return df.drop(columns=cols_drop) if cols_drop else df

    real_parts = {}
    if df_real_ztf is not None:
        real_parts['ztf'] = _drop_unsuffixed(df_real_ztf)
    if df_real_lsst is not None:
        real_parts['lsst'] = _drop_unsuffixed(df_real_lsst)
    if df_real_comb is not None:
        real_parts['comb'] = _drop_unsuffixed(df_real_comb)

    if not real_parts:
        print('  [Meta aug] Real features not found. Skipping augmentation.')
        return None, None

    df_real_feat = pd.concat(list(real_parts.values()), axis=1, join='outer')

    dup = df_real_feat.columns.duplicated().sum()
    if dup:
        df_real_feat = df_real_feat.loc[:, ~df_real_feat.columns.duplicated()]
    df_real_feat = df_real_feat.replace([np.inf, -np.inf], np.nan)

    if df_real_atlas is not None:
        df_real_feat = df_real_feat.join(_drop_unsuffixed(df_real_atlas), how='left')

    # ── Filtrar ───────────────────────────────────
    df_real = real_labels_clean.join(df_real_feat, how='inner')
    if len(df_real) == 0:
        print('  [Meta aug] WARNING: no real objects were found.')
        return None, None
    y_real = df_real['class_original']
    df_real_feat_strict = df_real.drop(columns=['class_original'])
    print(f'  [Meta aug] Strict branch objects: {len(y_real)}')

    all_real_cols = list(df_real_feat_strict.columns)

    # ── Construir X por modelo (misma lógica que en main()) ───────────────────
    # Modelo A
    _ztf_real = [c for c in ztf_cols if c in all_real_cols]
    X_A_real  = mtf.build_X(df_real_feat_strict, _ztf_real, add_ztf_diff=True)

    # Modelo B
    _sel_B_real = [c for c in sel_B if c in all_real_cols]
    X_B_real    = mtf.build_X(df_real_feat_strict, _sel_B_real)

    # Modelo C
    _c_ztf  = [c for c in sel_C_ztf  if c in all_real_cols]
    _c_lsst = [c for c in sel_C_lsst if c in all_real_cols]
    _c_atl  = [c for c in sel_C_atlas if c in all_real_cols]
    _sel_C_real = list(dict.fromkeys(_c_ztf + _c_lsst + _c_atl))
    _has_atlas_real = df_real_atlas is not None and len(_c_atl) > 0
    X_C_real = mtf.build_X(
        df_real_feat_strict, _sel_C_real,
        add_ztf_diff=True,
        add_atlas_diff=_has_atlas_real,
        add_cross_diff=True,
    )

    # Modelo D
    _sel_D_real = [c for c in sel_D if c in all_real_cols]
    X_D_real    = mtf.build_X(df_real_feat_strict, _sel_D_real)

    # Modelo E (solo objetos con detección ATLAS)
    X_E_real  = None
    y_real_E  = None
    if df_real_atlas is not None and atlas_cols:
        _atl_real = [c for c in atlas_cols if c in all_real_cols]
        if _atl_real:
            _atlas_mask_real = df_real_feat_strict[_atl_real].notna().any(axis=1)
            if _atlas_mask_real.sum() > 0:
                X_E_real = mtf.build_X(
                    df_real_feat_strict.loc[_atlas_mask_real],
                    _atl_real, add_atlas_diff=True,
                )
                y_real_E = y_real.loc[_atlas_mask_real]

    # ── Aplicar transformaciones CORAL ────────────────────────────────────────
    if coral_dir and os.path.isdir(coral_dir):
        def _apply(X, model_id):
            pkl = os.path.join(coral_dir, f'coral_model{model_id}.pkl')
            if not os.path.exists(pkl):
                return X
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            feats = [c for c in d['features'] if c in X.columns]
            if not feats:
                return X
            X_out = X.copy()
            arr   = X_out[feats].values.astype(float)
            arr[arr == -999] = np.nan
            for j in range(arr.shape[1]):
                nan_m = np.isnan(arr[:, j])
                if nan_m.any():
                    med = np.nanmedian(arr[:, j])
                    arr[nan_m, j] = med if np.isfinite(med) else 0.0
            arr_t = (arr - d['mu_s']) @ d['W'] + d['mu_t']
            orig_nan = X[feats].values == -999
            arr_t[orig_nan] = -999
            X_out[feats] = arr_t
            return X_out

        X_B_real = _apply(X_B_real, 'B')
        X_C_real = _apply(X_C_real, 'C_ztf')
        X_C_real = _apply(X_C_real, 'C_lsst')
        X_C_real = _apply(X_C_real, 'C_atlas')
        X_C_real = _apply(X_C_real, 'C_diff')
        X_D_real = _apply(X_D_real, 'D')

    X_real = {'A': X_A_real, 'B': X_B_real, 'C': X_C_real, 'D': X_D_real}
    if X_E_real is not None:
        X_real['E'] = X_E_real

    return X_real, y_real


# =============================================================================
# MAIN
# =============================================================================

def main():
    _ensure_dirs(MODELS_DIR_ADP, PLOTS_DIR_ADP, OOF_DIR_ADP, CORAL_DIR)

    print("Loading features consensus...")
    consensus = load_consensus(CONSENSUS_CSV)
    print(f"  {len(consensus)} features in consensus")

    print("\nLoading simulated features...")
    df_ztf      = mtf._load_features(FEATURES_ZTF_FILE)
    df_lsst     = mtf._load_features(FEATURES_LSST_FILE)
    df_combined = mtf._load_features(FEATURES_COMBINED_FILE)

    if FEATURES_ATLAS_FILE and Path(FEATURES_ATLAS_FILE).exists():
        df_atlas_feat = mtf._load_features(FEATURES_ATLAS_FILE)
    else:
        df_atlas_feat = None
        print("  ATLAS not available. Models C (ATLAS) and E skipped")

    ztf_cols      = mtf.get_ztf_cols(list(df_ztf.columns))
    lsst_cols     = mtf.get_lsst_cols(list(df_lsst.columns))
    combined_cols = mtf.get_combined_cols(list(df_combined.columns))
    atlas_cols    = mtf.get_atlas_cols(list(df_atlas_feat.columns)) if df_atlas_feat is not None else []

    print("\nLoading real features for CORAL...")
    df_real_ztf  = _load_real_strict(REAL_ZTF_STRICT)
    df_real_lsst = _load_real_strict(REAL_LSST_STRICT)
    df_real_comb  = _load_real_strict(REAL_COMB_STRICT)
    df_real_atlas = _load_real_strict(REAL_ATLAS_STRICT)

    print("\nSelecting features...")
    sel_B, coral_B = build_feature_list_for_model(
        consensus, "B", lsst_cols, "_lsst")
    sel_C_ztf,   coral_C_ztf   = build_feature_list_for_model(
        consensus, "C", ztf_cols,   "_ztf")
    sel_C_lsst,  coral_C_lsst  = build_feature_list_for_model(
        consensus, "C", lsst_cols,  "_lsst")
    sel_C_atlas, coral_C_atlas = build_feature_list_for_model(
        consensus, "C", atlas_cols, "_atlas") if atlas_cols else ([], [])
    sel_C   = list(dict.fromkeys(sel_C_ztf + sel_C_lsst + sel_C_atlas))
    coral_C = list(dict.fromkeys(coral_C_ztf + coral_C_lsst + coral_C_atlas))
    sel_D, coral_D = build_feature_list_for_model(
        consensus, "D", combined_cols, "_combined")

    print("\nJoining features...")
    df_feat = (
        df_ztf[ztf_cols]
        .join(df_lsst[lsst_cols],        how="inner")
        .join(df_combined[combined_cols], how="inner")
    )
    if df_atlas_feat is not None:
        df_feat = df_feat.join(df_atlas_feat[atlas_cols], how="left")
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

    # Labels 
    print("Loading labels...")
    p_labels = Path(LABELS_FILE)
    df_labels = (pd.read_parquet(p_labels) if p_labels.suffix == ".parquet"
                 else pd.read_csv(p_labels, index_col="oid"))
    if "class_original" not in df_labels.columns:
        df_labels["class_original"] = df_labels["classALeRCE"]
    labels = df_labels.loc[
        df_labels.class_original.isin(mtf.LABEL_ORDER), ["class_original"]
    ].copy()

    df = labels.join(df_feat, how="inner").replace([np.inf, -np.inf], np.nan)

    Y_original     = df["class_original"]
    Y_hierarchical = mtf.make_hierarchical_labels(Y_original)

    print(f"\nTotal objects (ZTF+LSST+combined): {len(Y_original)}")

    # ATLAS subset 
    if atlas_cols:
        atlas_mask = df[atlas_cols].notna().any(axis=1)
        df_atlas   = df.loc[atlas_mask].copy()
        Y_atlas    = Y_original.loc[atlas_mask]
        Yh_atlas   = mtf.make_hierarchical_labels(Y_atlas)
        print(f"Objects with ATLAS: {atlas_mask.sum()}")
    else:
        df_atlas = Y_atlas = Yh_atlas = None

    #  Build X con selección de features
    # Modelo A
    X_A = mtf.build_X(df, ztf_cols, add_ztf_diff=True)

    # Modelo B
    sel_B_available = [c for c in sel_B if c in df.columns]
    X_B_pre = mtf.build_X(df, sel_B_available)

    # Modelo C
    _sel_C_base = list(dict.fromkeys(
        [c for c in sel_C if c in df.columns]
    ))
    if df_atlas_feat is not None:
        atlas_for_C = sel_C_atlas if sel_C_atlas else atlas_cols
        _sel_C_atlas = [c for c in atlas_for_C if c in df.columns
                        and c not in set(_sel_C_base)]
        sel_C_available = _sel_C_base + _sel_C_atlas
    else:
        sel_C_available = _sel_C_base

    print(f"  [C] features for build_X: {len(sel_C_available)} "
          f"({len(set(sel_C_available))} unique) — "
          f"{'OK' if len(sel_C_available) == len(set(sel_C_available)) else '⚠ DUPLICATED'}")
    X_C_pre = mtf.build_X(df, sel_C_available,
                           add_ztf_diff=True, add_cross_diff=True,
                           add_atlas_diff=df_atlas_feat is not None)

    # Filtrar features diferenciales DROP del modelo C 
    if consensus is not None:
        from feature_selection import _compute_differential_gap  
        _ranking_C_path = os.path.join(os.path.dirname(CONSENSUS_CSV),
                                       "feature_ranking_modelC.csv")
        if os.path.exists(_ranking_C_path):
            _ranking_C = pd.read_csv(_ranking_C_path, index_col=0)
            def _is_differential(col):
                return any(x in col for x in [
                    'gr_ratio', 'gr_diff', 'co_ratio', 'co_diff',
                    'ztf_lsst', 'ztf_atlas', 'atlas_lsst', 'color_'])
            _drop_diff = set(_ranking_C[
                (_ranking_C['quadrant'] == 'DROP') &
                (_ranking_C.index.map(_is_differential))
            ].index)
            _cols_to_drop = [c for c in X_C_pre.columns if c in _drop_diff]
            if _cols_to_drop:
                X_C_pre = X_C_pre.drop(columns=_cols_to_drop)
                print(f"  [C] Dropping {len(_cols_to_drop)} DROP differential features"
                      f"({X_C_pre.shape[1]} features remaining)")
        else:
            print(f"  [C] feature_ranking_modelC.csv not found in "
                  f"{os.path.dirname(CONSENSUS_CSV)}")

    # Modelo D
    sel_D_available = [c for c in sel_D if c in df.columns]
    X_D_pre = mtf.build_X(df, sel_D_available)

    # Modelo E
    if df_atlas is not None and atlas_cols:
        X_E = mtf.build_X(df_atlas, atlas_cols, add_atlas_diff=True)
        Y_E  = Y_atlas
        Yh_E = Yh_atlas
    else:
        X_E = Y_E = Yh_E = None

    print(f"\npre-CORAL: dimension")
    print(f"  Model A (ZTF, no changes): {X_A.shape[1]:4d}")
    print(f"  Model B (LSST): {X_B_pre.shape[1]:4d}")
    print(f"  Model C (ZTF+LSST): {X_C_pre.shape[1]:4d}")
    print(f"  Model D (combined): {X_D_pre.shape[1]:4d}")
    if X_E is not None:
        print(f"  Model E (ATLAS, no changes): {X_E.shape[1]:4d}")

    # ── Aplicar CORAL ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CORAL DOMAIN ADAPTATION")
    print("=" * 60)

    _feat_sel_dir  = os.path.dirname(CONSENSUS_CSV)
    _ranking_B_csv = os.path.join(_feat_sel_dir, "feature_ranking_modelB.csv")
    _ranking_C_csv = os.path.join(_feat_sel_dir, "feature_ranking_modelC.csv")
    _ranking_D_csv = os.path.join(_feat_sel_dir, "feature_ranking_modelD.csv")

    # Modelo B: CORAL sobre features LSST

    coral_B_avail = _filter_by_ks(
        [c for c in coral_B if c in X_B_pre.columns],
        _ranking_B_csv, suffix_hint="B")
    if coral_B_avail and df_real_lsst is not None:
        X_B = fit_and_apply_coral_for_model(
            X_B_pre, df_real_lsst, coral_B_avail, "B", CORAL_DIR)
    else:
        print(f"  [CORAL B] Skipped (coral_features={len(coral_B_avail)}, "
              f"real_lsst={'available' if df_real_lsst is not None else 'missing'})")
        X_B = X_B_pre

    # Modelo C: CORAL separado para ZTF y LSST
    coral_C_ztf_avail  = _filter_by_ks(
        [c for c in coral_C_ztf  if c in X_C_pre.columns],
        _ranking_C_csv, suffix_hint="C_ztf")
    coral_C_lsst_avail = _filter_by_ks(
        [c for c in coral_C_lsst if c in X_C_pre.columns],
        _ranking_C_csv, suffix_hint="C_lsst")
    X_C = X_C_pre.copy()

    if coral_C_ztf_avail and df_real_ztf is not None:
        X_C = fit_and_apply_coral_for_model(
            X_C, df_real_ztf, coral_C_ztf_avail, "C_ztf", CORAL_DIR)
    if coral_C_lsst_avail and df_real_lsst is not None:
        X_C = fit_and_apply_coral_for_model(
            X_C, df_real_lsst, coral_C_lsst_avail, "C_lsst", CORAL_DIR)
    coral_C_atlas_avail = _filter_by_ks(
        [c for c in coral_C_atlas if c in X_C_pre.columns],
        _ranking_C_csv, suffix_hint="C_atlas")
    if coral_C_atlas_avail and df_real_atlas is not None:
        X_C = fit_and_apply_coral_for_model(
            X_C, df_real_atlas, coral_C_atlas_avail, "C_atlas", CORAL_DIR)
    elif coral_C_atlas_avail:
        print(f"  [CORAL C_atlas] Skipped. real_atlas missing")

    # ── CORAL sobre features diferenciales CORAL ─────────────────────────────
    if consensus is not None and os.path.exists(
            os.path.join(os.path.dirname(CONSENSUS_CSV), "feature_ranking_modelC.csv")):
        _ranking_C = pd.read_csv(
            os.path.join(os.path.dirname(CONSENSUS_CSV), "feature_ranking_modelC.csv"),
            index_col=0)
        def _is_diff(col):
            return any(x in col for x in [
                'gr_ratio', 'gr_diff', 'co_ratio', 'co_diff',
                'ztf_lsst', 'ztf_atlas', 'atlas_lsst', 'color_'])
        _coral_diff = [c for c in X_C.columns
                       if _is_diff(c)
                       and c in _ranking_C.index
                       and _ranking_C.loc[c, 'quadrant'] == 'CORAL']
        if _coral_diff:
            _real_parts = []
            if df_real_ztf is not None:
                _real_parts.append(df_real_ztf[[c for c in df_real_ztf.columns
                                                if c.endswith('_ztf')]])
            if df_real_lsst is not None:
                _real_parts.append(df_real_lsst[[c for c in df_real_lsst.columns
                                                  if c.endswith('_lsst')]])
            if df_real_atlas is not None:
                _real_parts.append(df_real_atlas[[c for c in df_real_atlas.columns
                                                   if c.endswith('_atlas')]])
            if _real_parts:
                import functools
                df_real_joined = functools.reduce(
                    lambda a, b: a.join(b, how='inner'), _real_parts)
                df_real_X = mtf.build_X(
                    df_real_joined,
                    [c for c in df_real_joined.columns],
                    add_ztf_diff=True, add_cross_diff=True,
                    add_atlas_diff=df_real_atlas is not None,
                )
                _coral_diff_avail = _filter_by_ks(
                    [c for c in _coral_diff if c in df_real_X.columns],
                    _ranking_C_csv, suffix_hint="C_diff")
                if _coral_diff_avail:
                    X_C = fit_and_apply_coral_for_model(
                        X_C, df_real_X, _coral_diff_avail,
                        "C_diff", CORAL_DIR)
                    print(f"  [CORAL C_diff] {len(_coral_diff_avail)} adapted differential features")

    # Modelo D: CORAL sobre features combined
    coral_D_avail = _filter_by_ks(
        [c for c in coral_D if c in X_D_pre.columns],
        _ranking_D_csv, suffix_hint="D")
    if coral_D_avail and df_real_comb is not None:
        X_D = fit_and_apply_coral_for_model(
            X_D_pre, df_real_comb, coral_D_avail, "D", CORAL_DIR)
    else:
        print(f"  [CORAL D] Skipped (coral_features={len(coral_D_avail)}, "
              f"real_comb={'available' if df_real_comb is not None else 'missing'})")
        X_D = X_D_pre

    print(f"\npost-CORAL dimension:")
    print(f"  Model B: {X_B.shape[1]:4d}")
    print(f"  Model C: {X_C.shape[1]:4d}")
    print(f"  Model D: {X_D.shape[1]:4d}")

    # ── Train/test split (mismo RANDOM_STATE que el original) ─────────────────
    print("\nSplit train/test...")
    split_main = model_selection.train_test_split(
        X_A, X_B, X_C, X_D, Y_original,
        test_size=0.2,
        stratify=Y_original,
        random_state=mtf.RANDOM_STATE,
    )
    XA_tr, XA_te = split_main[0], split_main[1]
    XB_tr, XB_te = split_main[2], split_main[3]
    XC_tr, XC_te = split_main[4], split_main[5]
    XD_tr, XD_te = split_main[6], split_main[7]
    yo_tr, yo_te  = split_main[8], split_main[9]
    yh_tr = mtf.make_hierarchical_labels(yo_tr)

    print(f"  ZTF+LSST: {len(yo_tr)} train | {len(yo_te)} test")

    if X_E is not None:
        atlas_tr_idx = yo_tr.index.intersection(df_atlas.index)
        atlas_te_idx = yo_te.index.intersection(df_atlas.index)
        XE_tr  = X_E.loc[atlas_tr_idx]
        XE_te  = X_E.loc[atlas_te_idx]
        yoE_tr = yo_tr.loc[atlas_tr_idx]
        yoE_te = yo_te.loc[atlas_te_idx]
        yhE_tr = mtf.make_hierarchical_labels(yoE_tr)
        print(f"  ATLAS:    {len(yoE_tr)} train | {len(yoE_te)} test")
    else:
        XE_tr = XE_te = yoE_tr = yoE_te = yhE_tr = None

    # ── Fase 1: OOF + calibración ─────────────────────────────────────────────
    print("\n=== PHASE 1: OOF predictions + calibration ===")
    oof_A = mtf.compute_oof_predictions(XA_tr, yo_tr,  "modelA")
    oof_B = mtf.compute_oof_predictions(XB_tr, yo_tr,  "modelB")
    oof_C = mtf.compute_oof_predictions(XC_tr, yo_tr,  "modelC")
    oof_D = mtf.compute_oof_predictions(XD_tr, yo_tr,  "modelD")
    oof_E = (mtf.compute_oof_predictions(XE_tr, yoE_tr, "modelE")
             if XE_tr is not None else None)

    calibrators = {}
    for model_name, oof_df, y_labels in [
        ("A", oof_A, yo_tr), ("B", oof_B, yo_tr),
        ("C", oof_C, yo_tr), ("D", oof_D, yo_tr),
    ]:
        print(f"\n  Calibrating model {model_name}:")
        oof_proba, _ = mtf._oof_to_proba_df(oof_df, model_name, mtf.LABEL_ORDER)
        cal_dict, method = mtf.fit_calibrators(oof_proba, y_labels,
                                               mtf.LABEL_ORDER, method="auto")
        calibrators[model_name] = cal_dict
        mtf.plot_calibration_curves(
            oof_proba, y_labels, cal_dict, mtf.LABEL_ORDER,
            model_name=f"Model {model_name}",
            save_path=f"{PLOTS_DIR_ADP}reliability_model{model_name}.pdf",
        )
        with open(f"{MODELS_DIR_ADP}calibrators_model{model_name}.pkl", "wb") as f:
            pickle.dump({"calibrators": cal_dict, "method": method,
                         "label_order": mtf.LABEL_ORDER}, f, pickle.HIGHEST_PROTOCOL)

    if oof_E is not None:
        oof_proba_E, _ = mtf._oof_to_proba_df(oof_E, "E", mtf.LABEL_ORDER)
        cal_dict_E, method_E = mtf.fit_calibrators(
            oof_proba_E, yoE_tr, mtf.LABEL_ORDER, method="auto")
        calibrators["E"] = cal_dict_E
        mtf.plot_calibration_curves(
            oof_proba_E, yoE_tr, cal_dict_E, mtf.LABEL_ORDER,
            model_name="Model E adapted",
            save_path=f"{PLOTS_DIR_ADP}reliability_modelE.pdf",
        )
        with open(f"{MODELS_DIR_ADP}calibrators_modelE.pkl", "wb") as f:
            pickle.dump({"calibrators": cal_dict_E, "method": method_E,
                         "label_order": mtf.LABEL_ORDER}, f, pickle.HIGHEST_PROTOCOL)

    def _calibrate_oof(oof_df, model_name):
        oof_cal = oof_df.copy()
        prob_cols  = [c for c in oof_df.columns if f"_model{model_name}" in c]
        cls_names  = np.array([c.replace("p_", "").replace(f"_model{model_name}", "")
                                for c in prob_cols])
        prob_matrix = oof_df[prob_cols].values
        prob_cal    = mtf.apply_calibrators(prob_matrix, cls_names, calibrators[model_name])
        for j, col in enumerate(prob_cols):
            oof_cal[col] = prob_cal[:, j]
        return oof_cal

    def _reindex_oof(oof_cal, ref_index, model_name, fill=mtf.PRIOR_PROB):
        cols   = [f"p_{c}_model{model_name}" for c in mtf.LABEL_ORDER]
        df_out = pd.DataFrame(fill, index=ref_index, columns=cols)
        shared = oof_cal.index.intersection(ref_index)
        df_out.loc[shared, cols] = oof_cal.loc[shared, cols].values
        # Binary mask: 1 = real predictions, 0 = prior fill
        mask_col = f"has_model{model_name}"
        df_out[mask_col] = 0
        df_out.loc[shared, mask_col] = 1
        return df_out

    meta_index = yo_tr.index
    oof_parts  = [
        _reindex_oof(_calibrate_oof(oof_A, "A"), meta_index, "A"),
        _reindex_oof(_calibrate_oof(oof_B, "B"), meta_index, "B"),
        _reindex_oof(_calibrate_oof(oof_C, "C"), meta_index, "C"),
        _reindex_oof(_calibrate_oof(oof_D, "D"), meta_index, "D"),
    ]
    if oof_E is not None:
        oof_E_cal  = _calibrate_oof(oof_E, "E")
        oof_parts.append(_reindex_oof(oof_E_cal, meta_index, "E"))

    oof_all = pd.concat(oof_parts, axis=1)
    oof_all["true_label"] = yo_tr
    oof_all.to_csv(f"{OOF_DIR_ADP}oof_all_models_calibrated.csv")
    print(f"\nOOF saved: {OOF_DIR_ADP}oof_all_models_calibrated.csv")

    # ── Fase 2: Reentrenamiento completo ──────────────────────────────────────
    print("\n=== PHASE 2: Retraining with complete set ===")

    retrain_specs = [
        ("A", XA_tr, yo_tr, yh_tr),
        ("B", XB_tr, yo_tr, yh_tr),
        ("C", XC_tr, yo_tr, yh_tr),
        ("D", XD_tr, yo_tr, yh_tr),
    ]
    if XE_tr is not None:
        retrain_specs.append(("E", XE_tr, yoE_tr, yhE_tr))

    models = {}
    for name, X_tr, yo, yh in retrain_specs:
        print(f"  Training Model {name}...")
        clf_hier, clfs = mtf.train_hierarchical_model(X_tr, yh, yo)
        models[name]   = (clf_hier, clfs)
        with open(f"{MODELS_DIR_ADP}model_{name}_hier.pkl", "wb") as f:
            pickle.dump(clf_hier, f, pickle.HIGHEST_PROTOCOL)
        for group, clf in clfs.items():
            with open(f"{MODELS_DIR_ADP}model_{name}_{group}.pkl", "wb") as f:
                pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

    # ── Fase 3: Predicciones sobre test ──────────────────────────────────────────────
    print("\n=== PHASE 3: Test predictions ===")

    test_specs = [
        ("A", XA_te, yo_te), ("B", XB_te, yo_te),
        ("C", XC_te, yo_te), ("D", XD_te, yo_te),
    ]
    if XE_te is not None:
        test_specs.append(("E", XE_te, yoE_te))

    test_probas = {}
    for name, X_te, yo_te_model in test_specs:
        clf_hier, clfs  = models[name]
        prob_matrix, cn = mtf.predict_hierarchical(X_te, clf_hier, clfs)
        mtf.evaluate_model(yo_te_model, prob_matrix, cn, mtf.LABEL_ORDER,
                           f"Model_{name}_adapted_raw")
        prob_cal = mtf.apply_calibrators(prob_matrix, cn, calibrators[name])
        mtf.evaluate_model(yo_te_model, prob_cal, cn, mtf.LABEL_ORDER,
                           f"Model_{name}_adapted_cal")
        test_probas[name] = (prob_cal, cn, yo_te_model)
        mtf.plot_feature_importances(
            clf_hier, list(X_te.columns),
            f"{PLOTS_DIR_ADP}feat_importance_model{name}_hier.pdf",
            color=MODEL_COLORS.get(name),
        )

    # ── Fase 4: Metamodelo ───────────────────────────────────────────────────
    print("\n=== PHASE 4: Metamodel ===")

    print('Training metamodel...')

    def proba_to_df(prob_matrix, class_names, model_name, ref_index,
                    prob_matrix_index, fill=mtf.PRIOR_PROB):
        cols   = [f"p_{c}_model{model_name}" for c in mtf.LABEL_ORDER]
        df_out = pd.DataFrame(fill, index=ref_index, columns=cols)
        shared = ref_index.intersection(prob_matrix_index)
        for j, cls in enumerate(mtf.LABEL_ORDER):
            idx = np.where(class_names == cls)[0]
            if len(idx):
                df_out.loc[shared, f"p_{cls}_model{model_name}"] = \
                    prob_matrix[prob_matrix_index.get_indexer(shared), idx[0]]
        # Binary mask: 1 = real predictions, 0 = prior fill
        mask_col = f"has_model{model_name}"
        df_out[mask_col] = 0
        df_out.loc[shared, mask_col] = 1
        return df_out

    meta_test_parts = []
    for name, (pm, cn, yo_te_m) in test_probas.items():
        meta_test_parts.append(
            proba_to_df(pm, cn, name, yo_te.index, yo_te_m.index))
    meta_X_test = pd.concat(meta_test_parts, axis=1)

    from sklearn.linear_model import LogisticRegression
    meta_X_train = oof_all.drop(columns=["true_label"])
    meta_y_train = oof_all["true_label"]

    # ── Aumentar con datos reales etiquetados ────────────────────────────────
    sample_weight = None
    if USE_REAL_META_AUG and REAL_META_UPWEIGHT > 0:
        print(f'\n  [Meta aug] Obtaining probabilities of real set...')
        X_real, y_real = _build_real_X_for_meta(
            df_real_ztf, df_real_lsst, df_real_comb, df_real_atlas,
            sel_B, sel_C_ztf, sel_C_lsst,
            sel_C_atlas if atlas_cols else [],
            sel_D, ztf_cols, atlas_cols,
            CORAL_DIR,
            REAL_LABELS_FILE,
        )

        if X_real is not None and y_real is not None:
            real_ref_index = y_real.index

            real_meta_parts = []
            for name in ['A', 'B', 'C', 'D', 'E']:
                if name not in X_real or name not in models:
                    # E puede estar ausente; rellenar con prior + mask=0
                    cols_n = [f"p_{c}_model{name}" for c in mtf.LABEL_ORDER]
                    if any(f'_model{name}' in c for c in meta_X_train.columns):
                        df_prior = pd.DataFrame(mtf.PRIOR_PROB,
                                                index=real_ref_index, columns=cols_n)
                        df_prior[f"has_model{name}"] = 0
                        real_meta_parts.append(df_prior)
                    continue

                X_r = X_real[name]
                clf_hier_r = models[name][0]
                if hasattr(clf_hier_r, 'feature_names_in_'):
                    expected_feats = list(clf_hier_r.feature_names_in_)
                    for col in expected_feats:
                        if col not in X_r.columns:
                            X_r[col] = -999
                    X_r = X_r[expected_feats]

                y_ref_r = y_real if name != 'E' else (
                    y_real.loc[X_r.index] if X_r is not None else y_real)
                pm_r, cn_r = mtf.predict_hierarchical(X_r, *models[name])
                pm_r_cal   = mtf.apply_calibrators(pm_r, cn_r, calibrators[name])

                real_meta_parts.append(
                    proba_to_df(pm_r_cal, cn_r, name,
                                real_ref_index, X_r.index))

            if real_meta_parts:
                meta_X_real = pd.concat(real_meta_parts, axis=1)

                expected_cols = list(meta_X_train.columns)
                for c in expected_cols:
                    if c not in meta_X_real.columns:
                        meta_X_real[c] = 0 if c.startswith('has_model') else mtf.PRIOR_PROB
                meta_X_real = meta_X_real[expected_cols]
                meta_y_real = y_real.loc[meta_X_real.index]

                meta_X_train = pd.concat([meta_X_train, meta_X_real], axis=0)
                meta_y_train = pd.concat([meta_y_train, meta_y_real], axis=0)
                sample_weight = np.concatenate([
                    np.ones(len(meta_X_train) - len(meta_X_real)),
                    np.full(len(meta_X_real), float(REAL_META_UPWEIGHT)),
                ])
                print(f'  [Meta aug] Metamodel training set: '
                      f'{len(meta_X_train)} rows '
                      f'({len(meta_X_train) - len(meta_X_real)} sim + '
                      f'{len(meta_X_real)} real, upweight×{REAL_META_UPWEIGHT})')
            else:
                print('  [Meta aug] Couldn´t obtain real probabilities. '
                      'Skipping metamodel augmentation.')
        else:
            print('  [Meta aug] Real data not available — '
                  'Skipping metamodel augmentation.')
    else:
        if USE_REAL_META_AUG:
            print('  [Meta aug] REAL_META_UPWEIGHT=0. Augmentation disabled.')
        else:
            print('  [Meta aug] USE_REAL_META_AUG=False. Augmentation disabled.')

    # ── Model dropout para robustez del metamodelo ─────────────────────────
    # Dropout a nivel de SURVEY, no de modelo individual.  Se dropea cada
    # survey (ZTF, LSST, ATLAS) con probabilidad META_DROPOUT_PROB y se
    # propaga la ausencia a los modelos que dependen de ese survey:
    #   A  requiere ZTF
    #   B  requiere LSST
    #   C  requiere ZTF + LSST  (ATLAS no es requisito)
    #   D  requiere ZTF + LSST  (curva combinada)
    #   E  requiere ATLAS
    # Al menos un survey sobrevive siempre por objeto.
    SURVEY_DEPS = {
        'A': {'ZTF'},
        'B': {'LSST'},
        'C': {'ZTF', 'LSST'},
        'D': {'ZTF', 'LSST'},
        'E': {'ATLAS'},
    }
    ALL_SURVEYS = ['ZTF', 'LSST', 'ATLAS']

    if META_DROPOUT_PROB > 0:
        print(f'\n  [Dropout] Applying survey dropout p={META_DROPOUT_PROB}...')
        rng = np.random.RandomState(mtf.RANDOM_STATE + 1)
        n_rows = len(meta_X_train)
        available_models = [m for m in SURVEY_DEPS
                            if f"has_model{m}" in meta_X_train.columns]

        survey_drop = rng.random((n_rows, len(ALL_SURVEYS))) < META_DROPOUT_PROB

        all_dropped = survey_drop.all(axis=1)
        if all_dropped.any():
            for i in np.where(all_dropped)[0]:
                keep = rng.randint(len(ALL_SURVEYS))
                survey_drop[i, keep] = False

        survey_dropped = {s: survey_drop[:, j] for j, s in enumerate(ALL_SURVEYS)}

        for model_name in available_models:
            mask_col  = f"has_model{model_name}"
            prob_cols = [c for c in meta_X_train.columns
                         if c.startswith("p_") and f"_model{model_name}" in c]
            required  = SURVEY_DEPS[model_name]
            model_drop = np.zeros(n_rows, dtype=bool)
            for survey in required:
                if survey in survey_dropped:
                    model_drop |= survey_dropped[survey]
            has_data = meta_X_train[mask_col] == 1
            to_drop  = has_data & pd.Series(model_drop, index=meta_X_train.index)
            meta_X_train.loc[to_drop, prob_cols] = mtf.PRIOR_PROB
            meta_X_train.loc[to_drop, mask_col]  = 0
            n_dropped = to_drop.sum()
            print(f'    Model {model_name} (req: {required}): '
                  f'{n_dropped}/{has_data.sum()} dropped '
                  f'({100*n_dropped/max(has_data.sum(),1):.1f}%)')

        _surv_active = {s: ~survey_dropped[s] for s in ALL_SURVEYS}
        _combos = pd.DataFrame(_surv_active).value_counts().sort_index()
        print(f'Survey combination distribution:')
        for combo, count in _combos.items():
            active = [s for s, v in zip(ALL_SURVEYS, combo) if v]
            print(f'      {"+".join(active) if active else "NONE":25s}: '
                  f'{count:5d} ({100*count/n_rows:.1f}%)')
    else:
        print('\n  [Dropout] Disabled (META_DROPOUT_PROB=0)')

    #Entrenar metamodelo
    meta_model = LogisticRegression(
        C=1.0, max_iter=2000, multi_class="multinomial",
        solver="lbfgs", random_state=mtf.RANDOM_STATE,
    )
    meta_model.fit(meta_X_train, meta_y_train, sample_weight=sample_weight)

    #Guardar
    import shutil
    meta_model_path    = f"{MODELS_DIR_ADP}meta_model.pkl"
    meta_baseline_path = f"{MODELS_DIR_ADP}meta_model_baseline.pkl"
    if USE_REAL_META_AUG and sample_weight is not None:
        if os.path.exists(meta_model_path) and not os.path.exists(meta_baseline_path):
            shutil.copy2(meta_model_path, meta_baseline_path)
            print(f'  [Meta aug] Baseline saved: {meta_baseline_path}')
        elif os.path.exists(meta_baseline_path):
            print(f'  [Meta aug] Baseline already exists, it won´t overwrite.')
    with open(meta_model_path, "wb") as f:
        pickle.dump(meta_model, f, pickle.HIGHEST_PROTOCOL)
    aug_tag = (f'augmentado (×{REAL_META_UPWEIGHT} real)'
               if USE_REAL_META_AUG and sample_weight is not None
               else 'baseline (solo simulado)')
    print(f'  Metamodel {aug_tag} saved: {meta_model_path}')

    meta_pred_proba = meta_model.predict_proba(meta_X_test)
    meta_pred       = meta_model.predict(meta_X_test)

    print('Metamodel done')

    # ── Fase 5: Evaluación y comparativa ──────────────────────────────────────
    print("\n=== PHSE 5: Final evaluation ===")

    print("\n=== ADAPTED METAMODEL ===")
    print("Balanced accuracy:",
          "%0.3f" % metrics.balanced_accuracy_score(yo_te, meta_pred))
    print("Macro F1:         ",
          "%0.3f" % metrics.f1_score(yo_te, meta_pred, average="macro"))
    print(metrics.classification_report(yo_te, meta_pred, digits=3))
    
    cm = metrics.confusion_matrix(yo_te, meta_pred, labels=mtf.LABEL_ORDER)
    mtf.plot_confusion_matrix(
        cm, mtf.LABEL_ORDER,
        f"{PLOTS_DIR_ADP}conf_matrix_metamodel.pdf",
        title="Metamodel",
        cmap=MODEL_CMAPS.get('Metamodel'),
    )

    # Tabla resumen
    _feat_dims = {name: X_te.shape[1] for name, X_te, _ in test_specs}
    summary_rows = []
    for name, (pm, cn, yo_te_m) in test_probas.items():
        pred = [cn[i] for i in np.argmax(pm, axis=1)]
        summary_rows.append({
            "Model":             f"Model_{name}_adapted",
            "balanced_accuracy": round(metrics.balanced_accuracy_score(yo_te_m, pred), 4),
            "macro_f1":          round(metrics.f1_score(yo_te_m, pred, average="macro"), 4),
            "n_features":        _feat_dims.get(name),
        })
    summary_rows.append({
        "Model":             "Metamodel_adapted",
        "balanced_accuracy": round(metrics.balanced_accuracy_score(yo_te, meta_pred), 4),
        "macro_f1":          round(metrics.f1_score(yo_te, meta_pred, average="macro"), 4),
        "n_features":        None,
    })
    summary_adapted = pd.DataFrame(summary_rows)
    summary_adapted.to_csv(f"{MODELS_DIR_ADP}model_comparison_summary.csv", index=False)
    print(f"\nResumen guardado: {MODELS_DIR_ADP}model_comparison_summary.csv")

    # =========================================================================
    # PHASE 5b: FULL ANALYSIS
    # =========================================================================

    TRANSIENT_CLASSES = ['SNIa', 'SNIbc', 'SNII', 'SLSN']

    def _compute_entropy(prob_matrix):
        eps = 1e-12
        p = np.clip(prob_matrix, eps, 1)
        return -np.sum(p * np.log2(p), axis=1)

    def _plot_entropy_by_class(entropy, y_true, model_name, save_path):
        from matplotlib.patches import Patch
        y_arr  = np.array(y_true)
        means  = np.array([entropy[y_arr == cls].mean() if (y_arr == cls).sum() > 0
                           else 0.0 for cls in mtf.LABEL_ORDER])
        stds   = np.array([entropy[y_arr == cls].std()  if (y_arr == cls).sum() > 0
                           else 0.0 for cls in mtf.LABEL_ORDER])
        group_colors = {'Transient': '#e05c5c', 'Stochastic': '#5c8ae0', 'Periodic': '#5cb85c'}
        cls_to_group = {cls: grp for grp, members in mtf.HIER_MAP.items() for cls in members}
        colors = [group_colors.get(cls_to_group.get(cls, ''), '#aaaaaa')
                  for cls in mtf.LABEL_ORDER]
        x = np.arange(len(mtf.LABEL_ORDER))
        fig, ax = plt.subplots(figsize=(14, 5))
        for xi, (m, s, c) in enumerate(zip(means, stds, colors)):
            ax.errorbar(xi, m, yerr=s, fmt='o', color=c, capsize=4,
                        linewidth=1.2, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        ax.legend(handles=[Patch(facecolor=c, alpha=0.85, label=g)
                            for g, c in group_colors.items()],
                  fontsize=10, loc='upper right')
        ax.set_xticks(x)
        ax.set_xticklabels(mtf.LABEL_ORDER, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('True class', fontsize=12)
        ax.set_ylabel('Mean prediction entropy (bits)', fontsize=12)
        ax.set_title(f'Entropy by class — {model_name}', fontsize=13)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=1)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Entropy plot saved: {save_path}")

    def _entropy_summary_df(entropy, y_true):
        y_arr = np.array(y_true)
        return pd.DataFrame([{
            'class':          cls,
            'mean_entropy':   round(float(entropy[y_arr == cls].mean()), 4),
            'median_entropy': round(float(np.median(entropy[y_arr == cls])), 4),
            'n':              int((y_arr == cls).sum()),
        } for cls in mtf.LABEL_ORDER if (y_arr == cls).sum() > 0])

    def _plot_transient_kde(prob_matrix, class_names, y_true, model_name, save_path):
        from scipy.stats import gaussian_kde
        class_names = np.array(class_names)
        y_arr = np.array(y_true)
        n_tr  = len(TRANSIENT_CLASSES)
        fig, axes = plt.subplots(n_tr, n_tr,
                                 figsize=(n_tr * 3.5, n_tr * 3.0), sharex=True)
        fig.suptitle(f'Transient probability distributions — {model_name}',
                     fontsize=13, y=1.01)
        x_grid = np.linspace(0, 1, 300)
        for ri, true_cls in enumerate(TRANSIENT_CLASSES):
            true_mask = y_arr == true_cls
            if not true_mask.any():
                for ci in range(n_tr): axes[ri, ci].set_visible(False)
                continue
            pm_sub   = prob_matrix[true_mask]
            pred_cls = class_names[np.argmax(pm_sub, axis=1)]
            for ci, pred_name in enumerate(TRANSIENT_CLASSES):
                ax = axes[ri, ci]
                col_pos = np.where(class_names == pred_name)[0]
                if not len(col_pos): ax.set_visible(False); continue
                probs = pm_sub[:, col_pos[0]]
                for mask, color, label in [
                    (pred_cls == true_cls,  '#2ca02c', 'Correct'),
                    (pred_cls != true_cls,  '#d62728', 'Misclassified'),
                ]:
                    vals = probs[mask]
                    if vals.sum() == 0 or len(vals) < 3: continue
                    if vals.std() < 1e-6:
                        ax.axvline(vals.mean(), color=color, lw=1.5, ls='--', label=label)
                        continue
                    try:
                        kde = gaussian_kde(vals, bw_method='scott')
                        ax.fill_between(x_grid, kde(x_grid), alpha=0.35, color=color)
                        ax.plot(x_grid, kde(x_grid), color=color, lw=1.5, label=label)
                    except Exception:
                        pass
                if ri == ci: ax.set_facecolor('#f5f5f5')
                ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
                ax.tick_params(labelsize=7)
                if ri == n_tr - 1: ax.set_xlabel(f'P({pred_name})', fontsize=9)
                if ci == 0:        ax.set_ylabel(f'True: {true_cls}', fontsize=9)
                if ri == 0 and ci == n_tr - 1: ax.legend(fontsize=7, loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight'); plt.close()
        print(f"  Transient KDE saved: {save_path}")

    def _plot_metamodel_reliability(meta_pred_proba, y_true, meta_model_obj, save_path):
        from sklearn.calibration import calibration_curve
        n_cols = 5
        n_rows = (len(mtf.LABEL_ORDER) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.5, n_rows * 3.5))
        axes = axes.flatten()
        class_order = meta_model_obj.classes_
        for idx, cls in enumerate(mtf.LABEL_ORDER):
            ax = axes[idx]
            col_idx = np.where(class_order == cls)[0]
            if not len(col_idx): ax.set_visible(False); continue
            p_cls = meta_pred_proba[:, col_idx[0]]
            y_bin = (np.array(y_true) == cls).astype(int)
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_bin, p_cls, n_bins=10, strategy='quantile')
                ax.plot(mean_pred, frac_pos, 's-', color='steelblue', label='Metamodel')
            except ValueError:
                ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center', fontsize=8)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_title(cls, fontsize=10)
            ax.set_xlabel('Mean predicted prob.', fontsize=8)
            ax.set_ylabel('Fraction of positives', fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for idx in range(len(mtf.LABEL_ORDER), len(axes)):
            axes[idx].set_visible(False)
        plt.suptitle('Reliability diagrams — Metamodel', fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight'); plt.close()
        print(f"  Reliability diagram saved: {save_path}")

    def _bootstrap_metrics(y_true, prob_matrix, class_names, n_bootstrap=1000):
        rng = np.random.RandomState(mtf.RANDOM_STATE)
        n   = len(y_true); y_arr = np.array(y_true); eps = 1e-12
        acc_b = np.empty(n_bootstrap); bacc_b = np.empty(n_bootstrap)
        f1_b  = np.empty(n_bootstrap); ent_b  = np.empty(n_bootstrap)
        per_class = {cls: {'precision': [], 'recall': [], 'f1': [], 'entropy': []}
                     for cls in mtf.LABEL_ORDER}
        for i in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            y_b = y_arr[idx]; pm_b = prob_matrix[idx]
            pred_b = [class_names[j] for j in np.argmax(pm_b, axis=1)]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                acc_b[i]  = metrics.accuracy_score(y_b, pred_b)
                bacc_b[i] = metrics.balanced_accuracy_score(y_b, pred_b)
                f1_b[i]   = metrics.f1_score(y_b, pred_b, average='macro', zero_division=0)
                ent_b[i]  = (-np.sum(np.clip(pm_b, eps, 1) *
                               np.log2(np.clip(pm_b, eps, 1)), axis=1)).mean()
                rep = metrics.classification_report(
                    y_b, pred_b, labels=mtf.LABEL_ORDER,
                    output_dict=True, zero_division=0)
            for cls in mtf.LABEL_ORDER:
                if cls in rep:
                    per_class[cls]['precision'].append(rep[cls]['precision'])
                    per_class[cls]['recall'].append(rep[cls]['recall'])
                    per_class[cls]['f1'].append(rep[cls]['f1-score'])
                mask_cls = y_b == cls
                if mask_cls.sum() > 0:
                    p_cls = np.clip(pm_b[mask_cls], eps, 1)
                    per_class[cls]['entropy'].append(
                        (-np.sum(p_cls * np.log2(p_cls), axis=1)).mean())
        g_std = {
            'accuracy_std':          round(float(acc_b.std()),  5),
            'balanced_accuracy_std': round(float(bacc_b.std()), 5),
            'macro_f1_std':          round(float(f1_b.std()),   5),
            'mean_entropy_std':      round(float(ent_b.std()),  5),
        }
        pc_rows = [{'class': cls,
                    **{f'{m}_std': round(float(np.std(per_class[cls][m])), 5)
                       if per_class[cls][m] else np.nan
                       for m in ['precision', 'recall', 'f1', 'entropy']}}
                   for cls in mtf.LABEL_ORDER]
        return g_std, pd.DataFrame(pc_rows)

    # ── Ejecutar análisis completo ─────────────────────────────────────────────
    print("\n=== FULL ANALYSIS ===")

    entropy_records   = []
    per_class_rows    = []
    conf_rows         = []
    bootstrap_rows    = []
    summary_full_rows = []
    _ent_lookup       = {}

    # Base models
    _X_te_map = {'A': XA_te, 'B': XB_te, 'C': XC_te, 'D': XD_te}
    if XE_te is not None:
        _X_te_map['E'] = XE_te
    
    # ── Check: clases ausentes en predicciones ───────────────────────────
    print("\nCHECK: classes without model predictions")
    for name, (pm, cn, yo_te_m) in test_probas.items():
        pred = [cn[i] for i in np.argmax(pm, axis=1)]
        pred_series = pd.Series(pred)
        ausentes = [c for c in mtf.LABEL_ORDER if c not in pred_series.unique()]
        sin_test  = [c for c in mtf.LABEL_ORDER if c not in yo_te_m.unique()]
        if ausentes:
            print(f"  Model {name}: does not predcit {ausentes}")
        if sin_test:
            print(f"  Model {name}: missing in test {sin_test}")
        if not ausentes and not sin_test:
            print(f"  Model {name}: OK")
    
    for name, (pm, cn, yo_te_m) in test_probas.items():
        pred = [cn[i] for i in np.argmax(pm, axis=1)]

        # Confusion matrix
        mtf.plot_confusion_matrix(
            metrics.confusion_matrix(yo_te_m, pred, labels=mtf.LABEL_ORDER),
            mtf.LABEL_ORDER,
            f"{PLOTS_DIR_ADP}conf_matrix_model{name}.pdf",
            title=f"Model {name}",
            cmap=MODEL_CMAPS.get(name),
        )
        cm_df = pd.DataFrame(
            metrics.confusion_matrix(yo_te_m, pred, labels=mtf.LABEL_ORDER),
            index=mtf.LABEL_ORDER, columns=mtf.LABEL_ORDER)
        cm_df.index.name = 'true_class'
        cm_df.insert(0, 'model', f'Model_{name}')
        conf_rows.append(cm_df.reset_index())

        # Feature importances
        if name in _X_te_map:
            mtf.plot_feature_importances(
                models[name][0], list(_X_te_map[name].columns),
                f"{PLOTS_DIR_ADP}feat_importance_model{name}_hier.pdf",
                color=MODEL_COLORS.get(name))

        # Entropy
        ent = _compute_entropy(pm)
        _plot_entropy_by_class(ent, yo_te_m, f"Model {name}",
                               f"{PLOTS_DIR_ADP}entropy_model{name}.pdf")
        df_ent = _entropy_summary_df(ent, yo_te_m)
        df_ent.insert(0, 'model', f'Model_{name}')
        entropy_records.append(df_ent)
        _ent_lookup[f'Model_{name}'] = round(float(ent.mean()), 4)

        # Transient KDE
        _plot_transient_kde(pm, cn, yo_te_m, f"Model {name}",
                            f"{PLOTS_DIR_ADP}transient_kde_model{name}.pdf")

        # Bootstrap
        print(f"  Bootstrap Model {name} ...")
        g_std, pc_std = _bootstrap_metrics(yo_te_m, pm, cn)
        pc_std.insert(0, 'model', f'Model_{name}')
        bootstrap_rows.append(pc_std)

        # Per-class metrics
        report = metrics.classification_report(
            yo_te_m, pred, labels=mtf.LABEL_ORDER,
            output_dict=True, zero_division=0)
        for cls in mtf.LABEL_ORDER:
            if cls in report:
                per_class_rows.append({
                    'model': f'Model_{name}', 'class': cls,
                    'precision': round(report[cls]['precision'], 4),
                    'recall':    round(report[cls]['recall'],    4),
                    'f1':        round(report[cls]['f1-score'],  4),
                    'support':   int(report[cls]['support']),
                })

        # Summary row
        summary_full_rows.append({
            'Model':             f'Model_{name}',
            'Accuracy':          round(metrics.accuracy_score(yo_te_m, pred),          4),
            'Balanced_Accuracy': round(metrics.balanced_accuracy_score(yo_te_m, pred), 4),
            'Macro_F1':          round(metrics.f1_score(yo_te_m, pred,
                                       average='macro', zero_division=0),               4),
            'Mean_Entropy':      _ent_lookup.get(f'Model_{name}', np.nan),
            'N_test':            len(yo_te_m),
            **g_std,
        })

    # Metamodel plots
    meta_ent = _compute_entropy(meta_pred_proba)
    _plot_entropy_by_class(meta_ent, yo_te, "Metamodel",
                           f"{PLOTS_DIR_ADP}entropy_metamodel.pdf")
    df_ent_meta = _entropy_summary_df(meta_ent, yo_te)
    df_ent_meta.insert(0, 'model', 'Metamodel')
    entropy_records.append(df_ent_meta)
    _ent_lookup['Metamodel'] = round(float(meta_ent.mean()), 4)

    _plot_transient_kde(meta_pred_proba, np.array(meta_model.classes_),
                        yo_te, "Metamodel adapted",
                        f"{PLOTS_DIR_ADP}transient_kde_metamodel.pdf")
    _plot_metamodel_reliability(meta_pred_proba, yo_te, meta_model,
                                f"{PLOTS_DIR_ADP}reliability_metamodel.pdf")

    mtf.plot_confusion_matrix(
        metrics.confusion_matrix(yo_te, meta_pred, labels=mtf.LABEL_ORDER),
        mtf.LABEL_ORDER,
        f"{PLOTS_DIR_ADP}conf_matrix_metamodel.pdf",
        title="Metamodel",
        cmap=MODEL_CMAPS.get('Metamodel'),
    )
    cm_meta = pd.DataFrame(
        metrics.confusion_matrix(yo_te, meta_pred, labels=mtf.LABEL_ORDER),
        index=mtf.LABEL_ORDER, columns=mtf.LABEL_ORDER)
    cm_meta.index.name = 'true_class'
    cm_meta.insert(0, 'model', 'Metamodel')
    conf_rows.append(cm_meta.reset_index())

    print("  Bootstrap Metamodel ...")
    g_std_meta, pc_std_meta = _bootstrap_metrics(
        yo_te, meta_pred_proba, np.array(meta_model.classes_))
    pc_std_meta.insert(0, 'model', 'Metamodel')
    bootstrap_rows.append(pc_std_meta)

    meta_pred_labels = meta_model.predict(meta_X_test)
    meta_report = metrics.classification_report(
        yo_te, meta_pred_labels, labels=mtf.LABEL_ORDER,
        output_dict=True, zero_division=0)
    for cls in mtf.LABEL_ORDER:
        if cls in meta_report:
            per_class_rows.append({
                'model': 'Metamodel', 'class': cls,
                'precision': round(meta_report[cls]['precision'], 4),
                'recall':    round(meta_report[cls]['recall'],    4),
                'f1':        round(meta_report[cls]['f1-score'],  4),
                'support':   int(meta_report[cls]['support']),
            })
    summary_full_rows.append({
        'Model':             'Metamodel',
        'Accuracy':          round(metrics.accuracy_score(yo_te, meta_pred_labels),          4),
        'Balanced_Accuracy': round(metrics.balanced_accuracy_score(yo_te, meta_pred_labels), 4),
        'Macro_F1':          round(metrics.f1_score(yo_te, meta_pred_labels,
                                   average='macro', zero_division=0),                        4),
        'Mean_Entropy':      _ent_lookup.get('Metamodel', np.nan),
        'N_test':            len(yo_te),
        **g_std_meta,
    })

    # Excel
    summary_full = pd.DataFrame(summary_full_rows)
    results_path = f"{MODELS_DIR_ADP}results_summary.xlsx"
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        summary_full.to_excel(
            writer, sheet_name='Summary',            index=False)
        pd.DataFrame(per_class_rows).to_excel(
            writer, sheet_name='Per_class_metrics',  index=False)
        pd.concat(conf_rows, ignore_index=True).to_excel(
            writer, sheet_name='Confusion_matrices', index=False)
        pd.concat(entropy_records, ignore_index=True).to_excel(
            writer, sheet_name='Entropy_by_class',   index=False)
        pd.concat(bootstrap_rows, ignore_index=True).to_excel(
            writer, sheet_name='Bootstrap_std',      index=False)
    print(f"\nResults saved: {results_path}")

    # ── Fase 5c: Matrices de confusión con percentiles ─────────────────────────
    from bootstrap_confmat import generate_bootstrap_confmats
    generate_bootstrap_confmats(
        test_probas      = test_probas,
        meta_pred_proba  = meta_pred_proba,
        meta_classes     = np.array(meta_model.classes_),
        yo_te            = yo_te,
        label_order      = mtf.LABEL_ORDER,
        plots_dir        = PLOTS_DIR_ADP,
        models_dir       = MODELS_DIR_ADP,
        n_bootstrap      = 1000,
        random_state     = mtf.RANDOM_STATE,
        hier_map         = mtf.HIER_MAP,
    )

    # ── Fase 6: SHAP analysis ──────────────────────────────────────────────────
    print("\n=== PHASE 6: SHAP analysis ===")

    def _shap_to_mean_abs(sv):
        if isinstance(sv, list):
            return np.mean([np.abs(a) for a in sv], axis=0)
        sv = np.array(sv)
        return np.mean(np.abs(sv), axis=2) if sv.ndim == 3 else np.abs(sv)

    def _shap_barplot(sv_raw, X_data, feature_names, title, save_path, top_n=40,
                      color=None):
        import shap
        sv_mean = _shap_to_mean_abs(sv_raw)
        exp = shap.Explanation(
            values=sv_mean,
            base_values=np.zeros(sv_mean.shape[0]),
            data=X_data.values if hasattr(X_data, 'values') else X_data,
            feature_names=feature_names,
        )
        plt.figure(figsize=(8, min(top_n * 0.35 + 1.5, 18)))
        shap.plots.bar(exp, max_display=top_n, show=False)
        if color is not None:
            for patch in plt.gca().patches:
                patch.set_facecolor(color)
        plt.title(title, fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def _compute_shap_for_model(model_name, clf_hier, clfs, X_te, yo_te_model):
        import shap
        top_n = 40
        print(f"  [SHAP] Model {model_name} — top-level ...")
        explainer_hier = shap.TreeExplainer(clf_hier)
        sv_hier        = explainer_hier.shap_values(X_te)
        _shap_barplot(sv_hier, X_te, list(X_te.columns),
                      f'SHAP — Model {model_name} (top-level)',
                      f'{PLOTS_DIR_ADP}shap_summary_{model_name}_hier.pdf', top_n,
                      color=MODEL_COLORS.get(model_name))

        shap_storage = {'hier': sv_hier}
        for group, clf_sub in clfs.items():
            cls_to_group = {cls: grp for grp, members in mtf.HIER_MAP.items()
                            for cls in members}
            mask  = (yo_te_model.map(cls_to_group) == group).values
            X_sub = X_te.loc[mask]
            if len(X_sub) == 0:
                continue
            print(f"  [SHAP] Model {model_name}. {group} ({mask.sum()} objects) ...")
            explainer_sub = shap.TreeExplainer(clf_sub)
            sv_sub        = explainer_sub.shap_values(X_sub)
            shap_storage[group] = sv_sub
            _shap_barplot(sv_sub, X_sub, list(X_sub.columns),
                          f'SHAP features — Model {model_name} ({group})',
                          f'{PLOTS_DIR_ADP}shap_summary_{model_name}_{group}.pdf', top_n,
                          color=MODEL_COLORS.get(model_name))

        shap_pkl = f'{MODELS_DIR_ADP}shap_values_{model_name}.pkl'
        with open(shap_pkl, 'wb') as f:
            pickle.dump({'shap_values': shap_storage,
                         'feature_names': list(X_te.columns),
                         'y_true': yo_te_model}, f, pickle.HIGHEST_PROTOCOL)
        print(f"  [SHAP] Saved: {shap_pkl}")

    shap_specs = [
        ('A', models['A'][0], models['A'][1], XA_te, yo_te),
        ('B', models['B'][0], models['B'][1], XB_te, yo_te),
        ('C', models['C'][0], models['C'][1], XC_te, yo_te),
        ('D', models['D'][0], models['D'][1], XD_te, yo_te),
    ]
    if 'E' in models and XE_te is not None:
        shap_specs.append(('E', models['E'][0], models['E'][1], XE_te, yoE_te))

    for model_name, clf_hier, clfs, X_te_shap, yo_te_shap in shap_specs:
        _compute_shap_for_model(model_name, clf_hier, clfs, X_te_shap, yo_te_shap)

    # ── Fase 7: SHAP violin plots (transient sub-classifier) ──────────────────
    print("\n=== PHASE 7: SHAP violin plots (Transient sub-classifier) ===")

    def _plot_shap_violin_transient(model_name, X_te, yo_te_model, top_n=20):
        import shap
        shap_pkl_path = f'{MODELS_DIR_ADP}shap_values_{model_name}.pkl'
        if not os.path.exists(shap_pkl_path):
            print(f"  Skipping {model_name}: .pkl not found"); return
        with open(shap_pkl_path, 'rb') as f:
            shap_data = pickle.load(f)
        sv_transient = shap_data['shap_values'].get('Transient')
        if sv_transient is None:
            print(f"  Skipping {model_name}: no Transient SHAP values"); return

        cls_to_group  = {cls: grp for grp, members in mtf.HIER_MAP.items()
                         for cls in members}
        transient_mask = (yo_te_model.map(cls_to_group) == 'Transient').values
        X_sub = X_te.loc[transient_mask]
        feat_names = list(X_sub.columns)

        sv = np.array(sv_transient)
        if sv.ndim == 2: sv = sv[:, :, np.newaxis]

        clf_sub = models[model_name][1].get('Transient')
        trans_cls_order = list(clf_sub.classes_) if clf_sub is not None else TRANSIENT_CLASSES

        for cls_idx, cls_name in enumerate(trans_cls_order[:sv.shape[2]]):
            sv_cls   = sv[:, :, cls_idx]
            mean_abs = np.abs(sv_cls).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[::-1][:top_n]
            exp = shap.Explanation(
                values=sv_cls[:, top_idx],
                base_values=np.zeros(sv_cls.shape[0]),
                data=X_sub.values[:, top_idx],
                feature_names=[feat_names[i] for i in top_idx],
            )
            plt.figure(figsize=(9, top_n * 0.38 + 1.5))
            shap.plots.violin(exp, max_display=top_n,
                              plot_type='layered_violin', show=False)
            plt.title(f'SHAP values — Transient ({cls_name}) Model {model_name}',
                      fontsize=11)
            plt.tight_layout()
            save_path = f'{PLOTS_DIR_ADP}shap_violin_{model_name}_{cls_name}.pdf'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {save_path}")

    for model_name, _, _, X_te_shap, yo_te_shap in shap_specs:
        _plot_shap_violin_transient(model_name, X_te_shap, yo_te_shap)

    print(f"\n✓ Trainin completed..")
    print(f"  Models: {MODELS_DIR_ADP}")
    print(f"  Plots:   {PLOTS_DIR_ADP}")
    print(f"  CORAL:   {CORAL_DIR}")


def parse_args():
    """
    CLI para model_training_adapted.py. Todos los argumentos son opcionales:
    sin pasar ninguno, el script usa los DEFAULTS definidos en la sección
    CONFIGURATION (pensado para lanzarse con runfile() desde Spyder).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Script for model training. Makes use of CORAL domain adaptation"
            "and metamodel augmentation with real data."
        )
    )

    g_in = parser.add_argument_group("Inputs")
    g_in.add_argument("--consensus-csv", default=CONSENSUS_CSV,
        help="SHAP/KS consensus CSV generated by feature_selection.py")
    g_in.add_argument("--features-ztf", default=FEATURES_ZTF_FILE,
        help="Simulated ZTF features parquet")
    g_in.add_argument("--features-lsst", default=FEATURES_LSST_FILE,
        help="Simulated LSST features parquet")
    g_in.add_argument("--features-atlas", default=FEATURES_ATLAS_FILE,
        help="Simulated ATLAS features parquet")
    g_in.add_argument("--features-combined", default=FEATURES_COMBINED_FILE,
        help="Simulated combined features parquet (Model D)")
    g_in.add_argument("--labels-file", default=LABELS_FILE,
        help="Labels CSV/parquet for simulated set (with class_original column)")
    g_in.add_argument("--real-ztf-strict", default=REAL_ZTF_STRICT,
        help="Real ZTF features parquet for CORAL target domain")
    g_in.add_argument("--real-lsst-strict", default=REAL_LSST_STRICT,
        help="Real LSST features parquet for CORAL target domain")
    g_in.add_argument("--real-comb-strict", default=REAL_COMB_STRICT,
        help="Real combined features parquet for CORAL target domain")
    g_in.add_argument("--real-atlas-strict", default=REAL_ATLAS_STRICT,
        help="Real ATLAS features parquet for CORAL target domain")
    g_in.add_argument("--real-labels-file", default=REAL_LABELS_FILE,
        help="Labels CSV/parquet for real set")

    g_out = parser.add_argument_group("Output")
    g_out.add_argument("--output-dir", default=OUTPUT_DIR,
        help="Output directory (contains models/, plots/, oof/, coral/)")

    g_coral = parser.add_argument_group("CORAL")
    g_coral.add_argument("--coral-lambda", type=float, default=CORAL_LAMBDA,
        help="Covariance regularization Regularización (λ·I) for CORAL")
    g_coral.add_argument("--coral-ks-threshold", type=float, default=CORAL_KS_THRESHOLD,
        help="Minimum KS threshold for CORAL domain adaptation")

    g_feat = parser.add_argument_group("Selección de features")
    g_feat.add_argument("--include-unknown", action="store_true", default=INCLUDE_UNKNOWN,
        help="Keep UNKNOWN features with high SHAP")
    g_feat.add_argument("--unknown-shap-percentile", type=int, default=UNKNOWN_SHAP_PERCENTILE,
        help="SHAP percentile threshold for keeping UNKNOWN features")

    g_meta = parser.add_argument_group("Metamodelo")
    g_meta.add_argument("--no-real-meta-aug", dest="use_real_meta_aug",
        action="store_false", default=USE_REAL_META_AUG,
        help="Disable metamodel augmentation")
    g_meta.add_argument("--real-meta-upweight", type=float, default=REAL_META_UPWEIGHT,
        help="Weight of the real objects for metamodel augmentation")
    g_meta.add_argument("--meta-dropout-prob", type=float, default=META_DROPOUT_PROB,
        help="Dropout probability for each survey")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    CONSENSUS_CSV           = args.consensus_csv
    FEATURES_ZTF_FILE       = args.features_ztf
    FEATURES_LSST_FILE      = args.features_lsst
    FEATURES_ATLAS_FILE     = args.features_atlas
    FEATURES_COMBINED_FILE  = args.features_combined
    LABELS_FILE             = args.labels_file
    REAL_ZTF_STRICT         = args.real_ztf_strict
    REAL_LSST_STRICT        = args.real_lsst_strict
    REAL_COMB_STRICT        = args.real_comb_strict
    REAL_ATLAS_STRICT       = args.real_atlas_strict
    REAL_LABELS_FILE        = args.real_labels_file

    OUTPUT_DIR     = args.output_dir
    MODELS_DIR_ADP = os.path.join(OUTPUT_DIR, "models") + "/"
    PLOTS_DIR_ADP  = os.path.join(OUTPUT_DIR, "plots")  + "/"
    OOF_DIR_ADP    = os.path.join(OUTPUT_DIR, "oof")    + "/"
    CORAL_DIR      = os.path.join(OUTPUT_DIR, "coral")  + "/"

    CORAL_LAMBDA            = args.coral_lambda
    CORAL_KS_THRESHOLD      = args.coral_ks_threshold
    INCLUDE_UNKNOWN         = args.include_unknown
    UNKNOWN_SHAP_PERCENTILE = args.unknown_shap_percentile
    USE_REAL_META_AUG       = args.use_real_meta_aug
    REAL_META_UPWEIGHT      = args.real_meta_upweight
    META_DROPOUT_PROB       = args.meta_dropout_prob

    main()