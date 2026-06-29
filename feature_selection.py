from __future__ import annotations

import argparse
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Defaults de rutas (Spyder runfile() / CLI sin argumentos)
# ---------------------------------------------------------------------------

DEFAULT_SHAP_DIR = "./output/models/"
DEFAULT_SIM_DIR  = "./data/simulated/"
DEFAULT_SIM_DIR  = "./data/simulated/"
DEFAULT_REAL_DIR = "./data/real/"
DEFAULT_OUT_DIR  = "./data/feature_selection/"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CLASSES = [
    "SNIa", "SNII", "SNIbc", "SLSN",
    "QSO", "AGN", "Blazar", "YSO", "CV/Nova",
    "RRL", "CEP", "DSCT", "LPV", "E", "Periodic-Other",
]


_MODEL_CONFIG = {
    "B": {
        "suffix":   "_lsst",
        "sim_file": "features_lsst.parquet",
        "real_file": "features_lsst_strict.parquet",
        "shap_file": "shap_values_B.pkl",
    },
    "C": {
        "suffix":        "_ztf",        
        "suffix2":       "_lsst",      
        "suffix3":       "_atlas",   
        "sim_file":      "features_lsst.parquet",   
        "sim_file_ztf":  "features_ztf.parquet",
        "sim_file_atlas": "features_atlas.parquet",
        "real_file":      "features_lsst_strict.parquet",
        "real_file_ztf":  "features_ztf_strict.parquet",
        "real_file_atlas": "features_atlas_strict.parquet",
        "shap_file": "shap_values_C.pkl",
    },
    "D": {
        "suffix":   "_combined",
        "sim_file": "features_lsst.parquet",   #
        "real_file": "features_comb_strict.parquet",
        "shap_file": "shap_values_D.pkl",
    },
}


_META_FEATURES = {"survey", "classALeRCE", "gal_b_lsst", "gal_l_lsst",
                  "gal_b_ztf", "gal_l_ztf", "gal_b_combined", "gal_l_combined",
                  "rb_lsst", "rb_ztf", "rb_combined",
                  "gal_b_atlas", "gal_l_atlas", "rb_atlas"}


_UNSTABLE_LSST = {f"GP_DRW_tau_{b}_lsst"     for b in range(1, 7)}
_UNSTABLE_COMB = {f"GP_DRW_tau_{b}_combined" for b in range(1, 7)}
_UNSTABLE      = _UNSTABLE_LSST | _UNSTABLE_COMB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_from_oid(oid: str) -> str:
    parts = str(oid).split("_")
    raw = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
    return "CV/Nova" if raw == "CV_Nova" else raw


def _load_sim(path: str, survey_id: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["classALeRCE"] = df.index.map(_label_from_oid)
    # Para combined: renombrar _lsst → _combined
    if survey_id == "combined":
        df.columns = [
            c.replace("_lsst", "_combined") if "_lsst" in c else c
            for c in df.columns
        ]
    return df


def _load_real(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _compute_gap(
    df_sim: pd.DataFrame,
    df_real: pd.DataFrame,
    feature_cols: List[str],
    nan_threshold: float = 0.5,
) -> pd.DataFrame:
    """KS statistic y Wasserstein normalizado para cada feature."""
    from scipy.stats import wasserstein_distance

    rows = []
    exclude = _META_FEATURES | _UNSTABLE
    for col in feature_cols:
        if col in exclude:
            continue
        if col not in df_sim.columns or col not in df_real.columns:
            continue
        sv = pd.to_numeric(df_sim[col],  errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rv = pd.to_numeric(df_real[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(sv) < 10 or len(rv) < 5:
            continue
        if df_sim[col].isnull().mean() >= nan_threshold:
            continue
        if df_real[col].isnull().mean() >= nan_threshold:
            continue
        ks, _ = ks_2samp(sv.values, rv.values)
        p1, p99 = np.nanpercentile(np.concatenate([sv.values, rv.values]), [1, 99])
        w = wasserstein_distance(sv.clip(p1, p99).values, rv.clip(p1, p99).values)
        std = sv.std()
        rows.append({
            "feature":      col,
            "ks_statistic": round(ks, 4),
            "wasserstein":  round(w / std if std > 1e-10 else 0.0, 4),
            "nan_sim":      round(df_sim[col].isnull().mean(),  3),
            "nan_real":     round(df_real[col].isnull().mean(), 3),
        })
    return pd.DataFrame(rows).set_index("feature") if rows else pd.DataFrame()


def _compute_shap_importance(
    shap_pkl: dict,
    shap_level: int = -1,
) -> pd.Series:
    """
    Importancia SHAP: |values|.mean sobre objetos y nivel jerárquico.

    shap_level: -1 = media de los 3 niveles, 0/1/2 = nivel específico.
    """
    arr          = shap_pkl["shap_values"]["hier"]   # (n_obj, n_feat, 3)
    feature_names = shap_pkl["feature_names"]

    abs_shap = np.abs(arr)                            # (n_obj, n_feat, 3)
    if shap_level == -1:
        importance = abs_shap.mean(axis=0).mean(axis=1)   # (n_feat,)
    else:
        importance = abs_shap[:, :, shap_level].mean(axis=0)

    return pd.Series(importance, index=feature_names, name="shap_mean")



def _compute_differential_gap(
    df_ztf_sim:   pd.DataFrame,
    df_lsst_sim:  Optional[pd.DataFrame],
    df_ztf_real:  pd.DataFrame,
    df_lsst_real: Optional[pd.DataFrame],
    diff_feature_cols: List[str],
    df_atlas_sim:  Optional[pd.DataFrame] = None,
    df_atlas_real: Optional[pd.DataFrame] = None,
    nan_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Calcula el domain gap para las features diferenciales generadas por
    build_X (ratios/diffs entre bandas y surveys) que no están en
    los parquets de features base.
    """
    import model_training_functions as _mtf

    def _classify(col):
        if col.endswith('_ztf') and ('gr_ratio' in col or 'gr_diff' in col):
            return 'ztf_intra'
        if col.endswith('_atlas') and ('co_ratio' in col or 'co_diff' in col):
            return 'atlas_intra'
        if 'ztf_lsst' in col:
            return 'ztf_lsst'
        if 'ztf_atlas' in col:
            return 'ztf_atlas'
        if 'atlas_lsst' in col:
            return 'atlas_lsst'
        if 'color_' in col and col.endswith('_lsst'):
            return 'lsst_color'
        return 'other'


    from collections import defaultdict
    by_family = defaultdict(list)
    for col in diff_feature_cols:
        by_family[_classify(col)].append(col)

    gap_parts = []


    def _gap_for_joined(df_sim_joined, df_real_joined,
                        col_list_sim, col_list_real,
                        target_cols,
                        add_ztf_diff=False, add_atlas_diff=False,
                        add_cross_diff=False):
        X_sim  = _mtf.build_X(df_sim_joined,  col_list_sim,
                               add_ztf_diff=add_ztf_diff,
                               add_atlas_diff=add_atlas_diff,
                               add_cross_diff=add_cross_diff)
        X_real = _mtf.build_X(df_real_joined, col_list_real,
                               add_ztf_diff=add_ztf_diff,
                               add_atlas_diff=add_atlas_diff,
                               add_cross_diff=add_cross_diff)

        available = [c for c in target_cols
                     if c in X_sim.columns and c in X_real.columns]
        if not available:
            return pd.DataFrame()
        return _compute_gap(X_sim, X_real, available, nan_threshold)


    if by_family['ztf_intra']:
        ztf_sim_cols  = _mtf.get_ztf_cols(list(df_ztf_sim.columns))
        ztf_real_cols = _mtf.get_ztf_cols(list(df_ztf_real.columns))
        g = _gap_for_joined(df_ztf_sim, df_ztf_real,
                            ztf_sim_cols, ztf_real_cols,
                            by_family['ztf_intra'],
                            add_ztf_diff=True)
        if not g.empty:
            gap_parts.append(g)
            print(f"Gap ztf_intra:  {len(g)} features")


    if by_family['atlas_intra'] and df_atlas_sim is not None and df_atlas_real is not None:
        atl_sim_cols  = _mtf.get_atlas_cols(list(df_atlas_sim.columns))
        atl_real_cols = _mtf.get_atlas_cols(list(df_atlas_real.columns))
        g = _gap_for_joined(df_atlas_sim, df_atlas_real,
                            atl_sim_cols, atl_real_cols,
                            by_family['atlas_intra'],
                            add_atlas_diff=True)
        if not g.empty:
            gap_parts.append(g)
            print(f"Gap atlas_intra: {len(g)} features")


    if by_family['lsst_color'] and df_lsst_sim is not None and df_lsst_real is not None:
        lsst_sim_cols  = _mtf.get_lsst_cols(list(df_lsst_sim.columns))
        lsst_real_cols = _mtf.get_lsst_cols(list(df_lsst_real.columns))
        g = _gap_for_joined(df_lsst_sim, df_lsst_real,
                            lsst_sim_cols, lsst_real_cols,
                            by_family['lsst_color'],
                            add_cross_diff=True)
        if not g.empty:
            gap_parts.append(g)
            print(f"Gap lsst_color: {len(g)} features")


    if by_family['ztf_lsst'] and df_lsst_sim is not None and df_lsst_real is not None:
        ztf_sim_cols   = _mtf.get_ztf_cols(list(df_ztf_sim.columns))
        lsst_sim_cols  = _mtf.get_lsst_cols(list(df_lsst_sim.columns))
        ztf_real_cols  = _mtf.get_ztf_cols(list(df_ztf_real.columns))
        lsst_real_cols = _mtf.get_lsst_cols(list(df_lsst_real.columns))

        df_sim_jn  = df_ztf_sim[ztf_sim_cols].join(
                         df_lsst_sim[lsst_sim_cols],  how='inner')
        df_real_jn = df_ztf_real[ztf_real_cols].join(
                         df_lsst_real[lsst_real_cols], how='inner')

        g = _gap_for_joined(df_sim_jn, df_real_jn,
                            ztf_sim_cols + lsst_sim_cols,
                            ztf_real_cols + lsst_real_cols,
                            by_family['ztf_lsst'],
                            add_ztf_diff=True, add_cross_diff=True)
        if not g.empty:
            gap_parts.append(g)
            print(f"Gap ztf_lsst:  {len(g)} features")


    if df_atlas_sim is not None and df_atlas_real is not None:
        ztf_sim_cols   = _mtf.get_ztf_cols(list(df_ztf_sim.columns))
        lsst_sim_cols  = _mtf.get_lsst_cols(list(df_lsst_sim.columns))
        atl_sim_cols   = _mtf.get_atlas_cols(list(df_atlas_sim.columns))
        ztf_real_cols  = _mtf.get_ztf_cols(list(df_ztf_real.columns))
        lsst_real_cols = _mtf.get_lsst_cols(list(df_lsst_real.columns))
        atl_real_cols  = _mtf.get_atlas_cols(list(df_atlas_real.columns))

        if by_family['ztf_atlas']:
            df_sim_jn  = df_ztf_sim[ztf_sim_cols].join(
                             df_atlas_sim[atl_sim_cols],  how='inner')
            df_real_jn = df_ztf_real[ztf_real_cols].join(
                             df_atlas_real[atl_real_cols], how='inner')
            g = _gap_for_joined(df_sim_jn, df_real_jn,
                                ztf_sim_cols + atl_sim_cols,
                                ztf_real_cols + atl_real_cols,
                                by_family['ztf_atlas'],
                                add_ztf_diff=True, add_atlas_diff=True,
                                add_cross_diff=True)
            if not g.empty:
                gap_parts.append(g)
                print(f"Gap ztf_atlas: {len(g)} features")

        if by_family['atlas_lsst'] and df_lsst_sim is not None and df_lsst_real is not None:
            lsst_sim_cols  = _mtf.get_lsst_cols(list(df_lsst_sim.columns))
            lsst_real_cols = _mtf.get_lsst_cols(list(df_lsst_real.columns))
            df_sim_jn  = df_atlas_sim[atl_sim_cols].join(
                             df_lsst_sim[lsst_sim_cols],  how='inner')
            df_real_jn = df_atlas_real[atl_real_cols].join(
                             df_lsst_real[lsst_real_cols], how='inner')
            g = _gap_for_joined(df_sim_jn, df_real_jn,
                                atl_sim_cols + lsst_sim_cols,
                                atl_real_cols + lsst_real_cols,
                                by_family['atlas_lsst'],
                                add_atlas_diff=True, add_cross_diff=True)
            if not g.empty:
                gap_parts.append(g)
                print(f"  Gap atlas_lsst:{len(g)} features")

    return pd.concat(gap_parts) if gap_parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Función principal de análisis por modelo
# ---------------------------------------------------------------------------

def analyze_model(
    model_id:      str,
    shap_dir:      str,
    sim_dir:       str,
    real_dir:      str,
    alpha:         float = 0.5,
    shap_level:    int   = -1,
    nan_threshold: float = 0.5,
    gap_csv:        Optional[str] = None,
    sim_atlas_dir:  Optional[str] = None,
    real_atlas_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cruza SHAP x domain gap para un modelo.
    """
    cfg = _MODEL_CONFIG[model_id]
    print(f"\n{'='*60}")
    print(f"Modelo {model_id}")
    print(f"{'='*60}")

    shap_path = os.path.join(shap_dir, cfg["shap_file"])
    with open(shap_path, "rb") as f:
        shap_pkl = pickle.load(f)
    shap_imp = _compute_shap_importance(shap_pkl, shap_level)
    print(f"SHAP: {len(shap_imp)} features loaded")

    if gap_csv and os.path.exists(gap_csv):
        gap_df = pd.read_csv(gap_csv, index_col="feature")
        print(f"Gap: loaded from {gap_csv} ({len(gap_df)} features)")
    else:
        suffix = cfg["suffix"]
        feature_cols = [f for f in shap_imp.index]

        if model_id == "C":
            ztf_cols   = [f for f in feature_cols if "_ztf"   in f]
            lsst_cols  = [f for f in feature_cols if "_lsst"  in f]
            atlas_cols = [f for f in feature_cols if "_atlas" in f]

            df_ztf_sim   = _load_sim(os.path.join(sim_dir,  cfg["sim_file_ztf"]),   "ztf")
            df_ztf_real  = _load_real(os.path.join(real_dir, cfg["real_file_ztf"]))
            df_lsst_sim  = _load_sim(os.path.join(sim_dir,  cfg["sim_file"]),        "lsst")
            df_lsst_real = _load_real(os.path.join(real_dir, cfg["real_file"]))

            gap_ztf  = _compute_gap(df_ztf_sim,  df_ztf_real,  ztf_cols,  nan_threshold)
            gap_lsst = _compute_gap(df_lsst_sim, df_lsst_real, lsst_cols, nan_threshold)
            gap_parts = [gap_ztf, gap_lsst]


            df_atlas_sim  = None
            df_atlas_real = None
            _atlas_sim_dir  = sim_atlas_dir  if sim_atlas_dir  is not None else sim_dir
            _atlas_real_dir = real_atlas_dir if real_atlas_dir is not None else real_dir
            atlas_sim_path  = os.path.join(_atlas_sim_dir,  cfg.get("sim_file_atlas",  ""))
            atlas_real_path = os.path.join(_atlas_real_dir, cfg.get("real_file_atlas", ""))
            if atlas_cols and os.path.exists(atlas_sim_path) and os.path.exists(atlas_real_path):
                df_atlas_sim  = _load_sim(atlas_sim_path,  "atlas")
                df_atlas_real = _load_real(atlas_real_path)
                gap_atlas = _compute_gap(df_atlas_sim, df_atlas_real,
                                         atlas_cols, nan_threshold)
                if not gap_atlas.empty:
                    gap_parts.append(gap_atlas)
                    print(f"Gap ATLAS: compued for {len(gap_atlas)} features")
            elif atlas_cols:
                print(f"Gap ATLAS skipped: sim or real not found")

            in_parquets = (set(df_ztf_sim.columns) | set(df_lsst_sim.columns)
                           | (set(df_atlas_sim.columns) if df_atlas_sim is not None else set()))
            diff_cols = [f for f in feature_cols if f not in in_parquets]
            if diff_cols:
                print(f"Computing gap for {len(diff_cols)} differential features...")
                gap_diff = _compute_differential_gap(
                    df_ztf_sim   = df_ztf_sim,
                    df_lsst_sim  = df_lsst_sim,
                    df_ztf_real  = df_ztf_real,
                    df_lsst_real = df_lsst_real,
                    diff_feature_cols = diff_cols,
                    df_atlas_sim  = df_atlas_sim,
                    df_atlas_real = df_atlas_real,
                    nan_threshold = nan_threshold,
                )
                if not gap_diff.empty:
                    gap_parts.append(gap_diff)
                    print(f"Differential gaps: {len(gap_diff)} features computed")

            gap_df = pd.concat(gap_parts) if gap_parts else pd.DataFrame()

        elif model_id == "D":
            df_sim  = _load_sim(os.path.join(sim_dir,  cfg["sim_file"]), "combined")
            df_real = _load_real(os.path.join(real_dir, cfg["real_file"]))
            gap_df  = _compute_gap(df_sim, df_real, feature_cols, nan_threshold)

        elif model_id == "B":
            df_sim  = _load_sim(os.path.join(sim_dir,  cfg["sim_file"]), "lsst")
            df_real = _load_real(os.path.join(real_dir, cfg["real_file"]))
            gap_df  = _compute_gap(df_sim, df_real, feature_cols, nan_threshold)

        print(f"Gap: compued for {len(gap_df)} features")

    df = shap_imp.to_frame()
    df = df.join(gap_df[["ks_statistic", "wasserstein",
                          "nan_sim", "nan_real"]], how="left")

    exclude = _META_FEATURES | _UNSTABLE
    df = df[~df.index.isin(exclude)]

    # Normalizar a [0, 1]
    shap_min, shap_max = df["shap_mean"].min(), df["shap_mean"].max()
    ks_min,   ks_max   = df["ks_statistic"].min(), df["ks_statistic"].max()

    df["shap_norm"] = (df["shap_mean"] - shap_min) / (shap_max - shap_min + 1e-12)
    df["ks_norm"]   = (df["ks_statistic"] - ks_min) / (ks_max - ks_min + 1e-12)

    df["score"] = df["shap_norm"] - alpha * df["ks_norm"]

    #Cuadrantes
    shap_med = df["shap_norm"].median()
    ks_med   = df["ks_norm"].median()

    conditions = [
        (df["shap_norm"] >= shap_med) & (df["ks_norm"] <  ks_med),
        (df["shap_norm"] >= shap_med) & (df["ks_norm"] >= ks_med),
        (df["shap_norm"] <  shap_med) & (df["ks_norm"] <  ks_med),
        (df["shap_norm"] <  shap_med) & (df["ks_norm"] >= ks_med),
    ]
    quadrant_labels = [
        "KEEP",          # alto SHAP, bajo KS  -> mantener
        "CORAL",         # alto SHAP, alto KS  -> adaptar con CORAL
        "OPTIONAL",      # bajo SHAP, bajo KS  -> mantener o eliminar (poco impacto)
        "DROP",          # bajo SHAP, alto KS  -> eliminar
    ]
    recommendations = [
        "Keep: high importance, low gap",
        "CORAL: high importance but high gap, adapt",
        "Opcional: low importance and gap",
        "Drop: low importance and high gap",
    ]
    df["quadrant"]       = np.select(conditions, quadrant_labels,       default="UNKNOWN")
    df["recommendation"] = np.select(conditions, recommendations, default="")

    df = df.sort_values("score", ascending=False)

    # ── Resumen terminal ─────────────────────────────────────────────────────
    counts = df["quadrant"].value_counts()
    print(f"\nQuadrant distribution:")
    for q in quadrant_labels:
        n = counts.get(q, 0)
        bar = "█" * int(n / max(counts.values) * 20)
        print(f"{q:<10}: {n:4d}  {bar}")

    print(f"\nTop 10 by score (KEEP + CORAL):")
    top = df[df["quadrant"].isin(["KEEP", "CORAL"])].head(10)
    print(top[["shap_mean", "ks_statistic", "score",
               "quadrant"]].to_string(float_format="{:.4f}".format))

    n_gap_missing = df["ks_statistic"].isna().sum()
    if n_gap_missing > 0:
        print(f"\n ⚠ {n_gap_missing} features without compued gap: treated as OPTIONAL/DROP")

    return df


# ---------------------------------------------------------------------------
# Consenso entre modelos
# ---------------------------------------------------------------------------

def build_consensus(
    results: Dict[str, pd.DataFrame],
    min_agreement: int = 2,
    alpha:          float = 0.5,
) -> pd.DataFrame:
    """
    Construye una lista de consenso de features seleccionadas.

    Una feature se incluye si aparece en al menos min_agreement modelos
    con quadrant en {KEEP, CORAL}.

    Para features compartidas entre modelos (misma base, distinto sufijo),
    se agrega por nombre base (eliminando sufijo _ztf/_lsst/_combined).
    """
    import re

    def base_name(feat: str) -> str:
        """'Amplitude_1_lsst' → 'Amplitude_1'"""
        return re.sub(r"_(ztf|lsst|combined|atlas)$", "", feat)

    rows = []
    for model_id, df in results.items():
        selected = df[df["quadrant"].isin(["KEEP", "CORAL"])].copy()
        selected["model"]        = model_id
        selected["feature_base"] = selected.index.map(base_name)
        rows.append(selected.reset_index())

    if not rows:
        return pd.DataFrame()

    all_sel = pd.concat(rows, ignore_index=True)

    # Agrupar por feature_base
    agg = (
        all_sel.groupby("feature_base")
        .agg(
            n_models      = ("model",          "nunique"),
            models        = ("model",          lambda x: "/".join(sorted(set(x)))),
            quadrants     = ("quadrant",        lambda x: "/".join(sorted(set(x)))),
            mean_score    = ("score",           "mean"),
            mean_shap     = ("shap_mean",       "mean"),
            mean_ks       = ("ks_statistic",    "mean"),
        )
        .reset_index()
    )

    # Filtrar por acuerdo mínimo
    consensus = agg[agg["n_models"] >= min_agreement].copy()
    consensus = consensus.sort_values("mean_score", ascending=False)

    # Recomendación de consenso
    def _rec(row):
        if "CORAL" in row["quadrants"] and row["mean_ks"] > 0.4:
            return "CORAL"
        elif row["n_models"] == len(results):
            return "KEEP (all models)"
        else:
            return f"KEEP ({row['n_models']}/{len(results)} models)"

    consensus["recommendation"] = consensus.apply(_rec, axis=1)

    print(f"\n{'='*60}")
    print(f"CONSENSUS (≥{min_agreement} models agree)")
    print(f"{'='*60}")
    print(f"Selected features: {len(consensus)}")
    coral = consensus[consensus["recommendation"].str.startswith("CORAL")]
    keep  = consensus[~consensus["recommendation"].str.startswith("CORAL")]
    print(f"  → KEEP:  {len(keep)}")
    print(f"  → CORAL: {len(coral)}")

    print(f"\nTop 20 by mean score:")
    print(consensus.head(20)[
        ["feature_base", "n_models", "mean_shap", "mean_ks",
         "mean_score", "recommendation"]
    ].to_string(index=False, float_format="{:.4f}".format))

    return consensus


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_shap_vs_gap(
    df:       pd.DataFrame,
    model_id: str,
    top_n:    int = 20,
    save_dir: Optional[str] = None,
):

    colors = {
        "KEEP":     "#1D9E75",
        "CORAL":    "#E24B4A",
        "OPTIONAL": "#aaaaaa",
        "DROP":     "#EF9F27",
        "UNKNOWN":  "#cccccc",
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    shap_med = df["shap_norm"].median()
    ks_med   = df["ks_norm"].median()
    ax.axvline(shap_med, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(ks_med,   color="gray", lw=0.8, ls="--", alpha=0.5)

    # Fondos de cuadrante
    ax.axhspan(0,       ks_med,   xmin=0,       xmax=0.5,  alpha=0.04, color="#1D9E75")
    ax.axhspan(ks_med,  1,        xmin=0,       xmax=0.5,  alpha=0.04, color="#EF9F27")
    ax.axhspan(0,       ks_med,   xmin=0.5,     xmax=1,    alpha=0.04, color="#aaaaaa")
    ax.axhspan(ks_med,  1,        xmin=0.5,     xmax=1,    alpha=0.04, color="#E24B4A")

    for q, sub in df.groupby("quadrant"):
        ax.scatter(sub["shap_norm"], sub["ks_norm"],
                   c=colors.get(q, "#cccccc"), alpha=0.55,
                   s=18, label=q, zorder=2)

    top = df.sort_values("score", ascending=False).head(top_n)
    for feat, row in top.iterrows():
        if row["quadrant"] in ("KEEP", "CORAL"):
            ax.annotate(
                feat, (row["shap_norm"], row["ks_norm"]),
                fontsize=6, alpha=0.8,
                xytext=(4, 2), textcoords="offset points",
            )

    ax.set_xlabel("SHAP importance norm.", fontsize=10)
    ax.set_ylabel("KS statistic (domain gap) norm.", fontsize=10)
    ax.set_title(
        f"SHAP × Domain Gap — Modelo {model_id}\n"
        f"(top-right = CORAL, top-left = DROP, bottom-right = KEEP)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    # Etiquetas de cuadrante
    ax.text(0.02,  0.02,  "DROP\n(bajo SHAP, bajo KS)",  fontsize=7, color="#EF9F27",
            transform=ax.transAxes, va="bottom")
    ax.text(0.52,  0.02,  "KEEP\n(alto SHAP, bajo KS)",  fontsize=7, color="#1D9E75",
            transform=ax.transAxes, va="bottom")
    ax.text(0.02,  0.72,  "DROP\n(bajo SHAP, alto KS)",  fontsize=7, color="gray",
            transform=ax.transAxes, va="bottom")
    ax.text(0.52,  0.72,  "CORAL\n(alto SHAP, alto KS)", fontsize=7, color="#E24B4A",
            transform=ax.transAxes, va="bottom")

    handles = [mpatches.Patch(color=c, label=q) for q, c in colors.items()
               if q != "UNKNOWN"]
    ax.legend(handles=handles, fontsize=8, loc="center right")

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"shap_vs_gap_model{model_id}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_top_features(
    df:       pd.DataFrame,
    model_id: str,
    top_n:    int = 40,
    save_dir: Optional[str] = None,
):

    colors = {"KEEP": "#1D9E75", "CORAL": "#E24B4A",
               "OPTIONAL": "#aaaaaa", "DROP": "#EF9F27"}

    sub = df.sort_values("score", ascending=False).head(top_n).copy()
    sub = sub.iloc[::-1]  # invertir para que el mejor quede arriba

    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, top_n * 0.32)),
                              sharey=True)
    y  = np.arange(len(sub))
    cs = [colors.get(q, "#cccccc") for q in sub["quadrant"]]

    axes[0].barh(y, sub["shap_norm"], color=cs, alpha=0.85)
    axes[0].set_xlabel("SHAP importance norm.", fontsize=9)
    axes[0].set_title("SHAP importance", fontsize=10)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub.index, fontsize=7)

    axes[1].barh(y, sub["ks_statistic"], color=cs, alpha=0.85)
    axes[1].axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_xlabel("KS statistic (domain gap)", fontsize=9)
    axes[1].set_title("Domain gap", fontsize=10)
    axes[1].set_xlim(0, 1.05)

    handles = [mpatches.Patch(color=c, label=q) for q, c in colors.items()]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")

    fig.suptitle(
        f"Top {top_n} features por score — Modelo {model_id}\n"
        f"score = SHAP_norm − {0.5}·KS_norm",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"top_features_model{model_id}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_consensus_summary(
    consensus: pd.DataFrame,
    save_dir:  Optional[str] = None,
):

    if consensus.empty:
        return

    sub  = consensus.head(50).copy()
    sub  = sub.iloc[::-1]
    cols = sub["recommendation"].map(
        lambda r: "#E24B4A" if "CORAL" in r else "#1D9E75"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, len(sub) * 0.32)),
                              sharey=True)
    y = np.arange(len(sub))

    axes[0].barh(y, sub["mean_shap"], color=cols, alpha=0.85)
    axes[0].set_xlabel("SHAP mean (avg. modelos)", fontsize=9)
    axes[0].set_title("Importancia SHAP", fontsize=10)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["feature_base"], fontsize=7)

    axes[1].barh(y, sub["mean_ks"], color=cols, alpha=0.85)
    axes[1].axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_xlabel("KS statistic (avg. modelos)", fontsize=9)
    axes[1].set_title("Domain gap", fontsize=10)

    handles = [
        mpatches.Patch(color="#1D9E75", label="KEEP"),
        mpatches.Patch(color="#E24B4A", label="CORAL prioritario"),
    ]
    axes[1].legend(handles=handles, fontsize=8)

    n_models_str = f"({sub['models'].iloc[0].count('/')+1} modelos)" if len(sub) else ""
    fig.suptitle(
        f"Consenso de features seleccionadas {n_models_str}\n"
        f"(top 50 por score medio)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "consensus_features.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_feature_selection(
    shap_dir:       str,
    sim_dir:        str,
    real_dir:       str,
    out_dir:        str,
    models:         List[str]      = None,
    alpha:          float          = 0.5,
    shap_level:     int            = -1,
    nan_threshold:  float          = 0.5,
    top_n:          int            = 40,
    min_agreement:  int            = 2,
    gap_csvs:       Dict[str, str] = None,
    sim_atlas_dir:  Optional[str]  = None,
    real_atlas_dir: Optional[str]  = None,
) -> Dict[str, pd.DataFrame]:
    """
    Ejecuta el análisis completo SHAP × gap para todos los modelos indicados.

    """
    if models is None:
        models = ["B", "C", "D"]
    if gap_csvs is None:
        gap_csvs = {}

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for m in models:
        cfg = _MODEL_CONFIG.get(m)
        if cfg is None:
            print(f"[WARN] Modelo {m} is not configured, skipping")
            continue

        shap_path = os.path.join(shap_dir, cfg["shap_file"])
        if not os.path.exists(shap_path):
            print(f"[WARN] Not found: {shap_path}. Skipping model {m}")
            continue

        df = analyze_model(
            model_id       = m,
            shap_dir       = shap_dir,
            sim_dir        = sim_dir,
            real_dir       = real_dir,
            alpha          = alpha,
            shap_level     = shap_level,
            nan_threshold  = nan_threshold,
            gap_csv        = gap_csvs.get(m),
            sim_atlas_dir  = sim_atlas_dir,
            real_atlas_dir = real_atlas_dir,
        )

        # Guardar CSV
        csv_path = os.path.join(out_dir, f"feature_ranking_model{m}.csv")
        df.to_csv(csv_path)
        print(f"\nCSV saved: {csv_path}")

        # Plots
        plot_shap_vs_gap(df, m, top_n=top_n, save_dir=out_dir)
        plot_top_features(df, m, top_n=top_n, save_dir=out_dir)

        results[m] = df

    # Consenso 
    if len(results) >= 2:
        consensus = build_consensus(results, min_agreement, alpha)
        csv_path  = os.path.join(out_dir, "consensus_features.csv")
        consensus.to_csv(csv_path, index=False)
        print(f"\nConsensus saved: {csv_path}")
        plot_consensus_summary(consensus, save_dir=out_dir)
        results["consensus"] = consensus
    else:
        print("\nOnly one model analysed, consensus not available")

    print(f"\n✓Done. Results in: {out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Selección de features: SHAP × domain gap"
    )
    parser.add_argument("--shap-dir",  default=DEFAULT_SHAP_DIR,
                        help="Directory with shap_values_B/C/D.pkl "
                             f"(default: {DEFAULT_SHAP_DIR})")
    parser.add_argument("--sim-dir",   default=DEFAULT_SIM_DIR,
                        help="Directory with simulated ZTF/LSST features"
                             f"(default: {DEFAULT_SIM_DIR})")
    parser.add_argument("--sim-atlas-dir", default=None,
                        help="Directory with simulated ATLAS features")
    parser.add_argument("--real-dir",  default=DEFAULT_REAL_DIR,
                        help="Directorio with real features"
                             f"(ZTF/LSST/comb) (default: {DEFAULT_REAL_DIR})")
    parser.add_argument("--real-atlas-dir", default=None,
                        help="Directory with real ATLAS features")
    parser.add_argument("--out-dir",   default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--models",    nargs="+", default=["B", "C", "D"],
                        help="Models used (default: B C D)")
    parser.add_argument("--alpha",     type=float, default=0.5,
                        help="KS weight (default: 0.5)")
    parser.add_argument("--shap-level", type=int, default=-1,
                        help="SHAP hierarchical level: -1=mean, 0/1/2 (default: -1)")
    parser.add_argument("--top-n",     type=int, default=40,
                        help="Features to be shown in plots (default: 40)")
    parser.add_argument("--min-agreement", type=int, default=2,
                        help="Minimum models for consensus (default: 2)")
    parser.add_argument("--nan-threshold", type=float, default=0.5,
                        help="NaN threshold for valid features (default: 0.5)")
    parser.add_argument("--gap-csv-B", default=None,
                        help="CSV with precalculated domain gap for model B")
    parser.add_argument("--gap-csv-C", default=None,
                        help="CSV with precalculated domain gap for model C")
    parser.add_argument("--gap-csv-D", default=None,
                        help="CSV with precalculated domain gap for model D")

    args = parser.parse_args()

    gap_csvs = {}
    if args.gap_csv_B: gap_csvs["B"] = args.gap_csv_B
    if args.gap_csv_C: gap_csvs["C"] = args.gap_csv_C
    if args.gap_csv_D: gap_csvs["D"] = args.gap_csv_D

    run_feature_selection(
        shap_dir       = args.shap_dir,
        sim_dir        = args.sim_dir,
        sim_atlas_dir  = args.sim_atlas_dir,
        real_dir       = args.real_dir,
        real_atlas_dir = args.real_atlas_dir,
        out_dir        = args.out_dir,
        models         = args.models,
        alpha          = args.alpha,
        shap_level     = args.shap_level,
        nan_threshold  = args.nan_threshold,
        top_n          = args.top_n,
        min_agreement  = args.min_agreement,
        gap_csvs       = gap_csvs,
    )