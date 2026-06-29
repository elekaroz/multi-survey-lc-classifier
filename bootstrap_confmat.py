from __future__ import annotations

import itertools
import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# COLORMAPS POR MODELO
# =============================================================================

# Teal para Modelo A
_teal_cmap = LinearSegmentedColormap.from_list(
    'teal_a',
    ['#E1F5EE', '#9FE1CB', '#5DCAA5', '#1D9E75', '#0F6E56', '#085041', '#04342C'],
)

# Amber para Modelo B
_amber_cmap = LinearSegmentedColormap.from_list(
    'amber_b',
    ['#FAEEDA', '#FAC775', '#EF9F27', '#BA7517', '#854F0B', '#633806', '#412402'],
)

# Purples para Modelo C
_purples_cmap = LinearSegmentedColormap.from_list(
    'purples_c',
    ['#EDE8F5', '#D9C9E8', '#C4AADA', '#9067B2', '#6A3495', '#4E2275', '#3F0070'],
)

# Green para Modelo D
_green_cmap = LinearSegmentedColormap.from_list(
    'green_d',
    ['#EBF3E0', '#C4DD9E', '#97C459', '#639922', '#3B6D11', '#27500A', '#173404'],
)

# Pink para Modelo E
_pink_cmap = LinearSegmentedColormap.from_list(
    'pink_e',
    ['#FBEAF0', '#F4C0D1', '#ED93B1', '#D4537E', '#993556', '#72243E', '#4B1528'],
)

#Metamodelo con azul por defecto

# Mapa modelo → colormap
MODEL_CMAPS = {
    'A':    _teal_cmap,
    'B':    _amber_cmap,
    'C':    _purples_cmap,
    'D':    _green_cmap,
    'E':    _pink_cmap,
    'meta': plt.cm.Blues,
}

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_training_full as mtfull
import model_training_functions as mtf

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Defaults usados si el script se lanza sin argumentos (runfile() en Spyder).
# Lanzado desde terminal, cualquier --argumento sobrescribe el default
# correspondiente (ver parse_args() al final del fichero).

RUN_SIM  = True   # test set simulado (regenera desde modelos guardados)
RUN_REAL = True   # test set real     (carga parquets de eval_real_testset)

OUTPUT_DIR    = "./output/"
MODELS_DIR    = os.path.join(OUTPUT_DIR, "models") + "/"
CORAL_DIR     = os.path.join(OUTPUT_DIR, "coral")  + "/"
PLOTS_DIR_SIM = os.path.join(OUTPUT_DIR, "plots")  + "/"
CONSENSUS_CSV = "./data/feature_selection/consensus_features.csv"

#Rutas específicas para test set real

EVAL_OUTPUT_DIR  = "./output/real_eval/"
PLOTS_DIR_REAL   = EVAL_OUTPUT_DIR
REAL_LABELS_FILE = "./data/real/labels_testset.csv"
REAL_branch      = "strict"   # "strict" o "relaxed"

N_BOOTSTRAP  = 1000
RANDOM_STATE = 42


# =============================================================================
# FUNCIONES REUTILIZABLES
# =============================================================================

def bootstrap_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: list,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple:
    """
    Bootstrap estratificado de matrices de confusión normalizadas.
    Muestrea con reemplazo dentro de cada clase para garantizar
    representación de clases minoritarias en cada iteración.
    """
    rng       = np.random.RandomState(random_state)
    n_classes = len(label_order)
    y_t       = np.array(y_true)
    y_p       = np.array(y_pred)

    class_indices = {cls: np.where(y_t == cls)[0] for cls in label_order}

    cms = np.empty((n_bootstrap, n_classes, n_classes))
    for b in range(n_bootstrap):
        boot_idx = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True)
            for cls, idx in class_indices.items() if len(idx) > 0
        ])
        cm_raw   = metrics.confusion_matrix(
            y_t[boot_idx], y_p[boot_idx], labels=label_order)
        row_sums = np.where(cm_raw.sum(axis=1, keepdims=True) == 0, 1,
                            cm_raw.sum(axis=1, keepdims=True))
        cms[b]   = (cm_raw.astype(float) / row_sums) * 100

    return (np.median(cms, axis=0),
            np.percentile(cms, 5,  axis=0),
            np.percentile(cms, 95, axis=0))


def plot_confmat_with_ci(
    cm_median: np.ndarray,
    cm_p5: np.ndarray,
    cm_p95: np.ndarray,
    label_order: list,
    title: str,
    save_path: str,
    hier_map: dict = None,
    cmap=None,
):
    """
    Matriz de confusión normalizada con mediana y errores asimétricos en todas las celdas.
    Cuadros negros delimitan los grupos jerárquicos.
    """
    if hier_map is None:
        hier_map = {
            'Transient':  ['SNIa', 'SNIbc', 'SNII', 'SLSN'],
            'Stochastic': ['QSO', 'AGN', 'Blazar', 'YSO', 'CV/Nova'],
            'Periodic':   ['LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other'],
        }
    if cmap is None:
        cmap = plt.cm.Blues
    n = len(label_order)
    fig, ax = plt.subplots(figsize=(14, 11.5))
    im = ax.imshow(cm_median, interpolation='nearest', cmap=cmap,
                   vmin=0, vmax=100)

    # Cuadros jerárquicos
    idx = 0
    for group_name in ['Transient', 'Stochastic', 'Periodic']:
        size = sum(1 for c in label_order if c in hier_map.get(group_name, []))
        rect = plt.Rectangle((idx - 0.5, idx - 0.5), size, size,
                              linewidth=2.5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        idx += size

    for i, j in itertools.product(range(n), range(n)):
        val    = cm_median[i, j]
        up_int = int(round(cm_p95[i, j] - val))
        dn_int = int(round(val - cm_p5[i, j]))
        v_int  = int(round(val))
        r, g, b, _ = cmap(val / 100.0)
        lum    = 0.2126 * r + 0.7152 * g + 0.0722 * b 
        color  = 'white' if lum < 0.40 else 'black'
        fs     = 7 if v_int >= 95 else 8 
        ax.text(j, i, f'${v_int}^{{+{up_int}}}_{{-{dn_int}}}$',
                ha='center', va='center', fontsize=fs, color=color)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(label_order, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(label_order, fontsize=10)
    ax.set_ylabel('True label', fontsize=13)
    ax.set_xlabel('Predicted label', fontsize=13)
    ax.set_title(title, fontsize=13, pad=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Recall (%)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved: {save_path}')


def generate_bootstrap_confmats(
    test_probas: dict,
    meta_pred_proba,
    meta_classes,
    yo_te,
    label_order: list,
    plots_dir: str,
    models_dir: str,
    tag: str = 'sim',
    n_bootstrap: int = 1000,
    random_state: int = 42,
    hier_map: dict = None,
    save_pkl: bool = True,
):
    """
    Genera PDFs de matrices de confusión bootstrap para todos los
    modelos base y el metamodelo.

    """
    os.makedirs(plots_dir, exist_ok=True)
    tag_label = {'sim': 'simulated test', 'strict': 'real strict',
                 'relaxed': 'real relaxed'}.get(tag, tag)

    print(f"\n=== Bootstrap confusion matrices [{tag}] "
          f"(ALeRCE Fig.7 style, B={n_bootstrap}) ===")

    #Modelos base
    for name, (pm, cn, yo_te_m) in test_probas.items():
        y_pred = np.array([cn[i] for i in np.argmax(pm, axis=1)])
        print(f"  Model {name} ({len(yo_te_m)} obj)...")
        cm_med, cm_p5, cm_p95 = bootstrap_confusion_matrices(
            np.array(yo_te_m), y_pred, label_order, n_bootstrap, random_state)
        plot_confmat_with_ci(
            cm_med, cm_p5, cm_p95, label_order,
            title=(f'Model {name} — {tag_label}'),
            save_path=f'{plots_dir}conf_matrix_bootstrap_model{name}_{tag}.pdf',
            hier_map=hier_map,
            cmap=MODEL_CMAPS.get(name, plt.cm.Blues),
        )

    #Metamodelo
    if meta_pred_proba is not None:
        y_pred_meta = np.array([meta_classes[i]
                                for i in np.argmax(meta_pred_proba, axis=1)])
        print(f"Metamodel ({len(yo_te)} obj)...")
        cm_med, cm_p5, cm_p95 = bootstrap_confusion_matrices(
            np.array(yo_te), y_pred_meta, label_order, n_bootstrap, random_state)
        plot_confmat_with_ci(
            cm_med, cm_p5, cm_p95, label_order,
            title=(f'Metamodel {tag_label}'),
            save_path=f'{plots_dir}conf_matrix_bootstrap_metamodel_{tag}.pdf',
            hier_map=hier_map,
            cmap=MODEL_CMAPS['meta'],
        )

    # ── Pickle (solo para simulado) ──────────────────────────────────────────
    if save_pkl and tag == 'sim':
        _preds = {
            'test_probas': {name: (pm.tolist(), cn.tolist(), list(yt))
                            for name, (pm, cn, yt) in test_probas.items()},
            'label_order': label_order,
        }
        if meta_pred_proba is not None:
            _preds['meta'] = {
                'pred_proba': meta_pred_proba.tolist(),
                'classes':    list(meta_classes),
                'y_true':     list(yo_te),
            }
        pkl_path = f"{models_dir}test_predictions.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(_preds, f, pickle.HIGHEST_PROTOCOL)
        print(f"Predictions saved: {pkl_path}")


# =============================================================================
# CARGAR PREDICCIONES REALES (parquets de eval_real_testset.py)
# =============================================================================

def load_real_predictions(
    eval_output_dir: str,
    labels_file: str,
    branch: str = 'strict',
    label_order: list = None,
) -> tuple:
    """
    Carga las probabilidades guardadas por eval_real_testset.py.

    """
    if label_order is None:
        label_order = mtf.LABEL_ORDER

    out = Path(eval_output_dir)

    # Labels
    lp = Path(labels_file)
    df_labels = (pd.read_parquet(lp) if lp.suffix == '.parquet'
                 else pd.read_csv(lp))
    if 'oid_combined' in df_labels.columns:
        df_labels = df_labels.set_index('oid_combined')
    elif df_labels.index.name != 'oid_combined':
        df_labels = df_labels.set_index(df_labels.columns[0])
    if 'class_original' not in df_labels.columns:
        df_labels['class_original'] = df_labels['classALeRCE']
    y_true_full = df_labels.loc[
        df_labels['class_original'].isin(label_order), 'class_original']

    # Modelos base
    test_probas = {}
    for name in ['A', 'B', 'C', 'D', 'E']:
        p = out / f'probas_model{name}_{branch}.parquet'
        if not p.exists():
            continue
        df_p   = pd.read_parquet(p)
        shared = y_true_full.index.intersection(df_p.index)
        if len(shared) == 0:
            print(f'  [WARN] Model {name}: sin índices comunes con labels — omitiendo.')
            continue
        y_te = y_true_full.loc[shared]
        cols = [f'p_{c}_model{name}' for c in label_order
                if f'p_{c}_model{name}' in df_p.columns]
        pm = df_p.loc[shared, cols].values
        cn = np.array([c.replace(f'p_', '').replace(f'_model{name}', '')
                       for c in cols])
        test_probas[name] = (pm, cn, y_te)
        print(f'  Loaded Model {name} [{branch}]: {len(y_te)} objetos')

    # Metamodelo
    meta_pred_proba, meta_classes, yo_te = None, None, None
    mp = out / f'probas_metamodel_{branch}.parquet'
    if mp.exists():
        df_m   = pd.read_parquet(mp)
        shared = y_true_full.index.intersection(df_m.index)
        if len(shared) > 0:
            yo_te  = y_true_full.loc[shared]
            # Columnas: p_{cls}_meta → extraer orden de label_order
            cols   = [f'p_{c}_meta' for c in label_order
                      if f'p_{c}_meta' in df_m.columns]
            meta_pred_proba = df_m.loc[shared, cols].values
            meta_classes    = np.array([c.replace('p_', '').replace('_meta', '')
                                        for c in cols])
            print(f'  Loaded Metamodel [{branch}]: {len(yo_te)} objetos')
    else:
        if test_probas:
            first = next(iter(test_probas.values()))
            yo_te = first[2]

    if yo_te is None and test_probas:
        yo_te = next(iter(test_probas.values()))[2]

    return test_probas, meta_pred_proba, meta_classes, yo_te


# =============================================================================
# STANDALONE: test simulado (regenera desde modelos guardados)
# =============================================================================

def _run_sim():
    from model_training_adapted import (
        load_consensus, build_feature_list_for_model,
    )
    from eval_real_testset import apply_coral_to_X

    os.makedirs(PLOTS_DIR_SIM, exist_ok=True)

    print("=" * 60)
    print("MODE: simulated test set")
    print("=" * 60)

    #Features simuladas
    print("\nLoading simulated features...")
    df_ztf      = mtf._load_features(mtfull.features_ztf_file)
    df_lsst     = mtf._load_features(mtfull.features_lsst_file)
    df_combined = mtf._load_features(mtfull.features_combined_file)
    df_atlas_feat = None
    if mtfull.features_atlas_file and Path(mtfull.features_atlas_file).exists():
        df_atlas_feat = mtf._load_features(mtfull.features_atlas_file)

    ztf_cols      = mtf.get_ztf_cols(list(df_ztf.columns))
    lsst_cols     = mtf.get_lsst_cols(list(df_lsst.columns))
    combined_cols = mtf.get_combined_cols(list(df_combined.columns))
    atlas_cols    = (mtf.get_atlas_cols(list(df_atlas_feat.columns))
                     if df_atlas_feat is not None else [])

    #Consensus
    print("Loading consensus...")
    consensus    = load_consensus(CONSENSUS_CSV)
    sel_B, _     = build_feature_list_for_model(consensus, "B", lsst_cols, "_lsst")
    sel_C_ztf,  _ = build_feature_list_for_model(consensus, "C", ztf_cols,  "_ztf")
    sel_C_lsst, _ = build_feature_list_for_model(consensus, "C", lsst_cols, "_lsst")
    sel_C_atlas, _ = (build_feature_list_for_model(
                          consensus, "C", atlas_cols, "_atlas")
                      if atlas_cols else ([], []))
    sel_D, _ = build_feature_list_for_model(consensus, "D", combined_cols, "_combined")

    #Join + labels
    print("Joining features and labels...")
    df_feat = (df_ztf[ztf_cols]
               .join(df_lsst[lsst_cols], how="inner")
               .join(df_combined[combined_cols], how="inner"))
    if df_atlas_feat is not None:
        df_feat = df_feat.join(df_atlas_feat[atlas_cols], how="left")
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

    p_labels = Path(mtfull.labels_file)
    df_labels = (pd.read_parquet(p_labels) if p_labels.suffix == ".parquet"
                 else pd.read_csv(p_labels, index_col="oid"))
    if "class_original" not in df_labels.columns:
        df_labels["class_original"] = df_labels["classALeRCE"]
    labels = df_labels.loc[
        df_labels.class_original.isin(mtf.LABEL_ORDER), ["class_original"]].copy()
    df = labels.join(df_feat, how="inner").replace([np.inf, -np.inf], np.nan)
    Y_original = df["class_original"]

    #Build X
    X_A = mtf.build_X(df, ztf_cols, add_ztf_diff=True)
    X_B = mtf.build_X(df, [c for c in sel_B if c in df.columns])
    sel_C_avail = list(dict.fromkeys(
        [c for c in sel_C_ztf + sel_C_lsst + sel_C_atlas if c in df.columns]))
    if df_atlas_feat is not None:
        sel_C_avail += [c for c in atlas_cols
                        if c in df.columns and c not in set(sel_C_avail)]
    X_C = mtf.build_X(df, sel_C_avail, add_ztf_diff=True,
                       add_cross_diff=True,
                       add_atlas_diff=df_atlas_feat is not None)
    X_D = mtf.build_X(df, [c for c in sel_D if c in df.columns])

    X_E = None
    if atlas_cols:
        atlas_mask = df[atlas_cols].notna().any(axis=1)
        df_atlas   = df.loc[atlas_mask]
        X_E = mtf.build_X(df_atlas, atlas_cols, add_atlas_diff=True)

    #CORAL
    if os.path.isdir(CORAL_DIR):
        print("Aplicando CORAL...")
        cp = Path(CORAL_DIR)
        X_B = apply_coral_to_X(X_B, cp, 'B')
        for cid in ['C_ztf', 'C_lsst', 'C_atlas', 'C_diff']:
            X_C = apply_coral_to_X(X_C, cp, cid)
        X_D = apply_coral_to_X(X_D, cp, 'D')

    #Split (mismo RANDOM_STATE)
    print("Train/test split...")
    sp = model_selection.train_test_split(
        X_A, X_B, X_C, X_D, Y_original,
        test_size=0.2, stratify=Y_original,
        random_state=mtf.RANDOM_STATE)
    XA_te, XB_te, XC_te, XD_te, yo_te = sp[1], sp[3], sp[5], sp[7], sp[9]

    XE_te, yoE_te = None, None
    if X_E is not None:
        atlas_te_idx = yo_te.index.intersection(df_atlas.index)
        XE_te  = X_E.loc[atlas_te_idx]
        yoE_te = yo_te.loc[atlas_te_idx]
    print(f"Test set: {len(yo_te)} objects")

    #Cargar modelos y predecir
    print("Loading models and predicting...")
    models_path = Path(MODELS_DIR)

    def _load(name):
        p = models_path / f'model_{name}_hier.pkl'
        if not p.exists(): return None
        with open(p, 'rb') as f: clf_h = pickle.load(f)
        clfs = {}
        for g in ['Periodic', 'Stochastic', 'Transient']:
            gp = models_path / f'model_{name}_{g}.pkl'
            if gp.exists():
                with open(gp, 'rb') as f: clfs[g] = pickle.load(f)
        return clf_h, clfs

    def _load_cal(name):
        p = models_path / f'calibrators_model{name}.pkl'
        if not p.exists(): return None
        with open(p, 'rb') as f: return pickle.load(f)['calibrators']

    test_specs = [("A", XA_te, yo_te), ("B", XB_te, yo_te),
                  ("C", XC_te, yo_te), ("D", XD_te, yo_te)]
    if XE_te is not None:
        test_specs.append(("E", XE_te, yoE_te))

    test_probas = {}
    for name, X_te, yo_m in test_specs:
        pkg = _load(name)
        if pkg is None:
            print(f"Model {name} not found: skipping.")
            continue
        clf_h, clfs = pkg
        cal = _load_cal(name)
        if hasattr(clf_h, 'feature_names_in_'):
            exp = list(clf_h.feature_names_in_)
            for c in exp:
                if c not in X_te.columns: X_te[c] = -999
            X_te = X_te[exp]
        pm, cn = mtf.predict_hierarchical(X_te, clf_h, clfs)
        pm_cal = mtf.apply_calibrators(pm, cn, cal) if cal else pm
        test_probas[name] = (pm_cal, cn, yo_m)
        pred = [cn[i] for i in np.argmax(pm_cal, axis=1)]
        print(f"Model {name}: bacc="
              f"{metrics.balanced_accuracy_score(yo_m, pred):.4f}")

    #Metamodelo
    meta_pred_proba, meta_classes = None, None
    mp = models_path / 'meta_model.pkl'
    if mp.exists():
        with open(mp, 'rb') as f: meta_model = pickle.load(f)
        parts = []
        for name, (pm, cn, yo_m) in test_probas.items():
            cols   = [f"p_{c}_model{name}" for c in mtf.LABEL_ORDER]
            df_out = pd.DataFrame(mtf.PRIOR_PROB, index=yo_te.index, columns=cols)
            shared = yo_te.index.intersection(yo_m.index)
            for j, cls in enumerate(mtf.LABEL_ORDER):
                ix = np.where(cn == cls)[0]
                if len(ix):
                    df_out.loc[shared, f"p_{cls}_model{name}"] = \
                        pm[yo_m.index.get_indexer(shared), ix[0]]
            df_out[f"has_model{name}"] = 0
            df_out.loc[shared, f"has_model{name}"] = 1
            parts.append(df_out)
        meta_X = pd.concat(parts, axis=1)
        if hasattr(meta_model, 'feature_names_in_'):
            exp = list(meta_model.feature_names_in_)
            for c in exp:
                if c not in meta_X.columns:
                    meta_X[c] = 0 if c.startswith('has_model') else mtf.PRIOR_PROB
            meta_X = meta_X[exp]
        meta_pred_proba = meta_model.predict_proba(meta_X)
        meta_classes    = np.array(meta_model.classes_)
        meta_pred       = meta_model.predict(meta_X)
        print(f"Metamodel: bacc="
              f"{metrics.balanced_accuracy_score(yo_te, meta_pred):.4f}")

    generate_bootstrap_confmats(
        test_probas, meta_pred_proba, meta_classes, yo_te,
        mtf.LABEL_ORDER, PLOTS_DIR_SIM, MODELS_DIR,
        tag='sim', n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE, hier_map=mtf.HIER_MAP,
        save_pkl=True,
    )


# =============================================================================
# STANDALONE: test set real (desde parquets de eval_real_testset.py)
# =============================================================================

def _run_real():
    print("=" * 60)
    print(f"MODE: real test set ({REAL_branch})")
    print("=" * 60)

    if not os.path.isdir(EVAL_OUTPUT_DIR):
        print(f"[ERROR] EVAL_OUTPUT_DIR doesn't exist': {EVAL_OUTPUT_DIR}")
        print("Run eval_real_testset.py first.")
        return

    print(f"\nLoading predictions: {EVAL_OUTPUT_DIR}")
    test_probas, meta_pred_proba, meta_classes, yo_te = load_real_predictions(
        EVAL_OUTPUT_DIR, REAL_LABELS_FILE, REAL_branch, mtf.LABEL_ORDER)

    if not test_probas and meta_pred_proba is None:
        print("[ERROR] Probability parquets not found.")
        print(f"Search for probas_model*_{REAL_branch}.parquet in {EVAL_OUTPUT_DIR}")
        return

    generate_bootstrap_confmats(
        test_probas, meta_pred_proba, meta_classes, yo_te,
        mtf.LABEL_ORDER, PLOTS_DIR_REAL, MODELS_DIR,
        tag=REAL_branch, n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE, hier_map=mtf.HIER_MAP,
        save_pkl=False,
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """
    CLI para bootstrap_confmat.py. Sin pasar ningún argumento, el script usa
    los DEFAULTS de la sección CONFIGURACIÓN (pensado para runfile() en
    Spyder, regenerando los plots de bootstrap del último run de
    model_training_adapted.py).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generates confusion matrices with errors. It needs to iterate the test evaluation over a bootstrap of the sample."
        )
    )

    g_modo = parser.add_argument_group("Modo de ejecución")
    g_modo.add_argument("--no-run-sim", dest="run_sim", action="store_false",
        default=RUN_SIM, help="Do not regenerate simulated test set matrices.")
    g_modo.add_argument("--no-run-real", dest="run_real", action="store_false",
        default=RUN_REAL, help="Do not regenerate real test set matrices.")

    g_sim = parser.add_argument_group("Simulated test set (shares output with model_training_adapted.py)")
    g_sim.add_argument("--output-dir", default=OUTPUT_DIR,
        help="model_training_adapted.py output directory (contains models/, coral/, plots/)")
    g_sim.add_argument("--consensus-csv", default=CONSENSUS_CSV,
        help="Consensus CSV generated by feature_selection.py")

    g_real = parser.add_argument_group("Test set real")
    g_real.add_argument("--eval-output-dir", default=EVAL_OUTPUT_DIR,
        help="eval_real_testset.py output directory")
    g_real.add_argument("--real-labels-file", default=REAL_LABELS_FILE,
        help="Real test set labels CSV/parquet")
    g_real.add_argument("--real-branch", default=REAL_branch, choices=["strict", "relaxed"],
        help="Real test set branch to evaluate")

    g_boot = parser.add_argument_group("Bootstrap")
    g_boot.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
        help="Bootstrap resample number")
    g_boot.add_argument("--random-state", type=int, default=RANDOM_STATE,
        help="Random seed")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    RUN_SIM  = args.run_sim
    RUN_REAL = args.run_real

    OUTPUT_DIR    = args.output_dir
    MODELS_DIR    = os.path.join(OUTPUT_DIR, "models") + "/"
    CORAL_DIR     = os.path.join(OUTPUT_DIR, "coral")  + "/"
    PLOTS_DIR_SIM = os.path.join(OUTPUT_DIR, "plots")  + "/"
    CONSENSUS_CSV = args.consensus_csv

    EVAL_OUTPUT_DIR  = args.eval_output_dir
    PLOTS_DIR_REAL   = EVAL_OUTPUT_DIR
    REAL_LABELS_FILE = args.real_labels_file
    REAL_branch      = args.real_branch

    N_BOOTSTRAP  = args.n_bootstrap
    RANDOM_STATE = args.random_state

    if RUN_SIM:
        _run_sim()
    if RUN_REAL:
        _run_real()
    print("\n✓ Done.")