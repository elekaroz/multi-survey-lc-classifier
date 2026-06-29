import os
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression as PlattScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# TAXONOMÍA
# =============================================================================

LABEL_ORDER = [
    'SNIa', 'SNIbc', 'SNII', 'SLSN',
    'QSO', 'AGN', 'Blazar',
    'YSO', 'CV/Nova',
    'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other',
]

HIER_MAP = {
    'Periodic'  : ['LPV', 'Periodic-Other', 'E', 'DSCT', 'RRL', 'CEP'],
    'Transient' : ['SNIa', 'SNIbc', 'SNII', 'SLSN'],
    'Stochastic': ['CV/Nova', 'YSO', 'AGN', 'QSO', 'Blazar'],
}

N_CLASSES  = len(LABEL_ORDER)
PRIOR_PROB = 1.0 / N_CLASSES

N_FOLDS      = 5
RANDOM_STATE = 42


# =============================================================================
# HIPERPARÁMETROS DEL RANDOM FOREST  (compartidos por todos los modelos base)
# =============================================================================

RF_PARAMS = dict(
    n_estimators       = 500,
    max_features       = 'sqrt',
    max_depth          = None,
    min_samples_split  = 10,
    min_samples_leaf   = 5,
    n_jobs             = -1,
    bootstrap          = False, #False para un training set no balanceado (se gestiona con replacement)
    class_weight       = None,
    criterion          = 'entropy',
    random_state       = RANDOM_STATE,
    sampling_strategy  = 'all',
    replacement        = True, #True para un training set no balanceado
)

# =============================================================================
# CONSTANTES DE BANDAS PARA FEATURES DIFERENCIALES CROSS-SURVEY
# =============================================================================
# Correspondencia física entre bandas de los tres surveys:
#   ZTF-g  (_1_ztf)  = LSST-g (_1_lsst)  = ATLAS-c (_1_atlas)
#   ZTF-r  (_2_ztf)  = LSST-r (_2_lsst)  = ATLAS-o (_2_atlas)
#   ZTF-i  (_3_ztf)  = LSST-i (_3_lsst)  xxxxxxxxxxxxxxxxxxx

ZTF_LSST_BAND_MATCH = {
    '1_ztf': '1_lsst',   # g = g
    '2_ztf': '2_lsst',   # r = r
    '3_ztf': '3_lsst',   # i = i
}

ZTF_ATLAS_BAND_MATCH = {
    '1_ztf': '1_atlas',  # g = c
    '2_ztf': '2_atlas',  # r = o
}

ATLAS_LSST_BAND_MATCH = {
    '1_atlas': '1_lsst',  # c = g
    '2_atlas': '2_lsst',  # o = r
}


# =============================================================================
# SELECCIÓN DE COLUMNAS
# =============================================================================

def get_ztf_cols(all_cols):
    """Devuelve todas las columnas ZTF / Modelo A (sufijo _ztf)."""
    return [c for c in all_cols if c.endswith('_ztf')]


def get_lsst_cols(all_cols):
    """Devuelve todas las columnas LSST / Modelo B (sufijo _lsst)."""
    return [c for c in all_cols if c.endswith('_lsst')]


def get_combined_cols(all_cols):
    """Devuelve todas las columnas de curvas combinada / Modelo D (sufijo _combined)."""
    return [c for c in all_cols if c.endswith('_combined')]


def get_atlas_cols(all_cols):
    """Devuelve todas las columnas ATLAS / Modelo E (sufijo _atlas)."""
    return [c for c in all_cols if c.endswith('_atlas')]


# =============================================================================
# LABELS JERÁRQUICAS
# =============================================================================

def make_hierarchical_labels(y_orig):
    """Mapea clases a su grupo jerárquico (Transient/Stochastic/Periodic)."""
    y_hier = y_orig.copy()
    for hier_cls, members in HIER_MAP.items():
        y_hier[y_orig.isin(members)] = hier_cls
    return y_hier


# =============================================================================
# FEATURES DIFERENCIALES POR SURVEY
# =============================================================================

def add_ztf_differential_features(df):
    """
    Diferenciales entre bandas (g vs r) para features ZTF.
    Espera columnas con nombre <feature>_1_ztf y <feature>_2_ztf.
    """
    eps = 1e-6
    df = df.copy()

    pairs = [
        ('Amplitude',       True),
        ('GP_DRW_tau',      True),
        ('GP_DRW_sigma',    True),
        ('SF_ML_gamma',     False),
        ('SF_ML_amplitude', True),
        ('delta_period',    False),
        ('MHPS_ratio',      False),
        ('Skew',            False),
        ('Gskew',           False),
        ('SmallKurtosis',   False),
        ('SPM_tau_rise',    True),
        ('SPM_tau_fall',    True),
        ('SPM_A',           True),
    ]

    for base, use_ratio in pairs:
        c1 = f'{base}_1_ztf'
        c2 = f'{base}_2_ztf'
        if c1 in df.columns and c2 in df.columns:
            if use_ratio:
                df[f'{base}_gr_ratio_ztf'] = df[c1] / (df[c2].abs() + eps)
            else:
                df[f'{base}_gr_diff_ztf'] = df[c1] - df[c2]

    return df


def add_atlas_differential_features(df):
    """
    Diferenciales entre bandas (c vs o) para features ATLAS.
    Espera columnas con nombre <feature>_1_atlas (c) y <feature>_2_atlas (o).
    """
    eps = 1e-6
    df = df.copy()

    pairs = [
        ('Amplitude',       True),
        ('GP_DRW_tau',      True),
        ('GP_DRW_sigma',    True),
        ('SF_ML_gamma',     False),
        ('SF_ML_amplitude', True),
        ('delta_period',    False),
        ('MHPS_ratio',      False),
        ('Skew',            False),
        ('Gskew',           False),
        ('SmallKurtosis',   False),
        ('SPM_tau_rise',    True),
        ('SPM_tau_fall',    True),
        ('SPM_A',           True),
    ]

    for base, use_ratio in pairs:
        c1 = f'{base}_1_atlas'
        c2 = f'{base}_2_atlas'
        if c1 in df.columns and c2 in df.columns:
            if use_ratio:
                df[f'{base}_co_ratio_atlas'] = df[c1] / (df[c2].abs() + eps)
            else:
                df[f'{base}_co_diff_atlas'] = df[c1] - df[c2]

    return df


# =============================================================================
# FEATURES DIFERENCIALES CROSS-SURVEY
# =============================================================================

def add_cross_survey_differential_features(df):
    """
    Diferenciales cross-survey entre bandas equivalentes de ZTF, LSST y ATLAS.
    También se añaden colores solo LSST (u, z, y) sin equivalente ZTF/ATLAS.
    """
    eps = 1e-6
    df = df.copy()

    pairs = [
        ('Amplitude',       True),
        ('GP_DRW_tau',      True),
        ('GP_DRW_sigma',    True),
        ('SF_ML_gamma',     False),
        ('SF_ML_amplitude', True),
        ('Skew',            False),
        ('SPM_tau_rise',    True),
        ('SPM_tau_fall',    True),
        ('SPM_A',           True),
        ('Std',             True),
        ('MedianAbsDev',    True),
        ('Q31',             True),
    ]

    def _make_pairs(band_match, tag):
        for src_band, dst_band in band_match.items():
            src_label = src_band.split('_')[0]
            for base, use_ratio in pairs:
                c_src = f'{base}_{src_band}'
                c_dst = f'{base}_{dst_band}'
                if c_src in df.columns and c_dst in df.columns:
                    if use_ratio:
                        df[f'{base}_b{src_label}_{tag}_ratio'] = \
                            df[c_src] / (df[c_dst].abs() + eps)
                    else:
                        df[f'{base}_b{src_label}_{tag}_diff'] = \
                            df[c_src] - df[c_dst]

    _make_pairs(ZTF_LSST_BAND_MATCH,  'ztf_lsst')
    _make_pairs(ZTF_ATLAS_BAND_MATCH,  'ztf_atlas')
    _make_pairs(ATLAS_LSST_BAND_MATCH, 'atlas_lsst')

    # Colores solo LSST: bandas sin equivalente ZTF ni ATLAS
    lsst_colour_pairs = [
        ('1_lsst', '2_lsst'),  # u-g
        ('2_lsst', '4_lsst'),  # g-i
        ('3_lsst', '4_lsst'),  # r-i
        ('4_lsst', '5_lsst'),  # i-z
        ('5_lsst', '6_lsst'),  # z-y
    ]
    for b1, b2 in lsst_colour_pairs:
        for base in ['Amplitude', 'Std', 'SPM_A']:
            c1 = f'{base}_{b1}'
            c2 = f'{base}_{b2}'
            if c1 in df.columns and c2 in df.columns:
                label = b1.split('_')[0] + b2.split('_')[0]
                df[f'{base}_color_{label}_lsst'] = df[c1] / (df[c2].abs() + eps)

    return df


# =============================================================================
# CLASIFICADOR JERÁRQUICO
# =============================================================================

def train_hierarchical_model(X_tr, yh_tr, yo_tr):
    """Entrena el clasificador jerárquico (nivel superior + 3 sub-clasificadores)."""
    clf_hier = RandomForestClassifier(**RF_PARAMS)
    clf_hier.fit(X_tr, yh_tr)

    clfs = {}
    for group in ['Periodic', 'Stochastic', 'Transient']:
        mask = yh_tr == group
        clf  = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr.loc[mask], yo_tr.loc[mask])
        clfs[group] = clf

    return clf_hier, clfs


def predict_hierarchical(X, clf_hier, clfs):
    """
    Genera probabilidades finales multiplicando probabilidades jerárquicas
    por las del sub-clasificador correspondiente.

    Devuelve (prob_matrix, class_names_array).
    """
    p_hier       = clf_hier.predict_proba(X)
    hier_classes = clf_hier.classes_

    all_probs = []
    all_names = []

    for group in ['Stochastic', 'Transient', 'Periodic']:
        idx     = np.where(hier_classes == group)[0][0]
        p_sub   = clfs[group].predict_proba(X)
        p_final = p_sub * p_hier[:, idx][:, np.newaxis]
        all_probs.append(p_final)
        all_names.extend(clfs[group].classes_)

    prob_matrix = np.concatenate(all_probs, axis=1)
    return prob_matrix, np.array(all_names)


# =============================================================================
# OOF
# =============================================================================

def compute_oof_predictions(X, y_orig, model_name, n_folds=N_FOLDS):
    """
    Genera predicciones de probabilidades out-of-fold con K folds.
    El modelo jerárquico completo se entrena y aplica en cada fold.
    """
    y_hier = make_hierarchical_labels(y_orig)
    skf    = model_selection.StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    oof_probs = np.full((len(X), len(LABEL_ORDER)), np.nan)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_orig)):
        print(f"  [{model_name}] OOF fold {fold + 1}/{n_folds} ...")

        X_tr  = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        yh_tr = y_hier.iloc[train_idx]
        yo_tr = y_orig.iloc[train_idx]

        clf_hier, clfs         = train_hierarchical_model(X_tr, yh_tr, yo_tr)
        prob_matrix, class_names = predict_hierarchical(X_val, clf_hier, clfs)

        for j, cls in enumerate(LABEL_ORDER):
            col_idx = np.where(class_names == cls)[0]
            if len(col_idx) > 0:
                oof_probs[val_idx, j] = prob_matrix[:, col_idx[0]]
            else:
                oof_probs[val_idx, j] = 0.0

    cols   = [f'p_{c}_{model_name}' for c in LABEL_ORDER]
    oof_df = pd.DataFrame(oof_probs, index=X.index, columns=cols)
    return oof_df


# =============================================================================
# CALIBRACIÓN
# =============================================================================

def select_calibration_method(y_true, min_samples_per_class=200):
    """
    Selecciona el método de calibración según el tamaño de la clase más pequeña.

    < 200 objetos -> Platt scalin
    ≥ 200 objetos -> Isotonic regression
    """
    counts    = pd.Series(y_true).value_counts()
    min_count = counts.min()
    method    = 'isotonic' if min_count >= min_samples_per_class else 'platt'
    print(f"  Smallest class: '{counts.idxmin()}' with {min_count} objects "
          f"→ using {method.upper()} calibration")
    return method


def fit_calibrators(oof_proba_df, y_true, label_order, method='auto'):
    """
    Ajusta un calibrador por clase sobre predicciones OOF.
    """
    if method == 'auto':
        method = select_calibration_method(y_true)

    calibrators = {}
    for cls in label_order:
        matching = [c for c in oof_proba_df.columns
                    if c == f'p_{cls}' or c.endswith(f'_{cls}')]
        if not matching:
            warnings.warn(f"No column found for class '{cls}', skipping.")
            continue

        p_cls = oof_proba_df[matching[0]].values
        y_bin = (y_true == cls).astype(int).values

        if method == 'platt':
            eps      = 1e-7
            log_odds = np.log(
                np.clip(p_cls, eps, 1 - eps) / (1 - np.clip(p_cls, eps, 1 - eps))
            )
            cal = PlattScaler(C=1e10, solver='lbfgs', max_iter=1000)
            cal.fit(log_odds.reshape(-1, 1), y_bin)
        else:
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(p_cls, y_bin)

        calibrators[cls] = (method, cal)

    return calibrators, method


def apply_calibrators(prob_matrix, class_names, calibrators):
    """
    Aplica los calibradores a una matriz de probabilidades y
    renormaliza las filas para que sumen 1.
    """
    prob_cal = prob_matrix.copy()

    for j, cls in enumerate(class_names):
        if cls not in calibrators:
            continue
        method, cal = calibrators[cls]
        p_col = prob_matrix[:, j]

        if method == 'platt':
            eps      = 1e-7
            log_odds = np.log(
                np.clip(p_col, eps, 1 - eps) / (1 - np.clip(p_col, eps, 1 - eps))
            )
            prob_cal[:, j] = cal.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        else:
            prob_cal[:, j] = cal.predict(p_col)

    row_sums = prob_cal.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return prob_cal / row_sums


def plot_calibration_curves(oof_proba_df, y_true, calibrators,
                            label_order, model_name, save_path):
    """
    Diagramas de fiabilidad por clase, antes y después de la calibración.
    """
    n_cols = 5
    n_rows = (len(label_order) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.5))
    axes = axes.flatten()

    for idx, cls in enumerate(label_order):
        ax       = axes[idx]
        matching = [c for c in oof_proba_df.columns
                    if c == f'p_{cls}' or c.endswith(f'_{cls}')]
        if not matching:
            ax.set_visible(False)
            continue

        p_raw = oof_proba_df[matching[0]].values
        y_bin = (y_true == cls).astype(int).values

        frac_pos_raw, mean_pred_raw = calibration_curve(
            y_bin, p_raw, n_bins=10, strategy='quantile')
        ax.plot(mean_pred_raw, frac_pos_raw, 's-',
                label='Uncalibrated', color='steelblue')

        if cls in calibrators:
            method, cal = calibrators[cls]
            if method == 'platt':
                eps    = 1e-7
                lo     = np.log(np.clip(p_raw, eps, 1 - eps) /
                                (1 - np.clip(p_raw, eps, 1 - eps)))
                p_cal  = cal.predict_proba(lo.reshape(-1, 1))[:, 1]
            else:
                p_cal  = cal.predict(p_raw)
            frac_pos_cal, mean_pred_cal = calibration_curve(
                y_bin, p_cal, n_bins=10, strategy='quantile')
            ax.plot(mean_pred_cal, frac_pos_cal, 's-',
                    label='Calibrated', color='tomato')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel('Mean predicted prob.', fontsize=8)
        ax.set_ylabel('Fraction of positives', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for idx in range(len(label_order), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Reliability diagrams — {model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Reliability diagrams saved: {save_path}")


def _oof_to_proba_df(oof_df, model_name, label_order):
    """
    Extrae las columnas de probabilidad de un DataFrame OOF y las renombra
    a 'p_<class>' para que fit_calibrators las encuentre independientemente
    del sufijo de modelo.
    """
    prob_cols = [c for c in oof_df.columns if f'_model{model_name}' in c]
    rename    = {c: c.replace(f'_model{model_name}', '') for c in prob_cols}
    return oof_df[prob_cols].rename(columns=rename), prob_cols


# =============================================================================
# UTILIDADES DE EVALUACIÓN
# =============================================================================

def evaluate_model(y_true, prob_matrix, class_names, label_order, model_name):
    """Imprime métricas para un modelo. No guarda plots (depende de PLOTS_DIR)."""
    class_final = [class_names[i] for i in np.argmax(prob_matrix, axis=1)]
    print(f"\n=== {model_name} ===")
    print("Accuracy:         ", "%0.3f" % metrics.accuracy_score(y_true, class_final))
    print("Balanced accuracy:", "%0.3f" % metrics.balanced_accuracy_score(y_true, class_final))
    print("Macro F1:         ", "%0.3f" % metrics.f1_score(y_true, class_final, average='macro'))
    print(metrics.classification_report(y_true, class_final, digits=3))
    return class_final


def plot_confusion_matrix(cm, classes, plot_name, normalize=True, title=None,
                         cmap=None):
    """Guarda una matriz de confusión normalizada como PDF."""
    if normalize:
        cm = np.round(
            (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
        )
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap or plt.cm.Blues)
    plt.title(title or '')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        label = "%d" % int(val) if np.isfinite(val) else "N/A"
        plt.text(j, i, label, ha='center',
                 color='white' if cm[i, j] > thresh else 'black', fontsize=13)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()


def plot_feature_importances(model, feature_names, save_path, top_n=60,
                            color=None):
    """Guarda un bar plot de las top-N features."""
    I   = np.argsort(model.feature_importances_)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(18, 5), tight_layout=True)
    x   = np.arange(len(I))
    plt.xticks(x, [feature_names[i] for i in I], rotation='vertical', fontsize=8)
    ax.bar(x, height=model.feature_importances_[I], color=color)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def bootstrap_metrics(y_true, prob_matrix, class_names, label_order,
                      n_bootstrap=1000, random_state=42):
    """
    Bootstrap std para accuracy, balanced accuracy, macro F1 y entropía,
    globales y por clase.
    """
    rng   = np.random.RandomState(random_state)
    n     = len(y_true)
    y_arr = np.array(y_true)

    acc_boot  = np.empty(n_bootstrap)
    bacc_boot = np.empty(n_bootstrap)
    f1_boot   = np.empty(n_bootstrap)
    ent_boot  = np.empty(n_bootstrap)

    per_class = {cls: {'precision': [], 'recall': [], 'f1': [], 'entropy': []}
                 for cls in label_order}

    eps = 1e-12
    for i in range(n_bootstrap):
        idx    = rng.randint(0, n, size=n)
        y_b    = y_arr[idx]
        pm_b   = prob_matrix[idx]
        pred_b = [class_names[j] for j in np.argmax(pm_b, axis=1)]

        acc_boot[i]  = metrics.accuracy_score(y_b, pred_b)
        bacc_boot[i] = metrics.balanced_accuracy_score(y_b, pred_b)
        f1_boot[i]   = metrics.f1_score(y_b, pred_b, average='macro', zero_division=0)
        p_clip        = np.clip(pm_b, eps, 1)
        ent_boot[i]  = (-np.sum(p_clip * np.log2(p_clip), axis=1)).mean()

        rep = metrics.classification_report(
            y_b, pred_b, labels=label_order,
            output_dict=True, zero_division=0)
        for cls in label_order:
            if cls in rep:
                per_class[cls]['precision'].append(rep[cls]['precision'])
                per_class[cls]['recall'].append(rep[cls]['recall'])
                per_class[cls]['f1'].append(rep[cls]['f1-score'])
            mask_cls = y_b == cls
            if mask_cls.sum() > 0:
                col_idx = np.where(class_names == cls)[0]
                if len(col_idx) > 0:
                    p_cls = np.clip(pm_b[mask_cls], eps, 1)
                    ent_cls = (-np.sum(p_cls * np.log2(p_cls), axis=1)).mean()
                    per_class[cls]['entropy'].append(ent_cls)

    global_std = {
        'accuracy_std':          round(float(acc_boot.std()),  5),
        'balanced_accuracy_std': round(float(bacc_boot.std()), 5),
        'macro_f1_std':          round(float(f1_boot.std()),   5),
        'mean_entropy_std':      round(float(ent_boot.std()),  5),
    }

    per_class_rows = []
    for cls in label_order:
        row = {'class': cls}
        for metric_name in ['precision', 'recall', 'f1', 'entropy']:
            vals = per_class[cls][metric_name]
            row[f'{metric_name}_std'] = round(float(np.std(vals)), 5) if vals else np.nan
        per_class_rows.append(row)

    return global_std, pd.DataFrame(per_class_rows)


# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def _load_features(path):
    """Carga un fichero parquet o CSV de features."""
    p  = Path(path)
    df = (pd.read_parquet(p) if p.suffix == '.parquet'
          else pd.read_csv(p, index_col='oid'))
    return df.replace([np.inf, -np.inf], np.nan)


def build_X(df, col_list, add_ztf_diff=False, add_atlas_diff=False,
            add_cross_diff=False):
    """
    Construye la matriz de features usando col_list, añadiendo
    opcionalmente features diferenciales.
    Las features diferenciales se calculan ANTES de rellenar NaN con -999
    """
    X = df[col_list].copy()
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]
    if add_ztf_diff:
        X = add_ztf_differential_features(X)
    if add_atlas_diff:
        X = add_atlas_differential_features(X)
    if add_cross_diff:
        X = add_cross_survey_differential_features(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-1e6, upper=1e6)
    X = X.fillna(-999)
    return X