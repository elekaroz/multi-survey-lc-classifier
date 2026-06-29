import argparse
import itertools
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn import metrics
from sklearn.calibration import calibration_curve

from model_training_functions import (
    LABEL_ORDER,
    HIER_MAP,
    N_CLASSES,
    PRIOR_PROB,
    get_ztf_cols,
    get_lsst_cols,
    get_atlas_cols,
    get_combined_cols,
    add_ztf_differential_features,
    add_atlas_differential_features,
    add_cross_survey_differential_features,
    make_hierarchical_labels,
    predict_hierarchical,
    apply_calibrators,
    build_X,
)


def _load_adapted_utils():
    from model_training_adapted import (
        load_consensus,
        build_feature_list_for_model,
        fit_and_apply_coral_for_model,
    )
    return load_consensus, build_feature_list_for_model, fit_and_apply_coral_for_model

TRANSIENT_CLASSES = ['SNIa', 'SNIbc', 'SNII', 'SLSN']

# ---------------------------------------------------------------------------
# Feature / label loading
# ---------------------------------------------------------------------------

def load_file(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    df = pd.read_csv(path)
    if 'oid_combined' in df.columns:
        df = df.set_index('oid_combined')
    else:
        df = df.set_index(df.columns[0])
    return df


def load_features_for_branch(features_dir: Path, atlas_dir, branch: str) -> pd.DataFrame:
    survey_files = {
        'ztf':      features_dir / f'features_ztf_{branch}.parquet',
        'lsst':     features_dir / f'features_lsst_{branch}.parquet',
        'combined': features_dir / f'features_comb_{branch}.parquet',
    }
    if atlas_dir is not None:
        survey_files['atlas'] = atlas_dir / f'features_atlas_{branch}.parquet'

    KNOWN_SUFFIXES = ('_ztf', '_lsst', '_atlas', '_combined')

    frames = {}
    for survey, path in survey_files.items():
        if path.exists():
            df = pd.read_parquet(path).replace([np.inf, -np.inf], np.nan)
            unsuffixed = [c for c in df.columns
                          if not any(c.endswith(s) for s in KNOWN_SUFFIXES)]
            if unsuffixed:
                print(f'[INFO] {survey}: dropping {len(unsuffixed)} unsuffixed'
                      f'columns: {unsuffixed[:5]}{"..." if len(unsuffixed) > 5 else ""}')
                df = df.drop(columns=unsuffixed)
            frames[survey] = df
            print(f'Loaded {survey:8s}: {len(df):5d} objects  ({path.name})')
        else:
            print(f'[WARN] Not found, skipping: {path}')

    if not frames:
        raise FileNotFoundError(
            f'No feature parquets found for branch "{branch}" in {features_dir}')

    merged = pd.concat(list(frames.values()), axis=1, join='outer')
    dup = merged.columns.duplicated().sum()
    if dup:
        merged = merged.loc[:, ~merged.columns.duplicated()]
        print(f'[WARN] Removed {dup} duplicate columns after merge')
    print(f'Merged feature matrix: {len(merged)} objects, '
          f'{merged.shape[1]} columns\n')
    return merged


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def load_model(models_dir: Path, name: str):
    hier_path = models_dir / f'model_{name}_hier.pkl'
    if not hier_path.exists():
        return None
    with open(hier_path, 'rb') as f:
        clf_hier = pickle.load(f)
    clfs = {}
    for group in ['Periodic', 'Stochastic', 'Transient']:
        p = models_dir / f'model_{name}_{group}.pkl'
        if p.exists():
            with open(p, 'rb') as f:
                clfs[group] = pickle.load(f)
    return clf_hier, clfs


def load_calibrators(models_dir: Path, name: str):
    path = models_dir / f'calibrators_model{name}.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)['calibrators']


def load_metamodel(models_dir: Path, filename: str = 'meta_model.pkl'):
    path = models_dir / filename
    if not path.exists():
        print(f'[Metamodel] Not found: {path}')
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_coral_transform(coral_dir: Path, model_id: str):
    path = coral_dir / f'coral_model{model_id}.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['features'], d['mu_s'], d['mu_t'], d['W']


def apply_coral_to_X(X: pd.DataFrame, coral_dir: Path, model_id: str) -> pd.DataFrame:

    result = load_coral_transform(coral_dir, model_id)
    if result is None:
        return X
    features, mu_s, mu_t, W = result
    available = [f for f in features if f in X.columns]
    missing_feats = [f for f in features if f not in X.columns]
    print(f"CORAL {model_id}] available: {len(available)}/{len(features)} "
          f"({100*len(available)/len(features):.0f}%)"
          + (f"missing: {missing_feats}" if missing_feats else ""))
    if not available:
        return X
    X_out = X.copy()
    n = len(X_out)

    arr_full = np.tile(np.asarray(mu_s, dtype=float), (n, 1))
    avail_idx = [features.index(f) for f in available]

    raw = X_out[available].values.astype(float)
    raw[raw == -999] = np.nan
    for k in range(raw.shape[1]):
        col_vals = raw[:, k]
        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            med = np.nanmedian(col_vals)
            col_vals[nan_mask] = med if np.isfinite(med) else 0.0
        arr_full[:, avail_idx[k]] = col_vals

    arr_aligned_full = (arr_full - mu_s) @ W + mu_t
    arr_aligned = arr_aligned_full[:, avail_idx]

    orig_nan = X[available].values == -999
    arr_aligned[orig_nan] = -999
    X_out[available] = arr_aligned
    return X_out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(y_true, pred, model_name):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='y_pred contains classes not in y_true')
        acc  = metrics.accuracy_score(y_true, pred)
        bacc = metrics.balanced_accuracy_score(y_true, pred)
        f1   = metrics.f1_score(y_true, pred, average='macro', zero_division=0)
        rep  = metrics.classification_report(y_true, pred, labels=LABEL_ORDER,
                                             digits=3, zero_division=0)
    print(f'\n{"="*60}')
    print(f'{model_name}')
    print(f'{"="*60}')
    print(f'Accuracy:          {acc:.3f}')
    print(f'Balanced accuracy: {bacc:.3f}')
    print(f'Macro F1:          {f1:.3f}')
    print(rep)


def per_class_metrics(y_true, pred) -> list:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='y_pred contains classes not in y_true')
        report = metrics.classification_report(
            y_true, pred, labels=LABEL_ORDER,
            output_dict=True, zero_division=0)
    rows = []
    for cls in LABEL_ORDER:
        if cls in report:
            rows.append({
                'class':     cls,
                'precision': round(report[cls]['precision'], 4),
                'recall':    round(report[cls]['recall'],    4),
                'f1':        round(report[cls]['f1-score'],  4),
                'support':   int(report[cls]['support']),
            })
    return rows


def per_class_f1(y_true, pred) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='y_pred contains classes not in y_true')
        f1s = metrics.f1_score(y_true, pred, labels=LABEL_ORDER,
                               average=None, zero_division=0)
    return dict(zip(LABEL_ORDER, f1s))


# ---------------------------------------------------------------------------
# Plots: confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes, save_path, normalize=True, title=None):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = np.round((cm.astype(float) / row_sums) * 100)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title or '', fontsize=13)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f'{int(cm[i, j])}', ha='center',
                color='white' if cm[i, j] > thresh else 'black', fontsize=11)
    ax.set_ylabel('True label', fontsize=14)
    ax.set_xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f'Saved: {save_path}')


# ---------------------------------------------------------------------------
# Plots: entropy (dot plot, matching model_training_functions.py style)
# ---------------------------------------------------------------------------

def compute_entropy(prob_matrix: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(prob_matrix, eps, 1)
    return -np.sum(p * np.log2(p), axis=1)


def plot_entropy_by_class(entropy: np.ndarray, y_true, label_order,
                          model_name: str, save_path: str):
    y_arr  = np.array(y_true)
    means  = np.array([entropy[y_arr == cls].mean() if (y_arr == cls).sum() > 0
                       else 0.0 for cls in label_order])
    stds   = np.array([entropy[y_arr == cls].std()  if (y_arr == cls).sum() > 0
                       else 0.0 for cls in label_order])

    group_colors = {'Transient': '#e05c5c', 'Stochastic': '#5c8ae0',
                    'Periodic': '#5cb85c'}
    cls_to_group = {cls: grp for grp, members in HIER_MAP.items()
                    for cls in members}
    colors = [group_colors.get(cls_to_group.get(cls, ''), '#aaaaaa')
              for cls in label_order]

    x = np.arange(len(label_order))
    fig, ax = plt.subplots(figsize=(14, 5))
    for xi, (m, s, c) in enumerate(zip(means, stds, colors)):
        ax.errorbar(xi, m, yerr=s, fmt='o', color=c,
                    capsize=4, linewidth=1.2, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=3)
    legend_elements = [Patch(facecolor=c, alpha=0.85, label=g)
                       for g, c in group_colors.items()]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('True class', fontsize=12)
    ax.set_ylabel('Mean prediction entropy (bits)', fontsize=12)
    ax.set_title(f'Entropy by class, {model_name}', fontsize=13)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Entropy plot saved: {save_path}')


def entropy_summary_df(entropy: np.ndarray, y_true, label_order) -> pd.DataFrame:
    y_arr = np.array(y_true)
    rows  = []
    for cls in label_order:
        mask = y_arr == cls
        if mask.sum() > 0:
            rows.append({
                'class':          cls,
                'mean_entropy':   round(float(entropy[mask].mean()),     4),
                'median_entropy': round(float(np.median(entropy[mask])), 4),
                'n':              int(mask.sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots: transient KDE (matches model_training_functions.py exactly)
# ---------------------------------------------------------------------------

def plot_transient_proba_kde(prob_matrix, class_names, y_true,
                              model_name, save_path):
    from scipy.stats import gaussian_kde

    class_names = np.array(class_names)
    y_true_arr  = np.array(y_true)
    n_tr        = len(TRANSIENT_CLASSES)

    fig, axes = plt.subplots(n_tr, n_tr,
                             figsize=(n_tr * 3.5, n_tr * 3.0),
                             sharex=True)
    fig.suptitle(f'Transient probability distributions, {model_name}',
                 fontsize=13, y=1.01)

    x_grid = np.linspace(0, 1, 300)

    for row_idx, true_cls in enumerate(TRANSIENT_CLASSES):
        true_mask = y_true_arr == true_cls
        if true_mask.sum() == 0:
            for col_idx in range(n_tr):
                axes[row_idx, col_idx].set_visible(False)
            continue

        pm_sub   = prob_matrix[true_mask]
        pred_cls = class_names[np.argmax(pm_sub, axis=1)]

        for col_idx, pred_cls_name in enumerate(TRANSIENT_CLASSES):
            ax = axes[row_idx, col_idx]
            col_pos = np.where(class_names == pred_cls_name)[0]
            if len(col_pos) == 0:
                ax.set_visible(False)
                continue
            probs = pm_sub[:, col_pos[0]]

            correct_mask   = pred_cls == true_cls
            incorrect_mask = ~correct_mask

            for mask, color, label in [
                (correct_mask,   '#2ca02c', 'Correct'),
                (incorrect_mask, '#d62728', 'Misclassified'),
            ]:
                vals = probs[mask]
                if vals.sum() == 0 or len(vals) < 3:
                    continue
                if vals.std() < 1e-6:
                    ax.axvline(vals.mean(), color=color,
                               linewidth=1.5, linestyle='--', label=label)
                    continue
                try:
                    kde = gaussian_kde(vals, bw_method='scott')
                    ax.fill_between(x_grid, kde(x_grid), alpha=0.35, color=color)
                    ax.plot(x_grid, kde(x_grid), color=color,
                            linewidth=1.5, label=label)
                except Exception:
                    pass

            if row_idx == col_idx:
                ax.set_facecolor('#f5f5f5')
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=7)
            if row_idx == n_tr - 1:
                ax.set_xlabel(f'P({pred_cls_name})', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f'True: {true_cls}', fontsize=9)
            if row_idx == 0 and col_idx == n_tr - 1:
                ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Transient KDE plot saved: {save_path}')


# ---------------------------------------------------------------------------
# Plots: metamodel reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(prob_matrix, class_names, y_true, label_order,
                              title, save_path):
    n_cols = 5
    n_rows = (len(label_order) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.5))
    axes = axes.flatten()
    class_order = np.array(class_names)

    for idx, cls in enumerate(label_order):
        ax = axes[idx]
        col_idx = np.where(class_order == cls)[0]
        if len(col_idx) == 0:
            ax.set_visible(False)
            continue
        p_cls = prob_matrix[:, col_idx[0]]
        y_bin = (np.array(y_true) == cls).astype(int)
        try:
            frac_pos, mean_pred = calibration_curve(
                y_bin, p_cls, n_bins=10, strategy='quantile')
            ax.plot(mean_pred, frac_pos, 's-', color='steelblue', label='Model')
        except ValueError:
            ax.text(0.5, 0.5, 'insufficient data',
                    ha='center', va='center', fontsize=8)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel('Mean predicted prob.', fontsize=8)
        ax.set_ylabel('Fraction of positives', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for idx in range(len(label_order), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Reliability diagram saved: {save_path}')


def plot_metamodel_reliability(meta_pred_proba, y_true, label_order,
                                meta_model_obj, save_path):
    n_cols = 5
    n_rows = (len(label_order) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.5))
    axes = axes.flatten()
    class_order = meta_model_obj.classes_

    for idx, cls in enumerate(label_order):
        ax = axes[idx]
        col_idx = np.where(class_order == cls)[0]
        if len(col_idx) == 0:
            ax.set_visible(False)
            continue
        p_cls = meta_pred_proba[:, col_idx[0]]
        y_bin = (np.array(y_true) == cls).astype(int)
        try:
            frac_pos, mean_pred = calibration_curve(
                y_bin, p_cls, n_bins=10, strategy='quantile')
            ax.plot(mean_pred, frac_pos, 's-', color='steelblue',
                    label='Metamodel')
        except ValueError:
            ax.text(0.5, 0.5, 'insufficient data',
                    ha='center', va='center', fontsize=8)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel('Mean predicted prob.', fontsize=8)
        ax.set_ylabel('Fraction of positives', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for idx in range(len(label_order), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Reliability diagrams: Metamodel (real test set)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Reliability diagram saved: {save_path}')


# ---------------------------------------------------------------------------
# Plots: confidence
# ---------------------------------------------------------------------------

def compute_confidence_df(prob_matrix, class_names, y_true, index) -> pd.DataFrame:
    pred_idx   = np.argmax(prob_matrix, axis=1)
    pred_class = np.array([class_names[i] for i in pred_idx])
    max_proba  = prob_matrix[np.arange(len(prob_matrix)), pred_idx]
    return pd.DataFrame({
        'true_class': np.array(y_true),
        'pred_class': pred_class,
        'max_proba':  max_proba,
        'entropy':    compute_entropy(prob_matrix),
        'correct':    (pred_class == np.array(y_true)),
    }, index=index)


def plot_confidence_distribution(conf_df, model_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 26)
    for correct, label, color in [(True,  'Correct',   'steelblue'),
                                   (False, 'Incorrect', 'tomato')]:
        vals = conf_df.loc[conf_df['correct'] == correct, 'max_proba']
        if len(vals) > 0:
            ax.hist(vals, bins=bins, alpha=0.65,
                    label=f'{label} (n={len(vals)})',
                    color=color, density=True)
    ax.set_xlabel('Max predicted probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Prediction confidence: {model_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f'Confidence plot saved: {save_path}')


def plot_confidence_branch_comparison(conf_strict, conf_relaxed,
                                     model_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins     = np.linspace(0, 1, 26)
    branch_cfg = [('strict',  conf_strict,  'steelblue'),
                ('relaxed', conf_relaxed, 'darkorange')]
    for ax, correct, title in zip(axes, [True, False],
                                   ['Correct', 'Incorrect']):
        for branch, conf_df, color in branch_cfg:
            vals = conf_df.loc[conf_df['correct'] == correct, 'max_proba']
            if len(vals) > 0:
                ax.hist(vals, bins=bins, alpha=0.55,
                        label=f'{branch} (n={len(vals)})',
                        color=color, density=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Max predicted probability', fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Density', fontsize=11)
    plt.suptitle(f'Confidence: strict vs. relaxed: {model_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f'Branch confidence comparison saved: {save_path}')


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_metrics(y_true, prob_matrix, class_names, label_order,
                      n_bootstrap=1000, random_state=42, macro_labels=None):
    if macro_labels is None:
        macro_labels = label_order

    rng   = np.random.RandomState(random_state)
    n     = len(y_true)
    y_arr = np.array(y_true)
    eps   = 1e-12

    acc_boot  = np.empty(n_bootstrap)
    bacc_boot = np.empty(n_bootstrap)
    f1_boot   = np.empty(n_bootstrap)
    ent_boot  = np.empty(n_bootstrap)
    prec_boot = np.empty(n_bootstrap)
    rec_boot  = np.empty(n_bootstrap)

    per_class = {cls: {'precision': [], 'recall': [], 'f1': [], 'entropy': []}
                 for cls in label_order}

    for i in range(n_bootstrap):
        idx    = rng.randint(0, n, size=n)
        y_b    = y_arr[idx]
        pm_b   = prob_matrix[idx]
        pred_b = [class_names[j] for j in np.argmax(pm_b, axis=1)]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='y_pred contains classes not in y_true')
            acc_boot[i]  = metrics.accuracy_score(y_b, pred_b)
            bacc_boot[i] = metrics.balanced_accuracy_score(y_b, pred_b)
            p_clip       = np.clip(pm_b, eps, 1)
            ent_boot[i]  = (-np.sum(p_clip * np.log2(p_clip), axis=1)).mean()
            rep = metrics.classification_report(
                y_b, pred_b, labels=label_order,
                output_dict=True, zero_division=0)
            prec_boot[i] = np.mean([rep[c]['precision'] for c in macro_labels])
            rec_boot[i]  = np.mean([rep[c]['recall']    for c in macro_labels])
            f1_boot[i]   = np.mean([rep[c]['f1-score']  for c in macro_labels])

        for cls in label_order:
            if cls in rep:
                per_class[cls]['precision'].append(rep[cls]['precision'])
                per_class[cls]['recall'].append(rep[cls]['recall'])
                per_class[cls]['f1'].append(rep[cls]['f1-score'])
            mask_cls = y_b == cls
            if mask_cls.sum() > 0:
                p_cls   = np.clip(pm_b[mask_cls], eps, 1)
                per_class[cls]['entropy'].append(
                    (-np.sum(p_cls * np.log2(p_cls), axis=1)).mean())

    global_std = {
        'accuracy_std':          round(float(acc_boot.std()),  5),
        'balanced_accuracy_std': round(float(bacc_boot.std()), 5),
        'macro_f1_std':          round(float(f1_boot.std()),   5),
        'mean_entropy_std':      round(float(ent_boot.std()),  5),
        'precision_macro_std':   round(float(prec_boot.std()), 5),
        'recall_macro_std':      round(float(rec_boot.std()),  5),
    }
    per_class_rows = []
    for cls in label_order:
        row = {'class': cls}
        for metric_name in ['precision', 'recall', 'f1', 'entropy']:
            vals = per_class[cls][metric_name]
            row[f'{metric_name}_std'] = round(float(np.std(vals)), 5) if vals else np.nan
        per_class_rows.append(row)

    return global_std, pd.DataFrame(per_class_rows)


# ---------------------------------------------------------------------------
# Cross-branch / domain-gap comparison plots
# ---------------------------------------------------------------------------

def plot_domain_gap(sim_f1, real_f1, save_path):
    models   = sorted(set(sim_f1.keys()) & set(real_f1.keys()))
    n_models = len(models)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, model_name in zip(axes, models):
        x = np.arange(len(LABEL_ORDER))
        w = 0.38
        sim_vals  = [sim_f1[model_name].get(c, 0) for c in LABEL_ORDER]
        real_vals = [real_f1[model_name].get(c, 0) for c in LABEL_ORDER]
        ax.bar(x - w/2, sim_vals,  w, label='Simulated test',
               color='steelblue', alpha=0.8)
        ax.bar(x + w/2, real_vals, w, label='Real test',
               color='tomato', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_ORDER, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Model {model_name}', fontsize=11)
        ax.set_ylabel('F1', fontsize=10)
        ax.legend(fontsize=8)
    plt.suptitle('Domain gap: simulated vs. real test set (per-class F1)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f'Domain-gap plot saved: {save_path}')


def plot_branch_comparison(branch_f1, save_path):
    branches    = list(branch_f1.keys())
    models   = sorted({m for t in branches for m in branch_f1[t].keys()})
    if not models:
        return
    colors   = {'strict': 'steelblue', 'relaxed': 'darkorange'}
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, model_name in zip(axes, models):
        x = np.arange(len(LABEL_ORDER))
        w = 0.35
        for k, branch in enumerate(branches):
            vals   = [branch_f1[branch].get(model_name, {}).get(c, 0)
                      for c in LABEL_ORDER]
            offset = (k - (len(branches) - 1) / 2) * w
            ax.bar(x + offset, vals, w, label=branch,
                   color=colors.get(branch, f'C{k}'), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_ORDER, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Model {model_name}', fontsize=11)
        ax.set_ylabel('F1', fontsize=10)
        ax.legend(fontsize=8)
    plt.suptitle('Strict vs. relaxed: per-class F1',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f'Branch comparison plot saved: {save_path}')


# ---------------------------------------------------------------------------
# Single-branch evaluation
# ---------------------------------------------------------------------------

def evaluate_branch(branch, df_feat, y_true, models_dir, out_dir, n_bootstrap,
                  adapted=False, consensus=None, coral_dir=None,
                  metamodel_filename='meta_model.pkl',
                  compare_metamodel_filename=None):
    print(f'\n{"━"*60}')
    print(f'BRANCH: {branch.upper()}  ({len(y_true)} labeled objects)')
    print(f'{"━"*60}')

    all_cols      = list(df_feat.columns)
    ztf_cols      = get_ztf_cols(all_cols)
    lsst_cols     = get_lsst_cols(all_cols)
    atlas_cols    = get_atlas_cols(all_cols)
    combined_cols = get_combined_cols(all_cols)
    has_atlas     = len(atlas_cols) > 0

    print(f'Feature columns:  ZTF={len(ztf_cols)}  LSST={len(lsst_cols)}  '
          f'ATLAS={len(atlas_cols)}  combined={len(combined_cols)}')

    #Feature selection for adapted mode
    if adapted and consensus is not None:
        _, build_feats, _ = _load_adapted_utils()
        sel_B, _  = build_feats(consensus, 'B', lsst_cols,     '_lsst')
        sel_D, _  = build_feats(consensus, 'D', combined_cols, '_combined')
        sel_C_ztf,  _ = build_feats(consensus, 'C', ztf_cols,  '_ztf')
        sel_C_lsst, _ = build_feats(consensus, 'C', lsst_cols, '_lsst')
        lsst_cols_use = [c for c in sel_B     if c in df_feat.columns]
        comb_cols_use = [c for c in sel_D     if c in df_feat.columns]
        c_ztf_use     = [c for c in sel_C_ztf  if c in df_feat.columns]
        c_lsst_use    = [c for c in sel_C_lsst if c in df_feat.columns]
        c_cols_use    = list(dict.fromkeys(c_ztf_use + c_lsst_use))
        print(f'[Adapted] Feature counts -> B:{len(lsst_cols_use)}, '
              f'C:{len(c_cols_use)}, D:{len(comb_cols_use)}')
    else:
        lsst_cols_use = lsst_cols
        comb_cols_use = combined_cols
        c_cols_use    = ztf_cols + lsst_cols + (atlas_cols if has_atlas else [])

    X = {
        'A': build_X(df_feat, ztf_cols,       add_ztf_diff=True),
        'B': build_X(df_feat, lsst_cols_use),
        'D': build_X(df_feat, comb_cols_use),
    }
    X['C'] = build_X(df_feat, c_cols_use + (atlas_cols if has_atlas else []),
                     add_ztf_diff=True,
                     add_atlas_diff=has_atlas,
                     add_cross_diff=True)

    #Apply CORAL in adapted mode
    if adapted and coral_dir is not None:
        coral_path = Path(coral_dir)
        X['B'] = apply_coral_to_X(X['B'], coral_path, 'B')
        X['C'] = apply_coral_to_X(X['C'], coral_path, 'C_ztf')
        X['C'] = apply_coral_to_X(X['C'], coral_path, 'C_lsst')
        X['C'] = apply_coral_to_X(X['C'], coral_path, 'C_atlas')
        X['C'] = apply_coral_to_X(X['C'], coral_path, 'C_diff')
        X['D'] = apply_coral_to_X(X['D'], coral_path, 'D')
        print(f'[CORAL] Transformations applied from {coral_dir}')

    #Align feature matrices to trained model's expected features
    if adapted:
        for name in list(X.keys()):
            model_pkg = load_model(models_dir, name)
            if model_pkg is None:
                continue
            clf_hier = model_pkg[0]
            if hasattr(clf_hier, 'feature_names_in_'):
                expected = list(clf_hier.feature_names_in_)
                available = [c for c in expected if c in X[name].columns]
                missing   = [c for c in expected if c not in X[name].columns]
                if missing:
                    # Fill missing columns with -999 (same sentinel as build_X)
                    for col in missing:
                        X[name][col] = -999
                    print(f'[Adapted align] Model {name}: '
                          f'{len(missing)} missing cols filled with -999')
                X[name] = X[name][expected]
            elif hasattr(clf_hier, 'n_features_in_'):
                n_exp = clf_hier.n_features_in_
                if X[name].shape[1] != n_exp:
                    print(f'[WARN] Model {name}: {X[name].shape[1]} features '
                          f'but model expects {n_exp}, cannot align without '
                          f'feature_names_in_. Skipping this model.')
                    del X[name]

    y_true_E = None
    if has_atlas:
        atlas_mask = df_feat[atlas_cols].notna().any(axis=1)
        df_e       = df_feat.loc[atlas_mask]
        y_true_E   = y_true.loc[atlas_mask]
        X['E']     = build_X(df_e, atlas_cols, add_atlas_diff=True)
        print(f'ATLAS coverage [{branch}]: {atlas_mask.sum()} / {len(df_feat)} objects')

    y_true_map = {name: (y_true_E if name == 'E' else y_true) for name in X}

    present_classes = [c for c in LABEL_ORDER if (y_true == c).sum() > 0]
    print(f'[{branch}] Classes with real objects ({len(present_classes)}/'
          f'{len(LABEL_ORDER)}): {present_classes}')

    summary_rows      = []
    per_class_rows    = []
    conf_matrix_rows  = []
    entropy_records   = []
    bootstrap_rows    = []

    real_f1_dict = {}
    prob_dfs     = {}
    conf_dfs     = {}

    # ------------------------------------------------------------------
    # Base models
    # ------------------------------------------------------------------
    for name, X_te in X.items():
        model_pkg = load_model(models_dir, name)
        if model_pkg is None:
            print(f'\nModel {name}: weights not found. Skipping.')
            continue

        calibs = load_calibrators(models_dir, name)
        if calibs is None:
            warnings.warn(f'Calibrators for Model {name} not found,'
                          'using raw probabilities.')

        clf_hier, clfs = model_pkg
        y_te = y_true_map[name]

        print(f'\nEvaluating Model {name} [{branch}] ({len(X_te)} objects)...')
        prob_matrix, class_names = predict_hierarchical(X_te, clf_hier, clfs)

        pred_raw = [class_names[i] for i in np.argmax(prob_matrix, axis=1)]
        print_report(y_te, pred_raw, f'Model {name}: raw [{branch}]')

        prob_cal = apply_calibrators(prob_matrix, class_names, calibs) \
            if calibs is not None else prob_matrix
        pred_cal = [class_names[i] for i in np.argmax(prob_cal, axis=1)]
        print_report(y_te, pred_cal, f'Model {name}: calibrated [{branch}]')

        cm = metrics.confusion_matrix(y_te, pred_cal, labels=LABEL_ORDER)
        cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
        cm_df.index.name = 'true_class'
        cm_df.insert(0, 'branch',  branch)
        cm_df.insert(0, 'model', f'Model_{name}')
        conf_matrix_rows.append(cm_df.reset_index())

        prob_df = pd.DataFrame(prob_cal, index=X_te.index,
                               columns=[f'p_{c}_model{name}' for c in class_names])
        ordered = [f'p_{c}_model{name}' for c in LABEL_ORDER
                   if f'p_{c}_model{name}' in prob_df.columns]
        prob_df[ordered].to_parquet(out_dir / f'probas_model{name}_{branch}.parquet')
        prob_dfs[name] = (prob_cal, class_names, y_te)


        ent = compute_entropy(prob_cal)
        plot_entropy_by_class(
            ent, y_te, LABEL_ORDER,
            model_name=f'Model {name} [{branch}]',
            save_path=str(out_dir / f'entropy_model{name}_{branch}.pdf'),
        )
        df_ent = entropy_summary_df(ent, y_te, LABEL_ORDER)
        df_ent.insert(0, 'branch',  branch)
        df_ent.insert(0, 'model', f'Model_{name}')
        entropy_records.append(df_ent)


        plot_transient_proba_kde(
            prob_cal, class_names, y_te,
            model_name=f'Model {name} [{branch}]',
            save_path=str(out_dir / f'transient_kde_model{name}_{branch}.pdf'),
        )


        conf_df = compute_confidence_df(prob_cal, class_names, y_te, X_te.index)
        conf_dfs[name] = conf_df
        conf_df.to_parquet(out_dir / f'confidence_model{name}_{branch}.parquet')
        plot_confidence_distribution(
            conf_df,
            model_name=f'Model {name} [{branch}]',
            save_path=str(out_dir / f'confidence_model{name}_{branch}.pdf'),
        )


        plot_reliability_diagram(
            prob_cal, class_names, y_te, LABEL_ORDER,
            title=f'Reliability diagrams: Model {name} [{branch}]',
            save_path=str(out_dir / f'reliability_model{name}_{branch}.pdf'),
        )


        print(f'Bootstrap [{branch}] Model {name} (n={n_bootstrap})...')
        g_std, pc_std = bootstrap_metrics(
            y_te, prob_cal, class_names, LABEL_ORDER, n_bootstrap,
            macro_labels=present_classes)
        pc_std.insert(0, 'branch',  branch)
        pc_std.insert(0, 'model', f'Model_{name}')
        bootstrap_rows.append(pc_std)


        for row in per_class_metrics(y_te, pred_cal):
            row.update({'model': f'Model_{name}', 'branch': branch})
            per_class_rows.append(row)


        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='y_pred contains classes not in y_true')
            acc  = metrics.accuracy_score(y_te, pred_cal)
            bacc = metrics.balanced_accuracy_score(y_te, pred_cal)

            rep = metrics.classification_report(
                y_te, pred_cal, labels=LABEL_ORDER,
                output_dict=True, zero_division=0)
            precision_macro = np.mean([rep[c]['precision'] for c in present_classes])
            recall_macro    = np.mean([rep[c]['recall']    for c in present_classes])
            macro_f1        = np.mean([rep[c]['f1-score']  for c in present_classes])

        row = {'Branch': branch, 'Model': f'Model_{name}',
               'Accuracy': round(acc, 4), 'Balanced_Accuracy': round(bacc, 4),
               'Macro_F1': round(macro_f1, 4),
               'Precision_macro': round(precision_macro, 4),
               'Recall_macro': round(recall_macro, 4),
               'Mean_Entropy': round(float(ent.mean()), 4),
               'N_objects': len(y_te),
               'N_classes_present': len(present_classes)}
        row.update(g_std)
        summary_rows.append(row)
        real_f1_dict[name] = per_class_f1(y_te, pred_cal)

    # ------------------------------------------------------------------
    # Helper: build meta_X from base model probabilities
    # ------------------------------------------------------------------
    def _build_meta_X(model_obj):
        parts = []
        for name, (prob_cal, class_names, y_te_model) in prob_dfs.items():
            cols   = [f'p_{c}_model{name}' for c in LABEL_ORDER]
            df_out = pd.DataFrame(PRIOR_PROB, index=y_true.index, columns=cols)
            shared = y_true.index.intersection(y_te_model.index)
            for j, cls in enumerate(LABEL_ORDER):
                idx = np.where(class_names == cls)[0]
                if len(idx) > 0:
                    df_out.loc[shared, f'p_{cls}_model{name}'] = \
                        prob_cal[y_te_model.index.get_indexer(shared), idx[0]]
            # Binary mask: 1 = real predictions, 0 = prior fill
            mask_col = f'has_model{name}'
            df_out[mask_col] = 0
            df_out.loc[shared, mask_col] = 1
            parts.append(df_out)
        mX = pd.concat(parts, axis=1)
        expected = (model_obj.feature_names_in_
                    if hasattr(model_obj, 'feature_names_in_')
                    else mX.columns)
        for col in expected:
            if col not in mX.columns:
                mX[col] = 0 if col.startswith('has_model') else PRIOR_PROB
        return mX[expected]

    # ------------------------------------------------------------------
    # Metamodel
    # ------------------------------------------------------------------
    meta_model = load_metamodel(models_dir, metamodel_filename)

    if meta_model is None:
        print('\nMetamodel not found, skipping.')
    elif not prob_dfs:
        print('\nNo base model probabilities available. Skipping metamodel.')
    else:
        print(f'\nEvaluating metamodel [{branch}] ({metamodel_filename})...')

        meta_X          = _build_meta_X(meta_model)
        meta_pred_proba = meta_model.predict_proba(meta_X)
        meta_pred       = meta_model.predict(meta_X)
        meta_classes    = np.array(meta_model.classes_)

        print_report(y_true, meta_pred, f'Metamodel [{branch}]')


        cm = metrics.confusion_matrix(y_true, meta_pred, labels=LABEL_ORDER)
        cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
        cm_df.index.name = 'true_class'
        cm_df.insert(0, 'branch',  branch)
        cm_df.insert(0, 'model', 'Metamodel')
        conf_matrix_rows.append(cm_df.reset_index())


        pd.DataFrame(meta_pred_proba, index=y_true.index,
                     columns=[f'p_{c}_meta' for c in meta_classes]
                     ).to_parquet(out_dir / f'probas_metamodel_{branch}.parquet')


        plot_metamodel_reliability(
            meta_pred_proba, y_true, LABEL_ORDER, meta_model,
            save_path=str(out_dir / f'reliability_metamodel_{branch}.pdf'),
        )


        meta_ent = compute_entropy(meta_pred_proba)
        plot_entropy_by_class(
            meta_ent, y_true, LABEL_ORDER,
            model_name=f'Metamodel [{branch}]',
            save_path=str(out_dir / f'entropy_metamodel_{branch}.pdf'),
        )
        df_ent_meta = entropy_summary_df(meta_ent, y_true, LABEL_ORDER)
        df_ent_meta.insert(0, 'branch',  branch)
        df_ent_meta.insert(0, 'model', 'Metamodel')
        entropy_records.append(df_ent_meta)


        plot_transient_proba_kde(
            meta_pred_proba, meta_classes, y_true,
            model_name=f'Metamodel [{branch}]',
            save_path=str(out_dir / f'transient_kde_metamodel_{branch}.pdf'),
        )


        meta_conf_df = compute_confidence_df(
            meta_pred_proba, meta_classes, y_true, y_true.index)
        conf_dfs['Meta'] = meta_conf_df
        meta_conf_df.to_parquet(out_dir / f'confidence_metamodel_{branch}.parquet')
        plot_confidence_distribution(
            meta_conf_df,
            model_name=f'Metamodel [{branch}]',
            save_path=str(out_dir / f'confidence_metamodel_{branch}.pdf'),
        )

 
        print(f'Bootstrap [{branch}] Metamodel (n={n_bootstrap})...')
        g_std_meta, pc_std_meta = bootstrap_metrics(
            y_true, meta_pred_proba, meta_classes, LABEL_ORDER, n_bootstrap,
            macro_labels=present_classes)
        pc_std_meta.insert(0, 'branch',  branch)
        pc_std_meta.insert(0, 'model', 'Metamodel')
        bootstrap_rows.append(pc_std_meta)


        for row in per_class_metrics(y_true, meta_pred):
            row.update({'model': 'Metamodel', 'branch': branch})
            per_class_rows.append(row)


        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='y_pred contains classes not in y_true')
            meta_acc  = metrics.accuracy_score(y_true, meta_pred)
            meta_bacc = metrics.balanced_accuracy_score(y_true, meta_pred)
            meta_rep = metrics.classification_report(
                y_true, meta_pred, labels=LABEL_ORDER,
                output_dict=True, zero_division=0)
            meta_precision_macro = np.mean([meta_rep[c]['precision'] for c in present_classes])
            meta_recall_macro    = np.mean([meta_rep[c]['recall']    for c in present_classes])
            meta_f1              = np.mean([meta_rep[c]['f1-score']  for c in present_classes])

        row = {'Branch': branch, 'Model': 'Metamodel',
               'Accuracy': round(meta_acc, 4),
               'Balanced_Accuracy': round(meta_bacc, 4),
               'Macro_F1': round(meta_f1, 4),
               'Precision_macro': round(meta_precision_macro, 4),
               'Recall_macro': round(meta_recall_macro, 4),
               'Mean_Entropy': round(float(meta_ent.mean()), 4),
               'N_objects': len(y_true),
               'N_classes_present': len(present_classes)}
        row.update(g_std_meta)
        summary_rows.append(row)
        real_f1_dict['Meta'] = per_class_f1(y_true, meta_pred)

        #Comparativa de metamodelos (--compare-metamodels)
        if compare_metamodel_filename is not None:
            meta_model_cmp = load_metamodel(models_dir, compare_metamodel_filename)
            if meta_model_cmp is None:
                print(f'[Compare] {compare_metamodel_filename} not found. Comparison skipped.')
            else:
                print(f'\n[Compare] Evaluating {compare_metamodel_filename}...')
                meta_X_cmp          = _build_meta_X(meta_model_cmp)
                meta_pred_proba_cmp = meta_model_cmp.predict_proba(meta_X_cmp)
                meta_pred_cmp       = meta_model_cmp.predict(meta_X_cmp)

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            message='y_pred contains classes not in y_true')
                    cmp_bacc = metrics.balanced_accuracy_score(y_true, meta_pred_cmp)
                    cmp_f1   = metrics.f1_score(y_true, meta_pred_cmp,
                                                average='macro', zero_division=0)
                    cmp_acc  = metrics.accuracy_score(y_true, meta_pred_cmp)

                pc_f1_main = per_class_f1(y_true, meta_pred)
                pc_f1_cmp  = per_class_f1(y_true, meta_pred_cmp)
                lbl_main   = metamodel_filename.replace('.pkl', '')
                lbl_cmp    = compare_metamodel_filename.replace('.pkl', '')

                print(f'\n{"="*70}')
                print(f'METAMODEL COMPARISON: {branch.upper()}')
                print(f'{"="*70}')
                print(f'{"Metric":<25} {lbl_main:>25} {lbl_cmp:>25}')
                print(f'{"-"*25} {"-"*25} {"-"*25}')
                print(f'{"Balanced Accuracy":<25} {meta_bacc:>25.4f} {cmp_bacc:>25.4f}')
                print(f'{"Macro F1":<25} {meta_f1:>25.4f} {cmp_f1:>25.4f}')
                print(f'{"Accuracy":<25} {meta_acc:>25.4f} {cmp_acc:>25.4f}')
                _delta_hdr = '\u0394'
                print(f'\n{"Class":<20} {lbl_main:>20} {lbl_cmp:>20} {_delta_hdr:>10}')
                print(f'  {"-"*20} {"-"*20} {"-"*20} {"-"*10}')
                _up, _dn = '\u25b2', '\u25bc'
                for cls in LABEL_ORDER:
                    f1_m   = pc_f1_main.get(cls, 0.0)
                    f1_c   = pc_f1_cmp.get(cls, 0.0)
                    delta  = f1_m - f1_c
                    marker = f' {_up}' if delta > 0.01 else (f' {_dn}' if delta < -0.01 else '')
                    print(f'{cls:<20} {f1_m:>20.4f} {f1_c:>20.4f} {delta:>+10.4f}{marker}')
                print(f'{"="*70}')

                # Guardar CSV con la comparativa
                cmp_rows = [
                    {'branch': branch, 'metric': 'Balanced_Accuracy',
                     lbl_main: round(meta_bacc, 4), lbl_cmp: round(cmp_bacc, 4),
                     'delta': round(meta_bacc - cmp_bacc, 4)},
                    {'branch': branch, 'metric': 'Macro_F1',
                     lbl_main: round(meta_f1, 4), lbl_cmp: round(cmp_f1, 4),
                     'delta': round(meta_f1 - cmp_f1, 4)},
                ]
                for cls in LABEL_ORDER:
                    f1_m = pc_f1_main.get(cls, 0.0)
                    f1_c = pc_f1_cmp.get(cls, 0.0)
                    cmp_rows.append({'branch': branch, 'metric': f'F1_{cls}',
                                     lbl_main: round(f1_m, 4),
                                     lbl_cmp:  round(f1_c, 4),
                                     'delta':  round(f1_m - f1_c, 4)})
                cmp_csv = out_dir / f'metamodel_comparison_{branch}.csv'
                import pandas as _pd
                _pd.DataFrame(cmp_rows).to_csv(cmp_csv, index=False)
                print(f'Comparison saved: {cmp_csv}')

    # ------------------------------------------------------------------
    # Bootstrap confusion matrices
    # ------------------------------------------------------------------
    try:
        from bootstrap_confmat import generate_bootstrap_confmats
        _meta_pp  = meta_pred_proba if meta_model is not None else None
        _meta_cls = meta_classes    if meta_model is not None else None
        generate_bootstrap_confmats(
            test_probas     = prob_dfs,
            meta_pred_proba = _meta_pp,
            meta_classes    = _meta_cls,
            yo_te           = y_true,
            label_order     = LABEL_ORDER,
            plots_dir       = str(out_dir) + '/',
            models_dir      = str(out_dir) + '/',
            tag             = branch,
            n_bootstrap     = n_bootstrap,
            random_state    = 42,
            hier_map        = HIER_MAP,
            save_pkl        = True,
        )
    except ImportError:
        print('[WARN] bootstrap_confmat.py not found'
              'Skipping bootstrap matrices.')

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------
    sheets = {
        'Summary':           pd.DataFrame(summary_rows),
        'Per_class_metrics': pd.DataFrame(per_class_rows),
        'Confusion_matrices': pd.concat(conf_matrix_rows, ignore_index=True)
                               if conf_matrix_rows else pd.DataFrame(),
        'Entropy_by_class':  pd.concat(entropy_records, ignore_index=True)
                               if entropy_records else pd.DataFrame(),
        'Bootstrap_std':     pd.concat(bootstrap_rows, ignore_index=True)
                               if bootstrap_rows else pd.DataFrame(),
    }

    # Print summary
    print(f'\n{"="*60}')
    print(f'SUMMARY [{branch}]')
    print(f'{"="*60}')
    print(sheets['Summary'].to_string(index=False))

    return real_f1_dict, conf_dfs, sheets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ensemble on real test set')
    p.add_argument('--features-dir', default='./data/real/',
                   help='Directory with features_{ztf,lsst,comb}_<branch>.parquet '
                        '(default: ./data/real/)')
    p.add_argument('--features-atlas-dir', default='./data/real/',
                   help='Directory with features_atlas_<branch>.parquet '
                        '(default: ./data/real/')
    p.add_argument('--labels', default='./data/real/labels_testset.csv',
                   help='Labels CSV/parquet for the real test set (default: ./data/real/labels_testset.csv)')
    p.add_argument('--models',
                   default='./output/models/',
                   help='Directory with trained models .pkl (default: ./output/models/)')
    p.add_argument('--output',
                   default='./output/real_eval/',
                   help='Output directory (default: ./output/real_eval/)')
    p.add_argument('--branches', nargs='+', default=['strict', 'relaxed'],
                   choices=['strict', 'relaxed'])
    p.add_argument('--bootstrap-n', type=int, default=1000)
    p.add_argument('--sim-summary',
                   default='./output/models/results_summary.xlsx',
                   help='results_summary.xlsx from baseline model (without domain adaptation)')

    p.add_argument('--adapted', action='store_true',
                   help='Evaluate adapted models (CORAL + feature selection). '
                        'Requires --consensus-csv and --coral-dir.')
    p.add_argument('--consensus-csv', default=None,
                   help='consensus_features.csv from select_features_coral.py')
    p.add_argument('--coral-dir', default=None,
                   help='Directory with coral_model*.pkl files from '
                        'model_training_adapted.py')
    p.add_argument('--compare-with', default=None,
                   help='Path to a results_summary_real.xlsx from a previous run '
                        '(baseline or adapted) to include in a side-by-side '
                        'comparison table at the end.')
    p.add_argument('--metamodel', default=None,
                   help='Filename of the metamodel pickle to load from --models dir '
                        '(default: meta_model.pkl). Use meta_model_baseline.pkl to '
                        'evaluate the pre-augmentation metamodel.')
    p.add_argument('--compare-metamodels', default=None,
                   metavar='FILENAME',
                   help='Second metamodel pickle filename to compare against --metamodel '
                        'in the same run (e.g. meta_model_baseline.pkl). Prints a '
                        'side-by-side table and saves metamodel_comparison_{branch}.csv.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    features_dir = Path(args.features_dir)
    atlas_dir    = Path(args.features_atlas_dir) if args.features_atlas_dir else None
    models_dir   = Path(args.models)
    out_dir      = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    branches        = args.branches
    n_bootstrap  = args.bootstrap_n

    #Adapted mode: load consensus
    consensus = None
    if args.adapted:
        if args.consensus_csv is None:
            raise ValueError('--adapted requires --consensus-csv')
        load_cons, _, _ = _load_adapted_utils()
        consensus = load_cons(args.consensus_csv)
        print(f'[Adapted] Consensus features loaded: {len(consensus)} entries')
        if args.coral_dir is None:
            print('[Adapted] --coral-dir not set; CORAL transforms will not be applied.')

    # Labels
    print('Loading labels...')
    df_labels = load_file(args.labels).replace([np.inf, -np.inf], np.nan)
    if 'class_original' not in df_labels.columns:
        if 'classALeRCE' in df_labels.columns:
            df_labels['class_original'] = df_labels['classALeRCE']
        else:
            raise ValueError("Labels file must have 'class_original' or "
                             "'classALeRCE' column.")
    labels_clean = df_labels.loc[
        df_labels['class_original'].isin(LABEL_ORDER), ['class_original']
    ].copy()
    print(f'Labeled objects in taxonomy: {len(labels_clean)}\n')

    # Accumulators for the combined Excel workbook (all branches)
    all_sheets: dict[str, list] = {
        'Summary': [], 'Per_class_metrics': [], 'Confusion_matrices': [],
        'Entropy_by_class': [], 'Bootstrap_std': [],
    }

    all_branch_f1  = {}
    all_conf_dfs = {}

    for branch in branches:
        print(f'\n{"▶"*4} Loading features: {branch} branch...')
        try:
            df_feat = load_features_for_branch(features_dir, atlas_dir, branch)
        except FileNotFoundError as e:
            print(f'[SKIP] {e}')
            continue

        df = labels_clean.join(df_feat, how='inner').replace([np.inf, -np.inf], np.nan)
        y_true          = df['class_original']
        df_feat_aligned = df.drop(columns=['class_original'])

        print(f'Objects with labels + features [{branch}]: {len(y_true)}')
        for cls in LABEL_ORDER:
            n = (y_true == cls).sum()
            if n > 0:
                print(f'  {cls:20s}: {n}')

        branch_f1, conf_dfs, sheets = evaluate_branch(
            branch, df_feat_aligned, y_true,
            models_dir, out_dir, n_bootstrap,
            adapted=args.adapted,
            consensus=consensus,
            coral_dir=args.coral_dir,
            metamodel_filename=args.metamodel or 'meta_model.pkl',
            compare_metamodel_filename=args.compare_metamodels,
        )
        all_branch_f1[branch]  = branch_f1
        all_conf_dfs[branch] = conf_dfs

        # Accumulate sheets across branches
        for sheet_name, df_sheet in sheets.items():
            if not df_sheet.empty:
                all_sheets[sheet_name].append(df_sheet)

    # ------------------------------------------------------------------
    # Write consolidated Excel workbook (all branches combined)
    # ------------------------------------------------------------------
    results_path = out_dir / 'results_summary_real.xlsx'
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        for sheet_name, dfs in all_sheets.items():
            if dfs:
                pd.concat(dfs, ignore_index=True).to_excel(
                    writer, sheet_name=sheet_name, index=False)
    print(f'\nNumeric results saved: {results_path}')
    print('Sheets: Summary | Per_class_metrics | Confusion_matrices | '
          'Entropy_by_class | Bootstrap_std')

    # CSV fallback (Summary sheet only)
    if all_sheets['Summary']:
        summary_csv = out_dir / 'model_comparison_summary_real.csv'
        pd.concat(all_sheets['Summary'], ignore_index=True).to_csv(
            summary_csv, index=False)
        print(f'CSV fallback: {summary_csv}')

    # ------------------------------------------------------------------
    # Cross-branch comparison
    # ------------------------------------------------------------------
    if len(all_branch_f1) >= 2:
        plot_branch_comparison(all_branch_f1,
                             str(out_dir / 'branch_comparison.pdf'))
        rows = []
        for branch, model_dict in all_branch_f1.items():
            for model_name, class_f1 in model_dict.items():
                for cls, f1 in class_f1.items():
                    rows.append({'branch': branch, 'model': model_name,
                                 'class': cls, 'f1': round(f1, 4)})
        pd.DataFrame(rows).to_csv(out_dir / 'branch_comparison.csv', index=False)
        print('Cross-branch F1 CSV saved: branch_comparison.csv')

        branches_list    = list(all_branch_f1.keys())
        t0, t1        = branches_list[0], branches_list[1]
        common_models = set(all_conf_dfs[t0]) & set(all_conf_dfs[t1])
        for model_name in sorted(common_models):
            plot_confidence_branch_comparison(
                conf_strict=all_conf_dfs[t0][model_name],
                conf_relaxed=all_conf_dfs[t1][model_name],
                model_name=f'Model {model_name}',
                save_path=str(out_dir / f'confidence_branch_cmp_{model_name}.pdf'),
            )

    # ------------------------------------------------------------------
    # Domain-gap comparison
    # ------------------------------------------------------------------
    if args.sim_summary is not None:
        sim_path = Path(args.sim_summary)
        if not sim_path.exists():
            print(f'\nSimulated summary not found at {sim_path}; '
                  'skipping domain-gap plot.')
        else:
            try:
                sim_pc = pd.read_excel(sim_path, sheet_name='Per_class_metrics')
                sim_f1_per_model = {}
                for model_name, grp in sim_pc.groupby('model'):
                    short = model_name.replace('Model_', '')
                    sim_f1_per_model[short] = dict(zip(grp['class'], grp['f1']))
                for branch, branch_f1 in all_branch_f1.items():
                    plot_domain_gap(
                        sim_f1_per_model, branch_f1,
                        save_path=str(out_dir / f'domain_gap_{branch}.pdf'),
                    )
            except Exception as e:
                print(f'\nFailed to load sim summary: {e}\n'
                      'Skipping domain-gap plot.')

    # ------------------------------------------------------------------
    # Optional comparison with another run (baseline vs adapted)
    # ------------------------------------------------------------------
    if args.compare_with is not None:
        cmp_path = Path(args.compare_with)
        if not cmp_path.exists():
            print(f'\n[Compare] File not found: {cmp_path}')
        else:
            try:
                df_other = pd.read_excel(cmp_path, sheet_name='Summary')
                if all_sheets['Summary']:
                    df_current = pd.concat(all_sheets['Summary'], ignore_index=True)
                    label_other  = 'baseline'
                    label_curr   = 'adapted' if args.adapted else 'current'
                    df_other['version']  = label_other
                    df_current['version'] = label_curr
                    compare = pd.concat([df_other, df_current], ignore_index=True)
                    cmp_cols = ['version', 'Branch', 'Model',
                                'Balanced_Accuracy', 'Macro_F1', 'Mean_Entropy']
                    cmp_cols = [c for c in cmp_cols if c in compare.columns]
                    print(f'\n{"="*70}')
                    print(f'COMPARISON: {label_other} vs {label_curr}')
                    print(f'{"="*70}')
                    print(compare[cmp_cols].sort_values(
                        ['Branch', 'Model']).to_string(
                        index=False, float_format='{:.4f}'.format))
                    cmp_out = out_dir / 'comparison_baseline_vs_adapted.csv'
                    compare[cmp_cols].to_csv(cmp_out, index=False)
                    print(f'\n  Comparison saved: {cmp_out}')
            except Exception as e:
                print(f'\n[Compare] Error reading {cmp_path}: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()