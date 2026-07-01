import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import shared utilities from model_training_functions
# ---------------------------------------------------------------------------

try:
    from model_training_functions import (
        LABEL_ORDER,
        N_CLASSES,
        PRIOR_PROB,
        get_ztf_cols,
        get_lsst_cols,
        get_atlas_cols,
        get_combined_cols,
        add_ztf_differential_features,
        add_atlas_differential_features,
        add_cross_survey_differential_features,
        predict_hierarchical,
        apply_calibrators,
        build_X,
    )
except ImportError as e:
    sys.exit(
        f"[ERROR] Cannot import from model_training_functions: {e}\n"
        "Make sure model_training_functions.py is on sys.path."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SURVEY_DEPS = {
    'A': ['ztf'],
    'B': ['lsst'],
    'C': ['ztf', 'lsst'],
    'D': ['ztf', 'lsst'],
    'E': ['atlas'],
}


MODEL_BUILD_KWARGS = {
    'A': dict(col_getter=get_ztf_cols,      add_ztf_diff=True),
    'B': dict(col_getter=get_lsst_cols),
    'C': dict(col_getter=None,              add_ztf_diff=True,
              add_atlas_diff=True, add_cross_diff=True),   # special case
    'D': dict(col_getter=get_combined_cols),
    'E': dict(col_getter=get_atlas_cols,    add_atlas_diff=True),
}


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------

def load_base_model(models_dir: str, name: str):
    """Load hierarchical RF + sub-classifiers for one base model."""
    hier_path = os.path.join(models_dir, f'model_{name}_hier.pkl')
    if not os.path.exists(hier_path):
        return None

    with open(hier_path, 'rb') as fh:
        clf_hier = pickle.load(fh)

    clfs = {}
    for group in ('Periodic', 'Stochastic', 'Transient'):
        grp_path = os.path.join(models_dir, f'model_{name}_{group}.pkl')
        if os.path.exists(grp_path):
            with open(grp_path, 'rb') as fh:
                clfs[group] = pickle.load(fh)

    return clf_hier, clfs


def load_calibrators(models_dir: str, name: str):
    """Load per-class calibrators for one base model."""
    cal_path = os.path.join(models_dir, f'calibrators_model{name}.pkl')
    if not os.path.exists(cal_path):
        return None
    with open(cal_path, 'rb') as fh:
        data = pickle.load(fh)
    return data['calibrators']   # dict  class → calibrator object


def load_meta_model(models_dir: str):
    """Load the LogisticRegression metamodel."""
    meta_path = os.path.join(models_dir, 'meta_model.pkl')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'rb') as fh:
        return pickle.load(fh)


def load_coral_params(coral_dir: str, name: str) -> dict | None:
    """
    Load CORAL transformation parameters saved by model_training_adapted.py.
    """
    if coral_dir is None:
        return None

    if name == "C":
        suffixes = ["ztf", "lsst", "diff", "atlas"]
        parts = []
        for suf in suffixes:
            path = os.path.join(coral_dir, f"coral_modelC_{suf}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    parts.append(pickle.load(fh))
                print(f" Loaded CORAL params for Model C ({suf}): "
                      f"{len(parts[-1]['features'])} features")
        if not parts:
            return None
        import scipy.linalg
        all_features, all_mu_s, all_mu_t, all_W = [], [], [], []
        for p in parts:
            all_features.extend(p["features"])
            all_mu_s.append(p["mu_s"])
            all_mu_t.append(p["mu_t"])
            all_W.append(p["W"])
        return {
            "features": all_features,
            "mu_s":     np.concatenate(all_mu_s),
            "mu_t":     np.concatenate(all_mu_t),
            "W":        scipy.linalg.block_diag(*all_W),
        }
    else:
        path = os.path.join(coral_dir, f"coral_model{name}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)


def load_artifacts(models_dir: str, active_models: list, use_meta: bool):
    """Load all required pkl artefacts and return them in a dict."""
    artifacts = {'base': {}, 'calibrators': {}, 'meta': None, 'coral': {}}

    for name in active_models:
        bm = load_base_model(models_dir, name)
        if bm is None:
            print(f"[WARN] Model {name}: pkl not found, skipping.")
            continue
        artifacts['base'][name] = bm
        print(f"Loaded Model {name}")

        cal = load_calibrators(models_dir, name)
        if cal is None:
            print(f"[WARN] No calibrators for Model {name}: raw probs will be used.")
        else:
            artifacts['calibrators'][name] = cal

    if use_meta:
        meta = load_meta_model(models_dir)
        if meta is None:
            print("[WARN] meta_model.pkl not found: skipping metamodel.")
        else:
            artifacts['meta'] = meta
            print(f"  Loaded meta-model")

    return artifacts


# ---------------------------------------------------------------------------
# Feature ingestion
# ---------------------------------------------------------------------------

def load_parquet_survey(path: str | None, survey_name: str) -> pd.DataFrame | None:
    """Read a feature parquet; return None if path is not provided."""
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[WARN] {survey_name} parquet not found at {path}")
        return None
    df = pd.read_parquet(path)
    if 'survey' in df.columns:
        df = df.drop(columns=['survey'])
    df.index = df.index.astype(str)
    return df


def merge_survey_frames(ztf_df, lsst_df, atlas_df, comb_df=None) -> pd.DataFrame:
    """
    Outer-join all available survey DataFrames on their index (oid_combined).
    Objects missing from a survey will have NaN in those survey's columns.
    """
    frames = [f for f in (ztf_df, lsst_df, atlas_df, comb_df) if f is not None]
    if not frames:
        raise ValueError("No survey feature DataFrames available.")
    merged = frames[0]
    for other in frames[1:]:
        merged = merged.join(other, how='outer', rsuffix='_dup')
        dup_cols = [c for c in merged.columns if c.endswith('_dup')]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)
    return merged


# ---------------------------------------------------------------------------
# Per-object coverage bookkeeping
# ---------------------------------------------------------------------------

def get_available_surveys(df: pd.DataFrame) -> dict:
    ztf_cols_present   = [c for c in df.columns if c.endswith('_ztf')]
    lsst_cols_present  = [c for c in df.columns if c.endswith('_lsst')]
    atlas_cols_present = [c for c in df.columns if c.endswith('_atlas')]

    def has_survey(cols):
        if not cols:
            return pd.Series(False, index=df.index)
        return df[cols].notna().any(axis=1)

    avail = pd.DataFrame({
        'ztf':   has_survey(ztf_cols_present),
        'lsst':  has_survey(lsst_cols_present),
        'atlas': has_survey(atlas_cols_present),
    })
    return avail


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _build_X_for_model(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Construct the feature matrix for one base model using build_X."""
    kwargs = MODEL_BUILD_KWARGS[name]

    if name == 'C':
        ztf_c   = get_ztf_cols(df.columns)
        lsst_c  = get_lsst_cols(df.columns)
        atlas_c = get_atlas_cols(df.columns)
        col_list = ztf_c + lsst_c + atlas_c
        has_atlas = len(atlas_c) > 0
        return build_X(df, col_list,
                       add_ztf_diff=True,
                       add_atlas_diff=has_atlas,
                       add_cross_diff=True)
    else:
        col_getter = kwargs['col_getter']
        col_list   = col_getter(df.columns)
        bkw = {k: v for k, v in kwargs.items() if k != 'col_getter'}
        return build_X(df, col_list, **bkw)


def _uniform_prior_df(index: pd.Index) -> pd.DataFrame:
    """Return a DataFrame of uniform priors (1/N_CLASSES) for given index."""
    return pd.DataFrame(
        np.full((len(index), N_CLASSES), PRIOR_PROB),
        index=index,
        columns=LABEL_ORDER,
    )


def run_base_model(name: str, df_full: pd.DataFrame,
                   clf_hier, clfs, calibrators,
                   coral_params: dict | None = None) -> pd.DataFrame:
    """
    Run one base model on every object that has the required survey columns.
    Objects without features for this model receive uniform priors.

    """

    avail = get_available_surveys(df_full)
    deps  = MODEL_SURVEY_DEPS[name]

    has_features = pd.Series(True, index=df_full.index)
    for survey in deps:
        if survey in avail.columns:
            has_features = has_features & avail[survey]
        else:
            has_features[:] = False 

    n_avail = has_features.sum()
    n_total = len(df_full)
    print(f"Model {name}: {n_avail}/{n_total} objects have required features")

    result = _uniform_prior_df(df_full.index)

    if n_avail == 0:
        print(f"Model {name}: no objects to classify, returning uniform prior")
        return result

    df_sub = df_full.loc[has_features]

    # Build feature matrix
    try:
        X = _build_X_for_model(name, df_sub)
    except Exception as exc:
        print(f"[ERROR] build_X failed for Model {name}: {exc}. Skipping")
        return result

    if hasattr(clf_hier, 'feature_names_in_'):
        expected = list(clf_hier.feature_names_in_)
        missing  = [c for c in expected if c not in X.columns]
        if missing:
            print(f"[WARN] Model {name}: {len(missing)} expected features missing, "
                  f"filling with NaN (e.g. {missing[:3]})")
        X = X.reindex(columns=expected, fill_value=np.nan)

    X = X.fillna(-999).clip(lower=-1e6, upper=1e6)

    if coral_params is not None:
        feats = [f for f in coral_params['features'] if f in X.columns]
        if feats:
            X_vals = X[feats].values.astype(float)
            X_vals = (X_vals - coral_params['mu_s']) @ coral_params['W'] + coral_params['mu_t']
            X = X.copy()
            X[feats] = X_vals
            print(f"Model {name}: CORAL applied to {len(feats)} features")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prob_matrix, class_names = predict_hierarchical(X, clf_hier, clfs)

    if calibrators:
        prob_matrix = apply_calibrators(prob_matrix, class_names, calibrators)

    prob_df = pd.DataFrame(prob_matrix, index=df_sub.index, columns=class_names)
    prob_df = prob_df.reindex(columns=LABEL_ORDER, fill_value=0.0)

    result.loc[has_features, LABEL_ORDER] = prob_df.values

    return result


# ---------------------------------------------------------------------------
# Confidence metrics
# ---------------------------------------------------------------------------

def compute_entropy(prob_matrix: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits.  0 = perfectly certain, log2(N) = uniform."""
    eps = 1e-12
    p = np.clip(prob_matrix, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def classify(
    models_dir: str,
    active_models: list,
    use_meta: bool,
    ztf_path:      str | None,
    lsst_path:     str | None,
    atlas_path:    str | None,
    combined_path: str | None = None,
    output_path:   str = 'classify_output',
    coral_dir:     str | None = None,
):
    print("\n=== Loading artefacts ===")
    arts = load_artifacts(models_dir, active_models, use_meta)

    if coral_dir is not None:
        for name in arts['base']:
            cp = load_coral_params(coral_dir, name)
            if cp is not None:
                arts['coral'][name] = cp
                print(f"Loaded CORAL params for Model {name} "
                      f"({len(cp['features'])} features)")
            else:
                print(f"[INFO] No CORAL params for Model {name}: no transform applied")

    loaded_models = list(arts['base'].keys())
    if not loaded_models:
        sys.exit("[ERROR] No base models could be loaded. Check --models-dir.")

    print("\n=== Loading feature parquets ===")
    ztf_df   = load_parquet_survey(ztf_path,      'ZTF')
    lsst_df  = load_parquet_survey(lsst_path,     'LSST')
    atlas_df = load_parquet_survey(atlas_path,    'ATLAS')
    comb_df  = load_parquet_survey(combined_path, 'Combined')

    print("\n=== Merging survey features ===")
    df_full = merge_survey_frames(ztf_df, lsst_df, atlas_df, comb_df)
    df_full = df_full.replace([np.inf, -np.inf], np.nan)
    print(f"  Total objects: {len(df_full)}")

    avail_surveys = get_available_surveys(df_full)
    n_models_used = pd.Series(0, index=df_full.index)
    model_had_features: dict[str, pd.Series] = {}

    print("\n=== Running base models ===")
    base_proba: dict[str, pd.DataFrame] = {}

    for name in loaded_models:
        clf_hier, clfs = arts['base'][name]
        calibrators    = arts['calibrators'].get(name, None)

        print(f"  --- Model {name} ---")
        coral_params = arts['coral'].get(name, None)
        prob_df = run_base_model(name, df_full, clf_hier, clfs, calibrators,
                                 coral_params=coral_params)
        base_proba[name] = prob_df

        deps = MODEL_SURVEY_DEPS[name]
        has_feat = pd.Series(True, index=df_full.index)
        for survey in deps:
            if survey in avail_surveys.columns:
                has_feat = has_feat & avail_surveys[survey]
            else:
                has_feat[:] = False
        n_models_used += has_feat.astype(int)
        model_had_features[name] = has_feat

    # ---------------------------------------------------------------------------
    # Final probabilities: metamodel OR mean of active base models
    # ---------------------------------------------------------------------------
    print("\n=== Computing final probabilities ===")

    if use_meta and arts['meta'] is not None:
        meta_parts = []
        all_expected = list(MODEL_SURVEY_DEPS.keys())

        for name in all_expected:
            if name in loaded_models:
                prob_df = base_proba[name]
                renamed = prob_df.copy()
                renamed.columns = [f'p_{cl}_model{name}' for cl in LABEL_ORDER]
                avail = get_available_surveys(df_full)
                deps  = MODEL_SURVEY_DEPS[name]
                has_feat = pd.Series(True, index=df_full.index)
                for survey in deps:
                    if survey in avail.columns:
                        has_feat = has_feat & avail[survey]
                    else:
                        has_feat[:] = False
                renamed[f'has_model{name}'] = has_feat.astype(int).values
                meta_parts.append(renamed)
            else:
                prior_df = _uniform_prior_df(df_full.index)
                prior_df.columns = [f'p_{cl}_model{name}' for cl in LABEL_ORDER]
                prior_df[f'has_model{name}'] = 0
                meta_parts.append(prior_df)

        meta_X = pd.concat(meta_parts, axis=1)

        meta_model_feats = list(arts['meta'].feature_names_in_) \
            if hasattr(arts['meta'], 'feature_names_in_') else list(meta_X.columns)
        fill_vals = {c: (0 if c.startswith('has_model') else PRIOR_PROB)
                     for c in meta_model_feats}
        meta_X = meta_X.reindex(columns=meta_model_feats)
        for col, fill in fill_vals.items():
            meta_X[col] = meta_X[col].fillna(fill)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            final_proba = arts['meta'].predict_proba(meta_X)

        final_class_names = list(arts['meta'].classes_)
        final_df = pd.DataFrame(final_proba, index=df_full.index,
                                columns=final_class_names)
        final_df = final_df.reindex(columns=LABEL_ORDER, fill_value=0.0)
        print("  Used meta-model for final probabilities.")

    else:
        stack = np.stack([base_proba[n].values for n in loaded_models], axis=0)
        mask_list = []
        for name in loaded_models:
            deps = MODEL_SURVEY_DEPS[name]
            has_feat = pd.Series(True, index=df_full.index)
            for survey in deps:
                if survey in avail_surveys.columns:
                    has_feat = has_feat & avail_surveys[survey]
                else:
                    has_feat[:] = False
            mask_list.append(has_feat.values)
        mask = np.array(mask_list)                  
        counts = mask.sum(axis=0, keepdims=True).T
        weighted = (stack * mask[:, :, np.newaxis]).sum(axis=0)
        has_any = (counts > 0).squeeze()
        final_proba = np.full_like(weighted, PRIOR_PROB)
        final_proba[has_any] = weighted[has_any] / counts[has_any]
        final_df = pd.DataFrame(final_proba, index=df_full.index,
                                columns=LABEL_ORDER)
        n_full = (counts.squeeze() == len(loaded_models)).sum()
        n_partial = (has_any & (counts.squeeze() < len(loaded_models))).sum()
        n_none = (~has_any).sum()
        print(f"Per-object average of {len(loaded_models)} base model(s):")
        print(f"All models available: {n_full} objects")
        print(f"Partial coverage:     {n_partial} objects")
        print(f"No models available:  {n_none} objects (uniform prior)")

    # ---------------------------------------------------------------------------
    # Build output DataFrame
    # ---------------------------------------------------------------------------
    print("\n=== Building output ===")

    out = pd.DataFrame(index=df_full.index)

    # Prediction + confidence
    out['predicted_class'] = final_df.idxmax(axis=1)
    out['max_prob']         = final_df.max(axis=1)
    out['entropy']          = compute_entropy(final_df.values)
    out['n_models_used']    = n_models_used

    # Second-best class and its probability
    def _second_best(row):
        top2 = row.nlargest(2)
        if len(top2) < 2:
            return pd.Series({'second_class': None, 'second_prob': np.nan})
        return pd.Series({'second_class': top2.index[1], 'second_prob': top2.iloc[1]})

    second = final_df.apply(_second_best, axis=1)
    out['second_class'] = second['second_class']
    out['second_prob']  = second['second_prob'].astype(float)

    # Final probability per class
    for cl in LABEL_ORDER:
        out[f'p_{cl}'] = final_df[cl].values

    # Per-model calibrated probabilities (for inspection).
    for name in loaded_models:
        had_feat = model_had_features[name]
        for cl in LABEL_ORDER:
            col = f'p_{cl}_model{name}'
            vals = base_proba[name][cl].copy()
            vals[~had_feat] = np.nan
            out[col] = vals

    # Per-model entropy (NaN where model had no features)
    for name in loaded_models:
        had_feat = model_had_features[name]
        prob_cols = [f'p_{cl}_model{name}' for cl in LABEL_ORDER]
        ent = pd.Series(np.nan, index=df_full.index)
        if had_feat.any():
            ent[had_feat] = compute_entropy(
                base_proba[name].loc[had_feat].values
            )
        out[f'entropy_model{name}'] = ent

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base_path = os.path.splitext(output_path)[0]
    parquet_path = base_path + '.parquet'
    csv_path     = base_path + '.csv'
    out.to_parquet(parquet_path)
    out.index = out.index.astype(str)
    out.to_csv(csv_path)
    print(f"\nResults saved -> {parquet_path}")
    print(f"             -> {csv_path}")
    print(f"Objects classified: {len(out)}")
    print("\nClass distribution (predicted_class):")
    print(out['predicted_class'].value_counts().to_string())
    print(f"\nMean entropy: {out['entropy'].mean():.3f} bits "
          f"(max possible = {np.log2(N_CLASSES):.3f})")

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Classify unlabeled objects with the multi-survey stacking ensemble."
    )
    p.add_argument('--models-dir', required=True,
                   help="Directory containing model_*.pkl, calibrators_*.pkl, meta_model.pkl")
    p.add_argument('--ztf',   default=None, help="Path to ZTF feature parquet")
    p.add_argument('--lsst',  default=None, help="Path to LSST feature parquet")
    p.add_argument('--atlas',    default=None, help="Path to ATLAS feature parquet")
    p.add_argument('--combined', default=None, help="Path to combined (ZTF+LSST) feature parquet (for Model D)")
    p.add_argument('--active-models', nargs='+', default=['A', 'B', 'C', 'D', 'E'],
                   choices=['A', 'B', 'C', 'D', 'E'],
                   help="Which base models to use (default: all five)")
    p.add_argument('--coral-dir', default=None,
                   help="Directory containing coral_model{X}.pkl files from "
                        "model_training_adapted.py (optional, omit for non-adapted models)")
    p.add_argument('--use-meta', action='store_true',
                   help="Use the metamodel for the final prediction (default: average)")
    p.add_argument('--output', required=True,
                   help="Output parquet path for classification results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Direct-run block (Spyder / IPython)
# ---------------------------------------------------------------------------

DIRECT_RUN = True   # False para usar desde terminal con argparse

if DIRECT_RUN:
    MODELS_DIR    = './output/models/'
    # Carpeta con las features de los objetos a clasificar
    DATA_DIR      = './data/unlabeled/'
    ZTF_PARQUET   = DATA_DIR + 'features_ztf_relaxed.parquet'   # None si no disponible
    LSST_PARQUET  = DATA_DIR + 'features_lsst_relaxed.parquet'  # None si no disponible
    ATLAS_PARQUET = DATA_DIR + 'features_atlas_relaxed.parquet'
    COMB_PARQUET  = DATA_DIR + 'features_comb_relaxed.parquet'
    ACTIVE_MODELS = ['A','B','C','D','E']
    USE_META      = True
    CORAL_DIR     = './output/coral/'
    OUTPUT_PATH   = './output/classify/classify_results.parquet'
    results = classify(
        models_dir    = MODELS_DIR,
        active_models = ACTIVE_MODELS,
        use_meta      = USE_META,
        ztf_path      = ZTF_PARQUET,
        lsst_path     = LSST_PARQUET,
        atlas_path    = ATLAS_PARQUET,
        combined_path = COMB_PARQUET,
        output_path   = OUTPUT_PATH,
        coral_dir     = CORAL_DIR,
    )


if __name__ == '__main__' and not DIRECT_RUN:
    args = _parse_args()
    classify(
        models_dir    = args.models_dir,
        active_models = args.active_models,
        use_meta      = args.use_meta,
        ztf_path      = args.ztf,
        lsst_path     = args.lsst,
        atlas_path    = args.atlas,
        combined_path = args.combined,
        output_path   = args.output,
        coral_dir     = args.coral_dir,
    )