import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import shap

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_training_functions as mtf

#%%

# ============================================================
# CONFIGURCIÓN
# ============================================================

labels_file             = './data/simulated/labels.csv'
features_ztf_file       = './data/simulated/features_ztf.parquet'
features_lsst_file      = './data/simulated/features_lsst.parquet'
features_atlas_file     = './data/simulated/features_atlas.parquet'
features_combined_file  = './data/simulated/features_comb.parquet'

# Rutas de salida
OUTPUT_DIR = './output/'
OOF_DIR    = os.path.join(OUTPUT_DIR, 'oof')    + '/'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models') + '/'
PLOTS_DIR  = os.path.join(OUTPUT_DIR, 'plots')  + '/'

# ── Model dropout para robustez del metamodelo ──────────────────────────────
# Durante entrenamiento del metamodelo, con probabilidad META_DROPOUT_PROB se
# dropea cada SURVEY (ZTF, LSST, ATLAS) y se propaga la ausencia a los
# modelos dependientes.  Esto enseña al metamodelo que "modelo sin datos" es
# un patrón válido de entrada y que debe funcionar con cualquier combinación
# de surveys (solo ZTF, solo LSST, solo ZTF+ATLAS, etc.).
# Dependencias: A-ZTF, B-LSST, C-ZTF+LSST, D-ZTF+LSST, E-ATLAS.
META_DROPOUT_PROB = 0.3   # 0.0 para desactivar


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OOF_DIR,    exist_ok=True)

    print("Loading features...")

    df_ztf      = mtf._load_features(features_ztf_file)
    df_lsst     = mtf._load_features(features_lsst_file)
    df_combined = mtf._load_features(features_combined_file)

    # ATLAS is optional: set features_atlas_file = None to skip Models C (ATLAS) and E
    if features_atlas_file is not None:
        if Path(features_atlas_file).exists():
            df_atlas_feat = mtf._load_features(features_atlas_file)
        else:
            print(f"  Warning: ATLAS features file not found ({features_atlas_file}), skipping.")
            df_atlas_feat = None
    else:
        df_atlas_feat = None



    all_ztf_cols      = list(df_ztf.columns)
    all_lsst_cols     = list(df_lsst.columns)
    all_combined_cols = list(df_combined.columns)

    ztf_cols      = mtf.get_ztf_cols(all_ztf_cols)
    lsst_cols     = mtf.get_lsst_cols(all_lsst_cols)
    combined_cols = mtf.get_combined_cols(all_combined_cols)

    if df_atlas_feat is not None:
        all_atlas_cols = list(df_atlas_feat.columns)
        atlas_cols     = mtf.get_atlas_cols(all_atlas_cols)
    else:
        atlas_cols = []

    print(f"  ZTF features:      {len(ztf_cols)}")
    print(f"  LSST features:     {len(lsst_cols)}")
    print(f"  ATLAS features:    {len(atlas_cols)}")
    print(f"  Combined features: {len(combined_cols)}")


    df_feat = (
        df_ztf[ztf_cols]
        .join(df_lsst[lsst_cols],         how='inner')
        .join(df_combined[combined_cols],  how='inner')
    )

    if df_atlas_feat is not None:
        df_feat = df_feat.join(df_atlas_feat[atlas_cols], how='left')

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

    print(f"\nFeature matrix shape after join: {df_feat.shape}")

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    print("Loading labels...")
    p_labels = Path(labels_file)
    if p_labels.suffix == '.parquet':
        df_labels = pd.read_parquet(p_labels)
    else:
        df_labels = pd.read_csv(p_labels, index_col='oid')

    if 'class_original' not in df_labels.columns:
        df_labels['class_original'] = df_labels['classALeRCE']

    labels = df_labels.loc[df_labels.class_original.isin(mtf.LABEL_ORDER),
                           ['class_original']].copy()

    all_cols = list(df_feat.columns)
    
    df = labels.join(df_feat, how='inner')
    df = df.replace([np.inf, -np.inf], np.nan)

    Y_original     = df['class_original']
    Y_hierarchical = mtf.make_hierarchical_labels(Y_original)

    print(f"\nTotal labeled objects (ZTF+LSST+combined): {len(Y_original)}")
    for cls in mtf.LABEL_ORDER:
        n = (Y_original == cls).sum()
        print(f"  {cls:20s}: {n}")


    if atlas_cols:
        atlas_mask  = df[atlas_cols].notna().any(axis=1)
        df_atlas    = df.loc[atlas_mask].copy()
        Y_atlas     = Y_original.loc[atlas_mask]
        Yh_atlas    = mtf.make_hierarchical_labels(Y_atlas)
        print(f"\nObjects with ATLAS data: {atlas_mask.sum()} "
              f"({atlas_mask.mean()*100:.1f}% of ZTF+LSST set)")
        for cls in mtf.LABEL_ORDER:
            n = (Y_atlas == cls).sum()
            if n > 0:
                print(f"  {cls:20s}: {n}")
    else:
        print("\nNo ATLAS features found. Models C (ATLAS-expanded) and E will be skipped.")
        df_atlas = None
        Y_atlas  = None

    # ============================================================
    # MATRICES DE FEATURES POR MODELO
    # ============================================================

    # Modelo A
    X_A = mtf.build_X(df, ztf_cols, add_ztf_diff=True)

    # Modelo B
    X_B = mtf.build_X(df, lsst_cols)

    # Modelo C
    if df_atlas is not None and atlas_cols:
        X_C = mtf.build_X(df, ztf_cols + lsst_cols + atlas_cols,
                      add_ztf_diff=True, add_atlas_diff=True, add_cross_diff=True)
        Y_C  = Y_original
        Yh_C = Y_hierarchical
    else:
        X_C = mtf.build_X(df, ztf_cols + lsst_cols,
                      add_ztf_diff=True, add_cross_diff=True)
        Y_C  = Y_original
        Yh_C = Y_hierarchical

    # Modelo D
    X_D = mtf.build_X(df, combined_cols)

    # Modelo E
    if df_atlas is not None and atlas_cols:
        X_E = mtf.build_X(df_atlas, atlas_cols, add_atlas_diff=True)
        Y_E  = Y_atlas
        Yh_E = Yh_atlas
    else:
        X_E = None

    print(f"\nFeature counts:")
    print(f"  Model A (ZTF): {X_A.shape[1]:4d}  ({len(X_A)} objects)")
    print(f"  Model B (LSST): {X_B.shape[1]:4d}  ({len(X_B)} objects)")
    print(f"  Model C (ZTF+LSST+ATLAS): {X_C.shape[1]:4d}  ({len(X_C)} objects)")
    print(f"  Model D (combined): {X_D.shape[1]:4d}  ({len(X_D)} objects)")
    if X_E is not None:
        print(f"  Model E (ATLAS): {X_E.shape[1]:4d}  ({len(X_E)} objects)")

    # TRAIN / TEST SPLIT

    split_main = model_selection.train_test_split(
        X_A, X_B, X_C, X_D, Y_original,
        test_size=0.2,
        stratify=Y_original,
        random_state=mtf.RANDOM_STATE
    )
    XA_tr, XA_te = split_main[0], split_main[1]
    XB_tr, XB_te = split_main[2], split_main[3]
    XC_tr, XC_te = split_main[4], split_main[5]
    XD_tr, XD_te = split_main[6], split_main[7]
    yo_tr, yo_te  = split_main[8], split_main[9]
    yh_tr = mtf.make_hierarchical_labels(yo_tr)

    print(f"\nSplit (ZTF+LSST): {len(yo_tr)} train | {len(yo_te)} test")

    if X_E is not None:

        atlas_tr_idx = yo_tr.index.intersection(df_atlas.index)
        atlas_te_idx = yo_te.index.intersection(df_atlas.index)

        XE_tr   = X_E.loc[atlas_tr_idx]
        XE_te   = X_E.loc[atlas_te_idx]
        yoE_tr  = yo_tr.loc[atlas_tr_idx]
        yoE_te  = yo_te.loc[atlas_te_idx]
        yhE_tr  = mtf.make_hierarchical_labels(yoE_tr)

        print(f"ATLAS subset: {len(yoE_tr)} train | {len(yoE_te)} test "
              f"({len(yoE_tr)/len(yo_tr)*100:.1f}% / {len(yoE_te)/len(yo_te)*100:.1f}% "
              f"of main split)")
    else:
        XE_tr = XE_te = yoE_tr = yoE_te = yhE_tr = None

    # ============================================================
    # PHASE 1: OOF PREDICTIONS + CALIBRATION
    # ============================================================

    print("\n=== PHASE 1: OOF predictions + calibration ===")

    oof_A = mtf.compute_oof_predictions(XA_tr, yo_tr,  'modelA')
    oof_B = mtf.compute_oof_predictions(XB_tr, yo_tr,  'modelB')
    oof_C = mtf.compute_oof_predictions(XC_tr, yo_tr,  'modelC')
    oof_D = mtf.compute_oof_predictions(XD_tr, yo_tr,  'modelD')

    if XE_tr is not None:
        oof_E = mtf.compute_oof_predictions(XE_tr, yoE_tr, 'modelE')
    else:
        oof_E = None

    calibrators = {}

    for model_name, oof_df, y_labels in [
        ('A', oof_A, yo_tr),
        ('B', oof_B, yo_tr),
        ('C', oof_C, yo_tr),
        ('D', oof_D, yo_tr),
    ]:
        print(f"\nCalibrating Model {model_name}:")
        oof_proba, _ = mtf._oof_to_proba_df(oof_df, model_name, mtf.LABEL_ORDER)
        cal_dict, method_used = mtf.fit_calibrators(oof_proba, y_labels, mtf.LABEL_ORDER, method='auto')
        calibrators[model_name] = cal_dict

        mtf.plot_calibration_curves(
            oof_proba, y_labels, cal_dict, mtf.LABEL_ORDER,
            model_name=f'Model_{model_name}',
            save_path=f'{PLOTS_DIR}reliability_model{model_name}.pdf'
        )
        with open(f'{MODELS_DIR}calibrators_model{model_name}.pkl', 'wb') as f:
            pickle.dump({'calibrators': cal_dict,
                         'method': method_used,
                         'label_order': mtf.LABEL_ORDER}, f, pickle.HIGHEST_PROTOCOL)
        print(f"Calibrators saved: {MODELS_DIR}calibrators_model{model_name}.pkl")

    if oof_E is not None:
        print(f"\nCalibrating Model E:")
        oof_proba_E, _ = mtf._oof_to_proba_df(oof_E, 'E', mtf.LABEL_ORDER)
        cal_dict_E, method_E = mtf.fit_calibrators(oof_proba_E, yoE_tr, mtf.LABEL_ORDER, method='auto')
        calibrators['E'] = cal_dict_E
        mtf.plot_calibration_curves(
            oof_proba_E, yoE_tr, cal_dict_E, mtf.LABEL_ORDER,
            model_name='Model_E',
            save_path=f'{PLOTS_DIR}reliability_modelE.pdf'
        )
        with open(f'{MODELS_DIR}calibrators_modelE.pkl', 'wb') as f:
            pickle.dump({'calibrators': cal_dict_E,
                         'method': method_E,
                         'label_order': mtf.LABEL_ORDER}, f, pickle.HIGHEST_PROTOCOL)
        print(f"Calibrators saved: {MODELS_DIR}calibrators_modelE.pkl")

    #Apply calibrators to OOF predictions

    def _calibrate_oof(oof_df, model_name, calibrators):
        oof_cal = oof_df.copy()
        prob_cols = [c for c in oof_df.columns if f'_model{model_name}' in c]
        cls_names = np.array([c.replace('p_', '').replace(f'_model{model_name}', '')
                              for c in prob_cols])
        prob_matrix = oof_df[prob_cols].values
        prob_cal = mtf.apply_calibrators(prob_matrix, cls_names, calibrators[model_name])
        for j, col in enumerate(prob_cols):
            oof_cal[col] = prob_cal[:, j]
        return oof_cal

    oof_A_cal = _calibrate_oof(oof_A, 'A', calibrators)
    oof_B_cal = _calibrate_oof(oof_B, 'B', calibrators)
    oof_C_cal = _calibrate_oof(oof_C, 'C', calibrators)
    oof_D_cal = _calibrate_oof(oof_D, 'D', calibrators)

    # Build the OOF matrix for the metamodel.
    def _reindex_oof(oof_cal, ref_index, model_name, fill_value=mtf.PRIOR_PROB):
        cols = [f'p_{c}_model{model_name}' for c in mtf.LABEL_ORDER]
        df_out = pd.DataFrame(fill_value, index=ref_index, columns=cols)
        shared = oof_cal.index.intersection(ref_index)
        df_out.loc[shared, cols] = oof_cal.loc[shared, cols].values
        # Binary mask: 1 = real predictions, 0 = prior fill
        mask_col = f'has_model{model_name}'
        df_out[mask_col] = 0
        df_out.loc[shared, mask_col] = 1
        return df_out

    meta_index  = yo_tr.index
    oof_A_meta  = _reindex_oof(oof_A_cal, meta_index, 'A')
    oof_B_meta  = _reindex_oof(oof_B_cal, meta_index, 'B')
    oof_C_meta  = _reindex_oof(oof_C_cal, meta_index, 'C')
    oof_D_meta  = _reindex_oof(oof_D_cal, meta_index, 'D')

    oof_parts = [oof_A_meta, oof_B_meta, oof_C_meta, oof_D_meta]

    if oof_E is not None:
        oof_E_cal  = _calibrate_oof(oof_E, 'E', calibrators)
        oof_E_meta = _reindex_oof(oof_E_cal, meta_index, 'E')
        oof_parts.append(oof_E_meta)

    oof_all = pd.concat(oof_parts, axis=1)
    oof_all['true_label'] = yo_tr
    oof_all.to_csv(f'{OOF_DIR}oof_all_models_calibrated.csv')
    n_mask = sum(1 for c in oof_all.columns if c.startswith('has_model'))
    n_prob = oof_all.shape[1] - 1 - n_mask
    print(f"\nCalibrated OOF saved ({n_prob} probability features + "
          f"{n_mask} availability binary flags = "
          f"{len(oof_parts)} models x {mtf.N_CLASSES} classes + masks, "
          f"{len(oof_all)} objects)")

    # ============================================================
    # PHASE 2: RETRAIN BASE MODELS ON FULL TRAINING SET
    # ============================================================
    print("\n=== PHASE 2: Retraining base models on full training set ===")

    # (model_name, X_train, y_orig_train, y_hier_train)
    retrain_specs = [
        ('A', XA_tr, yo_tr,  yh_tr),
        ('B', XB_tr, yo_tr,  yh_tr),
        ('C', XC_tr, yo_tr,  yh_tr),
        ('D', XD_tr, yo_tr,  yh_tr),
    ]
    if XE_tr is not None:
        retrain_specs.append(('E', XE_tr, yoE_tr, yhE_tr))

    models = {}
    for name, X_tr, yo, yh in retrain_specs:
        print(f"  Training Model {name}...")
        clf_hier, clfs = mtf.train_hierarchical_model(X_tr, yh, yo)
        models[name] = (clf_hier, clfs)
        with open(f'{MODELS_DIR}model_{name}_hier.pkl', 'wb') as f:
            pickle.dump(clf_hier, f, pickle.HIGHEST_PROTOCOL)
        for group, clf in clfs.items():
            with open(f'{MODELS_DIR}model_{name}_{group}.pkl', 'wb') as f:
                pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

    # ============================================================
    # PHASE 3: PREDICCIONES DEL TEST SET
    # ============================================================

    print("\n=== PHASE 3: Test-set predictions + calibration ===")

    test_specs = [
        ('A', XA_te, yo_te),
        ('B', XB_te, yo_te),
        ('C', XC_te, yo_te),
        ('D', XD_te, yo_te),
    ]
    if XE_te is not None:
        test_specs.append(('E', XE_te, yoE_te))

    test_probas = {}

    for name, X_te, yo_te_model in test_specs:
        clf_hier, clfs = models[name]
        prob_matrix, class_names = mtf.predict_hierarchical(X_te, clf_hier, clfs)

        mtf.evaluate_model(yo_te_model, prob_matrix, class_names, mtf.LABEL_ORDER,
                       f'Model_{name}_raw')

        prob_matrix_cal = mtf.apply_calibrators(prob_matrix, class_names, calibrators[name])
        mtf.evaluate_model(yo_te_model, prob_matrix_cal, class_names, mtf.LABEL_ORDER,
                       f'Model_{name}_cal')

        test_probas[name] = (prob_matrix_cal, class_names, yo_te_model)

        feat_names = list(X_te.columns)
        mtf.plot_feature_importances(clf_hier, feat_names,
                                 f'{PLOTS_DIR}feat_importance_model{name}_hier.pdf')

    # ============================================================
    # PHASE 4: METAMODELO
    # ============================================================
    print("\n=== PHASE 4: Training meta-model ===")

    meta_X_train = oof_all.drop(columns=['true_label'])
    meta_y_train = oof_all['true_label']

    n_meta_features = meta_X_train.shape[1]
    print(f"Metamodel input: {n_meta_features} features "
          f"({n_meta_features // mtf.N_CLASSES} models x {mtf.N_CLASSES} classes)")

    # ── Survey dropout para robustez del metamodelo ──────────────────────
    # Dropout a nivel de SURVEY, propagado a los modelos dependientes.
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
            print(f'Model {model_name} (req: {required}): '
                  f'{n_dropped}/{has_data.sum()} dropped '
                  f'({100*n_dropped/max(has_data.sum(),1):.1f}%)')

        _surv_active = {s: ~survey_dropped[s] for s in ALL_SURVEYS}
        _combos = pd.DataFrame(_surv_active).value_counts().sort_index()
        print(f'Survey combination distribution:')
        for combo, count in _combos.items():
            active = [s for s, v in zip(ALL_SURVEYS, combo) if v]
            print(f'{"+".join(active) if active else "NONE":25s}: '
                  f'{count:5d} ({100*count/n_rows:.1f}%)')
    else:
        print('\n[Dropout] Disabled (META_DROPOUT_PROB=0)')

    meta_model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=mtf.RANDOM_STATE,
    )
    meta_model.fit(meta_X_train, meta_y_train)

    with open(f'{MODELS_DIR}meta_model.pkl', 'wb') as f:
        pickle.dump(meta_model, f, pickle.HIGHEST_PROTOCOL)
    print(f"Metamodel saved: {MODELS_DIR}meta_model.pkl")

    # ============================================================
    # PHASE 5: EVALUACIÓN METAMODELO
    # ============================================================
    print("\n=== PHASE 5: Metamodel evaluation ===")

    def proba_to_df(prob_matrix, class_names, model_name, ref_index,
                    prob_matrix_index, fill_value=mtf.PRIOR_PROB):
        cols   = [f'p_{c}_model{model_name}' for c in mtf.LABEL_ORDER]
        df_out = pd.DataFrame(fill_value, index=ref_index, columns=cols)
        shared = ref_index.intersection(prob_matrix_index)
        for j, cls in enumerate(mtf.LABEL_ORDER):
            idx = np.where(class_names == cls)[0]
            if len(idx) > 0:
                df_out.loc[shared, f'p_{cls}_model{model_name}'] = \
                    prob_matrix[prob_matrix_index.get_indexer(shared), idx[0]]
        # Binary mask: 1 = real predictions, 0 = prior fill
        mask_col = f'has_model{model_name}'
        df_out[mask_col] = 0
        df_out.loc[shared, mask_col] = 1
        return df_out

    meta_test_parts = []
    for name, (prob_matrix_cal, class_names, yo_te_model) in test_probas.items():
        part = proba_to_df(prob_matrix_cal, class_names, name,
                           ref_index=yo_te.index,
                           prob_matrix_index=yo_te_model.index)
        meta_test_parts.append(part)

    meta_X_test = pd.concat(meta_test_parts, axis=1)

    meta_pred_proba = meta_model.predict_proba(meta_X_test)
    meta_pred       = meta_model.predict(meta_X_test)

    print("\n=== METAMODEL ===")
    print("Accuracy:         ", "%0.3f" % metrics.accuracy_score(yo_te, meta_pred))
    print("Balanced accuracy:", "%0.3f" % metrics.balanced_accuracy_score(yo_te, meta_pred))
    print("Macro F1:         ", "%0.3f" % metrics.f1_score(yo_te, meta_pred, average='macro'))
    print(metrics.classification_report(yo_te, meta_pred, digits=3))

    cm = metrics.confusion_matrix(yo_te, meta_pred, labels=mtf.LABEL_ORDER)
    mtf.plot_confusion_matrix(cm, mtf.LABEL_ORDER,
                          f'{PLOTS_DIR}conf_matrix_metamodel.pdf',
                          title='Meta-model')



    # ============================================================
    # RELIABILITY DIAGRAM — META-MODEL
    # ============================================================

    def plot_metamodel_reliability(meta_pred_proba, y_true, label_order,
                                    meta_model_obj, save_path):
        # Reliability diagram for the meta-model.
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
                        label='Meta-model')
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

        plt.suptitle('Reliability diagrams - Meta-model', fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Metamodel reliability diagrams saved: {save_path}")


    plot_metamodel_reliability(
        meta_pred_proba, yo_te, mtf.LABEL_ORDER, meta_model,
        f'{PLOTS_DIR}reliability_metamodel.pdf'
    )


    # ============================================================
    # TRANSIENT PROBABILITY DISTRIBUTION (KDE grid)
    # ============================================================

    TRANSIENT_CLASSES = ['SNIa', 'SNIbc', 'SNII', 'SLSN']

    def plot_transient_proba_kde(prob_matrix, class_names, y_true,
                                  model_name, save_path):
        from scipy.stats import gaussian_kde

        class_names = np.array(class_names)
        y_true_arr  = np.array(y_true)
        n_tr        = len(TRANSIENT_CLASSES)

        fig, axes = plt.subplots(n_tr, n_tr,
                                 figsize=(n_tr * 3.5, n_tr * 3.0),
                                 sharex=True)
        fig.suptitle(f'Transient probability distributions — {model_name}',
                     fontsize=13, y=1.01)

        x_grid = np.linspace(0, 1, 300)

        for row_idx, true_cls in enumerate(TRANSIENT_CLASSES):
            # Mask for objects truly of this class
            true_mask = y_true_arr == true_cls
            if true_mask.sum() == 0:
                for col_idx in range(n_tr):
                    axes[row_idx, col_idx].set_visible(False)
                continue

            pm_sub   = prob_matrix[true_mask]
            pred_cls = class_names[np.argmax(pm_sub, axis=1)]

            for col_idx, pred_cls_name in enumerate(TRANSIENT_CLASSES):
                ax = axes[row_idx, col_idx]

                # Index of this column class in prob_matrix
                col_pos = np.where(class_names == pred_cls_name)[0]
                if len(col_pos) == 0:
                    ax.set_visible(False)
                    continue
                probs = pm_sub[:, col_pos[0]]

                # Split into correct (diagonal) and incorrect
                correct_mask   = pred_cls == true_cls
                incorrect_mask = ~correct_mask

                for mask, color, label in [
                    (correct_mask,   '#2ca02c', 'Correct'),
                    (incorrect_mask, '#d62728', 'Misclassified'),
                ]:
                    vals = probs[mask]
                    if vals.sum() == 0 or len(vals) < 3:
                        continue
                    # Avoid KDE on degenerate distributions
                    if vals.std() < 1e-6:
                        ax.axvline(vals.mean(), color=color,
                                   linewidth=1.5, linestyle='--', label=label)
                        continue
                    try:
                        kde = gaussian_kde(vals, bw_method='scott')
                        ax.fill_between(x_grid, kde(x_grid),
                                        alpha=0.35, color=color)
                        ax.plot(x_grid, kde(x_grid),
                                color=color, linewidth=1.5, label=label)
                    except Exception:
                        pass

                # Diagonal: shade own-class probability region
                if row_idx == col_idx:
                    ax.set_facecolor('#f5f5f5')

                ax.set_xlim(0, 1)
                ax.set_ylim(bottom=0)
                ax.tick_params(labelsize=7)

                # Labels only on edges
                if row_idx == n_tr - 1:
                    ax.set_xlabel(f'P({pred_cls_name})', fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f'True: {true_cls}', fontsize=9)
                if row_idx == 0 and col_idx == n_tr - 1:
                    ax.legend(fontsize=7, loc='upper left')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Transient KDE plot saved: {save_path}")



    # ============================================================
    # ENTROPÍA
    # ============================================================

    def compute_entropy(prob_matrix):
        # Shannon entropy (bits) of each row of a probability matrix.
        eps = 1e-12
        p = np.clip(prob_matrix, eps, 1)
        return -np.sum(p * np.log2(p), axis=1)


    def plot_entropy_by_class(entropy, y_true, label_order, model_name, save_path):
        # Bar plot of mean entropy ± std per true class.
        y_true_arr = np.array(y_true)
        means = np.array([entropy[y_true_arr == cls].mean() if (y_true_arr == cls).sum() > 0
                          else 0.0 for cls in label_order])
        stds  = np.array([entropy[y_true_arr == cls].std()  if (y_true_arr == cls).sum() > 0
                          else 0.0 for cls in label_order])

        # Color bars by taxonomy group
        group_colors = {'Transient': '#e05c5c', 'Stochastic': '#5c8ae0', 'Periodic': '#5cb85c'}
        cls_to_group = {cls: grp for grp, members in mtf.HIER_MAP.items() for cls in members}
        colors = [group_colors.get(cls_to_group.get(cls, ''), '#aaaaaa') for cls in label_order]

        x = np.arange(len(label_order))
        fig, ax = plt.subplots(figsize=(14, 5))

        # Dot plot: point = mean, whiskers = ±1 std
        for xi, (m, s, c) in enumerate(zip(means, stds, colors)):
            ax.errorbar(xi, m, yerr=s, fmt='o', color=c,
                        capsize=4, linewidth=1.2, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5, zorder=3)

        # Legend for taxonomy groups
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, alpha=0.85, label=g)
                           for g, c in group_colors.items()]
        ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

        ax.set_xticks(x)
        ax.set_xticklabels(label_order, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('True class', fontsize=12)
        ax.set_ylabel('Mean prediction entropy (bits)', fontsize=12)
        ax.set_title(f'Entropy by class — {model_name}', fontsize=13)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=1)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Entropy plot saved: {save_path}")


    def entropy_summary_df(entropy, y_true, label_order):
        # Return a DataFrame with mean/median entropy per true class.
        y_true_arr = np.array(y_true)
        rows = []
        for cls in label_order:
            mask = y_true_arr == cls
            if mask.sum() > 0:
                rows.append({
                    'class':          cls,
                    'mean_entropy':   round(float(entropy[mask].mean()), 4),
                    'median_entropy': round(float(np.median(entropy[mask])), 4),
                    'n':              int(mask.sum()),
                })
        return pd.DataFrame(rows)


    entropy_records = []
    for name, (prob_matrix_cal, class_names, yo_te_model) in test_probas.items():
        ent = compute_entropy(prob_matrix_cal)
        plot_entropy_by_class(ent, yo_te_model, mtf.LABEL_ORDER,
                              f'Model_{name}',
                              f'{PLOTS_DIR}entropy_model{name}.pdf')
        df_ent = entropy_summary_df(ent, yo_te_model, mtf.LABEL_ORDER)
        df_ent.insert(0, 'model', f'Model_{name}')
        entropy_records.append(df_ent)

    meta_entropy = compute_entropy(meta_pred_proba)
    plot_entropy_by_class(meta_entropy, yo_te, mtf.LABEL_ORDER,
                          'Meta-model',
                          f'{PLOTS_DIR}entropy_metamodel.pdf')
    df_ent_meta = entropy_summary_df(meta_entropy, yo_te, mtf.LABEL_ORDER)
    df_ent_meta.insert(0, 'model', 'Meta-model')
    entropy_records.append(df_ent_meta)
    entropy_all = pd.concat(entropy_records, ignore_index=True)

    #Transient probability KDE plots
    print("\nGenerating transient probability KDE plots ...")
    for name, (prob_matrix_cal, class_names, yo_te_model) in test_probas.items():
        plot_transient_proba_kde(
            prob_matrix_cal, class_names, yo_te_model,
            f'Model_{name}',
            f'{PLOTS_DIR}transient_kde_model{name}.pdf'
        )

    plot_transient_proba_kde(
        meta_pred_proba, meta_model.classes_, yo_te,
        'Meta-model',
        f'{PLOTS_DIR}transient_kde_metamodel.pdf'
    )

    # ============================================================
    # SUMMARY + OUTPUT FILE
    # ============================================================

    print("\nDone. Summary of F1 scores per model:")
    summary_rows = []
    _ent_lookup = {}
    for _df in entropy_records:
        _mname = _df['model'].iloc[0]
        _ent_lookup[_mname] = round(float(_df['mean_entropy'].mean()), 4)

    for name, (prob_matrix_cal, class_names, yo_te_model) in test_probas.items():
        pred = [class_names[i] for i in np.argmax(prob_matrix_cal, axis=1)]
        f1   = metrics.f1_score(yo_te_model, pred, average='macro')
        acc  = metrics.accuracy_score(yo_te_model, pred)
        bacc = metrics.balanced_accuracy_score(yo_te_model, pred)
        summary_rows.append({
            'Model':             f'Model_{name}',
            'Accuracy':          round(acc,  4),
            'Balanced_Accuracy': round(bacc, 4),
            'Macro_F1':          round(f1,   4),
            'Mean_Entropy':      _ent_lookup.get(f'Model_{name}', np.nan),
            'N_test':            len(yo_te_model),
        })

    meta_f1   = metrics.f1_score(yo_te, meta_pred, average='macro')
    meta_acc  = metrics.accuracy_score(yo_te, meta_pred)
    meta_bacc = metrics.balanced_accuracy_score(yo_te, meta_pred)
    summary_rows.append({
        'Model':             'Meta-model',
        'Accuracy':          round(meta_acc,  4),
        'Balanced_Accuracy': round(meta_bacc, 4),
        'Macro_F1':          round(meta_f1,   4),
        'Mean_Entropy':      _ent_lookup.get('Meta-model', np.nan),
        'N_test':            len(yo_te),
    })

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    # ── Bootstrap std (1000 resamples of test set) ───────────────────────────
    print("\nComputing bootstrap std (n=1000) ...")
    bootstrap_std_rows = []

    for name, (prob_matrix_cal, class_names, yo_te_model) in test_probas.items():
        g_std, pc_std = mtf.bootstrap_metrics(
            yo_te_model, prob_matrix_cal, class_names, mtf.LABEL_ORDER)
        pc_std.insert(0, 'model', f'Model_{name}')
        bootstrap_std_rows.append(pc_std)
        # Annotate summary with global std
        mask = summary['Model'] == f'Model_{name}'
        for k, v in g_std.items():
            summary.loc[mask, k] = v
        print(f"Model_{name} done")

    g_std_meta, pc_std_meta = mtf.bootstrap_metrics(
        yo_te, meta_pred_proba, meta_model.classes_, mtf.LABEL_ORDER)
    pc_std_meta.insert(0, 'model', 'Meta-model')
    bootstrap_std_rows.append(pc_std_meta)
    mask_meta = summary['Model'] == 'Meta-model'
    for k, v in g_std_meta.items():
        summary.loc[mask_meta, k] = v
    print("Metamodel done")

    bootstrap_std_all = pd.concat(bootstrap_std_rows, ignore_index=True)

    per_class_rows = []
    for name, (prob_matrix_cal, class_names, yo_te_model) in test_probas.items():
        pred   = [class_names[i] for i in np.argmax(prob_matrix_cal, axis=1)]
        report = metrics.classification_report(
            yo_te_model, pred, labels=mtf.LABEL_ORDER,
            output_dict=True, zero_division=0)
        for cls in mtf.LABEL_ORDER:
            if cls in report:
                per_class_rows.append({
                    'model':     f'Model_{name}',
                    'class':     cls,
                    'precision': round(report[cls]['precision'], 4),
                    'recall':    round(report[cls]['recall'],    4),
                    'f1':        round(report[cls]['f1-score'],  4),
                    'support':   int(report[cls]['support']),
                })

    meta_report = metrics.classification_report(
        yo_te, meta_pred, labels=mtf.LABEL_ORDER,
        output_dict=True, zero_division=0)
    for cls in mtf.LABEL_ORDER:
        if cls in meta_report:
            per_class_rows.append({
                'model':     'Meta-model',
                'class':     cls,
                'precision': round(meta_report[cls]['precision'], 4),
                'recall':    round(meta_report[cls]['recall'],    4),
                'f1':        round(meta_report[cls]['f1-score'],  4),
                'support':   int(meta_report[cls]['support']),
            })

    per_class_df = pd.DataFrame(per_class_rows)

    conf_rows = []
    for name, (prob_matrix_cal, class_names, yo_te_model) in test_probas.items():
        pred = [class_names[i] for i in np.argmax(prob_matrix_cal, axis=1)]
        cm   = metrics.confusion_matrix(yo_te_model, pred, labels=mtf.LABEL_ORDER)
        mtf.plot_confusion_matrix(
            cm, mtf.LABEL_ORDER,
            f'{PLOTS_DIR}conf_matrix_model{name}.pdf',
            title=f'Model {name}',
        )
        cm_df = pd.DataFrame(cm, index=mtf.LABEL_ORDER, columns=mtf.LABEL_ORDER)
        cm_df.index.name = 'true_class'
        cm_df.insert(0, 'model', f'Model_{name}')
        conf_rows.append(cm_df.reset_index())

    cm_meta = metrics.confusion_matrix(yo_te, meta_pred, labels=mtf.LABEL_ORDER)
    cm_meta_df = pd.DataFrame(cm_meta, index=mtf.LABEL_ORDER, columns=mtf.LABEL_ORDER)
    cm_meta_df.index.name = 'true_class'
    cm_meta_df.insert(0, 'model', 'Meta-model')
    conf_rows.append(cm_meta_df.reset_index())
    conf_matrix_all = pd.concat(conf_rows, ignore_index=True)

    # Write everything to a single excel workbook
    results_path = f'{MODELS_DIR}results_summary.xlsx'
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        summary.to_excel(           writer, sheet_name='Summary',            index=False)
        per_class_df.to_excel(      writer, sheet_name='Per_class_metrics',  index=False)
        conf_matrix_all.to_excel(   writer, sheet_name='Confusion_matrices', index=False)
        entropy_all.to_excel(       writer, sheet_name='Entropy_by_class',   index=False)
        bootstrap_std_all.to_excel( writer, sheet_name='Bootstrap_std',      index=False)

    print(f"\nNumeric results saved: {results_path}")
    print("Sheets: Summary | Per_class_metrics | Confusion_matrices | Entropy_by_class | Bootstrap_std")

    # CSV fallback for quick inspection
    summary.to_csv(f'{MODELS_DIR}model_comparison_summary.csv', index=False)

    # ============================================================
    # PHASE 6: SHAP ANALYSIS (MODELS A, B, C)
    # ============================================================

    def _shap_to_mean_abs(sv):
        
        if isinstance(sv, list):
            return np.mean([np.abs(a) for a in sv], axis=0)
        sv = np.array(sv)
        if sv.ndim == 3:
            return np.mean(np.abs(sv), axis=2)
        return np.abs(sv)


    def _shap_barplot(sv_raw, X_data, feature_names, title, save_path, top_n=40):
        sv_mean = _shap_to_mean_abs(sv_raw)   # (n_samples, n_features)
        exp = shap.Explanation(
            values=sv_mean,
            base_values=np.zeros(sv_mean.shape[0]),
            data=X_data.values if hasattr(X_data, 'values') else X_data,
            feature_names=feature_names,
        )
        plt.figure(figsize=(8, min(top_n * 0.35 + 1.5, 18)))
        shap.plots.bar(exp, max_display=top_n, show=False)
        plt.title(title, fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def compute_shap_for_model(model_name, clf_hier, clfs, X_te, yo_te_model,
                                label_order, plots_dir, models_dir):
        top_n = 40
        print(f"\n[SHAP] Model {model_name} — top-level classifier ...")
        explainer_hier = shap.TreeExplainer(clf_hier)
        sv_hier        = explainer_hier.shap_values(X_te)

        hier_path = f'{plots_dir}shap_summary_{model_name}_hier.pdf'
        _shap_barplot(sv_hier, X_te, list(X_te.columns),
                      f'SHAP feature importance — Model {model_name} (top-level)',
                      hier_path, top_n)
        print(f"Saved: {hier_path}")

        # Sub-classifiers
        shap_storage = {'hier': sv_hier}
        for group, clf_sub in clfs.items():
            cls_to_group = {cls: grp for grp, members in mtf.HIER_MAP.items() for cls in members}
            mask  = (yo_te_model.map(cls_to_group) == group).values
            X_sub = X_te.loc[mask]
            if len(X_sub) == 0:
                continue
            print(f"[SHAP] Model {model_name}. {group} sub-classifier "
                  f"({mask.sum()} objects) ...")
            explainer_sub = shap.TreeExplainer(clf_sub)
            sv_sub        = explainer_sub.shap_values(X_sub)
            shap_storage[group] = sv_sub

            sub_path = f'{plots_dir}shap_summary_{model_name}_{group}.pdf'
            _shap_barplot(sv_sub, X_sub, list(X_sub.columns),
                          f'SHAP feature importance — Model {model_name} ({group})',
                          sub_path, top_n)
            print(f"Saved: {sub_path}")

        # Save raw SHAP values for offline analysis
        shap_pkl = f'{models_dir}shap_values_{model_name}.pkl'
        with open(shap_pkl, 'wb') as f:
            pickle.dump({'shap_values': shap_storage,
                         'feature_names': list(X_te.columns),
                         'y_true': yo_te_model}, f, pickle.HIGHEST_PROTOCOL)
        print(f"SHAP values saved: {shap_pkl}")


    def _load_models_from_disk(models_dir, atlas_available=True):
        loaded = {}
        names = ['A', 'B', 'C', 'D'] + (['E'] if atlas_available else [])
        for name in names:
            hier_path = f'{models_dir}model_{name}_hier.pkl'
            try:
                with open(hier_path, 'rb') as f:
                    clf_hier = pickle.load(f)
            except FileNotFoundError:
                continue
            clfs = {}
            for group in ['Periodic', 'Stochastic', 'Transient']:
                grp_path = f'{models_dir}model_{name}_{group}.pkl'
                try:
                    with open(grp_path, 'rb') as f:
                        clfs[group] = pickle.load(f)
                except FileNotFoundError:
                    pass
            loaded[name] = (clf_hier, clfs)
            print(f"Loaded Model {name} from disk")
        return loaded

    if 'models' not in dir() or not models:
        print("Models not in memory, loading from disk ...")
        models = _load_models_from_disk(MODELS_DIR,
                                        atlas_available=XE_te is not None
                                        if 'XE_te' in dir() else True)

    shap_specs = [
        ('A', models['A'][0], models['A'][1], XA_te, yo_te),
        ('B', models['B'][0], models['B'][1], XB_te, yo_te),
        ('D', models['D'][0], models['D'][1], XD_te, yo_te),
    ]
    if 'C' in models:
        shap_specs.append(('C', models['C'][0], models['C'][1], XC_te, yo_te))
    if 'E' in models:
        shap_specs.append(('E', models['E'][0], models['E'][1], XE_te, yoE_te))

    print("\n=== PHASE 6: SHAP analysis ===")
    for model_name, clf_hier, clfs, X_te_shap, yo_te_shap in shap_specs:
        compute_shap_for_model(
            model_name, clf_hier, clfs, X_te_shap, yo_te_shap,
            mtf.LABEL_ORDER, PLOTS_DIR, MODELS_DIR
        )

    if 'C' in models:
        print("\n [SHAP] Model C: cross-survey ratio feature analysis ...")
        shap_pkl_c = f'{MODELS_DIR}shap_values_C.pkl'
        with open(shap_pkl_c, 'rb') as f:
            shap_data_c = pickle.load(f)

        sv_hier_c     = shap_data_c['shap_values']['hier']
        feat_names_c  = shap_data_c['feature_names']
        ratio_mask    = np.array([any(tag in fn for tag in ('_ztf_lsst_ratio', '_ztf_lsst_diff', '_ztf_atlas_ratio', '_ztf_atlas_diff', '_atlas_lsst_ratio', '_atlas_lsst_diff')) for fn in feat_names_c])

        if ratio_mask.sum() > 0:
            sv_hier_c_mean = _shap_to_mean_abs(sv_hier_c)   # (n_samples, n_features)
            ratio_importance = sv_hier_c_mean[:, ratio_mask].mean(axis=0)
            ratio_names      = np.array(feat_names_c)[ratio_mask]
            sort_idx         = np.argsort(ratio_importance)[::-1]

            fig, ax = plt.subplots(figsize=(max(10, len(ratio_names) * 0.4), 5))
            ax.bar(np.arange(len(ratio_names)), ratio_importance[sort_idx])
            ax.set_xticks(np.arange(len(ratio_names)))
            ax.set_xticklabels(ratio_names[sort_idx],
                               rotation='vertical', fontsize=7)
            ax.set_ylabel('Mean |SHAP value|', fontsize=11)
            ax.set_title('SHAP — cross-survey differential features (Model C)',
                         fontsize=12)
            plt.tight_layout()
            ratio_path = f'{PLOTS_DIR}shap_ratio_features_modelC.pdf'
            plt.savefig(ratio_path, bbox_inches='tight')
            plt.close()
            print(f"Cross-survey ratio SHAP saved: {ratio_path}")
        else:
            print("No ratio features found for Model C.")


    # ============================================================
    # PHASE 7: SHAP VIOLIN PLOTS — TRANSIENT SUB-CLASSIFIER
    # ============================================================


    def plot_shap_violin_transient(sv_transient, X_sub, transient_classes,
                                    model_name, plots_dir, top_n=20):
        sv = np.array(sv_transient)
        if sv.ndim == 2:
            sv = sv[:, :, np.newaxis]
        elif isinstance(sv_transient, list):
            sv = np.stack(sv_transient, axis=2)

        feat_names = list(X_sub.columns)
        n_cls      = sv.shape[2]
        cls_indices = {cls: i for i, cls in enumerate(transient_classes[:n_cls])}

        for cls_name, cls_idx in cls_indices.items():
            sv_cls = sv[:, :, cls_idx]   # (n_samples, n_features)

            # Select top_n features by mean |SHAP| for this class
            mean_abs = np.abs(sv_cls).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[::-1][:top_n]

            exp = shap.Explanation(
                values=sv_cls[:, top_idx],
                base_values=np.zeros(sv_cls.shape[0]),
                data=X_sub.values[:, top_idx],
                feature_names=[feat_names[i] for i in top_idx],
            )

            plt.figure(figsize=(9, top_n * 0.38 + 1.5))
            shap.plots.violin(exp,
                              max_display=top_n,
                              plot_type="layered_violin",
                              show=False)
            plt.title(
                f"SHAP layered violin — Transient sub-clf, {cls_name} ({model_name})",
                fontsize=11)
            plt.tight_layout()
            save_path = f'{plots_dir}shap_violin_{model_name}_{cls_name}.pdf'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")


    # ── Reload models if needed (same as Phase 6) ────────────────────────────
    if 'models' not in dir() or not models:
        print("  models not in memory — loading from disk ...")
        models = _load_models_from_disk(MODELS_DIR)

    print("\n=== PHASE 7: SHAP violin plots (Transient sub-classifier) ===")
    for model_name_shap in ['A', 'B', 'C', 'D', 'E']:
        shap_pkl_path = f'{MODELS_DIR}shap_values_{model_name_shap}.pkl'
        try:
            with open(shap_pkl_path, 'rb') as f:
                shap_data = pickle.load(f)
        except FileNotFoundError:
            print(f"Skipping Model {model_name_shap}: .pkl not found")
            continue

        sv_transient = shap_data['shap_values'].get('Transient')
        if sv_transient is None:
            print(f"Skipping Model {model_name_shap}: no Transient SHAP values")
            continue

        feat_names = shap_data['feature_names']
        y_true_all = shap_data['y_true']


        _model_X_map = {
            'A': XA_te, 'B': XB_te, 'C': XC_te, 'D': XD_te, 'E': XE_te
        }
        _model_y_map = {
            'A': yo_te, 'B': yo_te, 'C': yo_te, 'D': yo_te, 'E': yoE_te
        }
        X_full = _model_X_map[model_name_shap]
        y_full = _model_y_map[model_name_shap]

        cls_to_group = {cls: grp for grp, members in mtf.HIER_MAP.items()
                        for cls in members}
        transient_mask = (y_full.map(cls_to_group) == 'Transient').values
        X_sub_violin   = X_full.loc[transient_mask]


        if model_name_shap in models:
            clf_sub = models[model_name_shap][1].get('Transient')
            if clf_sub is not None:
                trans_cls_order = list(clf_sub.classes_)
            else:
                trans_cls_order = TRANSIENT_CLASSES
        else:
            trans_cls_order = TRANSIENT_CLASSES

        print(f"[SHAP violin] Model {model_name_shap} "
              f"({X_sub_violin.shape[0]} transient objects) ...")
        plot_shap_violin_transient(
            sv_transient, X_sub_violin,
            trans_cls_order, model_name_shap,
            PLOTS_DIR, top_n=20
        )

def parse_args():
    """
    CLI para model_training_full.py (versión baseline, sin CORAL ni
    aumentation con datos reales). Sin pasar ningún argumento, el script
    usa los DEFAULTS de la sección CONFIGURATION (pensado para lanzarse
    con runfile() desde Spyder).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Main model training pipeline (no domain adaptation) "
        )
    )

    g_in = parser.add_argument_group("Inputs")
    g_in.add_argument("--labels-file", default=labels_file,
        help="Label CSV/parquet for training set")
    g_in.add_argument("--features-ztf", default=features_ztf_file,
        help="Training ZTF features parquet")
    g_in.add_argument("--features-lsst", default=features_lsst_file,
        help="Training LSST features parquet")
    g_in.add_argument("--features-atlas", default=features_atlas_file,
        help="Training ATLAS features parquet")
    g_in.add_argument("--features-combined", default=features_combined_file,
        help="Training combined features parquet")

    g_out = parser.add_argument_group("Output")
    g_out.add_argument("--output-dir", default=OUTPUT_DIR,
        help="Output directory (contains models/, plots/, oof/)")

    g_meta = parser.add_argument_group("Metamodelo")
    g_meta.add_argument("--meta-dropout-prob", type=float, default=META_DROPOUT_PROB,
        help="Survey dropout probability")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    labels_file             = args.labels_file
    features_ztf_file       = args.features_ztf
    features_lsst_file      = args.features_lsst
    features_atlas_file     = args.features_atlas
    features_combined_file  = args.features_combined

    OUTPUT_DIR = args.output_dir
    OOF_DIR    = os.path.join(OUTPUT_DIR, 'oof')    + '/'
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models') + '/'
    PLOTS_DIR  = os.path.join(OUTPUT_DIR, 'plots')  + '/'

    META_DROPOUT_PROB = args.meta_dropout_prob

    main()