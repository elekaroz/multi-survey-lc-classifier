from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd

#%%
# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------

BANDS_ZTF  = [1, 2, 3]
BANDS_LSST = [1, 2, 3, 4, 5, 6]

_BAND_NAMES = {1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y', 6: 'u'}

def _band_pairs(bands: List[int]) -> List[tuple]:
    return [
        (_BAND_NAMES[b1], _BAND_NAMES[b2])
        for i, b1 in enumerate(bands)
        for b2 in bands[i + 1:]
    ]

# ---------------------------------------------------------------------------
# Band features
# ---------------------------------------------------------------------------

def _per_band_features(bands: List[int]) -> List[str]:
    features = []
    for b in bands:
        features += [
            f'MHAOV_Period_{b}', f'Amplitude_{b}', f'AndersonDarling_{b}',
            f'Autocor_length_{b}', f'Beyond1Std_{b}', f'Con_{b}',
            f'Eta_e_{b}', f'ExcessVar_{b}',
            f'GP_DRW_sigma_{b}', f'GP_DRW_tau_{b}', f'Gskew_{b}',
            *[f'Harmonics_mag_{h}_{b}' for h in range(1, 8)],
            f'Harmonics_mse_{b}',
            *[f'Harmonics_phase_{h}_{b}' for h in range(2, 8)],
            f'IAR_phi_{b}', f'LinearTrend_{b}',
            f'MHPS_PN_flag_{b}', f'MHPS_high_{b}', f'MHPS_low_{b}',
            f'MHPS_non_zero_{b}', f'MHPS_ratio_{b}',
            f'MaxSlope_{b}', f'Mean_{b}', f'Meanvariance_{b}',
            f'MedianAbsDev_{b}', f'MedianBRP_{b}',
            f'PairSlopeTrend_{b}', f'PercentAmplitude_{b}',
            f'Psi_CS_{b}', f'Psi_eta_{b}', f'Pvar_{b}',
            f'Q31_{b}', f'Rcs_{b}',
            f'SF_ML_amplitude_{b}', f'SF_ML_gamma_{b}',
            f'SPM_A_{b}', f'SPM_beta_{b}', f'SPM_chi_{b}',
            f'SPM_gamma_{b}', f'SPM_t0_{b}',
            f'SPM_tau_fall_{b}', f'SPM_tau_rise_{b}',
            f'Skew_{b}', f'SmallKurtosis_{b}', f'Std_{b}', f'StetsonK_{b}',
            f'delta_mag_fid_{b}', f'delta_mjd_fid_{b}',
            f'dmag_first_det_fid_{b}', f'dmag_non_det_fid_{b}',
            f'first_mag_{b}',
            f'iqr_{b}',
            f'last_diffmaglim_before_fid_{b}', f'last_mjd_before_fid_{b}',
            f'max_diffmaglim_after_fid_{b}', f'max_diffmaglim_before_fid_{b}',
            f'mean_mag_{b}',
            f'median_diffmaglim_after_fid_{b}', f'median_diffmaglim_before_fid_{b}',
            f'min_mag_{b}',
            f'n_det_{b}', f'n_neg_{b}',
            f'n_non_det_after_fid_{b}', f'n_non_det_before_fid_{b}',
            f'n_pos_{b}', f'positive_fraction_{b}',
            f'delta_period_{b}',
        ]
    return features


def _color_features(bands: List[int]) -> List[str]:
    features = []
    for n1, n2 in _band_pairs(bands):
        features += [
            f'{n1}-{n2}_max', f'{n1}-{n2}_max_corr',
            f'{n1}-{n2}_mean', f'{n1}-{n2}_mean_corr',
        ]
    return features


GLOBAL_FEATURES: List[str] = [
    'Multiband_period', 'Period_fit',
    'Power_rate_1/2', 'Power_rate_1/3', 'Power_rate_1/4',
    'Power_rate_2', 'Power_rate_3', 'Power_rate_4',
    'gal_b', 'gal_l',
    'rb', 'sgscore1',
    'W1', 'W2', 'W3', 'W4', 'W1-W2', 'W2-W3',
    'r-W3', 'r-W2', 'g-W3', 'g-W2', 'g-r_ml',
]

# ---------------------------------------------------------------------------
# Full feature lists
# ---------------------------------------------------------------------------

FEATURE_LIST_ZTF: List[str] = (
    _per_band_features(BANDS_ZTF) +
    _color_features(BANDS_ZTF) +
    GLOBAL_FEATURES
)

FEATURE_LIST_LSST: List[str] = (
    _per_band_features(BANDS_LSST) +
    _color_features(BANDS_LSST) +
    GLOBAL_FEATURES
)

# ---------------------------------------------------------------------------
# Columns to drop
# ---------------------------------------------------------------------------

def _rm_nd_cols(bands: List[int]) -> List[str]:
    cols = []
    for b in bands:
        cols += [
            f'n_det_{b}', f'n_pos_{b}', f'n_neg_{b}',
            f'first_mag_{b}', f'MHPS_non_zero_{b}', f'MHPS_PN_flag_{b}',
            f'mean_mag_{b}', f'min_mag_{b}', f'iqr_{b}',
            f'delta_mjd_fid_{b}', f'last_mjd_before_fid_{b}',
            f'MHAOV_Period_{b}',
            f'dmag_first_det_fid_{b}',   
            f'positive_fraction_{b}',    
            f"SPM_gamma_{b}",
            f"SPM_beta_{b}",
            f"max_diffmaglim_before_fid_{b}", 
            f"max_diffmaglim_after_fid_{b}",
            f"median_diffmaglim_before_fid_{b}", 
            f"median_diffmaglim_after_fid_{b}",
            f"last_diffmaglim_before_fid_{b}",    
            f"dmag_non_det_fid_{b}",
            f"n_non_det_after_fid_{b}",     
            f"n_non_det_before_fid_{b}",
            f"delta_period_{b}",
            'Multiband_period'
        ]
    cols += ['W1', 'W2', 'W3', 'W4', 'g-r_ml', 'Period_fit', 'g-r_max_corr', 'g-r_mean_corr',
             'W1-W2', 'W2-W3', 'r-W3', 'r-W2', 'g-W3', 'g-W2']
    return cols


RM_ND_COLS_ZTF:  List[str] = _rm_nd_cols(BANDS_ZTF)
RM_ND_COLS_LSST: List[str] = _rm_nd_cols(BANDS_LSST)

# ---------------------------------------------------------------------------
# Survey ids
# ---------------------------------------------------------------------------

SURVEY_ID = {'ztf': 0, 'lsst': 1, 'combined': 2}

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def apply_feature_list(
    features: pd.DataFrame,
    feature_list: List[str],
    rm_nd_cols: List[str],
) -> pd.DataFrame:
    """
    Filter, drop noisy columns, and reorder a feature DataFrame.
    """
    final_list = [f for f in feature_list if f not in rm_nd_cols]
    ordered    = [c for c in final_list if c in features.columns]
    return features[ordered]




def postprocess_features(
    features: Dict[str, Optional[pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Apply apply_feature_list and add the 'survey' column to each
    feature DataFrame in the dict returned by pipeline.

    """
    result: Dict[str, pd.DataFrame] = {}

    #ZTF
    df_ztf = features.get('ztf')
    if df_ztf is not None and not df_ztf.empty:
        df_ztf = apply_feature_list(
            df_ztf,
            feature_list=[f'{c}_ztf' for c in FEATURE_LIST_ZTF],
            rm_nd_cols   =[f'{c}_ztf' for c in RM_ND_COLS_ZTF],
        ).copy()
        df_ztf['survey'] = SURVEY_ID['ztf']
    else:
        df_ztf = pd.DataFrame()
    result['ztf'] = df_ztf

    #LSST
    df_lsst = features.get('lsst')
    if df_lsst is not None and not df_lsst.empty:
        df_lsst = apply_feature_list(
            df_lsst,
            feature_list=[f'{c}_lsst' for c in FEATURE_LIST_LSST],
            rm_nd_cols   =[f'{c}_lsst' for c in RM_ND_COLS_LSST],
        ).copy()
        df_lsst['survey'] = SURVEY_ID['lsst']
    else:
        df_lsst = pd.DataFrame()
    result['lsst'] = df_lsst

    #Combined
    # Combined features use the LSST feature list (same 6 bands)
    df_comb = features.get('combined')
    if df_comb is not None and not df_comb.empty:
        df_comb = apply_feature_list(
            df_comb,
            feature_list=FEATURE_LIST_LSST,
            rm_nd_cols   =RM_ND_COLS_LSST,
        ).copy()
        df_comb.columns = [f'{c}_combined' for c in df_comb.columns]
        df_comb['survey'] = SURVEY_ID['combined']
    else:
        df_comb = pd.DataFrame()
    result['combined'] = df_comb

    return result
