from .base import GenericPreprocessor
import numpy as np
import pandas as pd

#%%

class ZTFLightcurvePreprocessor(GenericPreprocessor):
    def __init__(self, stream=False):
        super().__init__()
        self.not_null_columns = [
            'mjd',
            'fid',
            'magpsf',
            'sigmapsf',
            'magpsf_ml',
            'sigmapsf_ml',
            'ra',
            'dec',
            'rb'
        ]
        self.stream = stream
        if not self.stream:
            self.not_null_columns.append('sgscore1')
        self.column_translation = {
            'mjd': 'time',
            'fid': 'band',
            'magpsf_ml': 'magnitude',
            'sigmapsf_ml': 'error'
        }
        self.max_sigma = 1.0
        self.rb_threshold = 0.55

    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.not_null_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def discard_invalid_value_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections.replace([np.inf, -np.inf], np.nan)
        valid_alerts = detections[self.not_null_columns].notna().all(axis=1)
        detections = detections[valid_alerts.values]
        detections[self.not_null_columns] = detections[self.not_null_columns].apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        return detections

    def drop_duplicates(self, detections):
        """
        Sometimes the same source triggers two detections with slightly
        different positions.

        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()

        # keep the one with best rb
        detections = detections.sort_values("rb", ascending=False)
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'mjd'], keep='first')
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[((detections['sigmapsf_ml'] > 0.0) &
                                 (detections['sigmapsf_ml'] < self.max_sigma))
                                ]
        return detections

    def discard_bogus(self, detections):
        """

        :param detections:
        :return:
        """
        detections = detections[detections['rb'] >= self.rb_threshold]
        return detections

    def enough_alerts(self, detections, min_dets=5):
        objects = detections.groupby("oid")
        indexes = []
        for oid, group in objects:
            if len(group.fid == 1) > min_dets or len(group.fid == 2) > min_dets:
                indexes.append(oid)
        return detections.loc[indexes]

    def get_magpsf_ml(self, detections, objects):
        def magpsf_ml_not_stream(detections, objects_table):
            detections = detections.copy()
            oid = detections.index.values[0]
            is_corrected = objects_table.loc[[oid]].corrected.values[0]
            if is_corrected:
                detections["magpsf_ml"] = detections["magpsf_corr"]
                detections["sigmapsf_ml"] = detections["sigmapsf_corr_ext"]
            else:
                detections["magpsf_ml"] = detections["magpsf"]
                detections["sigmapsf_ml"] = detections["sigmapsf"]
            return detections

        def magpsf_ml_stream(detections):
            detections = detections.copy()
            is_corrected = detections.corrected.all()
            if is_corrected:
                detections["magpsf_ml"] = detections["magpsf_corr"]
                detections["sigmapsf_ml"] = detections["sigmapsf_corr_ext"]
            else:
                detections["magpsf_ml"] = detections["magpsf"]
                detections["sigmapsf_ml"] = detections["sigmapsf"]
            return detections

        grouped_detections = detections.groupby(level=0, sort=False, group_keys=False)
        if self.stream:
            detections = grouped_detections.apply(magpsf_ml_stream)
        else:
            detections = grouped_detections.apply(
                magpsf_ml_not_stream, objects_table=objects)
        return detections

    def preprocess(self, dataframe, objects=None):
        """
        :param dataframe:
        :param objects:
        :return:
        """
        if not self.stream and objects is None:
            raise Exception('ZTF Preprocessor requires objects dataframe')
        self.verify_dataframe(dataframe)
        dataframe = self.get_magpsf_ml(dataframe, objects)
        if not self.has_necessary_columns(dataframe):
            raise Exception('dataframe does not have all the necessary columns')
        dataframe = self.discard_bogus(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.enough_alerts(dataframe)
        dataframe = self.rename_columns_detections(dataframe)
        return dataframe

    def rename_columns_non_detections(self, non_detections):
        return non_detections.rename(
            columns=self.column_translation, errors='ignore')

    def rename_columns_detections(self, detections):
        return detections.rename(
            columns=self.column_translation, errors='ignore')

#%%
class ZTFForcedPhotometryLightcurvePreprocessor(GenericPreprocessor):
    """Preprocessing for lightcurves from ZTF forced photometry service."""
    def __init__(self):
        super().__init__()

        self.required_columns = [
            'mjd',
            'fid',
            'forcediffimflux',
            'forcediffimfluxunc',
            'mag_tot',
            'sigma_mag_tot',
        ]

        self.column_translation = {
            'mjd': 'time',
            'fid': 'band',
            'forcediffimflux': 'difference_flux',
            'forcediffimfluxunc': 'difference_flux_error',
            'mag_tot': 'magnitude',
            'sigma_mag_tot': 'error'
        }
        self.max_sigma = 1.0

        self.new_columns = []
        for c in self.required_columns:
            if c in self.column_translation.keys():
                self.new_columns.append(self.column_translation[c])
            else:
                self.new_columns.append(c)

        self.required_cols_metadata = ['ra', 'dec']

    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.required_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def metadata_has_necessary_columns(self, object_df):
        for c in self.required_cols_metadata:
            if c not in object_df.columns:
                return False
        return True

    def discard_invalid_value_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections.replace([np.inf, -np.inf], np.nan)
        detections[self.new_columns] = detections[self.new_columns].apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        return detections

    def drop_duplicates(self, detections):
        """
        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'time'])
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[
            ((detections['error'] > 0.0) &
             (detections['error'] < self.max_sigma)) | detections['error'].isna()
        ]
        return detections

    def enough_alerts(self, detections, min_dets=5):
        objects = detections.groupby("oid")
        indexes = []
        for oid, group in objects:
            if len(group.band == 1) > min_dets or len(group.band == 2) > min_dets:
                indexes.append(oid)
        return detections.loc[indexes]

    def discard_i_filter(self, detections):
        return detections[detections.band != 3]

    def preprocess(self, dataframe, objects=None):
        """
        :param dataframe:
        :param objects:
        :return:
        """
        self.verify_dataframe(dataframe)
        if not self.has_necessary_columns(dataframe):
            raise Exception(
                'Lightcurve dataframe does not have all the necessary columns')

        if not self.metadata_has_necessary_columns(objects):
            raise Exception(
                'Metadata dataframe does not have all the necessary columns')
        dataframe = self.rename_columns_detections(dataframe)
        dataframe = self.discard_i_filter(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.enough_alerts(dataframe)
        return dataframe

    def rename_columns_detections(self, detections):
        return detections.rename(
            columns=self.column_translation, errors='ignore')
    
#%%
class ZTFLightcurvePreprocessor3bands(GenericPreprocessor):
    def __init__(self, stream=False):
        super().__init__()
        self.not_null_columns = [
            'mjd',
            'fid',
            'magpsf',
            'sigmapsf',
            'magpsf_ml',
            'sigmapsf_ml',
            'ra',
            'dec',
            'rb'
        ]
        self.stream = stream
        if not self.stream:
            self.not_null_columns.append('sgscore1')
        self.column_translation = {
            'mjd': 'time',
            'fid': 'band',
            'magpsf_ml': 'magnitude',
            'sigmapsf_ml': 'error'
        }
        self.max_sigma = 1.0
        self.rb_threshold = 0.55
        self.valid_fids = [1,2,3]
        
    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.not_null_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def discard_invalid_value_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections.replace([np.inf, -np.inf], np.nan)
        valid_alerts = detections[self.not_null_columns].notna().all(axis=1)
        detections = detections[valid_alerts.values]
        detections[self.not_null_columns] = detections[self.not_null_columns].apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        return detections
    
    def discard_invalid_bands(self, detections):
        return detections[detections['fid'].isin(self.valid_fids)]

    def drop_duplicates(self, detections):
        """
        Sometimes the same source triggers two detections with slightly
        different positions.

        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()

        # keep the one with best rb
        detections = detections.sort_values("rb", ascending=False)
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'mjd'], keep='first')
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[((detections['sigmapsf_ml'] > 0.0) &
                                 (detections['sigmapsf_ml'] < self.max_sigma))
                                ]
        return detections

    def discard_bogus(self, detections):
        """

        :param detections:
        :return:
        """
        detections = detections[detections['rb'] >= self.rb_threshold]
        return detections
    
    def enough_alerts(self, detections, min_dets=5):
        valid_fids = detections['fid'].unique()  # detecta todas las bandas presentes
        objects = detections.groupby("oid")
        indexes = []
        for oid, group in objects:
            has_enough = any(
                len(group[group['fid'] == fid]) > min_dets
                for fid in valid_fids
            )
            if has_enough:
                indexes.append(oid)
        return detections.loc[indexes]

    def get_magpsf_ml(self, detections, objects):
        def magpsf_ml_not_stream(detections, objects_table):
            detections = detections.copy()
            oid = detections.index.values[0]
            is_corrected = objects_table.loc[[oid]].corrected.values[0]
            if is_corrected:
                detections["magpsf_ml"] = detections["magpsf_corr"]
                detections["sigmapsf_ml"] = detections["sigmapsf_corr_ext"]
            else:
                detections["magpsf_ml"] = detections["magpsf"]
                detections["sigmapsf_ml"] = detections["sigmapsf"]
            return detections

        def magpsf_ml_stream(detections):
            detections = detections.copy()
            is_corrected = detections.corrected.all()
            if is_corrected:
                detections["magpsf_ml"] = detections["magpsf_corr"]
                detections["sigmapsf_ml"] = detections["sigmapsf_corr_ext"]
            else:
                detections["magpsf_ml"] = detections["magpsf"]
                detections["sigmapsf_ml"] = detections["sigmapsf"]
            return detections

        grouped_detections = detections.groupby(level=0, sort=False, group_keys=False)
        if self.stream:
            detections = grouped_detections.apply(magpsf_ml_stream)
        else:
            detections = grouped_detections.apply(
                magpsf_ml_not_stream, objects_table=objects)
        return detections

    def preprocess(self, dataframe, objects=None):
        """
        :param dataframe:
        :param objects:
        :return:
        """
        if not self.stream and objects is None:
            raise Exception('ZTF Preprocessor requires objects dataframe')
        self.verify_dataframe(dataframe)
        dataframe = self.get_magpsf_ml(dataframe, objects)
        if not self.has_necessary_columns(dataframe):
            raise Exception('dataframe does not have all the necessary columns')
        dataframe = self.discard_bogus(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_invalid_bands(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.enough_alerts(dataframe)
        dataframe = self.rename_columns_detections(dataframe)
        return dataframe
    

    def rename_columns_non_detections(self, non_detections):
        return non_detections.rename(
            columns=self.column_translation, errors='ignore')

    def rename_columns_detections(self, detections):
        return detections.rename(
            columns=self.column_translation, errors='ignore')
    
#%%

class LSSTLightcurvePreprocessor(GenericPreprocessor):
    def __init__(self, stream=False):
        super().__init__()
        self.not_null_columns = [
            'mjd',
            'fid',
            'magpsf',
            'sigmapsf',
            'magpsf_ml',
            'sigmapsf_ml',
            'ra',
            'dec',
            'rb'
        ]
        self.stream = stream
        if not self.stream:
            self.not_null_columns.append('sgscore1')
        self.column_translation = {
            'mjd': 'time',
            'fid': 'band',
            'magpsf_ml': 'magnitude',
            'sigmapsf_ml': 'error'
        }
        self.max_sigma = 1.0
        self.rb_threshold = 0.55
        self.valid_fids = [6,1,2,3,4,5]
        
    def has_necessary_columns(self, dataframe):
        """
        :param dataframe:
        :return:
        """
        input_columns = set(dataframe.columns)
        constraint = set(self.not_null_columns)
        difference = constraint.difference(input_columns)
        return len(difference) == 0

    def discard_invalid_value_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections.replace([np.inf, -np.inf], np.nan)
        valid_alerts = detections[self.not_null_columns].notna().all(axis=1)
        detections = detections[valid_alerts.values]
        detections[self.not_null_columns] = detections[self.not_null_columns].apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        return detections
    
    def discard_invalid_bands(self, detections):
        return detections[detections['fid'].isin(self.valid_fids)]

    def drop_duplicates(self, detections):
        """
        Sometimes the same source triggers two detections with slightly
        different positions.

        :param detections:
        :return:
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()

        # keep the one with best rb
        detections = detections.sort_values("rb", ascending=False)
        detections['oid'] = detections.index
        #para que el preprocesador no se salte detecciones LSST ''duplicadas''
        dedup_cols = ['oid', 'mjd', 'tid'] if 'tid' in detections.columns else ['oid', 'mjd']
        detections = detections.drop_duplicates(dedup_cols, keep='first')
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        """
        :param detections:
        :return:
        """
        detections = detections[((detections['sigmapsf_ml'] > 0.0) &
                                 (detections['sigmapsf_ml'] < self.max_sigma))
                                ]
        return detections

    def discard_bogus(self, detections):
        """

        :param detections:
        :return:
        """
        detections = detections[detections['rb'] >= self.rb_threshold]
        return detections
    
    def enough_alerts(self, detections, min_dets=5):
        valid_fids = detections['fid'].unique()  # detecta todas las bandas presentes
        objects = detections.groupby("oid")
        indexes = []
        for oid, group in objects:
            has_enough = any(
                len(group[group['fid'] == fid]) > min_dets
                for fid in valid_fids
            )
            if has_enough:
                indexes.append(oid)
        return detections.loc[indexes]

    def get_magpsf_ml(self, detections, objects):
        def magpsf_ml_not_stream(detections, objects_table):
            detections = detections.copy()
            oid = detections.index.values[0]
            is_corrected = objects_table.loc[[oid]].corrected.values[0]
            if is_corrected:
                detections["magpsf_ml"] = detections["magpsf_corr"]
                detections["sigmapsf_ml"] = detections["sigmapsf_corr_ext"]
            else:
                detections["magpsf_ml"] = detections["magpsf"]
                detections["sigmapsf_ml"] = detections["sigmapsf"]
            return detections

        def magpsf_ml_stream(detections):
            detections = detections.copy()
            is_corrected = detections.corrected.all()
            if is_corrected:
                detections["magpsf_ml"] = detections["magpsf_corr"]
                detections["sigmapsf_ml"] = detections["sigmapsf_corr_ext"]
            else:
                detections["magpsf_ml"] = detections["magpsf"]
                detections["sigmapsf_ml"] = detections["sigmapsf"]
            return detections

        grouped_detections = detections.groupby(level=0, sort=False, group_keys=False)
        if self.stream:
            detections = grouped_detections.apply(magpsf_ml_stream)
        else:
            detections = grouped_detections.apply(
                magpsf_ml_not_stream, objects_table=objects)
        return detections

    def preprocess(self, dataframe, objects=None):
        """
        :param dataframe:
        :param objects:
        :return:
        """
        if not self.stream and objects is None:
            raise Exception('ZTF Preprocessor requires objects dataframe')
        self.verify_dataframe(dataframe)
        dataframe = self.get_magpsf_ml(dataframe, objects)
        if not self.has_necessary_columns(dataframe):
            raise Exception('dataframe does not have all the necessary columns')
        dataframe = self.discard_bogus(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_invalid_bands(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.enough_alerts(dataframe)
        dataframe = self.rename_columns_detections(dataframe)
        return dataframe
    
#versión debug ↴

    # def preprocess(self, dataframe, objects=None):
    #     if not self.stream and objects is None:
    #         raise Exception('ZTF Preprocessor requires objects dataframe')
    #     self.verify_dataframe(dataframe)
        
    #     def count_lsst(df, step):
    #         if 'tid' in df.columns:
    #             n = (df['tid'] == 'lsst').sum()
    #         else:
    #             n = 'columna tid no existe'
    #         print(f"  [{step}] LSST: {n}")
        
    #     dataframe = self.get_magpsf_ml(dataframe, objects);       count_lsst(dataframe, 'get_magpsf_ml')
    #     if not self.has_necessary_columns(dataframe):
    #         raise Exception('dataframe does not have all the necessary columns')
    #     dataframe = self.discard_bogus(dataframe);                count_lsst(dataframe, 'discard_bogus')
    #     dataframe = self.discard_invalid_value_detections(dataframe); count_lsst(dataframe, 'discard_invalid_value_detections')
    #     dataframe = self.discard_invalid_bands(dataframe);        count_lsst(dataframe, 'discard_invalid_bands')
    #     dataframe = self.discard_noisy_detections(dataframe);     count_lsst(dataframe, 'discard_noisy_detections')
    #     dataframe = self.drop_duplicates(dataframe);              count_lsst(dataframe, 'drop_duplicates')
    #     dataframe = self.enough_alerts(dataframe);                count_lsst(dataframe, 'enough_alerts')
    #     dataframe = self.rename_columns_detections(dataframe)
    #     return dataframe
    
    def rename_columns_non_detections(self, non_detections):
        return non_detections.rename(
            columns=self.column_translation, errors='ignore')

    def rename_columns_detections(self, detections):
        return detections.rename(
            columns=self.column_translation, errors='ignore')
    
#%%

class ATLASLightcurvePreprocessor(GenericPreprocessor):
    """
    Preprocessor for ATLAS forced photometry lightcurves.
 
    Input columns (ATLAS forced phot CSV or equivalent internal format):
        MJD / mjd     : Modified Julian Date
        m / magpsf    : PSF magnitude
        dm / sigmapsf : PSF magnitude error
        uJy           : difference flux in micro-Jansky (positive = brightening)
        duJy          : difference flux error
        F / filter    : filter string ('c' or 'o')
        chi/N         : chi^2/dof of the PSF fit
        Pass          : 0/1 global quality flag from ATLAS
 
    Band encoding (ATLAS):
        c = 0  (cyan,  ~g+r)
        o = 1  (orange, ~r+i)
 
    After preprocessing the DataFrame is ready to be passed directly to
    ATLASFeatureExtractor.  Column translation follows the same convention
    as ZTFLightcurvePreprocessor:
        mjd           -> time
        fid           -> band
        magpsf_ml     -> magnitude   (= magpsf; no host-galaxy correction in ATLAS)
        sigmapsf_ml   -> error       (= sigmapsf)
        uJy           -> difference_flux
        duJy          -> difference_flux_error
 
    Columns not available in ATLAS and not injected (no equivalent information):
        rb       — no real-bogus score; not used by ATLASFeatureExtractor
        sgscore1 — no star-galaxy separation; not used by ATLASFeatureExtractor
        magpsf_corr / sigmapsf_corr / sigmapsf_corr_ext = raw values
    """
 
    BAND_MAP   = {'c': 0, 'o': 1}
    VALID_FIDS = [0, 1]
 
    def __init__(self):
        super().__init__()
 
        self.not_null_columns = ['mjd', 'fid', 'magpsf', 'sigmapsf']
 
        # lc_classifier column names
        self.column_translation = {
            'mjd':         'time',
            'fid':         'band',
            'magpsf_ml':   'magnitude',
            'sigmapsf_ml': 'error',
            'uJy':         'difference_flux',
            'duJy':        'difference_flux_error',
        }
 
        self.max_sigma    = 0.5    # tighter than ZTF (forced phot is noisier)
        self.max_chi2dof  = 50.0   # secondary chi2 cut; primary is apply_quality_cuts (chi/N > 100)
        self.require_pass = True   # filter by ATLAS Pass flag when present
 
    # ── Column checks ─────────────────────────────────────────────────────────
 
    def has_necessary_columns(self, dataframe):
        missing = set(self.not_null_columns) - set(dataframe.columns)
        return len(missing) == 0
 
    # ── Normalisation (run before quality cuts) ───────────────────────────────
 
    def normalize_column_names(self, detections):
        """Rename raw ATLAS CSV columns to the internal pipeline schema."""
        rename_map = {
            'MJD':   'mjd',
            'm':     'magpsf',
            'dm':    'sigmapsf',
            'F':     'filter',
            'chi/N': 'chi2dof',
            'Pass':  'pass_flag',
        }
        rename_map = {k: v for k, v in rename_map.items() if k in detections.columns}
        return detections.rename(columns=rename_map)
 
    def map_filter_to_fid(self, detections):
        """Convert filter string ('c'/'o') to integer fid (0/1)."""
        if 'fid' in detections.columns and 'filter' not in detections.columns:
            # Already encoded as int — validate and return
            return detections[detections['fid'].isin(self.VALID_FIDS)]
        if 'filter' not in detections.columns:
            raise ValueError("No filter column found ('F' / 'filter' / 'fid').")
        detections = detections.copy()
        detections['fid'] = (
            detections['filter'].astype(str).str.lower().map(self.BAND_MAP)
        )
        unknown = detections['fid'].isna()
        if unknown.any():
            import warnings
            warnings.warn(
                f"Unknown ATLAS filters dropped: "
                f"{detections.loc[unknown, 'filter'].unique()}"
            )
        return detections[~unknown].copy()
 
    def inject_neutral_columns(self, detections):
        """
        Inject columns that ATLAS does not provide but lc_classifier expects.
        Also sets magpsf_ml / sigmapsf_ml (no host-galaxy correction in ATLAS).
        """
        detections = detections.copy()
 
        # No host-galaxy correction available -> ml = raw
        detections['magpsf_ml']   = detections['magpsf']
        detections['sigmapsf_ml'] = detections['sigmapsf']
 
        # Aliases expected by some lc_classifier sub-modules
        detections['magpsf_corr']       = detections['magpsf']
        detections['sigmapsf_corr']     = detections['sigmapsf']
        detections['sigmapsf_corr_ext'] = detections['sigmapsf']
 
        # isdiffpos from flux sign (uJy); default positive if unavailable
        if 'uJy' in detections.columns:
            detections['isdiffpos'] = np.where(
                detections['uJy'].fillna(0) >= 0, 1, -1
            ).astype(int)
        else:
            detections['isdiffpos'] = 1
            detections['uJy']  = np.nan
            detections['duJy'] = np.nan
 
        if 'duJy' not in detections.columns:
            detections['duJy'] = np.nan
 
        return detections
 
    # ── Quality cuts ──────────────────────────────────────────────────────────
 
    def discard_invalid_value_detections(self, detections):
        detections = detections.replace([np.inf, -np.inf], np.nan)
        valid = detections[self.not_null_columns].notna().all(axis=1)
        detections = detections[valid.values]
        # Reject ATLAS sentinel non-detection magnitude (99.0)
        detections = detections[detections['magpsf'] < 90.0]
        detections[self.not_null_columns] = detections[self.not_null_columns].apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        )
        return detections
 
    def discard_noisy_detections(self, detections):
        return detections[
            (detections['sigmapsf'] > 0.0) &
            (detections['sigmapsf'] < self.max_sigma)
        ]
 
    def discard_by_chi2(self, detections):
        if 'chi2dof' not in detections.columns:
            return detections
        return detections[
            detections['chi2dof'].isna() |
            (detections['chi2dof'] < self.max_chi2dof)
        ]
 
    def discard_by_pass_flag(self, detections):
        if 'pass_flag' not in detections.columns or not self.require_pass:
            return detections
        return detections[detections['pass_flag'] == 1]
 
    def discard_invalid_bands(self, detections):
        return detections[detections['fid'].isin(self.VALID_FIDS)]
 
    def drop_duplicates(self, detections):
        """
        Drop duplicate (oid, mjd, fid) triplets.
        Uses fid in the key (same fix as LSSTLightcurvePreprocessor) to avoid
        dropping valid simultaneous c/o observations on the same MJD.
        """
        assert detections.index.name == 'oid'
        detections = detections.copy()
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'mjd', 'fid'], keep='first')
        detections = detections[[c for c in detections.columns if c != 'oid']]
        return detections
 
    def enough_alerts(self, detections, min_dets=5):
        """Keep objects with at least min_dets detections in any single band."""
        objects = detections.groupby('oid')
        indexes = [
            oid for oid, group in objects
            if any(
                len(group[group['fid'] == fid]) > min_dets
                for fid in self.VALID_FIDS
            )
        ]
        return detections.loc[indexes]
 
    # ── Main entry point ──────────────────────────────────────────────────────
 
    def preprocess(self, dataframe, objects=None):
        """
        Preprocess an ATLAS forced photometry DataFrame.
 
        Parameters
        ----------
        dataframe : pd.DataFrame
            Raw ATLAS detections with index 'oid'.
            Accepts both raw CSV column names (MJD, m, dm, F, ...) and
            the internal pipeline schema (mjd, magpsf, sigmapsf, fid, ...).
        objects : ignored
            Accepted for API compatibility with ZTFLightcurvePreprocessor;
            not used (ATLAS has no objects table / host-galaxy correction).
 
        Returns
        -------
        pd.DataFrame
            Preprocessed detections with lc_classifier-compatible columns:
            time, band, magnitude, error, difference_flux,
            difference_flux_error, plus all pipeline columns.
        """
        self.verify_dataframe(dataframe)
        dataframe = self.normalize_column_names(dataframe)
        dataframe = self.map_filter_to_fid(dataframe)
        dataframe = self.discard_invalid_value_detections(dataframe)
        dataframe = self.discard_by_chi2(dataframe)
        dataframe = self.discard_by_pass_flag(dataframe)
        dataframe = self.discard_noisy_detections(dataframe)
        dataframe = self.discard_invalid_bands(dataframe)
        dataframe = self.inject_neutral_columns(dataframe)
        dataframe = self.drop_duplicates(dataframe)
        dataframe = self.enough_alerts(dataframe)
        dataframe = self.rename_columns_detections(dataframe)
        return dataframe
 
    def rename_columns_detections(self, detections):
        return detections.rename(columns=self.column_translation, errors='ignore')
 