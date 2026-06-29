from typing import Tuple, List
from functools import lru_cache

from ..core.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging

#%%

class ZTFColorFeatureExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'g-r_max', 'g-r_mean', 'g-r_max_corr', 'g-r_mean_corr'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'magpsf', 'magnitude'

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        """
        Parameters
        ----------
        detections 
        DataFrame with detections of an object.
        kwargs Not required.
        Returns :class:pandas.`DataFrame`
        -------
        """
        # pd.options.display.precision = 10
        def aux_function(oid_detections):
            oid = oid_detections.index.values[0]
            bands = oid_detections['band'].values
            unique_fids = np.unique(bands)
            if 1 not in unique_fids or 2 not in unique_fids:
                logging.debug(
                    f'extractor=COLOR  object={oid}  required_cols={self.get_required_keys()}  filters_qty=2')
                return self.nan_series()

            mag_corr = oid_detections['magnitude'].values
            g_band_mag_corr = mag_corr[bands == 1]
            r_band_mag_corr = mag_corr[bands == 2]

            mag = oid_detections['magpsf'].values
            g_band_mag = mag[bands == 1]
            r_band_mag = mag[bands == 2]

            g_r_max = g_band_mag.min() - r_band_mag.min()
            g_r_mean = g_band_mag.mean() - r_band_mag.mean()

            g_r_max_corr = g_band_mag_corr.min() - r_band_mag_corr.min()
            g_r_mean_corr = g_band_mag_corr.mean() - r_band_mag_corr.mean()

            if g_r_max == g_r_max_corr and g_r_mean == g_r_mean_corr:
                data = [g_r_max, g_r_mean, np.nan, np.nan]
            else:
                data = [g_r_max, g_r_mean, g_r_max_corr, g_r_mean_corr]
            oid_color = pd.Series(data=data, index=self.get_features_keys())
            return oid_color
        
        colors = detections.apply(aux_function)
        colors.index.name = 'oid'
        return colors

#%%

class ZTFColorForcedFeatureExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'g-r_max', 'g-r_mean'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'magnitude'

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        """
        Parameters
        ----------
        detections
        DataFrame with detections of an object.
        kwargs Not required.
        Returns :class:pandas.`DataFrame`
        -------
        """

        def aux_function(oid_detections):
            oid = oid_detections.index.values[0]
            bands = oid_detections['band'].values
            unique_fids = np.unique(bands)
            if 1 not in unique_fids or 2 not in unique_fids:
                logging.debug(
                    f'extractor=COLOR  object={oid}  required_cols={self.get_required_keys()}  filters_qty=2')
                return self.nan_series()

            mag = oid_detections['magnitude'].values
            g_band_mag = mag[bands == 1]
            r_band_mag = mag[bands == 2]

            g_r_max = g_band_mag.min() - r_band_mag.min()
            g_r_mean = g_band_mag.mean() - r_band_mag.mean()

            data = [g_r_max, g_r_mean]
            oid_color = pd.Series(data=data, index=self.get_features_keys())
            return oid_color

        colors = detections.apply(aux_function)
        colors.index.name = 'oid'
        return colors

#%%

class ElasticcColorFeatureExtractor(FeatureExtractor):
    def __init__(self, bands: List[str]) -> None:
        super().__init__()
        if len(bands) < 2:
            raise ValueError('Elasticc color feature extractor needs at least two bands')
        self.bands = bands

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return [f'{self.bands[i]}-{self.bands[i+1]}'for i in range(len(self.bands)-1)]

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'difference_flux'

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        """
        Parameters
        ----------
        detections
        DataFrame with detections of an object.
        kwargs Not required.
        Returns :class:pandas.`DataFrame`
        -------
        """

        def aux_function(oid_detections):
            oid = oid_detections.index.values[0]
            bands = oid_detections['band'].values
            fluxes = oid_detections['difference_flux'].values
            available_bands = np.unique(bands)
            d = {}
            for band in available_bands:
                band_mask = bands == band
                band_fluxes = fluxes[band_mask]
                band_fluxes_abs = np.abs(band_fluxes)
                band_90p = np.percentile(band_fluxes_abs, 90)
                d[band] = band_90p

            output = []
            for i in range(len(self.bands)-1):
                if (self.bands[i] not in available_bands 
                or self.bands[i+1] not in available_bands):
                    output.append(np.nan)
                    continue
                output.append(d[self.bands[i]]/(d[self.bands[i+1]] + 1))

            oid_color = pd.Series(data=output, index=self.get_features_keys())
            return oid_color

        colors = detections.apply(aux_function)
        colors.index.name = 'oid'
        return colors
    
#%%

class ZTFColorFeatureExtractor3bands(FeatureExtractor):
    """
    Versión extendida de ZTFColorFeatureExtractor que soporta
    cualquier combinación de bandas (g=1, r=2, i=3).
    Calcula colores para todos los pares consecutivos de bandas.
    """
    def __init__(self, bands=(1, 2, 3)):
        super().__init__()
        if len(bands) < 2:
            raise ValueError('Se necesitan al menos dos bandas')
        self.bands = sorted(bands)
        # Pares de bandas: (1,2), (1,3), (2,3)
        self.band_names = {1: 'g', 2: 'r', 3: 'i'}
        self.pairs = [
            (self.bands[i], self.bands[j])
            for i in range(len(self.bands))
            for j in range(i+1, len(self.bands))
        ]

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        keys = []
        for b1, b2 in self.pairs:
            n1 = self.band_names.get(b1, str(b1))
            n2 = self.band_names.get(b2, str(b2))
            keys += [
                f'{n1}-{n2}_max',
                f'{n1}-{n2}_mean',
                f'{n1}-{n2}_max_corr',
                f'{n1}-{n2}_mean_corr'
            ]
        return tuple(keys)

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'magpsf', 'magnitude'

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0), **kwargs)

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        def aux_function(oid_detections):
            oid = oid_detections.index.values[0]
            bands = oid_detections['band'].values
            unique_fids = np.unique(bands)

            mag_corr = oid_detections['magnitude'].values
            mag = oid_detections['magpsf'].values

            data = []
            for b1, b2 in self.pairs:
                n1 = self.band_names.get(b1, str(b1))
                n2 = self.band_names.get(b2, str(b2))

                if b1 not in unique_fids or b2 not in unique_fids:
                    logging.debug(
                        f'extractor=COLOR object={oid} '
                        f'par={n1}-{n2} no disponible'
                    )
                    data += [np.nan, np.nan, np.nan, np.nan]
                    continue

                mag1      = mag[bands == b1]
                mag2      = mag[bands == b2]
                mag1_corr = mag_corr[bands == b1]
                mag2_corr = mag_corr[bands == b2]

                color_max  = mag1.min()      - mag2.min()
                color_mean = mag1.mean()     - mag2.mean()
                color_max_corr  = mag1_corr.min()  - mag2_corr.min()
                color_mean_corr = mag1_corr.mean() - mag2_corr.mean()

                # Si corr == no corr, la corrección no se aplicó
                if color_max == color_max_corr and color_mean == color_mean_corr:
                    data += [color_max, color_mean, np.nan, np.nan]
                else:
                    data += [color_max, color_mean, color_max_corr, color_mean_corr]

            return pd.Series(data=data, index=self.get_features_keys())

        colors = detections.apply(aux_function)
        colors.index.name = 'oid'
        return colors
    
#%%

class LSSTColorFeatureExtractor(FeatureExtractor):
    """
    Versión extendida de ZTFColorFeatureExtractor que soporta
    cualquier combinación de bandas (u=6, g=1, r=2, i=3, z=4, y=5).
    Calcula colores para todos los pares consecutivos de bandas.
    """
    def __init__(self, bands=(1, 2, 3, 4, 5, 6)):
        super().__init__()
        if len(bands) < 2:
            raise ValueError('Se necesitan al menos dos bandas')
        self.bands = sorted(bands)
        self.band_names = {1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y', 6: 'u'}
        self.pairs = [
            (self.bands[i], self.bands[j])
            for i in range(len(self.bands))
            for j in range(i+1, len(self.bands))
        ]
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        keys = []
        for b1, b2 in self.pairs:
            n1 = self.band_names.get(b1, str(b1))
            n2 = self.band_names.get(b2, str(b2))
            keys += [
                f'{n1}-{n2}_max',
                f'{n1}-{n2}_mean',
                f'{n1}-{n2}_max_corr',
                f'{n1}-{n2}_mean_corr'
            ]
        return tuple(keys)

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'magpsf', 'magnitude'

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0), **kwargs)

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        def aux_function(oid_detections):
            oid = oid_detections.index.values[0]
            bands = oid_detections['band'].values
            unique_fids = np.unique(bands)

            mag_corr = oid_detections['magnitude'].values
            mag = oid_detections['magpsf'].values

            data = []
            for b1, b2 in self.pairs:
                n1 = self.band_names.get(b1, str(b1))
                n2 = self.band_names.get(b2, str(b2))

                if b1 not in unique_fids or b2 not in unique_fids:
                    logging.debug(
                        f'extractor=COLOR object={oid} '
                        f'par={n1}-{n2} no disponible'
                    )
                    data += [np.nan, np.nan, np.nan, np.nan]
                    continue
                    
                mag1      = mag[bands == b1]
                mag2      = mag[bands == b2]
                mag1_corr = mag_corr[bands == b1]
                mag2_corr = mag_corr[bands == b2]
                
                color_max  = mag1.min()      - mag2.min()
                color_mean = mag1.mean()     - mag2.mean()
                color_max_corr  = mag1_corr.min()  - mag2_corr.min()
                color_mean_corr = mag1_corr.mean() - mag2_corr.mean()


                # Si corr == no corr, la corrección no se aplicó
                if color_max == color_max_corr and color_mean == color_mean_corr:
                    data += [color_max, color_mean, np.nan, np.nan]
                else:
                    data += [color_max, color_mean, color_max_corr, color_mean_corr]

            return pd.Series(data=data, index=self.get_features_keys())

        colors = detections.apply(aux_function)
        colors.index.name = 'oid'
        return colors

#%%

class ATLASColorFeatureExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'c-o_max', 'c-o_mean', 'c-o_max_corr', 'c-o_mean_corr'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'band', 'magpsf', 'magnitude'

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(self, detections, **kwargs):
        """
        Parameters
        ----------
        detections 
        DataFrame with detections of an object.
        kwargs Not required.
        Returns :class:pandas.`DataFrame`
        -------
        """
        # pd.options.display.precision = 10
        def aux_function(oid_detections):
            oid = oid_detections.index.values[0]
            bands = oid_detections['band'].values
            unique_fids = np.unique(bands)
            if 1 not in unique_fids or 2 not in unique_fids:
                logging.debug(
                    f'extractor=COLOR  object={oid}  required_cols={self.get_required_keys()}  filters_qty=2')
                return self.nan_series()

            mag_corr = oid_detections['magnitude'].values
            c_band_mag_corr = mag_corr[bands == 0]
            o_band_mag_corr = mag_corr[bands == 1]

            mag = oid_detections['magpsf'].values
            c_band_mag = mag[bands == 0]
            o_band_mag = mag[bands == 1]

            c_o_max = c_band_mag.min() - o_band_mag.min()
            c_o_mean = c_band_mag.mean() - o_band_mag.mean()

            c_o_max_corr = c_band_mag_corr.min() - o_band_mag_corr.min()
            c_o_mean_corr = c_band_mag_corr.mean() - o_band_mag_corr.mean()

            if c_o_max == c_o_max_corr and c_o_mean == c_o_mean_corr:
                data = [c_o_max, c_o_mean, np.nan, np.nan]
            else:
                data = [c_o_max, c_o_mean, c_o_max_corr, c_o_mean_corr]
            oid_color = pd.Series(data=data, index=self.get_features_keys())
            return oid_color
        
        colors = detections.apply(aux_function)
        colors.index.name = 'oid'
        return colors