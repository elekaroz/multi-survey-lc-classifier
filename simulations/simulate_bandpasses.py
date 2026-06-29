from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

# Longitudes de onda efectivas y FWHM aproximados de las bandas ATLAS
# (para la aproximación gaussiana de fallback)
_ATLAS_BAND_APPROX = {
    'atlasc': {'lambda_eff': 5350.0, 'fwhm': 2300.0}, 
    'atlaso': {'lambda_eff': 6790.0, 'fwhm': 2600.0}
}

_REGISTERED = set()

# En register_atlas_bandpasses(), reemplazar la carga desde fichero local
# por descarga automática desde SVO si los ficheros no existen:

def _download_atlas_bandpasses(bp_dir: Path) -> None:
    """Descarga las curvas de transmisión de ATLAS desde el SVO."""
    from astroquery.svo_fps import SvoFps

    for band_name, svo_id, filename in [
        ('atlasc', ['Misc/Atlas.cyan.dat.txt',   'atlas_c.dat']),
        ('atlaso', ['Misc/Atlas.orange.dat.txt', 'atlas_o.dat']),
    ]:
        out_path = bp_dir / filename
        if out_path.exists():
            continue
        print(f"Downloading {svo_id} from SVO...")
        tbl = SvoFps.get_transmission_data(svo_id)
        np.savetxt(out_path,
                   np.column_stack([tbl['Wavelength'], tbl['Transmission']]))
        print(f"Saved: {out_path}")

def register_atlas_bandpasses(
    bandpass_dir: str | Path | None = None,
    force: bool = False,
) -> None:
    """
    Registra las bandas 'atlasc' y 'atlaso' en sncosmo.

    """
    import sncosmo

    bp_dir = Path(bandpass_dir) if bandpass_dir else _HERE

    for band_name, filename in [('atlasc', 'Misc_Atlas.cyan.dat.txt'), ('atlaso', 'Misc_Atlas.orange.dat.txt')]:
        if band_name in _REGISTERED and not force:
            print(f'Using transmission curve file for {band_name}')
            continue

        # Intentar descargar desde SVO si el fichero no existe
        bp_path = bp_dir / filename
        if not bp_path.exists():
            try:
                _download_atlas_bandpasses(bp_dir)
            except Exception as e:
                warnings.warn(
                    f"Couldn't download from SVO: {e}. ",
                    #f"Se usará aproximación gaussiana.",
                    UserWarning, stacklevel=2,
                )

        # Intentar cargar desde fichero real
        if bp_path.exists():
            data = np.loadtxt(bp_path)
            wavelengths   = data[:, 0]
            transmissions = data[:, 1]
            print(f'File for {band_name} found')
        else:
            warnings.warn(
                f"File not found: {bp_path}. "
                f"Using gaussian approximation '{band_name}'. "
                f"Download real bandpass files from ATLAS for best results ",
                UserWarning,
                stacklevel=2,
            )


        try:

            if transmissions[0] > 0.001:
                wavelengths   = np.concatenate([[wavelengths[0] - 10], wavelengths])
                transmissions = np.concatenate([[0.0], transmissions])
            if transmissions[-1] > 0.001:
                wavelengths   = np.concatenate([wavelengths, [wavelengths[-1] + 10]])
                transmissions = np.concatenate([transmissions, [0.0]])

            bp = sncosmo.Bandpass(wavelengths, transmissions, name=band_name)

            sncosmo.register(bp, force=True)
            _REGISTERED.add(band_name)
            print(f'Band {band_name} registered in sncosmo')
        except Exception as e:
            warnings.warn(f"Couldn't register '{band_name}': {e}", UserWarning)
