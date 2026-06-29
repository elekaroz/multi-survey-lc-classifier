import argparse
import glob as _glob
import io
import logging
import os
import time
import warnings
import zipfile
from pathlib import Path
 
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from tqdm import tqdm
 
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMY MAPS
# ─────────────────────────────────────────────────────────────────────────────
 
TNS_TYPE_MAP = {
    "SN Ia":           "SNIa",  "SN Ia-91T-like":  "SNIa",
    "SN Ia-91bg-like": "SNIa",  "SN Ia-02cx-like": "SNIa",
    "SN Ia-CSM":       "SNIa",  "SN Ia-pec":       "SNIa",
    "SN Ib":           "SNIbc", "SN Ic":            "SNIbc",
    "SN Ib/c":         "SNIbc", "SN Ic-BL":         "SNIbc",
    "SN Ibn":          "SNIbc", "SN Icn":            "SNIbc",
    "SN II":           "SNII",  "SN IIP":            "SNII",
    "SN IIL":          "SNII",  "SN IIn":            "SNII",
    "SN IIb":          "SNII",  "SN II-pec":         "SNII",
    "SLSN-I":          "SLSN",  "SLSN-II":           "SLSN",
    "SLSN-R":          "SLSN",
    "Nova":            "CV/Nova", "Nova-like": "CV/Nova", "CV": "CV/Nova",
}
 
SIMBAD_TYPE_MAP = {
    "SN":        "SNII",   "SNIa":      "SNIa",
    "SNIb":      "SNIbc",  "SNIc":      "SNIbc",
    "QSO":       "QSO",    "QSO_Candidate": "QSO",
    "Seyfert":   "AGN",    "Seyfert_1": "AGN",
    "Seyfert_2": "AGN",    "AGN":       "AGN",    "LINER": "AGN",
    "BLLac":     "Blazar", "Blazar":    "Blazar",
    "YSO":       "YSO",    "Orion_V*":  "YSO",    "TTau*": "YSO",
    "CV*":       "CV/Nova","Nova":      "CV/Nova", "DwarfNova": "CV/Nova",
    "Mira":      "LPV",    "OH/IR":     "LPV",    "SRstar": "LPV",
    "LPV*":      "LPV",
    "EclBin":    "E",      "EB*":       "E",      "SB*":   "E",
    "RRLyr":     "RRL",
    "Cepheid":   "CEP",    "deltaCep":  "CEP",
    "SX_Phe":    "DSCT",   "delta_Sct": "DSCT",   "DSCT":  "DSCT",
    "RoAp":      "Periodic-Other", "PulsV*": "Periodic-Other",
    "gammaDor":  "Periodic-Other",
}
 
MILLIQUAS_TYPE_MAP = {
    "Q": "QSO", "A": "AGN", "B": "Blazar",
    "K": "QSO", "N": "AGN", "S": "AGN",
}
 
VSX_TYPE_MAP = {
    "RRAB":  "RRL",  "RRC":   "RRL",  "RR":    "RRL",  "RRAB/RRC": "RRL",
    "DCEP":  "CEP",  "DCEPS": "CEP",  "CEP":   "CEP",  "CWA": "CEP", "CWB": "CEP",
    "DSCT":  "DSCT", "HADS":  "DSCT", "SX":    "DSCT",
    "MIRA":  "LPV",  "SR":    "LPV",  "SRB":   "LPV",  "SRA": "LPV",
    "SRD":   "LPV",  "L":     "LPV",  "LB":    "LPV",  "LC":  "LPV",
    "EA":    "E",    "EB":    "E",    "EW":    "E",    "ELL": "E",
    "CV":    "CV/Nova", "UG":  "CV/Nova", "UGSS": "CV/Nova",
    "UGSU":  "CV/Nova", "UGZ": "CV/Nova", "NL":   "CV/Nova",
    "ZAND":  "CV/Nova", "AM":  "CV/Nova",
    "ORION": "YSO",  "INT":   "YSO",  "IT":    "YSO",
    "SPB":   "Periodic-Other", "BCEP": "Periodic-Other",
    "GDOR":  "Periodic-Other", "ROAP": "Periodic-Other",
}
 
SOURCE_PRIORITY = ["TNS", "SIMBAD", "Milliquas", "VSX"]
 
LABEL_ORDER = [
    'SNIa', 'SNIbc', 'SNII', 'SLSN',
    'QSO', 'AGN', 'Blazar',
    'YSO', 'CV/Nova', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other'
]
 

FEATURE_PATTERNS = [
    "features_ztf_strict.parquet",
    "features_lsst_strict.parquet",
    "features_comb_strict.parquet",
    "features_ztf_relaxed.parquet",
    "features_lsst_relaxed.parquet",
    "features_comb_relaxed.parquet",
    "features_atlas_strict.parquet",
    "features_atlas_relaxed.parquet",
]
 
TRANSIENT_LABELS = {"SNIa", "SNIbc", "SNII", "SLSN"}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. TNS
# ─────────────────────────────────────────────────────────────────────────────
 
def query_tns_bulk(oids: list[str], api_key: str,
                   bot_id: str, bot_name: str) -> pd.DataFrame:
    """
    Hace crossmatch de una lista de OIDs en el CSV descargado de TNS.
    Puede tomar OIDs de cualquier survey y los busca en TNS por nombre interno.
    """
    log.info("Downloading TNS bulk CSV (this may take ~1 min)...")
    url = ("https://www.wis-tns.org/system/files/tns_public_objects/"
           "tns_public_objects.csv.zip")
    headers = {
        "User-Agent": f'tns_marker{{"tns_id":{bot_id},'
                      f'"type":"bot","name":"{bot_name}"}}',
    }
    resp = requests.get(url, headers=headers,
                        params={"api_key": api_key}, timeout=300)
    resp.raise_for_status()
 
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        fname = [n for n in zf.namelist() if n.endswith(".csv")][0]
        tns_df = pd.read_csv(zf.open(fname), low_memory=False, skiprows=1)
 
    log.info(f"TNS catalog loaded: {len(tns_df):,} objects")
    tns_df.columns = tns_df.columns.str.strip().str.lower().str.replace(" ", "_")
 
    tns_df["internal_names"] = tns_df["internal_names"].fillna("")
    tns_exploded = tns_df.assign(
        internal_name=tns_df["internal_names"].str.split(",")
    ).explode("internal_name")
    tns_exploded["internal_name"] = tns_exploded["internal_name"].str.strip()
 
    oids_set = set(oids)
    matched  = tns_exploded[tns_exploded["internal_name"].isin(oids_set)].copy()
    type_col = "type" if "type" in matched.columns else "obj_type"
    matched["label_tns"] = matched[type_col].map(TNS_TYPE_MAP)
 
    if "classification_source" in matched.columns:
        spec_mask = matched["classification_source"].str.contains(
            "Spectroscop", case=False, na=False)
        matched = matched[spec_mask | matched["label_tns"].notna()]
 
    result = matched[["internal_name", "name", type_col, "label_tns"]].copy()
    result.columns = ["oid", "tns_name", "tns_type_raw", "label_tns"]
    result = result.dropna(subset=["label_tns"]).drop_duplicates(subset="oid")
 
    log.info(f"TNS matches with valid label: {len(result)}")
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1b. TNS positional (for lsst_only mode, no ZTF internal names available)
# ─────────────────────────────────────────────────────────────────────────────
 
def query_tns_positional(coords_df: pd.DataFrame,
                         api_key: str, bot_id: str, bot_name: str,
                         radius_arcsec: float = 2.0) -> pd.DataFrame:
    """
    Crossmatch por coordenadas en el CSV de TNS. Sirve cuando los
    nombres internos de survey no están disponibles.
    """
    log.info("Downloading TNS bulk CSV for positional crossmatch...")
    url = ("https://www.wis-tns.org/system/files/tns_public_objects/"
           "tns_public_objects.csv.zip")
    headers = {
        "User-Agent": f'tns_marker{{"tns_id":{bot_id},'
                      f'"type":"bot","name":"{bot_name}"}}',
    }
    resp = requests.get(url, headers=headers,
                        params={"api_key": api_key}, timeout=300)
    resp.raise_for_status()
 
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        fname  = [n for n in zf.namelist() if n.endswith(".csv")][0]
        tns_df = pd.read_csv(zf.open(fname), low_memory=False, skiprows=1)
 
    log.info(f"TNS catalog loaded: {len(tns_df):,} objects")
    tns_df.columns = tns_df.columns.str.strip().str.lower().str.replace(" ", "_")
 
    # TNS bulk CSV columns for coordinates are 'ra' and 'declination'
    ra_col  = "ra"          if "ra"          in tns_df.columns else None
    dec_col = "declination" if "declination" in tns_df.columns else (
              "dec"         if "dec"         in tns_df.columns else None)
 
    if ra_col is None or dec_col is None:
        log.warning(f"TNS bulk CSV has no RA/Dec columns "
                    f"(found: {list(tns_df.columns[:10])}). Skipping positional TNS.")
        return pd.DataFrame(columns=["oid", "tns_name", "tns_type_raw", "label_tns"])
 
    type_col = "type" if "type" in tns_df.columns else "obj_type"
    tns_df   = tns_df.dropna(subset=[ra_col, dec_col]).copy()
    tns_df["label_tns"] = tns_df[type_col].map(TNS_TYPE_MAP)
    tns_df   = tns_df.dropna(subset=["label_tns"])
 
    if tns_df.empty:
        log.warning("No TNS objects with valid label after type mapping.")
        return pd.DataFrame(columns=["oid", "tns_name", "tns_type_raw", "label_tns"])
 
    log.info(f"TNS objects with valid label: {len(tns_df):,}. Running positional match...")
 
    tns_sky   = SkyCoord(ra=tns_df[ra_col].values * u.deg,
                         dec=tns_df[dec_col].values * u.deg)
    input_sky = SkyCoord(ra=coords_df["ra"].values * u.deg,
                         dec=coords_df["dec"].values * u.deg)
 
    idx, sep, _ = match_coordinates_sky(tns_sky, input_sky)
    mask = sep.arcsec < radius_arcsec
 
    tns_matched         = tns_df[mask].copy().reset_index(drop=True)
    tns_matched["oid"]  = coords_df.iloc[idx[mask]]["oid"].values
 
    name_col = "name" if "name" in tns_matched.columns else tns_matched.columns[0]
    result   = tns_matched[["oid", name_col, type_col, "label_tns"]].copy()
    result.columns = ["oid", "tns_name", "tns_type_raw", "label_tns"]
    result   = result.dropna(subset=["label_tns"]).drop_duplicates(subset="oid")
 
    log.info(f"TNS positional matches with valid label: {len(result)}")
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. SIMBAD
# ─────────────────────────────────────────────────────────────────────────────
 
def query_simbad_positional(coords_df: pd.DataFrame,
                            radius_arcsec: float = 1.5) -> pd.DataFrame:
    log.info("Querying SIMBAD (positional, batched)...")
 
    custom_simbad = Simbad()
    custom_simbad.reset_votable_fields()
    custom_simbad.add_votable_fields("otype", "otypes", "ra", "dec")
    custom_simbad.TIMEOUT = 120
 
    valid = coords_df.dropna(subset=["ra", "dec"])
    if len(valid) < len(coords_df):
        log.warning(f"Skipping {len(coords_df), len(valid)} objects "
                    f"with NaN coordinates in SIMBAD query")
 
    BATCH   = 500
    results = []
 
    for start in tqdm(range(0, len(valid), BATCH), desc="SIMBAD batches"):
        batch = valid.iloc[start:start + BATCH]
        sky   = SkyCoord(ra=batch["ra"].values * u.deg,
                         dec=batch["dec"].values * u.deg)
        try:
            result = custom_simbad.query_region(sky, radius=radius_arcsec * u.arcsec)
        except Exception as e:
            log.warning(f"SIMBAD batch {start}-{start+BATCH} failed: {e}")
            time.sleep(5)
            continue
 
        if result is None:
            continue
 
        sim_df  = result.to_pandas()
        ra_col  = "RA"  if "RA"  in sim_df.columns else "ra"
        dec_col = "DEC" if "DEC" in sim_df.columns else "dec"
 
        try:
            sim_coords = SkyCoord(
                ra=sim_df[ra_col].values,
                dec=sim_df[dec_col].values,
                unit=("hourangle", "deg")
                     if sim_df[ra_col].dtype == object else ("deg", "deg")
            )
        except Exception:
            continue
 
        idx, sep, _ = match_coordinates_sky(sim_coords, sky)
        mask = sep.arcsec < radius_arcsec
 
        sim_df        = sim_df[mask].copy()
        sim_df["oid"] = batch.iloc[idx[mask]]["oid"].values
 
        if "otypes.otype" in sim_df.columns:
            otypes_col = sim_df.groupby(sim_df.index)["otypes.otype"].apply(
                lambda x: "|".join(x.dropna().astype(str))
            )
            sim_df = sim_df[~sim_df.index.duplicated(keep="first")].copy()
            sim_df["otypes_flat"] = otypes_col
        elif "OTYPES" in sim_df.columns:
            sim_df["otypes_flat"] = sim_df["OTYPES"].fillna("").astype(str)
        else:
            sim_df["otypes_flat"] = sim_df.get(
                "otypes", pd.Series("", index=sim_df.index))
 
        otype_col = "OTYPE" if "OTYPE" in sim_df.columns else "otype"
        sim_df    = sim_df.rename(columns={otype_col: "otype"})
 
        results.append(sim_df[["oid", "otype", "otypes_flat"]])
        time.sleep(0.3)
 
    if not results:
        log.warning("No SIMBAD matches found.")
        return pd.DataFrame(columns=["oid", "simbad_otype", "label_simbad"])
 
    simbad_all = pd.concat(results, ignore_index=True).rename(columns={
        "otype":       "simbad_otype",
        "otypes_flat": "simbad_otypes",
    })
 
    def _map_simbad(row):
        label = SIMBAD_TYPE_MAP.get(row["simbad_otype"])
        if label is None and pd.notna(row.get("simbad_otypes")):
            for t in str(row["simbad_otypes"]).split("|"):
                label = SIMBAD_TYPE_MAP.get(t.strip())
                if label:
                    break
        return label
 
    simbad_all["label_simbad"] = simbad_all.apply(_map_simbad, axis=1)
    simbad_all = (simbad_all
                  .dropna(subset=["label_simbad"])
                  .drop_duplicates(subset="oid"))
 
    log.info(f"SIMBAD matches with valid label: {len(simbad_all)}")
    return simbad_all[["oid", "simbad_otype", "label_simbad"]]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Milliquas
# ─────────────────────────────────────────────────────────────────────────────
 
def query_milliquas(coords_df: pd.DataFrame,
                    radius_arcsec: float = 1.5) -> pd.DataFrame:
    log.info("Querying Milliquas via VizieR...")
 
    v = Vizier(columns=["RAJ2000", "DEJ2000", "Type", "Name"], row_limit=-1)
    v.TIMEOUT = 120
 
    input_sky = SkyCoord(ra=coords_df["ra"].values * u.deg,
                         dec=coords_df["dec"].values * u.deg)
    BATCH   = 1000
    results = []
 
    for start in tqdm(range(0, len(coords_df), BATCH), desc="Milliquas batches"):
        batch     = coords_df.iloc[start:start + BATCH]
        sky_batch = SkyCoord(ra=batch["ra"].values * u.deg,
                             dec=batch["dec"].values * u.deg)
        try:
            tables = v.query_region(sky_batch,
                                    radius=radius_arcsec * u.arcsec,
                                    catalog="VII/290")
        except Exception as e:
            log.warning(f"Milliquas batch {start} failed: {e}")
            time.sleep(3)
            continue
 
        if not tables:
            continue
 
        mq     = tables[0].to_pandas()
        mq_sky = SkyCoord(ra=mq["RAJ2000"].values * u.deg,
                          dec=mq["DEJ2000"].values * u.deg)
 
        idx, sep, _ = match_coordinates_sky(
            mq_sky, input_sky[start:start + BATCH])
        mask      = sep.arcsec < radius_arcsec
        mq        = mq[mask].copy()
        mq["oid"] = batch.iloc[idx[mask]]["oid"].values
        results.append(mq[["oid", "Type"]])
        time.sleep(0.3)
 
    if not results:
        log.warning("No Milliquas matches found.")
        return pd.DataFrame(columns=["oid", "milliquas_type", "label_milliquas"])
 
    mq_all = pd.concat(results, ignore_index=True)
    mq_all.columns = ["oid", "milliquas_type"]
    mq_all["label_milliquas"] = (mq_all["milliquas_type"]
                                 .astype(str).str.strip().str[0]
                                 .map(MILLIQUAS_TYPE_MAP))
    mq_all = (mq_all
              .dropna(subset=["label_milliquas"])
              .drop_duplicates(subset="oid"))
 
    log.info(f"Milliquas matches with valid label: {len(mq_all)}")
    return mq_all[["oid", "milliquas_type", "label_milliquas"]]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. VSX
# ─────────────────────────────────────────────────────────────────────────────
 
def query_vsx(coords_df: pd.DataFrame,
              radius_arcsec: float = 1.5) -> pd.DataFrame:
    log.info("Querying VSX via VizieR...")
 
    v = Vizier(columns=["RAJ2000", "DEJ2000", "Type", "Name"], row_limit=-1)
    v.TIMEOUT = 120
 
    input_sky = SkyCoord(ra=coords_df["ra"].values * u.deg,
                         dec=coords_df["dec"].values * u.deg)
    BATCH   = 1000
    results = []
 
    for start in tqdm(range(0, len(coords_df), BATCH), desc="VSX batches"):
        batch     = coords_df.iloc[start:start + BATCH]
        sky_batch = SkyCoord(ra=batch["ra"].values * u.deg,
                             dec=batch["dec"].values * u.deg)
        try:
            tables = v.query_region(sky_batch,
                                    radius=radius_arcsec * u.arcsec,
                                    catalog="B/vsx")
        except Exception as e:
            log.warning(f"VSX batch {start} failed: {e}")
            time.sleep(3)
            continue
 
        if not tables:
            continue
 
        vsx     = tables[0].to_pandas()
        vsx_sky = SkyCoord(ra=vsx["RAJ2000"].values * u.deg,
                           dec=vsx["DEJ2000"].values * u.deg)
 
        idx, sep, _ = match_coordinates_sky(
            vsx_sky, input_sky[start:start + BATCH])
        mask       = sep.arcsec < radius_arcsec
        vsx        = vsx[mask].copy()
        vsx["oid"] = batch.iloc[idx[mask]]["oid"].values
        results.append(vsx[["oid", "Type"]])
        time.sleep(0.3)
 
    if not results:
        log.warning("No VSX matches found.")
        return pd.DataFrame(columns=["oid", "vsx_type", "label_vsx"])
 
    vsx_all = pd.concat(results, ignore_index=True)
    vsx_all.columns = ["oid", "vsx_type"]
 
    def _map_vsx(t):
        if pd.isna(t):
            return None
        t = str(t).strip().upper().split("+")[0].split("/")[0]
        return VSX_TYPE_MAP.get(t)
 
    vsx_all["label_vsx"] = vsx_all["vsx_type"].map(_map_vsx)
    vsx_all = (vsx_all
               .dropna(subset=["label_vsx"])
               .drop_duplicates(subset="oid"))
 
    log.info(f"VSX matches with valid label: {len(vsx_all)}")
    return vsx_all[["oid", "vsx_type", "label_vsx"]]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Merge + priority resolution
# ─────────────────────────────────────────────────────────────────────────────
 
def merge_labels(coords_df, tns, simbad, milliquas, vsx) -> pd.DataFrame:
    df = coords_df[["oid", "ra", "dec"]].copy()
    df = df.merge(tns[["oid", "label_tns", "tns_type_raw"]],
                  on="oid", how="left")
    df = df.merge(simbad[["oid", "label_simbad", "simbad_otype"]],
                  on="oid", how="left")
    df = df.merge(milliquas[["oid", "label_milliquas", "milliquas_type"]],
                  on="oid", how="left")
    df = df.merge(vsx[["oid", "label_vsx", "vsx_type"]],
                  on="oid", how="left")
 
    label_cols   = ["label_tns", "label_simbad", "label_milliquas", "label_vsx"]
    source_names = ["TNS", "SIMBAD", "Milliquas", "VSX"]
 
    def _resolve(row):
        chosen_label = chosen_source = None
        all_labels   = []
        for col, src in zip(label_cols, source_names):
            val = row[col]
            if pd.notna(val):
                all_labels.append(val)
                if chosen_label is None:
                    chosen_label  = val
                    chosen_source = src
        return pd.Series({
            "classALeRCE":    chosen_label,
            "label_source":   chosen_source,
            "label_conflict": len(set(all_labels)) > 1,
            "all_labels":     "|".join(all_labels) if all_labels else None,
        })
 
    resolved = df.apply(_resolve, axis=1)
    df       = pd.concat([df, resolved], axis=1)
    labeled  = df.dropna(subset=["classALeRCE"]).copy()
 
    log.info(f"\nLabel resolution summary:")
    log.info(f"Total objects queried  : {len(coords_df):>6,}")
    log.info(f"Objects with any label : {len(labeled):>6,}")
    log.info(f"Conflicting labels     : {labeled['label_conflict'].sum():>6,}")
    log.info(f"\n Source breakdown:")
    for src in source_names:
        log.info(f"{src:<12}: {(labeled['label_source'] == src).sum():>5,}")
    log.info(f"\nlass distribution:")
    for cls, n in labeled["classALeRCE"].value_counts().items():
        log.info(f"{cls:<20}: {n:>5,}")
 
    return labeled
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Quality filters
# ─────────────────────────────────────────────────────────────────────────────
 
def apply_quality_filters(labeled_df: pd.DataFrame,
                          min_per_class: int = 10) -> pd.DataFrame:
    df = labeled_df.copy()
 
    conflict_mask = df["label_conflict"]
    has_tns       = df["label_tns"].notna()
    drop_mask     = conflict_mask & ~has_tns
    if drop_mask.sum():
        log.warning(f"Dropping {drop_mask.sum()} objects with conflicting "
                    f"labels and no TNS spectroscopic classification.")
    df = df[~drop_mask].copy()
 
    for cls, n in df["classALeRCE"].value_counts().items():
        if n < min_per_class:
            log.warning(f"Class '{cls}' has only {n} objects.")
 
    transient_in_vsx = (
        df["classALeRCE"].isin(TRANSIENT_LABELS) & df["label_vsx"].notna()
    )
    if transient_in_vsx.sum():
        log.warning(f"{transient_in_vsx.sum()} transient-labeled objects "
                    f"also appear in VSX, check 'transient_vsx_flag'.")
    df["transient_vsx_flag"] = transient_in_vsx
 
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. Split feature parquets into testset / unlabeled
# ─────────────────────────────────────────────────────────────────────────────
 
def split_feature_parquets(feats_dir: Path,
                           labeled_combined: set,
                           all_combined: set) -> None:
    """
    Divide los parquets de features existentes en testset (labeled) y unlabeled,
    según se ha encontrado o no una clase en el crossmatch para cada objeto.
    """
    unlabeled_combined = all_combined - labeled_combined
 
    found = 0
    for fname in FEATURE_PATTERNS:
        src = feats_dir / fname
        if not src.exists():
            log.warning(f"Feature file not found, skipping: {src.name}")
            continue
 
        found += 1
        df_feat  = pd.read_parquet(src)
        oids_in  = df_feat.index.get_level_values(0)
 
        df_test      = df_feat[oids_in.isin(labeled_combined)]
        df_unlabeled = df_feat[oids_in.isin(unlabeled_combined)]
 
        stem      = src.stem   # e.g. "features_ztf_strict"
        test_path = feats_dir / f"{stem}_testset.parquet"
        unl_path  = feats_dir / f"{stem}_unlabeled.parquet"
 
        df_test.to_parquet(test_path)
        df_unlabeled.to_parquet(unl_path)
 
        log.info(f"{fname}: {len(df_test)} test-set, "
                 f"{len(df_unlabeled)} unlabeled")
 
    if found == 0:
        log.error(f"No feature parquets found in {feats_dir}. "
                  f"Expected files matching: {FEATURE_PATTERNS}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Crossmatch objects against classification catalogs. "
                    "Supports ZTF+LSST (both), LSST-only, or ZTF-only survey modes."
    )
    parser.add_argument(
        "--features-dir", default=None,
        help="Directory containing feature parquets produced by ztf_lsst.py. "
             "Required unless --coords-csv is provided.")
    parser.add_argument(
        "--survey-mode", default="both",
        choices=["both", "lsst_only", "ztf_only"],
        help="Survey mode matching the ztf_lsst.py SURVEY_MODE used to produce "
             "the features. Affects OID format and RA/Dec source. "
             "both: oid={lsst}_{ztf}, RA/Dec from obj_ztf_* checkpoints. "
             "lsst_only: oid=lsst_id, RA/Dec from obj_lsst_* checkpoints. "
             "ztf_only: oid=ztf_id, RA/Dec from obj_ztf_* checkpoints. "
             "(default: both)")
    parser.add_argument(
        "--obj-dir", default=None,
        help="Directory with obj_*.parquet checkpoints (for RA/Dec). "
             "Defaults to --features-dir if not provided.")
    parser.add_argument(
        "--coords-csv", default=None,
        help="CSV with columns 'oid', 'ra', 'dec' to use directly as the "
             "crossmatch input, bypassing feature parquet loading and "
             "obj_* checkpoint loading. Intended for pre-crossmatch use "
             "(e.g. lsst_only_for_crossmatch.csv from babamul_alerts.py). "
             "When provided, --features-dir is optional and feature parquet "
             "splitting is skipped.")
    parser.add_argument(
        "--output", required=True,
        help="Output path for the labels CSV")
    parser.add_argument(
        "--radius", type=float, default=1.5,
        help="Crossmatch radius in arcseconds (default: 1.5)")
    parser.add_argument(
        "--tns-api-key", default=os.environ.get("TNS_API_KEY"),
        help="TNS API key. Default: {TNS_API_KEY}"
             "If not given, TNS queries are skipped.")
    parser.add_argument(
        "--tns-bot-id", default=os.environ.get("TNS_BOT_ID"),
        help="TNS bot ID. Default: {TNS_BOT_ID}")
    parser.add_argument(
        "--tns-bot-name", default=os.environ.get("TNS_BOT_NAME"),
        help="TNS bot name. Default: {TNS_BOT_NAME}")
    parser.add_argument(
        "--skip-simbad",    action="store_true")
    parser.add_argument(
        "--skip-milliquas", action="store_true")
    parser.add_argument(
        "--skip-vsx",       action="store_true")
    parser.add_argument(
        "--min-per-class", type=int, default=10,
        help="Warns if a class has fewer than N objects")
    args = parser.parse_args()
 
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    survey_mode = args.survey_mode

    # ── Validate args ──────────────────────────────────────────────────────────
    if args.coords_csv is None and args.features_dir is None:
        parser.error("--features-dir is required unless --coords-csv is provided.")

    feats_dir    = Path(args.features_dir) if args.features_dir else None
    coords_csv   = Path(args.coords_csv)   if args.coords_csv   else None
    using_coords_csv = coords_csv is not None

    if using_coords_csv:
        log.info(f"Loading coords from {coords_csv} ...")
        coords_df = pd.read_csv(coords_csv)
        required  = {"oid", "ra", "dec"}
        missing   = required - set(coords_df.columns)
        if missing:
            log.error(f"--coords-csv is missing columns: {missing}")
            return
        coords_df = coords_df[["oid", "ra", "dec"]].drop_duplicates("oid")
        # In coords-csv mode oid IS already the oid_combined key
        coords_df["oid_combined"] = coords_df["oid"]
        all_combined = set(coords_df["oid_combined"].tolist())
        log.info(f"Objects to crossmatch: {len(coords_df):,} (from CSV, survey_mode={survey_mode!r})")

    else:
        log.info("Collecting OIDs from all feature parquets...")
        all_combined = set()
        for fname in FEATURE_PATTERNS:
            src = feats_dir / fname
            if src.exists():
                df = pd.read_parquet(src, columns=[])  # only index
                all_combined.update(df.index.get_level_values(0).unique())

        log.info(f"Total unique objects across all parquets: {len(all_combined):,}")

        if not all_combined:
            log.error("No feature parquets found. Check --features-dir.")
            return

        ckpt_dir = Path(args.obj_dir) if args.obj_dir else feats_dir

        if survey_mode == "lsst_only":
            ckpt_pattern = str(ckpt_dir / "obj_lsst_*.parquet")
            ra_key, dec_key = "meanra", "meandec"
        else:
            ckpt_pattern = str(ckpt_dir / "obj_ztf_*.parquet")
            ra_key, dec_key = "ra", "meanra"

        obj_files = sorted(_glob.glob(ckpt_pattern))
        ra_map = dec_map = {}

        if obj_files:
            obj_df  = pd.concat([pd.read_parquet(f) for f in obj_files])
            ra_col  = ra_key  if ra_key  in obj_df.columns else "meanra"
            dec_col = dec_key if dec_key in obj_df.columns else "meandec"
            if ra_col in obj_df.columns and dec_col in obj_df.columns:
                ra_map  = obj_df[ra_col].to_dict()
                dec_map = obj_df[dec_col].to_dict()
                log.info(f"Loaded RA/Dec for {len(ra_map):,} objects from {ckpt_dir}")
            else:
                log.warning("RA/Dec columns not found in checkpoints: "
                            "positional crossmatch will use NaN coordinates.")
        else:
            log.warning(f"No checkpoints found matching {ckpt_pattern}: "
                        "positional crossmatch will use NaN coordinates.")

        def _ztf_part(oid_combined: str) -> str:
            for p in reversed(oid_combined.split("_")):
                if p.startswith("ZTF"):
                    return p
            return oid_combined

        coords_rows = []
        for oid_comb in sorted(all_combined):
            if survey_mode in ("lsst_only", "ztf_only"):
                query_oid = oid_comb
            else:
                query_oid = _ztf_part(oid_comb)
            coords_rows.append({
                "oid":          query_oid,
                "oid_combined": oid_comb,
                "ra":           ra_map.get(oid_comb, float("nan")),
                "dec":          dec_map.get(oid_comb, float("nan")),
            })

        coords_df = pd.DataFrame(coords_rows).drop_duplicates("oid")
        log.info(f"Objects to crossmatch: {len(coords_df):,} (survey_mode={survey_mode!r})")

    tns = pd.DataFrame(columns=["oid", "tns_type_raw", "label_tns"])
    if args.tns_api_key and not (args.tns_bot_id and args.tns_bot_name):
        log.warning("--tns-api-key given, but --tns-bot-id/--tns-bot-name missing"
                    "(or TNS_BOT_ID/TNS_BOT_NAME variables): "
                    "skipping TNS query.")
    elif args.tns_api_key:
        tns = query_tns_bulk(
            oids=coords_df["oid"].tolist(),
            api_key=args.tns_api_key,
            bot_id=args.tns_bot_id,
            bot_name=args.tns_bot_name,
        )

        if survey_mode == "lsst_only" and len(tns) == 0:
            log.info("TNS name match found 0 objects: trying positional fallback...")
            tns = query_tns_positional(
                coords_df,
                api_key=args.tns_api_key,
                bot_id=args.tns_bot_id,
                bot_name=args.tns_bot_name,
                radius_arcsec=args.radius,
            )
 
    simbad = pd.DataFrame(columns=["oid", "simbad_otype", "label_simbad"])
    if not args.skip_simbad:
        simbad = query_simbad_positional(coords_df, radius_arcsec=args.radius)
 
    milliquas = pd.DataFrame(columns=["oid", "milliquas_type", "label_milliquas"])
    if not args.skip_milliquas:
        milliquas = query_milliquas(coords_df, radius_arcsec=args.radius)
 
    vsx = pd.DataFrame(columns=["oid", "vsx_type", "label_vsx"])
    if not args.skip_vsx:
        vsx = query_vsx(coords_df, radius_arcsec=args.radius)
 
    labeled = merge_labels(coords_df, tns, simbad, milliquas, vsx)
 
    labeled = apply_quality_filters(labeled, min_per_class=args.min_per_class)
 
    oid_to_combined  = coords_df.set_index("oid")["oid_combined"].to_dict()
    labeled["oid_combined"] = labeled["oid"].map(oid_to_combined)
    labeled_combined = set(labeled["oid_combined"].dropna())
 
    log.info(f"\nTest set : {len(labeled_combined):,} objects with label")
    log.info(f"Unlabeled: {len(all_combined) - len(labeled_combined):,} objects")
 
    if not using_coords_csv and feats_dir is not None:
        log.info("\nSplitting feature parquets into testset / unlabeled...")
        split_feature_parquets(feats_dir, labeled_combined, all_combined)
    elif using_coords_csv:
        log.info("\nFeature parquet split skipped (--coords-csv mode). "
                 "Run again with --features-dir after extracting features.")
 
    labeled.to_csv(out_path, index=False)
    log.info(f"\nSaved {len(labeled):,} labeled objects to {out_path}")
 
    conflicts = labeled[labeled["label_conflict"]]
    if len(conflicts):
        conflict_path = out_path.with_name(out_path.stem + "_conflicts.csv")
        conflicts.to_csv(conflict_path, index=False)
        log.info(f"Saved {len(conflicts)} conflicting objects to {conflict_path}")

    try:
        import matplotlib.pyplot as plt
        counts = (labeled["classALeRCE"]
                  .value_counts()
                  .reindex(LABEL_ORDER, fill_value=0))
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#c0392b"] * 4 + ["#2980b9"] * 3 + ["#27ae60"] * 8
        bars = ax.bar(counts.index, counts.values, color=colors)
        ax.set_xlabel("Class")
        ax.set_ylabel("N objects")
        ax.set_title(f"Test set class distribution ({survey_mode})")
        plt.xticks(rotation=45, ha="right")
        for bar, n in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, str(n),
                    ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plot_path = out_path.with_name(out_path.stem + "_distribution.pdf")
        plt.savefig(plot_path)
        log.info(f"Class distribution plot saved to {plot_path}")
    except ImportError:
        pass
 
 
if __name__ == "__main__":
    main()