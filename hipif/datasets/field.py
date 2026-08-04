"""Field telemetry loader — manifest-first (WP6 / A9).

Loads flights strictly through data/manifests/field_manifest.csv
(filename_in_zip, decoded_name, pack, reference_soh_pct, reference_source).
Files absent from the manifest, or manifest rows with an empty pack, are a
HARD FAIL (P7): no substring guessing, no 'Unknown' bucket.
Reference SoH lives in the manifest with per-pack provenance; the loader
returns it in a separate evaluation-only frame.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..config import HIPIFConfig

MANIFEST = Path("data/manifests/field_manifest.csv")


def load_field_manifest(manifest: Path = MANIFEST) -> pd.DataFrame:
    if not manifest.exists():
        raise FileNotFoundError(
            f"field manifest not found: {manifest}\n"
            f"Run scripts/prepare_field_manifest.py, fill pack/reference "
            f"columns, and keep the provenance evidence (WP6)")
    m = pd.read_csv(manifest, encoding="utf-8-sig")
    bad = m[m["pack"].isna() | (m["pack"].astype(str).str.strip() == "")]
    if len(bad):
        raise ValueError(
            f"manifest rows without pack assignment: "
            f"{bad['decoded_name'].tolist()} — fill them in (hard fail, P7)")
    return m


def load_field_flights(cfg: HIPIFConfig, csv_dir: Path,
                       manifest: Path = MANIFEST, verbose: bool = True):
    """Returns (telemetry_df, reference_df). telemetry has pack_id per row;
    reference_df is evaluation-only (pack_id, reference_soh_pct, source)."""
    m = load_field_manifest(manifest)
    frames = []
    for _, r in m.iterrows():
        p = Path(csv_dir) / r["filename_in_zip"]
        if not p.exists():
            alt = Path(csv_dir) / r["decoded_name"]
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError(
                    f"manifest file missing on disk: {r['filename_in_zip']} "
                    f"(or {r['decoded_name']}) under {csv_dir}")
        df = pd.read_csv(p)
        df["pack_id"] = str(r["pack"])
        df["flight_file"] = r["decoded_name"]
        frames.append(df)
    tel = pd.concat(frames, ignore_index=True)
    ref = (m.groupby("pack")
             .agg(reference_soh_pct=("reference_soh_pct", "first"),
                  reference_source=("reference_source", "first"))
             .reset_index().rename(columns={"pack": "pack_id"}))
    if verbose:
        print(f"[field] {len(m)} flights, packs="
              f"{sorted(tel['pack_id'].unique())}")
    return tel, ref
