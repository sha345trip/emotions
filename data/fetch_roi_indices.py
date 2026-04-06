"""
fetch_roi_indices.py — One-time Setup Script
=============================================
Downloads the HCP Glasser 360-parcel annotation for fsaverage5 and extracts
the vertex indices for every parcel required by Emotional Weight.

Run this ONCE locally:
    python data/fetch_roi_indices.py

It will print Python list literals ready to paste into roi_map.py.

The Glasser atlas is available via:
  - templateflow (pip install templateflow)
  - neuromaps   (pip install neuromaps)
  - nibabel + a manually downloaded .label.gii file from HCP

This script tries templateflow first, then neuromaps as a fallback.

NOTE: This script is NOT run at inference time — it is a developer utility.
      The output (vertex index lists) is hardcoded into roi_map.py.

Reference:
    Glasser et al. (2016). A multi-modal parcellation of human cerebral cortex.
    Nature, 536, 171–178. https://doi.org/10.1038/nature18933
"""

import sys
import numpy as np

# Target parcels and which hemisphere(s) to use.
# Language ROIs: left hemisphere only (index offset = 0)
# TPJ, DMN:      bilateral (add +10242 for right hemisphere indices)
TARGET_PARCELS = {
    # region_key : (parcel_names_in_atlas, hemispheres)
    # hemispheres: "L", "R", or "LR"
    "PGi":   ("PGi",   "LR"),   # TPJ — bilateral
    "TE1a":  ("TE1a",  "L"),    # MTG — left only
    "44":    ("44",    "L"),    # Broca / IFG — left only
    "45":    ("45",    "L"),    # Broca / IFG — left only
    "IFSp":  ("IFSp",  "L"),    # Broca / IFG — left only
    "STSva": ("STSva", "L"),    # STS — left only
    "STSvp": ("STSvp", "L"),    # STS — left only
    "d32":   ("d32",   "LR"),   # DMN / ACC — bilateral
    "10pp":  ("10pp",  "LR"),   # DMN / frontal pole — bilateral
    "10d":   ("10d",   "LR"),   # DMN / frontal pole — bilateral
}

# fsaverage5 has 10,242 vertices per hemisphere
N_VERTS_PER_HEMI = 10_242
RH_OFFSET = N_VERTS_PER_HEMI  # right hemisphere indices = LH_idx + 10242


def fetch_via_templateflow():
    """Try downloading Glasser annotation via templateflow."""
    try:
        import templateflow.api as tflow
        lh_path = tflow.get(
            "fsaverage5",
            hemi="L",
            desc="Glasser",
            suffix="dseg",
            extension=".label.gii",
        )
        rh_path = tflow.get(
            "fsaverage5",
            hemi="R",
            desc="Glasser",
            suffix="dseg",
            extension=".label.gii",
        )
        return str(lh_path), str(rh_path)
    except Exception as e:
        print(f"  templateflow failed: {e}", file=sys.stderr)
        return None, None


def fetch_via_neuromaps():
    """Try downloading Glasser annotation via neuromaps."""
    try:
        from neuromaps.datasets import fetch_annotation
        # neuromaps stores Glasser as a surface parcellation
        from neuromaps import datasets
        files = datasets.fetch_atlas("glasser", "fsaverage5")
        # Returns a tuple (lh_path, rh_path)
        return files[0], files[1]
    except Exception as e:
        print(f"  neuromaps failed: {e}", file=sys.stderr)
        return None, None


def parse_label_gii(path: str) -> tuple[np.ndarray, dict]:
    """
    Parse a .label.gii file.
    Returns:
        labels_array : (n_vertices,) int array — parcel index per vertex
        label_dict   : {parcel_index: parcel_name}
    """
    import nibabel as nib
    img = nib.load(path)
    labels_array = img.darrays[0].data.astype(int)

    # Build index→name lookup from the GIFTI label table
    label_table = img.labeltable.labels
    label_dict = {entry.key: entry.label for entry in label_table}
    return labels_array, label_dict


def extract_indices(labels_array, label_dict, parcel_name, hemi_offset=0):
    """Return vertex indices (with optional hemisphere offset) for a named parcel."""
    # Find the label index whose name matches parcel_name
    target_key = None
    for key, name in label_dict.items():
        if name == parcel_name:
            target_key = key
            break

    if target_key is None:
        print(f"  ⚠️  Parcel '{parcel_name}' not found in atlas!", file=sys.stderr)
        return np.array([], dtype=int)

    # Vertices where the label == target_key
    verts = np.where(labels_array == target_key)[0]
    return verts + hemi_offset


def main():
    print("🔍 Fetching Glasser fsaverage5 annotation…")

    lh_path, rh_path = fetch_via_templateflow()
    if lh_path is None:
        print("Trying neuromaps…")
        lh_path, rh_path = fetch_via_neuromaps()

    if lh_path is None:
        print(
            "\n❌ Could not download atlas automatically.\n"
            "   Download the GIFTI annotation files manually:\n"
            "   LH: tpl-fsaverage5_hemi-L_desc-Glasser_dseg.label.gii\n"
            "   RH: tpl-fsaverage5_hemi-R_desc-Glasser_dseg.label.gii\n"
            "   from https://templateflow.s3.amazonaws.com/templateflow/\n"
            "   Then run:  python data/fetch_roi_indices.py --lh <file> --rh <file>",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  LH: {lh_path}")
    print(f"  RH: {rh_path}")

    lh_labels, lh_dict = parse_label_gii(lh_path)
    rh_labels, rh_dict = parse_label_gii(rh_path)

    print("\n# ── Paste this into data/roi_map.py ──────────────────────────────\n")

    results = {}
    for parcel_key, (parcel_name, hemis) in TARGET_PARCELS.items():
        indices = np.array([], dtype=int)

        if "L" in hemis:
            lh_idx = extract_indices(lh_labels, lh_dict, parcel_name, hemi_offset=0)
            indices = np.concatenate([indices, lh_idx])

        if "R" in hemis:
            rh_idx = extract_indices(rh_labels, rh_dict, parcel_name, hemi_offset=RH_OFFSET)
            indices = np.concatenate([indices, rh_idx])

        results[parcel_key] = sorted(indices.tolist())
        count = len(results[parcel_key])
        print(f'    "{parcel_key}": {results[parcel_key]},  # {count} vertices ({hemis})')

    print("\n# ────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
