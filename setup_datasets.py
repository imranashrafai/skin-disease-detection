#!/usr/bin/env python3
"""
setup_datasets.py — One-script dataset preparation for local training
======================================================================
Run this BEFORE train_model.py. It downloads and organises both datasets.

Usage:
    python setup_datasets.py

Requirements:
    pip install kaggle datasets pandas pillow tqdm

Kaggle API setup (one time only):
    1. Go to https://www.kaggle.com/settings → API → "Create New Token"
    2. Download kaggle.json
    3. Place it at:
         Linux/Mac : ~/.kaggle/kaggle.json
         Windows   : C:\\Users\\<you>\\.kaggle\\kaggle.json
    4. chmod 600 ~/.kaggle/kaggle.json   (Linux/Mac only)
"""

import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
KAGGLE_ZIP   = "31-classes-of-skin-disease.zip"
KAGGLE_DIR   = "dataset_kaggle"
HF_SUBSETS   = ["district_hospital", "dermatology_clinic", "rural_health_centre"]

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Kaggle image dataset
# ══════════════════════════════════════════════════════════════════════════════

def download_kaggle():
    try:
        import kaggle
    except ImportError:
        log.error("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    if os.path.isdir(KAGGLE_DIR):
        log.info("Kaggle dataset folder '%s' already exists — skipping download.", KAGGLE_DIR)
        return

    log.info("Downloading Kaggle dataset (kelixo25/31-classes-of-skin-disease)...")
    os.system(f"kaggle datasets download -d kelixo25/31-classes-of-skin-disease -p .")

    if not os.path.exists(KAGGLE_ZIP):
        log.error(
            "Download failed or zip not found. "
            "Make sure kaggle.json is in place and you have internet access."
        )
        sys.exit(1)

    log.info("Extracting Kaggle zip...")
    with zipfile.ZipFile(KAGGLE_ZIP, "r") as zf:
        zf.extractall(".")

    # The Kaggle dataset may unzip into various layouts.
    # Normalise to: dataset_kaggle/<ClassName>/...
    _normalise_kaggle_layout()

    log.info("Kaggle dataset ready → %s/", KAGGLE_DIR)


def _normalise_kaggle_layout():
    """
    Handle the most common Kaggle zip layouts:
      1. Class folders directly in zip root       → rename root to KAGGLE_DIR
      2. Single parent folder containing class folders → rename parent to KAGGLE_DIR
    """
    if os.path.isdir(KAGGLE_DIR):
        return

    # Find the extracted root
    extracted_roots = [
        d for d in os.listdir(".")
        if os.path.isdir(d)
        and d not in (KAGGLE_DIR, "model", "static", "templates", "__pycache__")
        and not d.startswith(".")
    ]

    image_extensions = {".jpg", ".jpeg", ".png"}

    for candidate in extracted_roots:
        candidate_path = Path(candidate)
        # Check if it directly contains class subfolders with images
        subdirs = [d for d in candidate_path.iterdir() if d.is_dir()]
        if subdirs:
            # Check if sub-subdirs contain images (2-level layout) or root contains images
            img_count = sum(
                1 for f in candidate_path.rglob("*")
                if f.suffix.lower() in image_extensions
            )
            if img_count > 0:
                log.info("Renaming '%s' → '%s'", candidate, KAGGLE_DIR)
                shutil.move(candidate, KAGGLE_DIR)
                return

    log.warning(
        "Could not auto-detect extracted folder layout. "
        "Please manually rename/move your class folders into '%s/'", KAGGLE_DIR
    )


def verify_kaggle_structure():
    if not os.path.isdir(KAGGLE_DIR):
        log.error("Dataset folder '%s' not found.", KAGGLE_DIR)
        return False

    classes = [d for d in os.listdir(KAGGLE_DIR) if os.path.isdir(os.path.join(KAGGLE_DIR, d))]
    total_images = 0
    class_counts = {}
    for cls in classes:
        cls_path = os.path.join(KAGGLE_DIR, cls)
        imgs = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        class_counts[cls] = len(imgs)
        total_images += len(imgs)

    log.info("Kaggle dataset summary:")
    log.info("  Classes   : %d", len(classes))
    log.info("  Total imgs: %d", total_images)

    # Show class breakdown
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        status = "✓" if count >= 50 else "⚠ (< 50 images — may be skipped in training)"
        log.info("    %-35s  %4d imgs  %s", cls, count, status)

    return len(classes) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — HuggingFace tabular dataset
# ══════════════════════════════════════════════════════════════════════════════

def download_hf_metadata():
    """Download HF CSVs for metadata enrichment (no images, optional)."""
    all_present = all(os.path.exists(f"hf_{s}.csv") for s in HF_SUBSETS)
    if all_present:
        log.info("HuggingFace CSV files already present — skipping download.")
        return

    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        log.warning(
            "Missing packages for HF download. Run: pip install datasets pandas\n"
            "The HF metadata is optional — training will proceed without it."
        )
        return

    log.info("Downloading HuggingFace skin-diseases-dermatology dataset (3 subsets)...")
    for subset in HF_SUBSETS:
        csv_path = f"hf_{subset}.csv"
        if os.path.exists(csv_path):
            log.info("  %s already exists — skipping.", csv_path)
            continue
        try:
            log.info("  Downloading subset: %s", subset)
            ds = load_dataset("electricsheepafrica/skin-diseases-dermatology", subset)
            df = ds["train"].to_pandas()
            df.to_csv(csv_path, index=False)
            log.info("  Saved %d rows → %s", len(df), csv_path)
        except Exception as exc:
            log.warning("  Could not download '%s': %s — skipping.", subset, exc)

    log.info("HuggingFace metadata CSVs ready.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("DermAI Dataset Setup")
    log.info("=" * 60)

    # ── Kaggle images ─────────────────────────────────────────────────────────
    log.info("\n[1/2] Kaggle image dataset")
    download_kaggle()
    ok = verify_kaggle_structure()
    if not ok:
        log.error(
            "\nDataset folder '%s' is missing or empty.\n"
            "Manual setup:\n"
            "  1. Download from https://www.kaggle.com/datasets/kelixo25/31-classes-of-skin-disease\n"
            "  2. Unzip into this folder\n"
            "  3. Make sure the layout is:\n"
            "       dataset_kaggle/\n"
            "           Acne/\n"
            "           Basal Cell Carcinoma/\n"
            "           ...\n",
            KAGGLE_DIR,
        )
        sys.exit(1)

    # ── HuggingFace metadata ──────────────────────────────────────────────────
    log.info("\n[2/2] HuggingFace metadata CSVs (optional — for disease info enrichment)")
    download_hf_metadata()

    log.info("\n" + "=" * 60)
    log.info("Setup complete! Now run:")
    log.info("  python train_model.py")
    log.info("=" * 60)
