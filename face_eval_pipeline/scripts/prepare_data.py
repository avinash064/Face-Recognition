"""
prepare_data.py
===============
Downloads all three datasets used in the face recognition evaluation pipeline.

Datasets
--------
1. AT&T Database of Faces (ORL) – ~4 MB zip, 40 subjects × 10 images
2. IMFDB – Kaggle dataset (requires API credentials)
3. IMDB-WIKI crop – ETHZ server, large (~7 GB tar); a mini subset is
   downloaded by default (first ~500 MB = first 9 sub-directories).

Usage
-----
    python scripts/prepare_data.py --data_dir data/ [--dataset all|att|imfdb|imdb_wiki]

Environment variables (for Kaggle):
    KAGGLE_USERNAME, KAGGLE_KEY
    – OR – place ~/.kaggle/kaggle.json
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress-aware download helper
# ---------------------------------------------------------------------------

class _ProgressReporter:
    """Callback for urllib.request.urlretrieve – prints download progress."""

    def __init__(self, filename: str):
        self.filename = filename
        self._prev = -1

    def __call__(self, block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = int(block_count * block_size * 100 / total_size)
        pct = min(pct, 100)
        if pct != self._prev and pct % 5 == 0:
            logger.info(f"  {self.filename}: {pct}%")
            self._prev = pct


def _download(url: str, dest: Path, desc: str = "") -> Path:
    """Download *url* to *dest* (file path). Skips if already present."""
    if dest.exists():
        logger.info(f"  Already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {desc or url}  →  {dest}")
    try:
        urllib.request.urlretrieve(
            url, str(dest), reporthook=_ProgressReporter(dest.name)
        )
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        if dest.exists():
            dest.unlink()
        raise
    return dest


# ---------------------------------------------------------------------------
# 1. AT&T Database of Faces
# ---------------------------------------------------------------------------

ATT_URLS = [
    # Primary mirror (Cambridge)
    "https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/att_faces.zip",
    # Fallback mirrors
    "https://www.kaggle.com/datasets/kasikrit/att-database-of-faces/download/att_faces.zip",
]

ATT_DIRECT_URL = (
    "https://github.com/Qucs/qucs/raw/master/qucs-doc/technical/faces.tar.gz"
)

# Best reliable mirror
ATT_RELIABLE_URL = "https://www.cl.cam.ac.uk/Research/DTG/attarchive:pub/data/att_faces.zip"


def download_att(data_dir: Path) -> Path:
    """Download and extract AT&T Faces to data_dir/att_faces/."""
    out_dir = data_dir / "att_faces"
    if out_dir.exists() and any(out_dir.rglob("*.pgm")):
        logger.info(f"AT&T dataset already present at {out_dir}")
        return out_dir

    zip_path = data_dir / "att_faces.zip"

    # Try multiple mirrors
    urls_to_try = [
        "https://github.com/Noob-can-Compile/Facial-Recognition/raw/master/att_faces.zip",
        "https://www.kaggle.com/datasets/tavarez/the-orl-database-for-training-and-testing/download/att_faces.zip",
        # Manual fallback
        ATT_RELIABLE_URL,
    ]

    downloaded = False
    for url in urls_to_try:
        try:
            _download(url, zip_path, "AT&T Faces")
            downloaded = True
            break
        except Exception:
            logger.warning(f"  Mirror failed: {url}")
            if zip_path.exists():
                zip_path.unlink()

    if not downloaded:
        logger.warning(
            "\n⚠  Auto-download failed for AT&T.\n"
            "   Please download manually from:\n"
            "     https://www.kaggle.com/datasets/kasikrit/att-database-of-faces\n"
            "   and extract to: data/att_faces/\n"
            "   Expected structure: data/att_faces/s1/*.pgm … s40/*.pgm\n"
        )
        return out_dir

    # Extract
    logger.info(f"Extracting AT&T Faces → {out_dir}")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(data_dir))
    except zipfile.BadZipFile:
        logger.warning("Bad zip file – trying as tar.gz …")
        try:
            with tarfile.open(str(zip_path), 'r:gz') as tf:
                tf.extractall(str(data_dir))
        except Exception as exc:
            logger.error(f"Extraction failed: {exc}")
            return out_dir

    # Normalise: some archives embed in a sub-folder named "att_faces" already
    if (data_dir / "att_faces").exists():
        pass  # already correct
    elif (data_dir / "orl_faces").exists():
        (data_dir / "orl_faces").rename(out_dir)

    zip_path.unlink(missing_ok=True)
    logger.info(f"✓ AT&T dataset ready: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# 2. IMFDB
# ---------------------------------------------------------------------------

IMFDB_KAGGLE_SLUG = "anirudhsimhachalam/indian-movie-faces-datasetimfdb-face-recognition"


def download_imfdb(data_dir: Path) -> Path:
    """Download IMFDB via Kaggle API."""
    out_dir = data_dir / "imfdb"

    if out_dir.exists() and any(
        f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        for f in out_dir.rglob('*')
    ):
        logger.info(f"IMFDB already present at {out_dir}")
        return out_dir

    # Try kaggle CLI
    kaggle_ok = shutil.which("kaggle") is not None
    if not kaggle_ok:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "kaggle"],
                check=True, capture_output=True
            )
            kaggle_ok = True
        except subprocess.CalledProcessError:
            pass

    if not kaggle_ok:
        logger.warning(
            "\n⚠  Kaggle CLI not available.\n"
            "   Install: pip install kaggle\n"
            "   Then set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
            "   (or place kaggle.json at ~/.kaggle/kaggle.json)\n"
            "   and re-run this script with --dataset imfdb\n"
        )
        return out_dir

    logger.info(f"Downloading IMFDB via Kaggle API → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", IMFDB_KAGGLE_SLUG,
                "-p", str(out_dir),
                "--unzip",
            ],
            check=True,
            env=env,
        )
        logger.info(f"✓ IMFDB ready: {out_dir}")
    except subprocess.CalledProcessError as exc:
        logger.error(
            f"Kaggle download failed: {exc}\n"
            "  Ensure KAGGLE_USERNAME and KAGGLE_KEY are set, or\n"
            "  download manually from:\n"
            f"  https://www.kaggle.com/datasets/{IMFDB_KAGGLE_SLUG}\n"
            "  and unzip to: data/imfdb/"
        )
    return out_dir


# ---------------------------------------------------------------------------
# 3. IMDB-WIKI crop (mini subset by default)
# ---------------------------------------------------------------------------

IMDB_META_URL = (
    "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar"
)
IMDB_CROP_URL = (
    "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
)
WIKI_META_URL = (
    "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_meta.tar"
)
WIKI_CROP_URL = (
    "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"
)


def download_imdb_wiki(
    data_dir: Path,
    full: bool = False,
    use_wiki: bool = False,
) -> Path:
    """
    Download IMDB-WIKI crop dataset.

    Parameters
    ----------
    full     : download the complete ~7 GB tar (slow); default False → wiki only (~1 GB)
    use_wiki : download the Wiki subset (~1 GB) instead of IMDB (~7 GB)
    """
    dataset_key = "wiki" if (use_wiki or not full) else "imdb"
    out_dir = data_dir / f"{dataset_key}_crop"

    if out_dir.exists() and any(out_dir.rglob("*.mat")):
        logger.info(f"IMDB-WIKI ({dataset_key}) already present at {out_dir}")
        return out_dir

    meta_url = WIKI_META_URL if dataset_key == "wiki" else IMDB_META_URL
    crop_url = WIKI_CROP_URL if dataset_key == "wiki" else IMDB_CROP_URL

    logger.info(
        f"Downloading IMDB-WIKI ({dataset_key}) …\n"
        "  This is a large download – wiki_crop ~1 GB, imdb_crop ~7 GB."
    )

    # ── metadata tar (small) ──
    meta_tar = data_dir / f"{dataset_key}_meta.tar"
    try:
        _download(meta_url, meta_tar, f"{dataset_key}_meta.tar")
        with tarfile.open(str(meta_tar), 'r') as tf:
            tf.extractall(str(data_dir))
        meta_tar.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Metadata download failed: {exc}")

    # ── crop images tar (large) ──
    crop_tar = data_dir / f"{dataset_key}_crop.tar"
    try:
        _download(crop_url, crop_tar, f"{dataset_key}_crop.tar")
        logger.info(f"Extracting {crop_tar} … (may take a few minutes)")
        with tarfile.open(str(crop_tar), 'r') as tf:
            tf.extractall(str(data_dir))
        crop_tar.unlink(missing_ok=True)
    except Exception as exc:
        logger.error(
            f"IMDB-WIKI download/extract failed: {exc}\n"
            "  Download manually from:\n"
            f"  {crop_url}\n"
            f"  and extract to: {data_dir}/"
        )
        return out_dir

    logger.info(f"✓ IMDB-WIKI ({dataset_key}) ready: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets for the face recognition evaluation pipeline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory to store downloaded datasets (default: data/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "att", "imfdb", "imdb_wiki"],
        help="Which dataset(s) to download (default: all)",
    )
    parser.add_argument(
        "--imdb_full",
        action="store_true",
        help="Download full IMDB crop (~7 GB) instead of Wiki crop (~1 GB)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    target = args.dataset.lower()

    if target in ("all", "att"):
        logger.info("\n── AT&T Database of Faces ──────────────────────────")
        download_att(data_dir)

    if target in ("all", "imfdb"):
        logger.info("\n── IMFDB ────────────────────────────────────────────")
        download_imfdb(data_dir)

    if target in ("all", "imdb_wiki"):
        logger.info("\n── IMDB-WIKI ────────────────────────────────────────")
        download_imdb_wiki(
            data_dir,
            full=args.imdb_full,
            use_wiki=not args.imdb_full,
        )

    logger.info("\n✓ Dataset preparation complete.")
    logger.info(f"  Data directory: {data_dir}")
    logger.info(
        "\nExpected structure:\n"
        "  data/\n"
        "    att_faces/   s1/ … s40/\n"
        "    imfdb/       ActorName/ images…\n"
        "    wiki_crop/   wiki.mat  00/ 01/ …\n"
    )


if __name__ == "__main__":
    main()
