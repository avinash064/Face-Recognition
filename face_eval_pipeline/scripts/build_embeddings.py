"""
build_embeddings.py
===================
Pre-compute and cache face embeddings + pose info for all images
across all configured datasets and models.

Outputs one .npz file per (dataset, model) combination:
    embeddings/{dataset}_{model}.npz

Each .npz contains:
  - paths       : array of image file paths
  - labels      : array of identity labels (str)
  - embeddings  : float32 array shape (N, 512)
  - yaw         : float32 array (N,)
  - pitch       : float32 array (N,)
  - roll        : float32 array (N,)
  - det_scores  : float32 array (N,)  — -1 if detection failed

Usage
-----
  # Build for all datasets and default models
  python scripts/build_embeddings.py

  # Specific dataset + model
  python scripts/build_embeddings.py --dataset att --models buffalo_l

  # All supported models
  python scripts/build_embeddings.py --all_models
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import ATTDataLoader, IMFDBDataLoader, IMDBWikiDataLoader
from src.model_manager import ModelManager
from src.embedding import FaceEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_ATT_DIR   = str(ROOT / "data")
DEFAULT_IMFDB_DIR = str(ROOT.parent / "datasets" / "IMFDB FR dataset" / "IMFDB FR dataset")
DEFAULT_IMDB_WIKI_DIR = ""
DEFAULT_EMBED_DIR = str(ROOT / "embeddings")


# ─── Dataset configs ───────────────────────────────────────────────────────────

def build_dataset_configs(args) -> list:
    configs = []

    att_dir   = Path(args.att_dir)   if args.att_dir   else None
    imfdb_dir = Path(args.imfdb_dir) if args.imfdb_dir else None
    imdb_dir  = Path(args.imdb_wiki_dir) if args.imdb_wiki_dir else None

    if att_dir and att_dir.exists():
        if any(d.name.lower().startswith('s') and d.is_dir() for d in att_dir.iterdir()):
            configs.append(("att", ATTDataLoader(str(att_dir), seed=args.seed)))
            logger.info(f"  ✓ AT&T dataset found at {att_dir}")

    if imfdb_dir and imfdb_dir.exists():
        if any(d.is_dir() for d in imfdb_dir.iterdir()):
            configs.append(("imfdb", IMFDBDataLoader(str(imfdb_dir), seed=args.seed)))
            logger.info(f"  ✓ IMFDB dataset found at {imfdb_dir}")

    if imdb_dir and imdb_dir.exists():
        mat_file = "imdb.mat" if args.use_imdb else "wiki.mat"
        configs.append((
            "imdb_wiki",
            IMDBWikiDataLoader(
                str(imdb_dir),
                mat_file=mat_file,
                min_face_score=args.imdb_min_score,
                min_images_per_identity=args.imdb_min_images,
                max_identities=args.imdb_max_identities,
                seed=args.seed,
            ),
        ))
        logger.info(f"  ✓ IMDB-WIKI dataset found at {imdb_dir}")

    return configs


# ─── Extraction ───────────────────────────────────────────────────────────────

def build_embeddings_for(
    dataset_name: str,
    loader,
    embedder: FaceEmbedder,
    model_name: str,
    embed_dir: Path,
    overwrite: bool,
):
    out_path = embed_dir / f"{dataset_name}_{model_name}.npz"

    if out_path.exists() and not overwrite:
        logger.info(f"  Cache already exists, skipping: {out_path.name}")
        return out_path

    logger.info(f"\n--- Building embeddings: dataset={dataset_name}  model={model_name} ---")
    identity_images = loader.load()

    if not identity_images:
        logger.warning(f"No images found for {dataset_name}. Skip.")
        return None

    total_images = sum(len(v) for v in identity_images.values())
    logger.info(f"  {len(identity_images)} identities, {total_images} images total.")

    paths_out      = []
    labels_out     = []
    embeddings_out = []
    yaw_out        = []
    pitch_out      = []
    roll_out       = []
    det_scores_out = []

    t0 = time.perf_counter()
    processed = 0
    failed = 0

    for identity, paths in identity_images.items():
        for img_path in paths:
            info = embedder.get_face_info(img_path)

            paths_out.append(img_path)
            labels_out.append(identity)

            if info is not None:
                embeddings_out.append(info['embedding'])
                yaw_out.append(info['pose']['yaw'])
                pitch_out.append(info['pose']['pitch'])
                roll_out.append(info['pose']['roll'])
                det_scores_out.append(info['det_score'])
            else:
                # Sentinel: zero embedding, -1 det_score
                embeddings_out.append(np.zeros(512, dtype=np.float32))
                yaw_out.append(0.0)
                pitch_out.append(0.0)
                roll_out.append(0.0)
                det_scores_out.append(-1.0)
                failed += 1

            processed += 1
            if processed % 200 == 0:
                elapsed = time.perf_counter() - t0
                rate = processed / elapsed
                remaining = (total_images - processed) / rate
                logger.info(
                    f"  {processed}/{total_images} images  |  "
                    f"{failed} failed  |  ~{remaining:.0f}s left"
                )

    elapsed = time.perf_counter() - t0
    logger.info(f"  Done in {elapsed:.1f}s  ({failed}/{total_images} failed detections).")

    # Save
    embed_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        paths       = np.array(paths_out, dtype=object),
        labels      = np.array(labels_out, dtype=object),
        embeddings  = np.array(embeddings_out, dtype=np.float32),
        yaw         = np.array(yaw_out, dtype=np.float32),
        pitch       = np.array(pitch_out, dtype=np.float32),
        roll        = np.array(roll_out, dtype=np.float32),
        det_scores  = np.array(det_scores_out, dtype=np.float32),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"  Saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute face embeddings for all datasets and models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--att_dir",      type=str, default=DEFAULT_ATT_DIR)
    parser.add_argument("--imfdb_dir",    type=str, default=DEFAULT_IMFDB_DIR)
    parser.add_argument("--imdb_wiki_dir",type=str, default=DEFAULT_IMDB_WIKI_DIR)
    parser.add_argument("--embed_dir",    type=str, default=DEFAULT_EMBED_DIR, help="Where to save .npz files.")
    parser.add_argument("--dataset",      type=str, default="all", choices=["all","att","imfdb","imdb_wiki"])
    parser.add_argument("--models",       type=str, default="buffalo_l,buffalo_sc", help="Comma-separated model names.")
    parser.add_argument("--all_models",   action="store_true", help="Build for all supported models.")
    parser.add_argument("--det_thresh",   type=float, default=0.5)
    parser.add_argument("--det_size",     type=int,   default=640)
    parser.add_argument("--overwrite",    action="store_true", help="Re-build even if cache file exists.")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--use_imdb",     action="store_true")
    parser.add_argument("--imdb_min_score",    type=float, default=1.0)
    parser.add_argument("--imdb_min_images",   type=int,   default=3)
    parser.add_argument("--imdb_max_identities",type=int,  default=200)
    return parser.parse_args()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    embed_dir = Path(args.embed_dir)

    manager = ModelManager(
        device="auto",
        det_size=(args.det_size, args.det_size),
        det_thresh=args.det_thresh,
    )

    models_to_run = (
        list(manager.SUPPORTED_MODELS.keys()) if args.all_models
        else [m.strip() for m in args.models.split(',')]
    )

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║          Reference Embedding Builder                  ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"  GPU Provider : {manager.active_providers[0]}")
    logger.info(f"  Models       : {', '.join(models_to_run)}")
    logger.info(f"  Output dir   : {embed_dir}")

    all_configs = build_dataset_configs(args)

    if args.dataset != "all":
        all_configs = [(n, l) for n, l in all_configs if n == args.dataset]

    if not all_configs:
        logger.error("No datasets found. Check paths.")
        sys.exit(1)

    built = []
    for model_name in models_to_run:
        logger.info(f"\n{'='*55}")
        logger.info(f"  Loading model: {model_name}")
        logger.info(f"{'='*55}")

        try:
            app = manager.load_model(model_name)
            embedder = FaceEmbedder(app)
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue

        for dataset_name, loader in all_configs:
            try:
                out = build_embeddings_for(
                    dataset_name  = dataset_name,
                    loader        = loader,
                    embedder      = embedder,
                    model_name    = model_name,
                    embed_dir     = embed_dir,
                    overwrite     = args.overwrite,
                )
                if out:
                    built.append(out)
            except Exception as e:
                logger.error(f"Failed for {dataset_name}/{model_name}: {e}", exc_info=True)

    logger.info(f"\n✓ Done. {len(built)} embedding file(s) saved to {embed_dir}/")
    for p in built:
        data = np.load(str(p), allow_pickle=True)
        n_total  = len(data['labels'])
        n_failed = int((data['det_scores'] < 0).sum())
        logger.info(f"   {p.name}  —  {n_total} images, {n_failed} failed detections")


if __name__ == "__main__":
    main()
