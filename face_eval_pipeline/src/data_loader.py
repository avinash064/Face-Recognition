"""
data_loader.py
==============
Loads face datasets organized by identity and produces reproducible
train/test (gallery/probe) splits.

Supported datasets:
  - AT&T Database of Faces  (ATTDataLoader)
  - IMFDB                    (IMFDBDataLoader)
  - IMDB-WIKI crop           (IMDBWikiDataLoader)
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Abstract base class for dataset loaders.

    Sub-classes must implement `load()` which returns a dict mapping
    identity label -> list of absolute image paths.
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.tif', '.tiff'}

    def __init__(self, dataset_path: str, seed: int = 42):
        self.dataset_path = Path(dataset_path)
        self.seed = seed
        # Fix random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    def load(self) -> Dict[str, List[str]]:
        """Return {identity_label: [image_path, ...]} for every identity."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def train_test_split(
        self,
        identity_images: Dict[str, List[str]],
        n_train: int = 2,
        min_test: int = 1,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Per-identity reproducible split into gallery (train) and probe (test).

        Args:
            identity_images : {identity: [paths]}
            n_train         : Number of gallery images per identity (1–3 per spec).
            min_test        : Minimum probe images required; identities with too
                              few images are skipped.

        Returns:
            train_dict, test_dict  (same key structure as input)
        """
        train_dict: Dict[str, List[str]] = {}
        test_dict:  Dict[str, List[str]] = {}

        rng = random.Random(self.seed)   # isolated RNG for reproducibility

        for identity, paths in identity_images.items():
            paths_copy = list(paths)
            rng.shuffle(paths_copy)

            # Skip identities that do not have enough images
            if len(paths_copy) < n_train + min_test:
                logger.warning(
                    f"Skipping '{identity}': only {len(paths_copy)} images "
                    f"(need ≥ {n_train + min_test})."
                )
                continue

            # First n_train → gallery; rest → probe
            actual_train = min(n_train, len(paths_copy) - min_test)
            train_dict[identity] = paths_copy[:actual_train]
            test_dict[identity]  = paths_copy[actual_train:]

        n_train_imgs = sum(len(v) for v in train_dict.values())
        n_test_imgs  = sum(len(v) for v in test_dict.values())
        logger.info(
            f"Split complete: {len(train_dict)} identities | "
            f"{n_train_imgs} gallery imgs | {n_test_imgs} probe imgs."
        )
        return train_dict, test_dict


# ---------------------------------------------------------------------------
# AT&T Database of Faces
# ---------------------------------------------------------------------------

class ATTDataLoader(DataLoader):
    """
    Loader for the AT&T (ORL) Database of Faces.

    Expected directory structure::

        att_faces/
          s1/   1.pgm  2.pgm … 10.pgm
          s2/   1.pgm  2.pgm … 10.pgm
          …
          s40/  1.pgm  2.pgm … 10.pgm

    Each sub-folder ``sN`` is treated as a distinct identity.
    """

    def load(self) -> Dict[str, List[str]]:
        identity_images: Dict[str, List[str]] = {}

        for subject_dir in sorted(self.dataset_path.iterdir()):
            if not subject_dir.is_dir():
                continue
            if not subject_dir.name.lower().startswith('s'):
                continue

            identity = subject_dir.name
            image_paths = sorted(
                str(p)
                for p in subject_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.VALID_EXTENSIONS
            )

            if image_paths:
                identity_images[identity] = image_paths

        logger.info(f"[AT&T] Loaded {len(identity_images)} identities.")
        return identity_images


# ---------------------------------------------------------------------------
# IMFDB (Indian Movie Face Database)
# ---------------------------------------------------------------------------

class IMFDBDataLoader(DataLoader):
    """
    Loader for the IMFDB dataset.

    Expected directory structure::

        imfdb/
          ActorName/
            image1.jpg
            image2.jpg
            …

    Each actor directory is treated as a distinct identity.
    Some archive versions nest images inside sub-directories; this loader
    handles that with ``rglob``.
    """

    def load(self) -> Dict[str, List[str]]:
        identity_images: Dict[str, List[str]] = {}

        for actor_dir in sorted(self.dataset_path.iterdir()):
            if not actor_dir.is_dir():
                continue

            identity = actor_dir.name
            image_paths = sorted(
                str(p)
                for p in actor_dir.rglob('*')
                if p.is_file() and p.suffix.lower() in self.VALID_EXTENSIONS
            )

            if image_paths:
                identity_images[identity] = image_paths

        logger.info(f"[IMFDB] Loaded {len(identity_images)} identities.")
        return identity_images


# ---------------------------------------------------------------------------
# IMDB-WIKI Crop
# ---------------------------------------------------------------------------

class IMDBWikiDataLoader(DataLoader):
    """
    Loader for the IMDB-WIKI crop dataset.

    The dataset ships with a MATLAB ``.mat`` metadata file (``imdb.mat`` or
    ``wiki.mat``) that stores per-image metadata:

    * ``full_path``         – relative path inside the crop directory
    * ``name``              – celebrity name (identity label)
    * ``face_score``        – MTCNN detection confidence  (NaN = no face)
    * ``second_face_score`` – score of second face (if > 0, multiple faces)

    Image filtering applied:
      * ``face_score  ≥ min_face_score``
      * ``second_face_score`` is NaN or ≤ 0  (single face images only)
      * Identity must appear ≥ ``min_images_per_identity`` times after filtering
      * A random subset of ``max_identities`` is sampled for tractability.

    Expected directory structure::

        imdb_crop/          (or wiki_crop/)
          imdb.mat
          00/   image files
          01/   image files
          …
    """

    def __init__(
        self,
        dataset_path: str,
        mat_file: str = 'imdb.mat',
        min_face_score: float = 1.0,
        min_images_per_identity: int = 3,
        max_identities: int = 200,
        seed: int = 42,
    ):
        super().__init__(dataset_path, seed)
        self.mat_file = mat_file
        self.min_face_score = min_face_score
        self.min_images_per_identity = min_images_per_identity
        self.max_identities = max_identities

    # ------------------------------------------------------------------
    def load(self) -> Dict[str, List[str]]:
        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy is required for IMDB-WIKI loading. pip install scipy")

        mat_path = self.dataset_path / self.mat_file
        if not mat_path.exists():
            raise FileNotFoundError(
                f"MAT file not found: {mat_path}\n"
                "Download imdb_crop.tar from:\n"
                "  https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/"
            )

        logger.info(f"[IMDB-WIKI] Loading metadata from {mat_path} …")
        mat = scipy.io.loadmat(str(mat_path))

        # Navigate MATLAB nested struct
        db_key = 'imdb' if 'imdb' in mat else 'wiki'
        db = mat[db_key][0, 0]

        full_paths         = db['full_path'][0]
        names              = db['name'][0]
        face_scores        = db['face_score'][0]
        try:
            second_face_scores = db['second_face_score'][0]
        except (KeyError, ValueError):
            second_face_scores = np.full(len(face_scores), np.nan)

        identity_images: Dict[str, List[str]] = {}

        for i in range(len(full_paths)):
            # ---- face score filter ----
            score = face_scores[i]
            if np.isnan(score) or float(score) < self.min_face_score:
                continue

            # ---- single-face filter ----
            sec = second_face_scores[i]
            if not np.isnan(sec) and float(sec) > 0.0:
                continue

            # ---- parse name ----
            try:
                raw_name = names[i]
                name = (
                    str(raw_name[0])
                    if hasattr(raw_name, '__getitem__') and len(raw_name) > 0
                    else str(raw_name)
                )
            except Exception:
                continue
            if not name or name in ('', 'nan'):
                continue

            # ---- build absolute path ----
            try:
                rel = full_paths[i]
                rel_str = str(rel[0]) if hasattr(rel, '__getitem__') else str(rel)
                abs_path = str(self.dataset_path / rel_str)
            except Exception:
                continue

            if not os.path.isfile(abs_path):
                continue

            identity_images.setdefault(name, []).append(abs_path)

        # ---- apply filters ----
        identity_images = {
            k: v
            for k, v in identity_images.items()
            if len(v) >= self.min_images_per_identity
        }

        # ---- random subset for tractability ----
        if len(identity_images) > self.max_identities:
            rng = random.Random(self.seed)
            chosen = rng.sample(list(identity_images.keys()), self.max_identities)
            identity_images = {k: identity_images[k] for k in chosen}

        logger.info(
            f"[IMDB-WIKI] {len(identity_images)} identities "
            f"(≥{self.min_images_per_identity} images each)."
        )
        return identity_images


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dataloader(
    dataset_name: str,
    dataset_path: str,
    seed: int = 42,
    **kwargs,
) -> DataLoader:
    """
    Factory function – returns the appropriate DataLoader for the given dataset.

    ``dataset_name`` values: ``'att'``, ``'imfdb'``, ``'imdb_wiki'``.
    Extra ``**kwargs`` are forwarded to the loader constructor.
    """
    registry = {
        'att':       ATTDataLoader,
        'imfdb':     IMFDBDataLoader,
        'imdb_wiki': IMDBWikiDataLoader,
    }
    key = dataset_name.lower().replace('-', '_').replace(' ', '_')
    if key not in registry:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[key](dataset_path, seed=seed, **kwargs)
