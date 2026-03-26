"""
build_reference_gallery.py
--------------------------
Builds a reference gallery for the ORL (AT&T) Faces dataset.

Strategy:
  - For each subject subfolder (s1 … s40), sort the images numerically.
  - Pick the FIRST 2 and LAST 2 images  (4 images per subject).
  - Copy them into:
      <orl_faces>/reference_gallery/
          s1/
              1.pgm   (first)
              2.pgm   (first)
              9.pgm   (last)
              10.pgm  (last)
          s2/ ...
          ...

Usage:
    python build_reference_gallery.py \
        --orl_root face_eval_pipeline/orl_faces \
        --gallery_root face_eval_pipeline/orl_faces/reference_gallery
"""

import argparse
import shutil
from pathlib import Path


def sorted_images(subject_dir: Path) -> list[Path]:
    """Return .pgm files inside *subject_dir* sorted by numeric stem."""
    images = list(subject_dir.glob("*.pgm"))
    images.sort(key=lambda p: int(p.stem))
    return images


def build_gallery(orl_root: Path, gallery_root: Path) -> None:
    gallery_root.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(
        [d for d in orl_root.iterdir() if d.is_dir() and d.name != gallery_root.name],
        key=lambda d: int(d.name[1:]),   # s1 → 1, s10 → 10, …
    )

    if not subject_dirs:
        print(f"[ERROR] No subject sub-directories found in {orl_root}")
        return

    total_copied = 0
    for subj_dir in subject_dirs:
        images = sorted_images(subj_dir)
        if len(images) < 4:
            print(f"[WARN] {subj_dir.name} has only {len(images)} image(s), skipping.")
            continue

        selected = images[:2] + images[-2:]   # first 2 + last 2

        dest_dir = gallery_root / subj_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src in selected:
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            total_copied += 1

        print(
            f"  {subj_dir.name}: copied "
            + ", ".join(p.name for p in selected)
        )

    print(f"\n✓ Reference gallery built at: {gallery_root}")
    print(f"  {len(subject_dirs)} subjects × 4 images = {total_copied} total files")


def main():
    parser = argparse.ArgumentParser(description="Build ORL reference gallery")
    parser.add_argument(
        "--orl_root",
        type=Path,
        default=Path(__file__).parent.parent / "orl_faces",
        help="Path to orl_faces root directory",
    )
    parser.add_argument(
        "--gallery_root",
        type=Path,
        default=None,
        help="Destination for the gallery (default: <orl_root>/reference_gallery)",
    )
    args = parser.parse_args()

    orl_root: Path = args.orl_root.resolve()
    gallery_root: Path = (
        args.gallery_root.resolve()
        if args.gallery_root
        else orl_root / "reference_gallery"
    )

    if not orl_root.exists():
        raise FileNotFoundError(f"ORL root not found: {orl_root}")

    print(f"ORL root   : {orl_root}")
    print(f"Gallery    : {gallery_root}")
    print()
    build_gallery(orl_root, gallery_root)


if __name__ == "__main__":
    main()
