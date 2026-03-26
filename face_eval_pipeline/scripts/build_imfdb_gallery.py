"""
build_imfdb_gallery.py (FIXED)

Select top 5–10 frontal images per identity from IMFDB dataset
using InsightFace (GPU-safe).

Usage:
    python build_imfdb_gallery.py
"""

import cv2
import math
import shutil
from pathlib import Path
from insightface.app import FaceAnalysis

# ── Config ────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/avinash/Desktop/Bidaal/datasets/IMFDB FR dataset/IMFDB FR dataset")
GALLERY_ROOT = DATASET_ROOT.parent / "reference_gallery"

MIN_K = 5
MAX_K = 10
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── INIT MODEL (GPU SAFE) ─────────────────────────────────────────────────
app = FaceAnalysis(
    name="buffalo_l",
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(320, 320))  # safer for RTX 3050


# ── SCORING (using pose from InsightFace) ─────────────────────────────────
def score_face(face):
    if face.pose is None:
        return 0.0

    yaw, pitch, roll = face.pose

    # frontal preference
    yaw_score = max(0.0, 1 - abs(yaw) / 90) * 100
    pitch_score = max(0.0, 1 - abs(pitch) / 90) * 100

    return round(0.5 * yaw_score + 0.5 * pitch_score, 2)


def score_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return path, 0.0

    faces = app.get(img)
    if len(faces) == 0:
        return path, 0.0

    # pick largest face
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))

    if face.det_score < 0.6:
        return path, 0.0

    score = score_face(face)
    return path, score


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    GALLERY_ROOT.mkdir(parents=True, exist_ok=True)
    identity_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])

    total, skipped = 0, 0

    for i, id_dir in enumerate(identity_dirs, 1):
        images = [p for p in id_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

        if len(images) < MIN_K:
            print(f"[{i:3}/{len(identity_dirs)}] SKIP {id_dir.name} ({len(images)} images)")
            skipped += 1
            continue

        # ✅ NO THREADING (SAFE)
        scored = []
        for path in images:
            scored.append(score_image(path))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:MAX_K]

        dest = GALLERY_ROOT / id_dir.name
        dest.mkdir(parents=True, exist_ok=True)

        for src, sc in selected:
            shutil.copy2(src, dest / src.name)

        names = ", ".join(f"{p.name}({s:.1f})" for p, s in selected)
        print(f"[{i:3}/{len(identity_dirs)}] {id_dir.name}: {names}")

        total += len(selected)

    print(f"\n✓ Gallery: {GALLERY_ROOT}")
    print(f"  {len(identity_dirs)-skipped} identities | {total} images | {skipped} skipped")


if __name__ == "__main__":
    main()
