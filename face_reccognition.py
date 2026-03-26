# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from collections import defaultdict
# from sklearn.metrics.pairwise import cosine_similarity

# from insightface.app import FaceAnalysis


# # ==============================
# # CONFIG
# # ==============================

# DATASET_PATH = "/home/avinash/datasets/imdb_wiki/wiki_crop"
# MODELS = ["buffalo_l", "buffalo_m", "buffalo_s"]
# MAX_REF = 3


# # ==============================
# # INIT MODEL
# # ==============================

# def load_model(model_name):
#     print(f"\n[INFO] Loading model: {model_name}")
#     app = FaceAnalysis(
#         name=model_name,
#         providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
#     )
#     app.prepare(ctx_id=0, det_size=(640, 640))
#     return app


# # ==============================
# # FACE PROCESSING
# # ==============================

# def process_image(app, img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         return None

#     faces = app.get(img)

#     if len(faces) == 0:
#         return None

#     # select largest face
#     face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

#     embedding = face.embedding
#     pose = face.pose  # (yaw, pitch, roll)

#     return {
#         "embedding": embedding,
#         "pose": pose,
#         "path": img_path
#     }


# # ==============================
# # POSE-BASED SELECTION
# # ==============================

# def select_reference(images):
#     if len(images) <= MAX_REF:
#         return images, []

#     # sort by yaw (frontal first)
#     images_sorted = sorted(images, key=lambda x: abs(x["pose"][0]))

#     ref = []

#     # 1. frontal
#     ref.append(images_sorted[0])

#     # 2. extreme yaw
#     images_yaw = sorted(images, key=lambda x: -abs(x["pose"][0]))
#     if len(images_yaw) > 1:
#         ref.append(images_yaw[0])

#     # 3. diverse pitch/roll
#     images_pitch = sorted(images, key=lambda x: abs(x["pose"][1]))
#     if len(images_pitch) > 2:
#         ref.append(images_pitch[-1])

#     # remove duplicates
#     ref = list({x["path"]: x for x in ref}.values())[:MAX_REF]

#     ref_paths = set([x["path"] for x in ref])
#     query = [x for x in images if x["path"] not in ref_paths]

#     if len(query) <= len(ref):
#         query = images[len(ref):]

#     return ref, query


# # ==============================
# # LOAD DATASET
# # ==============================

# def load_dataset(app):
#     data = {}

#     identities = sorted(os.listdir(DATASET_PATH))

#     for identity in tqdm(identities, desc="Processing identities"):
#         identity_path = os.path.join(DATASET_PATH, identity)

#         if not os.path.isdir(identity_path):
#             continue

#         images = []

#         for img_name in os.listdir(identity_path):
#             img_path = os.path.join(identity_path, img_name)

#             result = process_image(app, img_path)
#             if result:
#                 images.append(result)

#         if len(images) < 2:
#             continue

#         ref, query = select_reference(images)

#         if len(query) == 0:
#             continue

#         data[identity] = {
#             "ref": ref,
#             "query": query
#         }

#     return data


# # ==============================
# # MATCHING + METRICS
# # ==============================

# def evaluate(data):
#     correct_rank1 = 0
#     correct_top5 = 0
#     total = 0

#     gallery_embeddings = []
#     gallery_labels = []

#     # build gallery
#     for identity, items in data.items():
#         for r in items["ref"]:
#             gallery_embeddings.append(r["embedding"])
#             gallery_labels.append(identity)

#     gallery_embeddings = np.array(gallery_embeddings)

#     # evaluate queries
#     for identity, items in data.items():
#         for q in items["query"]:
#             total += 1

#             sim = cosine_similarity(
#                 [q["embedding"]],
#                 gallery_embeddings
#             )[0]

#             sorted_idx = np.argsort(-sim)

#             ranked_labels = [gallery_labels[i] for i in sorted_idx]

#             # Rank-1
#             if ranked_labels[0] == identity:
#                 correct_rank1 += 1

#             # Top-5
#             if identity in ranked_labels[:5]:
#                 correct_top5 += 1

#     rank1 = correct_rank1 / total if total > 0 else 0
#     top5 = correct_top5 / total if total > 0 else 0

#     return {
#         "rank1": rank1,
#         "top5": top5,
#         "total_queries": total
#     }


# # ==============================
# # MAIN
# # ==============================

# def main():
#     results = {}

#     for model_name in MODELS:
#         app = load_model(model_name)

#         print("[INFO] Loading dataset...")
#         data = load_dataset(app)

#         print("[INFO] Evaluating...")
#         metrics = evaluate(data)

#         results[model_name] = metrics

#         print(f"\n[RESULT] {model_name}")
#         print(metrics)

#     print("\n=== FINAL RESULTS ===")
#     print(results)


# if __name__ == "__main__":
#     main()



import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis


# ==============================
# CONFIG
# ==============================

DATASET_PATH = "/home/avinash/datasets/imdb_wiki/wiki_crop"
MODELS = ["buffalo_l", "buffalo_m", "buffalo_s"]
MAX_REF = 3
MIN_IMAGES_PER_ID = 5   # filter noisy identities


# ==============================
# INIT MODEL (FIXED)
# ==============================

def load_model(model_name):
    print(f"\n[INFO] Loading model: {model_name}")

    app = FaceAnalysis(
        name=model_name,
        allowed_modules=['detection', 'recognition'],
        providers=['CUDAExecutionProvider']
    )

    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


# ==============================
# FACE PROCESSING (IMPROVED)
# ==============================

def process_image(app, img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    faces = app.get(img)

    if len(faces) == 0:
        return None

    # select largest face
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    # FILTER LOW QUALITY
    if face.det_score < 0.6:
        return None

    return {
        "embedding": face.embedding,
        "pose": face.pose,
        "path": img_path
    }


# ==============================
# POSE-BASED SELECTION (IMPROVED)
# ==============================

def select_reference(images):
    if len(images) <= MAX_REF:
        return images, []

    ref = []

    # 1. frontal (min yaw)
    frontal = min(images, key=lambda x: abs(x["pose"][0]))
    ref.append(frontal)

    # 2. max yaw
    profile = max(images, key=lambda x: abs(x["pose"][0]))
    ref.append(profile)

    # 3. max pitch
    pitch_var = max(images, key=lambda x: abs(x["pose"][1]))
    ref.append(pitch_var)

    # remove duplicates
    ref = list({x["path"]: x for x in ref}.values())[:MAX_REF]

    ref_paths = set(x["path"] for x in ref)
    query = [x for x in images if x["path"] not in ref_paths]

    if len(query) <= len(ref):
        query = images[len(ref):]

    return ref, query


# ==============================
# LOAD DATASET (FIXED LOGIC)
# ==============================

def load_dataset(app):
    data = {}

    identities = sorted(os.listdir(DATASET_PATH))

    for identity in tqdm(identities, desc="Processing identities"):
        identity_path = os.path.join(DATASET_PATH, identity)

        if not os.path.isdir(identity_path):
            continue

        images = []

        for img_name in os.listdir(identity_path):
            img_path = os.path.join(identity_path, img_name)

            result = process_image(app, img_path)
            if result:
                images.append(result)

        # FILTER BAD IDENTITIES
        if len(images) < MIN_IMAGES_PER_ID:
            continue

        ref, query = select_reference(images)

        if len(query) == 0:
            continue

        data[identity] = {
            "ref": ref,
            "query": query
        }

    return data


# ==============================
# MATCHING + METRICS
# ==============================

def evaluate(data):
    correct_rank1 = 0
    correct_top5 = 0
    total = 0

    gallery_embeddings = []
    gallery_labels = []

    # build gallery
    for identity, items in data.items():
        for r in items["ref"]:
            gallery_embeddings.append(r["embedding"])
            gallery_labels.append(identity)

    gallery_embeddings = np.array(gallery_embeddings)

    # evaluate queries
    for identity, items in data.items():
        for q in items["query"]:
            total += 1

            sim = cosine_similarity(
                [q["embedding"]],
                gallery_embeddings
            )[0]

            sorted_idx = np.argsort(-sim)
            ranked_labels = [gallery_labels[i] for i in sorted_idx]

            if ranked_labels[0] == identity:
                correct_rank1 += 1

            if identity in ranked_labels[:5]:
                correct_top5 += 1

    return {
        "rank1": correct_rank1 / total if total else 0,
        "top5": correct_top5 / total if total else 0,
        "total_queries": total
    }


# ==============================
# MAIN
# ==============================

def main():
    results = {}

    for model_name in MODELS:
        try:
            app = load_model(model_name)
        except Exception as e:
            print(f"[ERROR] Skipping {model_name}: {e}")
            continue

        print("[INFO] Loading dataset...")
        data = load_dataset(app)

        print(f"[INFO] Identities used: {len(data)}")

        print("[INFO] Evaluating...")
        metrics = evaluate(data)

        results[model_name] = metrics

        print(f"\n[RESULT] {model_name}")
        print(metrics)

    print("\n=== FINAL RESULTS ===")
    print(results)


if __name__ == "__main__":
    main()