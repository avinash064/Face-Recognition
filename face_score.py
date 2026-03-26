# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from insightface.app import FaceAnalysis
# import mediapipe as mp

# # ==============================
# # CONFIG
# # ==============================

# DATASET_PATH = "/home/avinash/datasets/imdb_wiki/wiki_crop"
# REF_PER_ID = 5

# # ==============================
# # INIT MODELS
# # ==============================

# # InsightFace (GPU)
# face_app = FaceAnalysis(
#     name="buffalo_l",
#     providers=['CUDAExecutionProvider']
# )
# face_app.prepare(ctx_id=0, det_size=(640, 640))

# # MediaPipe FaceMesh (CPU)
# mp_face_mesh = mp.solutions.face_mesh
# mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# # ==============================
# # POSE ESTIMATION (MediaPipe)
# # ==============================

# def get_pose(img):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = mesh.process(img_rgb)

#     if not results.multi_face_landmarks:
#         return None

#     landmarks = results.multi_face_landmarks[0]

#     # key points (approx)
#     left_eye = landmarks.landmark[33]
#     right_eye = landmarks.landmark[263]
#     nose = landmarks.landmark[1]

#     # simple yaw estimation
#     yaw = (right_eye.x - left_eye.x)

#     # pitch estimation (nose vertical shift)
#     pitch = nose.y - ((left_eye.y + right_eye.y) / 2)

#     # roll estimation
#     roll = (right_eye.y - left_eye.y)

#     return np.array([yaw, pitch, roll])


# # ==============================
# # IMAGE PROCESSING
# # ==============================

# def process_image(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         return None

#     faces = face_app.get(img)
#     if len(faces) == 0:
#         return None

#     face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))

#     if face.det_score < 0.6:
#         return None

#     pose = get_pose(img)
#     if pose is None:
#         return None

#     return {
#         "embedding": face.embedding,
#         "pose": pose,
#         "path": img_path
#     }


# # ==============================
# # SCORING FUNCTION
# # ==============================

# def compute_score(pose):
#     yaw, pitch, roll = pose

#     # want diversity → penalize near-zero all
#     score = (
#         abs(yaw) * 2 +
#         abs(pitch) * 1.5 +
#         abs(roll) * 1.0
#     )

#     return score


# # ==============================
# # SELECT BEST 5
# # ==============================

# def select_best(images):
#     # compute scores
#     for img in images:
#         img["score"] = compute_score(img["pose"])

#     # sort by score descending
#     images = sorted(images, key=lambda x: -x["score"])

#     return images[:REF_PER_ID]


# # ==============================
# # MAIN
# # ==============================

# all_embeddings = []
# all_paths = []
# all_ids = []

# identities = sorted(os.listdir(DATASET_PATH))

# for identity in tqdm(identities, desc="Processing IDs"):
#     identity_path = os.path.join(DATASET_PATH, identity)

#     if not os.path.isdir(identity_path):
#         continue

#     images = []

#     for img_name in os.listdir(identity_path):
#         img_path = os.path.join(identity_path, img_name)

#         result = process_image(img_path)
#         if result:
#             images.append(result)

#     if len(images) < REF_PER_ID:
#         continue

#     refs = select_best(images)

#     for r in refs:
#         all_embeddings.append(r["embedding"])
#         all_paths.append(r["path"])
#         all_ids.append(identity)


# # ==============================
# # SAVE
# # ==============================

# all_embeddings = np.array(all_embeddings)

# np.save("reference_embeddings.npy", all_embeddings)

# with open("reference_paths.txt", "w") as f:
#     for p in all_paths:
#         f.write(p + "\n")

# with open("reference_ids.txt", "w") as f:
#     for i in all_ids:
#         f.write(i + "\n")

# print(f"\nSaved {len(all_embeddings)} reference embeddings")



import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ==============================
# CONFIG
# ==============================

DATASET_PATH = "/home/avinash/datasets/imdb_wiki/wiki_crop"
REF_PER_ID = 5
MIN_IMAGES_PER_ID = 5

# ==============================
# INIT MODEL (GPU ONLY)
# ==============================

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=['detection', 'recognition'],
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))


# ==============================
# PROCESS IMAGE
# ==============================

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    faces = app.get(img)
    if len(faces) == 0:
        return None

    # pick largest face
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))

    # filter bad detections
    if face.det_score < 0.6:
        return None

    # pose might be None → skip
    if face.pose is None:
        return None

    return {
        "embedding": face.embedding,
        "pose": face.pose,
        "path": img_path
    }


# ==============================
# POSE SCORING FUNCTION
# ==============================

def compute_score(pose):
    yaw, pitch, roll = pose

    # encourage diversity
    score = (
        abs(yaw) * 2.0 +
        abs(pitch) * 1.5 +
        abs(roll) * 1.0
    )

    return score


# ==============================
# SELECT BEST 5 REFERENCE IMAGES
# ==============================

def select_best(images):
    if len(images) <= REF_PER_ID:
        return images

    # compute scores
    for img in images:
        img["score"] = compute_score(img["pose"])

    # sort by score (descending)
    images = sorted(images, key=lambda x: -x["score"])

    # take top 5
    selected = images[:REF_PER_ID]

    return selected


# ==============================
# MAIN
# ==============================

all_embeddings = []
all_paths = []
all_ids = []

identities = sorted(os.listdir(DATASET_PATH))

for identity in tqdm(identities, desc="Processing IDs"):
    identity_path = os.path.join(DATASET_PATH, identity)

    if not os.path.isdir(identity_path):
        continue

    images = []

    for img_name in os.listdir(identity_path):
        img_path = os.path.join(identity_path, img_name)

        result = process_image(img_path)
        if result:
            images.append(result)

    # filter weak identities
    if len(images) < MIN_IMAGES_PER_ID:
        continue

    refs = select_best(images)

    for r in refs:
        all_embeddings.append(r["embedding"])
        all_paths.append(r["path"])
        all_ids.append(identity)


# ==============================
# SAVE OUTPUT
# ==============================

all_embeddings = np.array(all_embeddings)

np.save("reference_embeddings.npy", all_embeddings)

with open("reference_paths.txt", "w") as f:
    for p in all_paths:
        f.write(p + "\n")

with open("reference_ids.txt", "w") as f:
    for i in all_ids:
        f.write(i + "\n")

print("\n==============================")
print(f"Saved embeddings: {len(all_embeddings)}")
print("==============================")