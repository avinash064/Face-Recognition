#!/usr/bin/env bash
# run_pipeline.sh
# Runs the full face recognition evaluation pipeline end-to-end.

set -e

echo "============================================"
echo " Face Recognition Evaluation Pipeline"
echo "============================================"

echo ""
echo "[Step 1/2] Extracting Gallery and Probe Embeddings..."
python face_eval_pipeline/scripts/extract_all_features.py

echo ""
echo "[Step 2/2] Running Cosine Similarity Evaluation..."
python face_eval_pipeline/scripts/evaluate_similarities.py

echo ""
echo "============================================"
echo " Pipeline complete! Check results/"
echo "============================================"
