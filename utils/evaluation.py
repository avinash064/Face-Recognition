"""
utils/evaluation.py

Evaluation and reporting utilities for face recognition experiments.
"""

import json
import numpy as np
from pathlib import Path


def compute_rank1(predictions: list) -> float:
    """
    Compute Rank-1 closed-set accuracy from prediction records.

    Args:
        predictions: List of dicts with keys 'predicted_identity', 'ground_truth_identity'.

    Returns:
        Rank-1 accuracy as a float (0.0 - 1.0).
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if p['predicted_identity'] == p['ground_truth_identity'])
    return correct / len(predictions)


def print_comparison_table(metrics: dict, datasets: list, models: list):
    """
    Print a formatted model comparison table to the terminal.

    Args:
        metrics: Dict of {dataset: {model: rank1_acc}}.
        datasets: List of dataset names.
        models:   List of model names.
    """
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON (Cosine Similarity Rank-1 Accuracy)")
    print("=" * 80)
    header = f"| {'Model':<12} | " + " | ".join(f"{d:<10} Rank-1" for d in datasets) + " | Avg Accuracy |"
    print(header)
    print("-" * len(header))

    model_avgs = {}
    for m in models:
        accs = [metrics.get(d, {}).get(m, 0.0) for d in datasets]
        avg = float(np.mean(accs))
        model_avgs[m] = avg
        row = f"| {m:<12} | " + " | ".join(f"{a:<17.4f}" for a in accs) + f" | {avg:<12.4f} |"
        print(row)
    print("=" * 80)
    return model_avgs


def save_results(metrics: dict, model_avgs: dict, output_path: Path):
    """
    Save the final model comparison metrics to a JSON file.

    Args:
        metrics:     Dict of {dataset: {model: rank1_acc}}.
        model_avgs:  Dict of {model: avg_accuracy}.
        output_path: Path to the output .json file.
    """
    best_model = max(model_avgs, key=model_avgs.get) if model_avgs else "N/A"
    output = {
        "model_comparison": {},
        "best_model": best_model
    }
    for m, avg in model_avgs.items():
        output["model_comparison"][m] = {d: metrics.get(d, {}).get(m, 0.0) for d in metrics}
        output["model_comparison"][m]["avg"] = avg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\n✓ Saved results to {output_path}")
    print(f"\nBest Model: {best_model} (Avg Accuracy: {model_avgs.get(best_model, 0):.4f})")
