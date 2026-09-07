"""
Cleanlab-based label noise defense for NSL-KDD.

Uses Confident Learning algorithm to identify and remove mislabeled samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = Path(SCRIPT_DIR).resolve().parents[2]

from tadr.configs import NSLKDDResources
from tadr.preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from tadr.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

try:
    from cleanlab.filter import find_label_issues
except ImportError:
    logger.error("cleanlab not installed. Please: pip install cleanlab")
    raise


def resolve_repo_path(path_text: str) -> Path:
    """Resolve path relative to repo root or current directory."""
    path = Path(path_text)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(SCRIPT_DIR) / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "src" / "tadr" / "NSL_KDD" / path,
        PROJECT_ROOT / "src" / "tadr" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path(SCRIPT_DIR) / path


def encode_dataset(df: pd.DataFrame, preprocessor: NSLKDDPreprocessor, num_encoder: str) -> pd.DataFrame:
    """Encode dataset for classifier training."""
    df_enc = preprocessor.select_features_and_label(df)

    if num_encoder == "minmax":
        df_enc = preprocessor.preprocess_encode_numerical_features_minmax(df_enc)
    elif num_encoder == "quantile_uniform":
        df_enc = preprocessor.preprocess_encode_numerical_features_quantile_uniform(df_enc)
    else:
        raise ValueError(f"Unknown numerical encoder: {num_encoder}")

    df_enc = preprocessor.preprocess_encode_binary_features(df_enc)
    df_enc = preprocessor.preprocess_encode_label(df_enc)
    df_enc = preprocessor.preprocess_encode_categorical_features(df_enc)
    return df_enc


def inverse_transform_data(df_encoded: pd.DataFrame, preprocessor: NSLKDDPreprocessor, num_encoder: str) -> pd.DataFrame:
    """Inverse transform encoded features back to original space."""
    numerical_inverse = "minmax" if num_encoder == "minmax" else "standard"
    return preprocessor.inverse_transform(df_encoded, numerical_inverse=numerical_inverse)


def train_detector_model(X: np.ndarray, y: np.ndarray, model_type: str, n_jobs: int = -1) -> Tuple[object, np.ndarray]:
    """
    Train a lightweight detector model using cross-validation to get out-of-fold predictions.
    
    Args:
        X: Encoded feature matrix
        y: Encoded labels
        model_type: 'rf' (RandomForest) or 'lgbm' (LightGBM)
        n_jobs: Number of parallel jobs
        
    Returns:
        (trained_model, pred_probs) where pred_probs are out-of-fold probabilities
    """
    logger.info(f"[+] Training {model_type} detector model with cross-validation...")
    
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=n_jobs,
            verbose=0
        )
    elif model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=150,
                max_depth=8,
                num_leaves=31,
                random_state=42,
                n_jobs=n_jobs,
                verbose=-1
            )
        except ImportError:
            logger.warning("LightGBM not available, falling back to RandomForest")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=n_jobs,
                verbose=0
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Get out-of-fold predictions using cross_val_predict
    pred_probs = cross_val_predict(
        model,
        X,
        y,
        cv=5,
        method="predict_proba",
        n_jobs=n_jobs
    )

    # Train final model on full data for later use
    model.fit(X, y)
    logger.info(f"[+] Model training complete")
    
    return model, pred_probs


def apply_cleanlab_filter(
    y: np.ndarray,
    pred_probs: np.ndarray,
    candidate_mask: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply cleanlab confident learning to identify label issues.
    
    Args:
        y: Encoded labels
        pred_probs: Model prediction probabilities (out-of-fold)
        candidate_mask: Boolean mask of samples eligible for filtering
        threshold: Label quality score threshold (lower = more aggressive filtering)
        
    Returns:
        (issues_idx, label_quality_scores, issue_info)
    """
    logger.info(f"[+] Running cleanlab.find_label_issues with threshold={threshold}")

    # cleanlab expects labels in contiguous range [0, K-1] matching pred_probs columns.
    unique_labels = np.unique(y)
    if pred_probs.shape[1] != len(unique_labels):
        raise ValueError(
            f"pred_probs has {pred_probs.shape[1]} columns but observed labels are {len(unique_labels)}. "
            "Ensure detector probabilities align with observed classes."
        )
    y_local = np.searchsorted(unique_labels, y)
    
    ranked_issue_indices = find_label_issues(
        labels=y_local,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        filter_by="confident_learning"
    )

    if isinstance(ranked_issue_indices, np.ndarray) and ranked_issue_indices.dtype == bool:
        ranked_issue_indices = np.where(ranked_issue_indices)[0]
    ranked_issue_indices = np.asarray(ranked_issue_indices, dtype=int)

    row_idx = np.arange(len(y))
    all_quality_scores = pred_probs[row_idx, y_local]

    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    flagged_by_cleanlab = np.zeros(len(y), dtype=bool)
    if len(ranked_issue_indices) > 0:
        flagged_by_cleanlab[ranked_issue_indices] = True

    threshold_mask = all_quality_scores < float(threshold)
    final_issue_mask = flagged_by_cleanlab & threshold_mask & candidate_mask
    issues_idx = np.where(final_issue_mask)[0]
    
    issue_info = {
        "total_issues_detected": int(len(issues_idx)),
        "issue_rate": float(len(issues_idx) / len(y)) if len(y) > 0 else 0.0,
        "threshold_used": float(threshold),
        "candidate_samples": int(candidate_mask.sum()),
        "candidate_rate": float(candidate_mask.mean()) if len(candidate_mask) > 0 else 0.0,
    }
    
    logger.info(f"[+] Found {issue_info['total_issues_detected']} label issues ({issue_info['issue_rate']*100:.2f}%)")
    
    return issues_idx, all_quality_scores, issue_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanlab-based label noise defense for NSL-KDD")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="src/tadr/NSL_KDD/resources/NSLKDD/clean_merged/nslkdd_train_clean_merged_noise_30.csv"
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--num-encoder", "-n", type=str, default="minmax", choices=["minmax", "quantile_uniform"])
    parser.add_argument("--model-type", "-m", type=str, default="lgbm", choices=["rf", "lgbm"])
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Label quality threshold (lower = stricter)")
    parser.add_argument("--only-benign", action="store_true", help="Only filter rows currently labeled as Benign")
    parser.add_argument("--benign-label", type=str, default="Benign")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    setup_logging(args.log_level)

    if not (0.0 < float(args.threshold) <= 1.0):
        raise SystemExit("--threshold must be in (0.0, 1.0]")

    args.input = str(resolve_repo_path(args.input))
    if args.output is not None:
        args.output = str(resolve_repo_path(args.output))
    if args.report is not None:
        args.report = str(resolve_repo_path(args.report))

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_cleanlab_sanitized{ext}"
    if args.report is None:
        base, _ = os.path.splitext(args.output)
        args.report = f"{base}_report.json"

    pre = NSLKDDPreprocessor()
    if not pre.load_encoders():
        raise SystemExit("Encoders not found. Please fit/load encoders first.")

    df_in = pd.read_csv(args.input, low_memory=False)
    logger.info(f"[+] Input shape: {df_in.shape}")

    meta_cols = ["original_label", "is_noisy", "__source__", "detected_as_noise"]
    meta_saved = {c: df_in[c].copy() for c in meta_cols if c in df_in.columns}
    df_model = df_in.drop(columns=meta_cols, errors="ignore")

    # Encode dataset
    df_enc = encode_dataset(df_model, pre, args.num_encoder)
    X = df_enc.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df_enc.iloc[:, -1].to_numpy(dtype=np.int64)

    # Train detector and get pred_probs
    model, pred_probs = train_detector_model(X, y, args.model_type, n_jobs=args.n_jobs)

    if args.only_benign:
        benign_encoded = int(pre.encoders["label"].transform([args.benign_label])[0])
        candidate_mask = (y == benign_encoded)
    else:
        candidate_mask = np.ones(len(y), dtype=bool)

    # Apply cleanlab filter
    issues_idx, label_quality_scores, issue_info = apply_cleanlab_filter(
        y,
        pred_probs,
        candidate_mask=candidate_mask,
        threshold=args.threshold
    )

    # Create mask for samples to keep (NOT in issues list)
    keep_mask = np.ones(len(df_in), dtype=bool)
    keep_mask[issues_idx] = False

    # Filter data
    df_enc_filtered = df_enc[keep_mask].copy()

    # Inverse transform
    df_out = inverse_transform_data(df_enc_filtered, pre, args.num_encoder)

    # Restore metadata columns
    for col in meta_cols:
        if col in meta_saved:
            df_out[col] = meta_saved[col][keep_mask].values

    # Add cleanlab-specific columns
    df_out["cleanlab_label_quality"] = label_quality_scores[keep_mask]
    df_out["cleanlab_flagged"] = False

    # Save cleaned CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)
    logger.info(f"[+] Saved cleaned CSV: {args.output} ({len(df_out)} rows, {len(df_in) - len(df_out)} removed)")

    # Recompute per-class stats
    per_class_removal = {}
    for label_encoded in np.unique(y):
        label_name = str(pre.encoders["label"].inverse_transform([int(label_encoded)])[0])
        mask_label = (y == label_encoded)
        total_count = int(mask_label.sum())
        removed_count = int(np.isin(np.where(mask_label)[0], issues_idx).sum())
        per_class_removal[label_name] = {
            "total": total_count,
            "removed": removed_count,
            "removal_rate": float(removed_count / total_count) if total_count > 0 else 0.0,
        }

    report = {
        "input_csv": args.input,
        "output_csv": args.output,
        "resource": "nslkdd",
        "defense_method": "cleanlab_confident_learning",
        "model_type": args.model_type,
        "threshold": float(args.threshold),
        "only_benign": bool(args.only_benign),
        "benign_label": args.benign_label,
        "total_samples": int(len(df_in)),
        "samples_removed": int(len(df_in) - len(df_out)),
        "samples_kept": int(len(df_out)),
        "removal_rate": float((len(df_in) - len(df_out)) / len(df_in)) if len(df_in) > 0 else 0.0,
        "label_issues_detected": issue_info,
        "per_class_stats": per_class_removal,
        "label_quality_stats": {
            "min": float(np.min(label_quality_scores)),
            "max": float(np.max(label_quality_scores)),
            "mean": float(np.mean(label_quality_scores)),
            "median": float(np.median(label_quality_scores)),
            "std": float(np.std(label_quality_scores)),
        },
    }

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[+] Saved report: {args.report}")


if __name__ == "__main__":
    main()
