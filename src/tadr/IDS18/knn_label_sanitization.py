"""
KNN label sanitization defense for CIC-IDS2018 / NSL-KDD style datasets.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = Path(SCRIPT_DIR).resolve().parents[2]

from tadr.configs import CIC2018Resources, NSLKDDResources
from tadr.preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from tadr.preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from tadr.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

REGISTRY = {
    "cic2018": (CIC2018Resources, CIC2018Preprocessor),
    "nslkdd": (NSLKDDResources, NSLKDDPreprocessor),
}


def resolve_repo_path(path_text: str) -> Path:
    """Resolve a path against common project locations and return the first match."""
    path = Path(path_text)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(SCRIPT_DIR) / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "src" / "tadr" / "IDS18" / path,
        PROJECT_ROOT / "src" / "tadr" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path(SCRIPT_DIR) / path


def encode_dataset(df: pd.DataFrame, preprocessor, num_encoder: str) -> pd.DataFrame:
    """Select model features and apply the configured preprocessing pipeline."""
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


def inverse_transform_data(df_encoded: pd.DataFrame, preprocessor, num_encoder: str) -> pd.DataFrame:
    """Invert the encoded frame back to the original feature representation."""
    numerical_inverse = "minmax" if num_encoder == "minmax" else "standard"
    return preprocessor.inverse_transform(df_encoded, numerical_inverse=numerical_inverse)


def majority_with_tie_break(neighbor_labels: np.ndarray, neighbor_distances: np.ndarray) -> tuple[int, float]:
    """Pick the majority label among neighbors and break ties by nearest appearance."""
    counts: dict[int, int] = {}
    for lbl in neighbor_labels:
        key = int(lbl)
        counts[key] = counts.get(key, 0) + 1

    max_count = max(counts.values())
    tied_labels = [lbl for lbl, cnt in counts.items() if cnt == max_count]

    if len(tied_labels) == 1:
        mode_label = tied_labels[0]
    else:
        mode_label = None
        for lbl, dist in zip(neighbor_labels, neighbor_distances):
            if int(lbl) in tied_labels:
                mode_label = int(lbl)
                break
        if mode_label is None:
            mode_label = tied_labels[0]

    confidence = float(max_count / max(len(neighbor_labels), 1))
    return mode_label, confidence


def apply_knn_sanitization(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    eta: float,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Relabel candidate rows when their neighbors strongly agree on a different class."""
    if len(X) == 0:
        return y.copy(), np.zeros(0, dtype=bool), np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    # Fit KNN on the encoded samples and query each row against its neighborhood.
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    y_new = y.copy()
    changed = np.zeros(len(y), dtype=bool)
    confidence_arr = np.zeros(len(y), dtype=float)
    mode_arr = np.full(len(y), -1, dtype=int)

    for i in range(len(y)):
        if not candidate_mask[i]:
            continue

        # Skip the first neighbor because it is the sample itself.
        neigh_idx = indices[i, 1:]
        neigh_dist = distances[i, 1:]

        if len(neigh_idx) == 0:
            continue

        neigh_labels = y[neigh_idx]
        mode_label, conf = majority_with_tie_break(neigh_labels, neigh_dist)
        confidence_arr[i] = conf
        mode_arr[i] = int(mode_label)

        if conf >= eta and int(mode_label) != int(y[i]):
            y_new[i] = int(mode_label)
            changed[i] = True

    return y_new, changed, confidence_arr, mode_arr


def main() -> None:
    """Run the full KNN sanitization workflow and write sanitized outputs plus a report."""
    parser = argparse.ArgumentParser(description="Multi-class KNN label sanitization defense")
    parser.add_argument("--input", "-i", type=str, default="src/tadr/IDS18/resources/IDS18/clean_merged/cic2018_train_original_undersampled_noise_30.csv")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--resource", "-r", type=str, default="cic2018", choices=list(REGISTRY.keys()))
    parser.add_argument("--num-encoder", "-n", type=str, default="minmax", choices=["minmax", "quantile_uniform"])
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--only-benign", action="store_true", help="Only sanitize rows currently labeled as Benign")
    parser.add_argument("--benign-label", type=str, default="Benign")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.k <= 0:
        raise SystemExit("--k must be > 0")
    if not (0.5 <= float(args.eta) <= 1.0):
        raise SystemExit("--eta must be in [0.5, 1.0]")

    args.input = str(resolve_repo_path(args.input))
    if args.output is not None:
        args.output = str(resolve_repo_path(args.output))
    if args.report is not None:
        args.report = str(resolve_repo_path(args.report))

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_knn_sanitized{ext}"
    if args.report is None:
        base, _ = os.path.splitext(args.output)
        args.report = f"{base}_report.json"

    # Load the fitted encoders and preprocessing utilities for the selected resource.
    _, PreprocessorClass = REGISTRY[args.resource]
    pre = PreprocessorClass()

    if not pre.load_encoders():
        raise SystemExit("Encoders not found. Please fit/load encoders first.")

    # Read raw data and keep metadata columns aside before preprocessing.
    df_in = pd.read_csv(args.input, low_memory=False)
    logger.info(f"[+] Input shape: {df_in.shape}")

    meta_cols = ["original_label", "is_noisy", "__source__", "detected_as_noise"]
    meta_saved = {c: df_in[c].copy() for c in meta_cols if c in df_in.columns}
    df_model = df_in.drop(columns=meta_cols, errors="ignore")

    # Encode only the model features, then split features and current labels.
    df_enc = encode_dataset(df_model, pre, args.num_encoder)
    X = df_enc.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df_enc.iloc[:, -1].to_numpy(dtype=np.int64)

    # Build the set of rows that are allowed to be sanitized.
    if args.only_benign:
        benign_encoded = int(pre.encoders["label"].transform([args.benign_label])[0])
        candidate_mask = (y == benign_encoded)
    else:
        candidate_mask = np.ones(len(y), dtype=bool)

    # Apply the KNN defense only to candidate rows.
    y_new, changed_mask, conf_arr, mode_arr = apply_knn_sanitization(
        X,
        y,
        k=int(args.k),
        eta=float(args.eta),
        candidate_mask=candidate_mask,
    )

    # Write the relabelled class back into the encoded frame and invert it.
    df_enc_new = df_enc.copy()
    df_enc_new.iloc[:, -1] = y_new.astype(df_enc_new.iloc[:, -1].dtype)
    df_out = inverse_transform_data(df_enc_new, pre, args.num_encoder)

    # Restore any metadata columns that were removed before encoding.
    for col, series in meta_saved.items():
        df_out[col] = series

    # Convert predicted mode labels back to human-readable form.
    inv_mode = np.full(len(mode_arr), "", dtype=object)
    valid_mode = mode_arr >= 0
    if np.any(valid_mode):
        inv_mode[valid_mode] = pre.encoders["label"].inverse_transform(mode_arr[valid_mode].astype(int))

    # Add defense outputs for inspection and auditing.
    df_out["knn_confidence"] = conf_arr
    df_out["knn_mode_label"] = inv_mode
    df_out["knn_relabelled"] = changed_mask

    # Save the sanitized CSV.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)

    changed_count = int(changed_mask.sum())
    total = int(len(df_out))

    changed_per_class = (
        pd.DataFrame({"old": y, "changed": changed_mask})
        .groupby("old")["changed"]
        .sum()
        .to_dict()
    )
    changed_per_class_readable = {
        str(pre.encoders["label"].inverse_transform([int(k)])[0]): int(v)
        for k, v in changed_per_class.items()
    }

    # Assemble the final summary report.
    report = {
        "input_csv": args.input,
        "output_csv": args.output,
        "resource": args.resource,
        "k": int(args.k),
        "eta": float(args.eta),
        "only_benign": bool(args.only_benign),
        "total_samples": total,
        "relabelled_samples": changed_count,
        "relabelled_rate": float(changed_count / total) if total > 0 else 0.0,
        "relabelled_per_class": changed_per_class_readable,
    }

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[+] Saved sanitized CSV: {args.output}")
    logger.info(f"[+] Saved report JSON: {args.report}")


if __name__ == "__main__":
    main()
