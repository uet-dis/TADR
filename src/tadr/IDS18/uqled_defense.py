"""
UQ-LED CL-MCD-E label-error defense for CIC-IDS2018.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from tadr.EdgeIIoTset.uqled_defense import (
    OOFMCDResult,
    UQLEDConfig,
    UQLEDMLP,
    UQLEDResult,
    _parse_hidden_dims,
    _write_outputs,
    build_scope_masks,
    classwise_thresholds,
    compute_entropy_filtered_confident_joint,
    enable_mc_dropout,
    generate_oof_mcd_predictions,
    predictive_entropy,
    prune_by_noise_rate,
    resolve_device,
    run_uqled_detector,
    set_global_seed,
)
from tadr.preprocessing.cic2018_preprocessor import CIC2018Preprocessor


def _encode_ids18(df: pd.DataFrame, num_encoder: str) -> tuple[np.ndarray, np.ndarray, Any]:
    preprocessor = CIC2018Preprocessor()
    encoders_dir = SCRIPT_DIR / "encoders"
    if not preprocessor.load_encoders(str(encoders_dir), fixed_label_encoder=True):
        raise RuntimeError(f"CIC-IDS2018 encoders were not found in {encoders_dir}")

    encoded = preprocessor.select_features_and_label(df)
    if num_encoder == "minmax":
        encoded = preprocessor.preprocess_encode_numerical_features_minmax(encoded)
    elif num_encoder == "quantile_uniform":
        encoded = preprocessor.preprocess_encode_numerical_features_quantile_uniform(encoded)
    else:
        raise ValueError(f"Unsupported numerical encoder: {num_encoder}")
    encoded = preprocessor.preprocess_encode_binary_features(encoded)
    encoded = preprocessor.preprocess_encode_label(encoded)
    encoded = preprocessor.preprocess_encode_categorical_features(encoded)
    X = encoded.drop(columns=[preprocessor.label_column]).to_numpy(dtype=np.float32)
    y = encoded[preprocessor.label_column].to_numpy(dtype=np.int64)
    return X, y, preprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="UQ-LED CL-MCD-E defense for CIC-IDS2018")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output-global", required=True)
    parser.add_argument("--output-benign-only", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--benign-label", default="Benign")
    parser.add_argument("--num-encoder", default="minmax", choices=["minmax", "quantile_uniform"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--mc-passes", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden-dims", type=_parse_hidden_dims, default=(256, 256, 128, 128, 64))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")
    if args.n_splits < 2 or args.mc_passes < 2:
        raise SystemExit("UQ-LED requires at least two folds and two MCD passes")
    if not 0.0 < args.dropout < 1.0:
        raise SystemExit("--dropout must be in (0, 1)")
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in (0, 1)")

    df = pd.read_csv(input_path, low_memory=False)
    X, labels, preprocessor = _encode_ids18(df, args.num_encoder)
    try:
        benign_class_id = int(preprocessor.encoders["label"].transform([args.benign_label])[0])
    except ValueError as exc:
        raise SystemExit(f"Unknown benign label {args.benign_label!r}") from exc

    config = UQLEDConfig(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        n_splits=args.n_splits,
        mc_passes=args.mc_passes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )
    print(
        f"[UQ-LED] CL-MCD-E: samples={len(X)}, features={X.shape[1]}, "
        f"classes={len(np.unique(labels))}, device={resolve_device(config.device)}"
    )
    result = run_uqled_detector(X, labels, benign_class_id, config)
    _write_outputs(
        df,
        labels,
        preprocessor,
        result,
        Path(args.output_global).resolve(),
        Path(args.output_benign_only).resolve(),
        Path(args.report).resolve(),
        Path(args.artifacts).resolve(),
        config,
        input_path,
    )
    print(
        f"[UQ-LED] completed: global_removed={int(result.global_issue_mask.sum())}, "
        f"benign_only_removed={int(result.benign_only_issue_mask.sum())}"
    )


if __name__ == "__main__":
    main()
