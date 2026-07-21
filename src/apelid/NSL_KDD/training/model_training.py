"""Train all NSL-KDD models for the ablation pipeline.

Overall flow:
1) Resolve the training CSV passed by the ablation runner.
2) Load the fixed NSL-KDD test CSV.
3) Load the fitted encoders and prepare train/validation/test arrays.
4) Train every supported model with the JSON parameters for that model.
5) Save each model, its metrics JSON, and its confusion matrix PNG.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder

from apelid.utils.logging import get_logger, setup_logging

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from apelid.configs.nslkdd import NSLKDDResources
from apelid.preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from apelid.preprocessing.prepare import PrepareData
from apelid.training.bagging import BaggingModel
from apelid.training.catb import CatBoostModel
from apelid.training.dnn import DNNModel
from apelid.training.gbm import GBMModel
from apelid.training.histgbm import HistGBMModel
from apelid.training.lgbm import LGBMModel
from apelid.training.model import confusion_matrix_from_indices, evaluate_classification, save_confusion_matrix_png
from apelid.training.rf import RandomForestModel
from apelid.training.xgb import XGBModel

logger = get_logger(__name__)

RESOURCE_ROOT = SCRIPT_DIR.parent / "resources" / "NSLKDD" / "clean_merged"
PARAMS_DIR = SCRIPT_DIR.parent / "params" / "training"
DEFAULT_TRAIN_FILE = RESOURCE_ROOT / "nslkdd_train_clean_merged.csv"
DEFAULT_TEST_FILE = RESOURCE_ROOT / "nslkdd_test_clean_merged.csv"
MODEL_ORDER = ["xgb", "catb", "bagging", "histgbm", "gbm", "lgbm", "rf", "dnn"]


def _resolve_path(path_text: str | None, default: Path) -> Path:
    """Resolve a possibly relative path against common workspace anchors."""
    if not path_text:
        return default

    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path

    for candidate in (Path.cwd() / path, SCRIPT_DIR / path, SRC_ROOT / path):
        if candidate.exists():
            return candidate

    return path


def _load_params(model_name: str) -> dict:
    """Load the JSON hyperparameter file for one model."""
    path = PARAMS_DIR / f"{model_name}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_safe_ordinal_encoder(pre, df_train: pd.DataFrame) -> None:
    """Replace legacy ordinal encoders with an unknown-safe version when needed."""
    if "ordinal" not in pre.encoders:
        return

    ordinal = pre.encoders["ordinal"]
    if (
        getattr(ordinal, "handle_unknown", None) == "use_encoded_value"
        and getattr(ordinal, "unknown_value", None) == -1
    ):
        return

    safe_ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    safe_ordinal.fit(df_train[pre.encoded_categorical_features])
    pre.encoders["ordinal"] = safe_ordinal
    logger.warning("[!] Legacy ordinal encoder detected. Re-fitted with unknown handling (unknown_value=-1).")


def _evaluate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, class_names: list[str]) -> dict:
    """Compute binary metrics for the Benign-vs-Attack case."""
    metrics = {
        "accuracy_binary": float(accuracy_score(y_true, y_pred)),
        "f1_binary": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_binary": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_binary": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is None:
        return metrics

    if y_proba.ndim == 1:
        attack_scores = y_proba
    else:
        attack_idx = class_names.index("Attack") if "Attack" in class_names else 1
        if y_proba.shape[1] <= attack_idx:
            return metrics
        attack_scores = y_proba[:, attack_idx]

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, attack_scores))
    except Exception:
        pass

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, attack_scores))
    except Exception:
        pass

    return metrics


def _log_input_distribution(df: pd.DataFrame, dataset_name: str) -> None:
    """Log the label distribution of a dataset for quick inspection."""
    if "Label" not in df.columns:
        logger.warning(f"[!] {dataset_name} does not contain 'Label' column. Skip distribution logging.")
        return

    total = len(df)
    logger.info(f"[+] {dataset_name} rows: {total}")
    if total == 0:
        logger.warning(f"[!] {dataset_name} is empty. Skip distribution logging.")
        return

    dist = df["Label"].value_counts(dropna=False)
    logger.info(f"[+] {dataset_name} label distribution:")
    for label, count in dist.items():
        logger.info(f"    - {label}: {count} ({(count / total) * 100:.2f}%)")


def _build_output_paths(output_dir: Path, model_name: str) -> tuple[Path, Path, Path]:
    """Build the three artifact paths for one model run."""
    model_ext = ".pth" if model_name == "dnn" else ".pkl"
    model_path = output_dir / f"{model_name}{model_ext}"
    report_path = output_dir / f"{model_name}_metrics.json"
    cm_path = output_dir / f"{model_name}_cm.png"
    return model_path, report_path, cm_path


def _build_model(model_name: str, params: dict, num_classes: int, args, pre, X_tr: np.ndarray, meta_tr: dict):
    """Instantiate one model implementation from its JSON configuration."""
    if model_name == "xgb":
        model_params = params.get("params", {}).copy()
        if args.device in ("GPU", "auto"):
            model_params["device"] = "cuda"
            model_params["tree_method"] = "hist"
        return XGBModel(
            num_class=num_classes,
            params=model_params,
            num_round=int(params.get("num_round", 1200)),
            early_stopping=int(params.get("early_stopping", 20)),
            random_state=42,
        )

    if model_name == "catb":
        model_params = params.get("params", {}).copy()
        if args.device in ("GPU", "auto"):
            model_params["task_type"] = "GPU"
        return CatBoostModel(num_class=num_classes, params=model_params, random_state=42)

    if model_name == "bagging":
        return BaggingModel(params=params.get("params", {}), random_state=42)

    if model_name == "histgbm":
        return HistGBMModel(num_class=num_classes, params=params.get("params", {}), random_state=42)

    if model_name == "gbm":
        return GBMModel(num_class=num_classes, params=params.get("params", {}), random_state=42)

    if model_name == "lgbm":
        return LGBMModel(
            num_class=num_classes,
            params=params.get("params", {}),
            num_round=int(params.get("num_round", 200)),
            early_stopping=int(params.get("early_stopping", 20)),
            random_state=42,
        )

    if model_name == "rf":
        return RandomForestModel(num_class=num_classes, params=params.get("params", {}), random_state=42)

    if model_name == "dnn":
        # Collect the feature metadata needed by the DNN embedding and normalization stack.
        cat_cardinalities = pre.extract_categorical_cardinalities()
        cat_feature_indices = meta_tr["cat_feature_indices"]
        cont_feature_indices = meta_tr["cont_feature_indices"]
        binary_feature_indices = meta_tr.get("binary_feature_indices", [])
        cont_means = meta_tr.get("cont_means")
        cont_stds = meta_tr.get("cont_stds")
        inputnorm_eps = float(meta_tr.get("inputnorm_eps", 1e-6))

        return DNNModel(
            input_dim=X_tr.shape[1],
            num_class=num_classes,
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
            batch_size=int(params.get("batch_size", 512)),
            max_epochs=int(params.get("epochs", 50)),
            patience=int(params.get("patience", 8)),
            device=None if args.device == "auto" else args.device,
            random_state=int(params.get("random_state", 42)),
            cat_cardinalities=cat_cardinalities,
            cat_feature_indices=cat_feature_indices,
            binary_feature_indices=binary_feature_indices,
            cont_feature_indices=cont_feature_indices,
            embedding_dim=int(params.get("embedding_dim", 16)),
            hidden_dims=[256, 128],
            cont_means=cont_means,
            cont_stds=cont_stds,
            inputnorm_eps=inputnorm_eps,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def train_model(model_name: str, args, pre, X_tr, X_val, y_tr, y_val, X_te, y_te, meta_tr, class_names: list[str], output_dir: Path) -> dict:
    """Train one model, evaluate it, and persist its artifacts."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"[+] Training model: {model_name.upper()}")
    logger.info(f"{'=' * 80}")

    # Load the model-specific parameters and construct the estimator.
    params = _load_params(model_name)
    logger.info(f"[+] Loading params: {params}")

    # Fit on the prepared training split and validate on the held-out split.
    model = _build_model(model_name, params, len(class_names), args, pre, X_tr, meta_tr)
    model.fit(X_tr, y_tr, X_val, y_val)

    # Generate predictions on the fixed test set.
    y_pred_idx = model.predict(X_te)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_te)
    except Exception as exc:
        logger.warning(f"[!] predict_proba not available for {model_name}: {exc}")

    # Compute the multiclass metrics used by the ablation summary.
    metrics = evaluate_classification(y_te, y_pred_idx, class_names)
    logger.info(
        f"[+] Metrics (Macro): accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}, "
        f"precision={metrics['precision_macro']:.4f}, recall={metrics['recall_macro']:.4f}"
    )
    logger.info(
        f"[+] Metrics (Weighted): f1={metrics['f1_weighted']:.4f}, precision={metrics['precision_weighted']:.4f}, "
        f"recall={metrics['recall_weighted']:.4f}"
    )

    # Add binary metrics when the selected label encoder is binary.
    if len(class_names) == 2:
        metrics.update(_evaluate_binary_metrics(y_te, y_pred_idx, y_proba, class_names))
        logger.info(
            f"[+] Metrics (Binary): accuracy={metrics['accuracy_binary']:.4f}, f1={metrics['f1_binary']:.4f}, "
            f"precision={metrics['precision_binary']:.4f}, recall={metrics['recall_binary']:.4f}, "
            f"roc_auc={metrics.get('roc_auc', float('nan')):.4f}, pr_auc={metrics.get('pr_auc', float('nan')):.4f}"
        )

    # Save the model artifact, metrics JSON, and confusion matrix PNG.
    model_path, report_path, cm_path = _build_output_paths(output_dir, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    cm_mat = confusion_matrix_from_indices(y_te, y_pred_idx, num_classes=len(class_names))
    save_confusion_matrix_png(cm_mat, class_names, str(cm_path))

    logger.info(f"[+] Model saved -> {model_path}")
    logger.info(f"[+] Metrics JSON -> {report_path}")
    logger.info(f"[+] Confusion matrix PNG -> {cm_path}")

    return metrics


def main() -> None:
    """Parse CLI arguments, prepare data, train all models, and print a summary."""
    parser = argparse.ArgumentParser(description="NSL-KDD all-model training entrypoint")
    parser.add_argument("--model", "-m", default="all", choices=["all"], help="Only 'all' is supported")
    parser.add_argument("--resource", "-r", default="nslkdd", choices=["nslkdd"])
    parser.add_argument("--train-in", type=str, default=None, help="Training CSV to use for the ablation run")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where models, metrics, and CMs are written")
    parser.add_argument("--device", type=str, default="auto", choices=["CPU", "auto"])
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Resolve the training input from the ablation runner or fall back to the default clean file.
    train_in = _resolve_path(args.train_in, DEFAULT_TRAIN_FILE)
    # The test set is fixed for this ablation workflow.
    test_in = _resolve_path(None, DEFAULT_TEST_FILE)

    if not train_in.exists():
        raise SystemExit(f"Train file not found: {train_in}")
    if not test_in.exists():
        raise SystemExit(f"Test file not found: {test_in}")

    logger.info(f"[+] Loading inputs:\n  train={train_in}\n  test={test_in}")
    logger.info("[+] Training mode: ABLATION_ALL_MODELS")

    # Load raw CSVs before preprocessing.
    df_train = pd.read_csv(train_in, low_memory=False)
    df_test = pd.read_csv(test_in, low_memory=False)

    # Load the NSL-KDD preprocessor and its fitted encoders.
    pre = NSLKDDPreprocessor()
    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Fit encoders first.")

    # Make sure ordinal encoding can tolerate unseen test-time categories.
    _ensure_safe_ordinal_encoder(pre, df_train)

    if "label" not in pre.encoders:
        raise SystemExit("Label encoder not loaded.")
    class_names = [str(value) for value in pre.encoders["label"].classes_]

    # Prepare train/validation arrays and the fixed test arrays.
    X_tr, X_val, y_tr, y_val, meta_tr = PrepareData.prepare_training_data(
        df_train,
        pre,
        use_validation=True,
        val_size=float(args.val_frac),
        random_state=42,
    )
    X_te, y_te, _ = PrepareData.prepare_input_data(df_test, pre, include_label=True)

    # Train the full model suite expected by the ablation pipeline.
    logger.info(f"[+] Training all models: {MODEL_ORDER}")
    all_metrics: dict[str, dict] = {}
    output_dir = Path(args.output_dir)

    for model_name in MODEL_ORDER:
        try:
            # Keep going even if one model fails so the batch run still yields partial results.
            all_metrics[model_name] = train_model(
                model_name,
                args,
                pre,
                X_tr,
                X_val,
                y_tr,
                y_val,
                X_te,
                y_te,
                meta_tr,
                class_names,
                output_dir,
            )
        except Exception as exc:
            logger.error(f"[!] Failed to train {model_name}: {exc}")
            import traceback

            traceback.print_exc()

    # Print a compact end-of-run summary for the ablation log parser.
    logger.info(f"\n{'=' * 80}")
    logger.info("[+] TRAINING SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info("[+] Input files:")
    logger.info(f"    - Train: {train_in}")
    logger.info(f"    - Test : {test_in}")
    _log_input_distribution(df_train, "Train")
    _log_input_distribution(df_test, "Test")

    for model_name in MODEL_ORDER:
        metrics = all_metrics.get(model_name)
        if metrics is None:
            continue
        logger.info(f"  {model_name.upper():12s} - Accuracy: {metrics['accuracy']:.4f}, F1 (Macro): {metrics['f1_macro']:.4f}")

    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()