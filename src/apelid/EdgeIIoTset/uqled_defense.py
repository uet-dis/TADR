"""
UQ-LED CL-MCD-E label-error defense for EdgeIIoTset.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def predictive_entropy(pred_probs: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return entropy of each categorical predictive distribution."""
    probs = np.asarray(pred_probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("pred_probs must have shape (n_samples, n_classes)")
    safe = np.clip(probs, epsilon, 1.0)
    return -(probs * np.log(safe)).sum(axis=1)


def classwise_thresholds(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    entropy: np.ndarray,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the CL-MCD-E confidence and entropy means per observed class."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(pred_probs, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)
    if probs.ndim != 2 or len(labels) != len(probs) or len(entropy) != len(labels):
        raise ValueError("labels, pred_probs, and entropy must describe the same samples")
    classes = probs.shape[1] if num_classes is None else int(num_classes)
    if probs.shape[1] != classes:
        raise ValueError("num_classes must match pred_probs columns")

    confidence = np.empty(classes, dtype=np.float64)
    entropy_thresholds = np.empty(classes, dtype=np.float64)
    for class_id in range(classes):
        mask = labels == class_id
        if not np.any(mask):
            raise ValueError(f"Observed class {class_id} has no samples")
        confidence[class_id] = probs[mask, class_id].mean()
        entropy_thresholds[class_id] = entropy[mask].mean()
    return confidence, entropy_thresholds


def compute_entropy_filtered_confident_joint(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    entropy: np.ndarray,
    confidence_thresholds: np.ndarray,
    entropy_thresholds: np.ndarray,
    *,
    calibrate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the confident joint after enforcing the CL-MCD-E entropy gate."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(pred_probs, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)
    confidence_thresholds = np.asarray(confidence_thresholds, dtype=np.float64)
    entropy_thresholds = np.asarray(entropy_thresholds, dtype=np.float64)
    num_classes = probs.shape[1]
    if confidence_thresholds.shape != (num_classes,) or entropy_thresholds.shape != (num_classes,):
        raise ValueError("Threshold arrays must contain one value per class")

    confident_bins = probs >= (confidence_thresholds - 1e-6)
    num_confident = confident_bins.sum(axis=1)
    any_confident = num_confident > 0
    multiple_confident = num_confident > 1
    single_choice = confident_bins.argmax(axis=1)
    probability_choice = probs.argmax(axis=1)
    assignments = np.where(multiple_confident, probability_choice, single_choice).astype(np.int64)
    entropy_passes = entropy <= entropy_thresholds[labels]
    valid = any_confident & entropy_passes
    assignments[~valid] = -1

    joint = np.zeros((num_classes, num_classes), dtype=np.int64)
    if np.any(valid):
        np.add.at(joint, (labels[valid], assignments[valid]), 1)
    np.fill_diagonal(joint, np.clip(np.diag(joint), 1, None))

    if calibrate:
        from cleanlab.count import calibrate_confident_joint

        joint = calibrate_confident_joint(joint, labels)
    return joint, assignments


class UQLEDMLP(nn.Module):
    """Five-block tabular MLP adapting the paper's five-dropout design."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Iterable[int] = (256, 256, 128, 128, 64),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(value) for value in hidden_dims]]
        if len(dims) != 6:
            raise ValueError("UQ-LED tabular adaptation requires exactly five hidden blocks")
        blocks: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Linear(dims[-1], int(num_classes))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable stochastic dropout while keeping BatchNorm statistics frozen."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
        elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def prune_by_noise_rate(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    confident_joint: np.ndarray,
) -> np.ndarray:
    """Run PBNR using the entropy-filtered confident joint."""
    from cleanlab.filter import find_label_issues

    labels = np.asarray(labels, dtype=np.int64)
    ranked = find_label_issues(
        labels=labels,
        pred_probs=np.asarray(pred_probs, dtype=np.float64),
        confident_joint=np.asarray(confident_joint, dtype=np.int64),
        filter_by="prune_by_noise_rate",
        return_indices_ranked_by="self_confidence",
        n_jobs=1,
    )
    if isinstance(ranked, np.ndarray) and ranked.dtype == bool:
        return ranked.copy()
    mask = np.zeros(len(labels), dtype=bool)
    mask[np.asarray(ranked, dtype=np.int64)] = True
    return mask


def build_scope_masks(
    global_issue_mask: np.ndarray,
    labels: np.ndarray,
    benign_class_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive Global and Benign-Only removal masks from one detector run."""
    global_mask = np.asarray(global_issue_mask, dtype=bool).copy()
    labels = np.asarray(labels, dtype=np.int64)
    if len(global_mask) != len(labels):
        raise ValueError("Issue mask and labels must have equal length")
    benign_mask = global_mask & (labels == int(benign_class_id))
    return global_mask, benign_mask


@dataclass(frozen=True)
class UQLEDConfig:
    """Paper parameters."""

    hidden_dims: tuple[int, ...] = (256, 256, 128, 128, 64)
    dropout: float = 0.5
    n_splits: int = 4
    mc_passes: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    validation_fraction: float = 0.1
    gradient_clip_norm: float = 5.0
    seed: int = 42
    device: str = "auto"


@dataclass
class OOFMCDResult:
    mcd_passes: np.ndarray
    mean_probs: np.ndarray
    fold_ids: np.ndarray
    fold_epochs: list[int]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray | None,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    tensors: tuple[torch.Tensor, ...]
    features = torch.from_numpy(np.asarray(X, dtype=np.float32))
    if y is None:
        tensors = (features,)
    else:
        tensors = (features, torch.from_numpy(np.asarray(y, dtype=np.int64)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=max(1, min(int(batch_size), len(features))),
        shuffle=shuffle,
        drop_last=drop_last and len(features) >= int(batch_size),
        generator=generator,
        num_workers=0,
    )


def _validation_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            loss = criterion(model(features), labels)
            total_loss += float(loss.item()) * len(features)
            total_samples += len(features)
    return total_loss / max(total_samples, 1)


def _train_fold_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    num_classes: int,
    config: UQLEDConfig,
    fold_seed: int,
    device: torch.device,
) -> tuple[UQLEDMLP, int]:
    set_global_seed(fold_seed)
    model = UQLEDMLP(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = _make_loader(
        X_train,
        y_train,
        config.batch_size,
        shuffle=True,
        drop_last=True,
        seed=fold_seed,
    )
    valid_loader = _make_loader(
        X_valid,
        y_valid,
        config.batch_size,
        shuffle=False,
        drop_last=False,
        seed=fold_seed,
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    completed_epochs = 0
    for epoch in range(int(config.max_epochs)):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
        completed_epochs = epoch + 1
        current_loss = _validation_loss(model, valid_loader, device)
        if current_loss < best_loss - 1e-8:
            best_loss = current_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config.patience):
                break
    if best_state is None:
        raise RuntimeError("Fold training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, completed_epochs


def _mc_predict(
    model: nn.Module,
    X: np.ndarray,
    config: UQLEDConfig,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    loader = _make_loader(
        X,
        None,
        config.batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    all_passes: list[np.ndarray] = []
    with torch.no_grad():
        for pass_id in range(int(config.mc_passes)):
            set_global_seed(seed + pass_id)
            enable_mc_dropout(model)
            batches: list[np.ndarray] = []
            for (features,) in loader:
                probs = torch.softmax(model(features.to(device)), dim=1)
                batches.append(probs.cpu().numpy())
            all_passes.append(np.concatenate(batches, axis=0))
    return np.stack(all_passes, axis=1).astype(np.float32)


def generate_oof_mcd_predictions(
    X: np.ndarray,
    y: np.ndarray,
    config: UQLEDConfig,
) -> OOFMCDResult:
    """
    Train stratified fold models and return five-pass OOF MCD predictions.
    
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X and y must have shapes (n, d) and (n,)")
    unique = np.unique(y)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Labels must be contiguous integers starting at zero")
    counts = np.bincount(y)
    if counts.min() < int(config.n_splits):
        raise ValueError("Every observed class needs at least n_splits samples")

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    num_classes = len(unique)
    passes = np.full((len(X), config.mc_passes, num_classes), np.nan, dtype=np.float32)
    fold_ids = np.full(len(X), -1, dtype=np.int64)
    fold_epochs: list[int] = []
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    for fold_id, (outer_train, holdout) in enumerate(splitter.split(X, y)):
        fold_seed = config.seed + 1000 * (fold_id + 1)
        train_idx, valid_idx = train_test_split(
            outer_train,
            test_size=config.validation_fraction,
            random_state=fold_seed,
            stratify=y[outer_train],
        )
        model, epochs = _train_fold_model(
            X[train_idx],
            y[train_idx],
            X[valid_idx],
            y[valid_idx],
            num_classes,
            config,
            fold_seed,
            device,
        )
        passes[holdout] = _mc_predict(model, X[holdout], config, device, fold_seed + 500)
        fold_ids[holdout] = fold_id
        fold_epochs.append(epochs)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if np.isnan(passes).any() or np.any(fold_ids < 0):
        raise RuntimeError("OOF assembly left samples without predictions")
    mean_probs = passes.mean(axis=1, dtype=np.float64).astype(np.float32)
    return OOFMCDResult(
        mcd_passes=passes,
        mean_probs=mean_probs,
        fold_ids=fold_ids,
        fold_epochs=fold_epochs,
    )


@dataclass
class UQLEDResult:
    """All auditable outputs from one CL-MCD-E detector run."""

    oof: OOFMCDResult
    entropy: np.ndarray
    confidence_thresholds: np.ndarray
    entropy_thresholds: np.ndarray
    confident_joint: np.ndarray
    confident_assignments: np.ndarray
    global_issue_mask: np.ndarray
    benign_only_issue_mask: np.ndarray
    self_confidence: np.ndarray
    suggested_labels: np.ndarray
    issue_score: np.ndarray


def run_uqled_detector(
    X: np.ndarray,
    observed_labels: np.ndarray,
    benign_class_id: int,
    config: UQLEDConfig,
) -> UQLEDResult:
    """Run the selected UQ-LED CL-MCD-E variant without ground-truth access."""
    labels = np.asarray(observed_labels, dtype=np.int64)
    oof = generate_oof_mcd_predictions(X, labels, config)
    entropy = predictive_entropy(oof.mean_probs)
    confidence, entropy_thresholds = classwise_thresholds(labels, oof.mean_probs, entropy)
    confident_joint, assignments = compute_entropy_filtered_confident_joint(
        labels,
        oof.mean_probs,
        entropy,
        confidence,
        entropy_thresholds,
        calibrate=True,
    )
    issue_mask = prune_by_noise_rate(labels, oof.mean_probs, confident_joint)
    global_mask, benign_mask = build_scope_masks(issue_mask, labels, benign_class_id)
    rows = np.arange(len(labels))
    self_confidence = oof.mean_probs[rows, labels]
    suggested = oof.mean_probs.argmax(axis=1)
    issue_score = oof.mean_probs[rows, suggested] - self_confidence
    return UQLEDResult(
        oof=oof,
        entropy=entropy,
        confidence_thresholds=confidence,
        entropy_thresholds=entropy_thresholds,
        confident_joint=confident_joint,
        confident_assignments=assignments,
        global_issue_mask=global_mask,
        benign_only_issue_mask=benign_mask,
        self_confidence=self_confidence,
        suggested_labels=suggested,
        issue_score=issue_score,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


def _encode_edgeiot(df: pd.DataFrame, num_encoder: str) -> tuple[np.ndarray, np.ndarray, Any]:
    from apelid.configs.edgeiot import EDGEIOTResources
    from apelid.preprocessing.edgeiot_preprocessor import EDGEIOTPreprocessor

    preprocessor = EDGEIOTPreprocessor()
    encoders_dir = SCRIPT_DIR / EDGEIOTResources.ENCODERS_FOLDER.lstrip(os.sep)
    if not preprocessor.load_encoders(str(encoders_dir)):
        raise RuntimeError(f"EdgeIIoT encoders were not found in {encoders_dir}")
    model_df = preprocessor.select_features_and_label(df)
    if num_encoder == "minmax":
        encoded = preprocessor.preprocess_encode_numerical_features_minmax(model_df)
    elif num_encoder == "quantile_uniform":
        encoded = preprocessor.preprocess_encode_numerical_features_quantile_uniform(model_df)
    else:
        raise ValueError(f"Unsupported numerical encoder: {num_encoder}")
    encoded = preprocessor.preprocess_encode_binary_features(encoded)
    encoded = preprocessor.preprocess_encode_label(encoded)
    encoded = preprocessor.preprocess_encode_categorical_features(encoded)
    X = encoded.drop(columns=[preprocessor.label_column]).to_numpy(dtype=np.float32)
    y = encoded[preprocessor.label_column].to_numpy(dtype=np.int64)
    return X, y, preprocessor


def _detection_metrics(
    issue_mask: np.ndarray,
    is_noisy: np.ndarray,
    issue_scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    issue_mask = np.asarray(issue_mask, dtype=bool)
    truth = np.asarray(is_noisy, dtype=bool)
    tp = int(np.sum(issue_mask & truth))
    fp = int(np.sum(issue_mask & ~truth))
    fn = int(np.sum(~issue_mask & truth))
    tn = int(np.sum(~issue_mask & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics: dict[str, float | int] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if issue_scores is not None:
        from sklearn.metrics import average_precision_score

        metrics["auprc"] = float(average_precision_score(truth, issue_scores))
    return metrics


def _write_outputs(
    df: pd.DataFrame,
    labels: np.ndarray,
    preprocessor: Any,
    result: UQLEDResult,
    output_global: Path,
    output_benign_only: Path,
    report_path: Path,
    artifacts_path: Path,
    config: UQLEDConfig,
    input_path: Path,
) -> None:
    for path in (output_global, output_benign_only, report_path, artifacts_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[~result.global_issue_mask].to_csv(output_global, index=False)
    df.loc[~result.benign_only_issue_mask].to_csv(output_benign_only, index=False)

    class_names = [str(value) for value in preprocessor.encoders["label"].classes_]
    issue_report_csv = report_path.with_name(f"{report_path.stem}_issues.csv")
    pd.DataFrame(
        {
            "sample_id": np.arange(len(df), dtype=np.int64),
            "fold_id": result.oof.fold_ids,
            "observed_label": df[preprocessor.label_column].astype(str).to_numpy(),
            "suggested_label": [class_names[index] for index in result.suggested_labels],
            "self_confidence": result.self_confidence,
            "suggested_confidence": result.oof.mean_probs.max(axis=1),
            "predictive_entropy": result.entropy,
            "confidence_threshold_observed": result.confidence_thresholds[labels],
            "entropy_threshold_observed": result.entropy_thresholds[labels],
            "prediction_margin": result.issue_score,
            "confident_assignment": result.confident_assignments,
            "global_issue": result.global_issue_mask,
            "benign_only_issue": result.benign_only_issue_mask,
        }
    ).to_csv(issue_report_csv, index=False)

    np.savez_compressed(
        artifacts_path,
        mcd_passes=result.oof.mcd_passes,
        mean_probs=result.oof.mean_probs,
        fold_ids=result.oof.fold_ids,
        entropy=result.entropy,
        confidence_thresholds=result.confidence_thresholds,
        entropy_thresholds=result.entropy_thresholds,
        confident_joint=result.confident_joint,
        confident_assignments=result.confident_assignments,
        global_issue_mask=result.global_issue_mask,
        benign_only_issue_mask=result.benign_only_issue_mask,
    )
    report: dict[str, Any] = {
        "method": "UQ-LED",
        "variant": "CL-MCD-E",
        "preprocessing_protocol": "fixed_label_agnostic_project_encoders_before_model_oof",
        "input_csv": str(input_path),
        "output_global_csv": str(output_global),
        "output_benign_only_csv": str(output_benign_only),
        "issue_report_csv": str(issue_report_csv),
        "artifacts_npz": str(artifacts_path),
        "total_samples": int(len(df)),
        "global_samples_removed": int(result.global_issue_mask.sum()),
        "benign_only_samples_removed": int(result.benign_only_issue_mask.sum()),
        "global_samples_kept": int((~result.global_issue_mask).sum()),
        "benign_only_samples_kept": int((~result.benign_only_issue_mask).sum()),
        "fold_epochs": result.oof.fold_epochs,
        "class_names": class_names,
        "confidence_thresholds": result.confidence_thresholds.tolist(),
        "entropy_thresholds": result.entropy_thresholds.tolist(),
        "confident_joint": result.confident_joint.tolist(),
        "config": asdict(config),
    }
    if "is_noisy" in df.columns:
        truth = df["is_noisy"].astype(bool).to_numpy()
        report["global_detection_metrics"] = _detection_metrics(
            result.global_issue_mask,
            truth,
            result.issue_score,
        )
        report["benign_only_detection_metrics"] = _detection_metrics(
            result.benign_only_issue_mask,
            truth,
        )
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(dims) != 5 or any(item <= 0 for item in dims):
        raise argparse.ArgumentTypeError("--hidden-dims requires five positive comma-separated integers")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(description="UQ-LED CL-MCD-E defense for EdgeIIoTset")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output-global", required=True)
    parser.add_argument("--output-benign-only", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--benign-label", default="Normal")
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
    X, labels, preprocessor = _encode_edgeiot(df, args.num_encoder)
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
