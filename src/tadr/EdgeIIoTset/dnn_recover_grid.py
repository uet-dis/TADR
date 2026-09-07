"""EdgeIIoT DNN recovery pipeline.

Overall flow:
1) Load the filtered clean CSV and the remaining-noise CSV.
2) Load the fitted EdgeIIoT encoders and encode both inputs.
3) Train one fixed DNN recovery model on the clean set.
4) Predict labels and confidence scores for the noise set.
5) Accept only high-confidence recoveries using fixed thresholds.
6) Cap recovered samples per predicted class to keep the recovery conservative.
7) Export the final clean CSV, remaining-noise CSV, and recovery report files.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from tadr.configs.edgeiot import EDGEIOTResources
from tadr.preprocessing.edgeiot_preprocessor import EDGEIOTPreprocessor
from tadr.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

ENCODERS_DIR = SCRIPT_DIR / EDGEIOTResources.ENCODERS_FOLDER.lstrip(os.sep)

DEFAULT_CLEAN_INPUT = str(Path(EDGEIOTResources.CLEAN_MERGED_FOLDER) / "edgeiot_train_clean_merged.csv")
DEFAULT_NOISE_INPUT = str(Path(EDGEIOTResources.CLEAN_MERGED_FOLDER) / "edgeiot_train_clean_merged_noise.csv")
DEFAULT_BENIGN_LABEL = "Normal"
DEFAULT_ATTACK_THRESHOLD = 0.9
DEFAULT_BENIGN_THRESHOLD = 0.9
DEFAULT_RECOVER_CAP_RATIO = 0.5


def _distribution_rows(values: pd.Series) -> list[dict]:
    counts = values.astype(str).value_counts(dropna=False)
    total = int(len(values))
    rows = []
    for label_name, count in counts.items():
        rows.append(
            {
                "label": str(label_name),
                "count": int(count),
                "rate": float(count / total) if total > 0 else 0.0,
            }
        )
    return rows


def encode_dataset(df: pd.DataFrame, preprocessor: EDGEIOTPreprocessor) -> pd.DataFrame:
    """Apply the EdgeIIoT encoding pipeline to one dataframe."""
    logger.info("[+] Encoding dataset: categorical (OneHot) + numerical (MinMax) + label")

    df_enc = preprocessor.select_features_and_label(df)
    df_enc = preprocessor.preprocess_encode_numerical_features_minmax(df_enc)
    df_enc = preprocessor.preprocess_encode_binary_features(df_enc)
    df_enc = preprocessor.preprocess_encode_categorical_features(df_enc)
    df_enc = preprocessor.preprocess_encode_label(df_enc)

    logger.info(f"[+] Encoding complete: {df_enc.shape}")
    return df_enc


class RobustMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one_model(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    label_smoothing: float,
    device: str,
    seed: int,
) -> tuple[RobustMLP, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = max(1, len(dataset) - val_size)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = RobustMLP(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    class_counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_state = model.state_dict()
    best_val_acc = -1.0
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                val_correct += int((pred == yb).sum().item())
                val_total += len(yb)

        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[+] Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model, {"val_acc": float(best_val_acc)}


def predict_proba(model: RobustMLP, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    probs = []
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i : i + batch_size].to(device)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.vstack(probs)


def recovery_metrics(y_true: np.ndarray, y_pred: np.ndarray, accept_mask: np.ndarray) -> dict:
    accepted = int(accept_mask.sum())
    if accepted > 0:
        tp = int(np.sum((y_pred == y_true) & accept_mask))
        precision = tp / accepted
    else:
        tp = 0
        precision = 0.0

    total = len(y_true)
    recall = tp / total if total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accepted": accepted,
        "tp": tp,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def log_label_distribution(name: str, df: pd.DataFrame, label_col: str = "Attack_type") -> None:
    if label_col not in df.columns:
        logger.info(f"[DIST] {name}: missing column {label_col}")
        return

    counts = df[label_col].astype(str).value_counts(dropna=False)
    total = int(len(df))
    logger.info(f"[DIST] {name}: total={total}")
    for label, count in counts.items():
        ratio = (count / total) if total > 0 else 0.0
        logger.info(f"[DIST] {name}: {label} -> {int(count)} ({ratio:.4f})")


def per_class_recovery_stats(df_recovered: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    rows = []
    if len(df_recovered) == 0:
        for class_name in class_names:
            rows.append(
                {
                    "class": class_name,
                    "recovered_count": 0,
                    "correct_count": 0,
                    "correct_rate": 0.0,
                }
            )
        return pd.DataFrame(rows)

    if "dnn_pred_label" not in df_recovered.columns or "original_label" not in df_recovered.columns:
        raise SystemExit("Recovered data must contain 'dnn_pred_label' and 'original_label' for class statistics.")

    for class_name in class_names:
        class_mask = df_recovered["dnn_pred_label"].astype(str) == str(class_name)
        recovered_count = int(class_mask.sum())
        correct_count = int(
            (class_mask & (df_recovered["original_label"].astype(str) == str(class_name))).sum()
        )
        correct_rate = (correct_count / recovered_count) if recovered_count > 0 else 0.0
        rows.append(
            {
                "class": class_name,
                "recovered_count": recovered_count,
                "correct_count": correct_count,
                "correct_rate": float(correct_rate),
            }
        )

    return pd.DataFrame(rows)


def build_accept_mask(
    pred: np.ndarray,
    conf: np.ndarray,
    y_noise_current: np.ndarray,
    benign_encoded: int,
    attack_threshold: float,
    benign_threshold: float,
    class_caps: dict[int, int] | None = None,
) -> np.ndarray:
    """Accept only high-confidence Normal rows, then enforce optional per-class caps."""
    current_benign = y_noise_current == benign_encoded
    attack_accept = current_benign & (pred != benign_encoded) & (conf >= float(attack_threshold))
    benign_accept = current_benign & (pred == benign_encoded) & (conf >= float(benign_threshold))
    accept = attack_accept | benign_accept

    if class_caps is None:
        return accept

    capped_accept = np.zeros_like(accept, dtype=bool)
    for cls, cap in class_caps.items():
        if cap <= 0:
            continue
        cls_idx = np.where(accept & (pred == cls))[0]
        if len(cls_idx) == 0:
            continue
        if len(cls_idx) <= cap:
            capped_accept[cls_idx] = True
        else:
            order = np.argsort(conf[cls_idx])[::-1]
            keep = cls_idx[order[:cap]]
            capped_accept[keep] = True

    return capped_accept


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeIIoT DNN recovery pipeline")
    parser.add_argument("--clean-in", type=str, required=True, help="Input clean CSV after the defense stage")
    parser.add_argument("--noise-in", type=str, required=True, help="Input remaining-noise CSV after the defense stage")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for recovery artifacts")
    parser.add_argument("--name", type=str, default=None, help="Output file prefix when --output-dir is specified")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if not os.path.exists(args.clean_in):
        raise FileNotFoundError(f"Clean input not found: {args.clean_in}")
    if not os.path.exists(args.noise_in):
        raise FileNotFoundError(f"Noise input not found: {args.noise_in}")

    if args.output_dir is not None:
        args.output_dir = str(Path(args.output_dir).expanduser())
        os.makedirs(args.output_dir, exist_ok=True)

    if args.output_dir is not None:
        prefix = args.name if args.name else Path(args.clean_in).stem
        ext = Path(args.clean_in).suffix
        args.output_clean = os.path.join(args.output_dir, f"{prefix}_recovered_final{ext}")
        args.output_noise = os.path.join(args.output_dir, f"{prefix}_remaining_after_recover{ext}")
    else:
        clean_base, clean_ext = os.path.splitext(args.clean_in)
        args.output_clean = f"{clean_base}_recovered_final{clean_ext}"
        args.output_noise = f"{clean_base}_remaining_after_recover{clean_ext}"

    logger.info(f"\n{'=' * 80}")
    logger.info("EdgeIIoT DNN RECOVERY PIPELINE")
    logger.info(f"{'=' * 80}")
    logger.info(f"[+] Clean input: {args.clean_in}")
    logger.info(f"[+] Noise input: {args.noise_in}")

    preprocessor = EDGEIOTPreprocessor()
    logger.info("[+] Loading encoders for EdgeIIoT...")
    if not preprocessor.load_encoders(str(ENCODERS_DIR)):
        raise SystemExit("Encoders not found. Please run preprocessing first.")

    df_clean = pd.read_csv(args.clean_in, low_memory=False)
    df_noise = pd.read_csv(args.noise_in, low_memory=False)
    logger.info(f"[+] Clean shape: {df_clean.shape}")
    logger.info(f"[+] Noise shape: {df_noise.shape}")
    log_label_distribution("clean_before", df_clean)
    log_label_distribution("noise_before", df_noise)

    has_original_label = "original_label" in df_noise.columns
    has_is_noisy = "is_noisy" in df_noise.columns
    has_source = "__source__" in df_noise.columns

    original_label_col = df_noise["original_label"].copy() if has_original_label else None
    is_noisy_col = df_noise["is_noisy"].copy() if has_is_noisy else None
    source_col = df_noise["__source__"].copy() if has_source else None

    drop_meta = ["original_label", "is_noisy", "__source__", "detected_as_noise"]
    df_clean_model = df_clean.drop(columns=drop_meta, errors="ignore")
    df_noise_model = df_noise.drop(columns=drop_meta, errors="ignore")

    # STEP 1: Encode clean and noise datasets
    logger.info(f"\n{'=' * 80}")
    logger.info("[+] STEP 1: ENCODE INPUTS")
    logger.info(f"{'=' * 80}")
    df_clean_enc = encode_dataset(df_clean_model, preprocessor)
    df_noise_enc = encode_dataset(df_noise_model, preprocessor)

    X_clean = df_clean_enc.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_clean = df_clean_enc.iloc[:, -1].to_numpy(dtype=np.int64)
    y_noise_current = df_noise_enc.iloc[:, -1].to_numpy(dtype=np.int64)
    X_noise = df_noise_enc.iloc[:, :-1].to_numpy(dtype=np.float32)

    y_noise_true = None
    if has_original_label:
        y_noise_true = preprocessor.encoders["label"].transform(df_noise["original_label"].astype(str).values)

    # STEP 2: Train recovery model
    logger.info(f"\n{'=' * 80}")
    logger.info("[+] STEP 2: TRAIN RECOVERY MODEL")
    logger.info(f"{'=' * 80}")
    n_classes = len(preprocessor.encoders["label"].classes_)
    benign_encoded = int(preprocessor.encoders["label"].transform([DEFAULT_BENIGN_LABEL])[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[+] Device: {device}")

    hidden_dims = [256, 128]
    dropout = 0.3
    lr = 5e-4
    weight_decay = 1e-4
    label_smoothing = 0.0
    batch_size = 512
    max_epochs = 60
    patience = 8
    seed = 42

    model, train_info = train_one_model(
        X=X_clean,
        y=y_clean,
        num_classes=n_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        label_smoothing=label_smoothing,
        device=device,
        seed=seed,
    )
    logger.info(f"[+] Clean validation accuracy: {train_info['val_acc']:.4f}")

    # STEP 3: Predict noise set
    logger.info(f"\n{'=' * 80}")
    logger.info("[+] STEP 3: PREDICT AND FILTER NOISE")
    logger.info(f"{'=' * 80}")
    probs = predict_proba(model, X_noise, batch_size=batch_size, device=device)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    attack_threshold = DEFAULT_ATTACK_THRESHOLD
    benign_threshold = DEFAULT_BENIGN_THRESHOLD
    recover_cap_ratio = DEFAULT_RECOVER_CAP_RATIO

    clean_class_counts = np.bincount(y_clean, minlength=n_classes)
    class_caps: dict[int, int] = {
        cls: int(max(0, np.floor(clean_class_counts[cls] * float(recover_cap_ratio))))
        for cls in range(n_classes)
    }

    accept_mask = build_accept_mask(
        pred=pred,
        conf=conf,
        y_noise_current=y_noise_current,
        benign_encoded=benign_encoded,
        attack_threshold=attack_threshold,
        benign_threshold=benign_threshold,
        class_caps=class_caps,
    )
    inv_labels = preprocessor.encoders["label"].inverse_transform(pred.astype(int))

    logger.info(f"[+] attack_threshold={attack_threshold}")
    logger.info(f"[+] benign_threshold={benign_threshold}")
    logger.info(f"[+] recover_cap_ratio={recover_cap_ratio}")
    logger.info(f"[+] Accepted for recovery: {int(accept_mask.sum())} / {len(accept_mask)}")

    # STEP 4: Split recovered vs remaining
    logger.info(f"\n{'=' * 80}")
    logger.info("[+] STEP 4: SPLIT RECOVERED VS REMAINING")
    logger.info(f"{'=' * 80}")
    df_noise_corrected = df_noise.copy()
    df_noise_corrected["dnn_pred_label"] = inv_labels
    df_noise_corrected["dnn_confidence"] = conf
    df_noise_corrected["selected_for_recover"] = accept_mask

    df_recovered = df_noise_corrected[accept_mask].copy()
    if len(df_recovered) > 0:
        df_recovered["Attack_type"] = df_recovered["dnn_pred_label"]

    df_remaining_noise = df_noise_corrected[~accept_mask].copy()
    df_final_clean = pd.concat([df_clean, df_recovered], ignore_index=True)

    # STEP 5: Build reports
    logger.info(f"\n{'=' * 80}")
    logger.info("[+] STEP 5: BUILD REPORTS")
    logger.info(f"{'=' * 80}")
    class_names = [str(c) for c in preprocessor.encoders["label"].classes_]
    class_recovery_report = per_class_recovery_stats(df_recovered, class_names)

    summary_report = {
        "clean_input_csv": str(args.clean_in),
        "noise_input_csv": str(args.noise_in),
        "output_clean_csv": str(args.output_clean),
        "output_noise_csv": str(args.output_noise),
        "attack_threshold": float(attack_threshold),
        "benign_threshold": float(benign_threshold),
        "recover_cap_ratio": float(recover_cap_ratio),
        "total_noise_rows": int(len(df_noise_corrected)),
        "total_recovered_count": int(accept_mask.sum()),
        "clean_input_label_distribution": _distribution_rows(df_clean["Attack_type"].astype(str))
        if "Attack_type" in df_clean.columns
        else [],
        "noise_input_label_distribution": _distribution_rows(df_noise["Attack_type"].astype(str))
        if "Attack_type" in df_noise.columns
        else [],
    }

    if y_noise_true is not None:
        metrics = recovery_metrics(y_noise_true, pred, accept_mask)
        summary_report["recovery_metrics_vs_original_label"] = metrics
        logger.info(
            f"[+] Recovery metrics -> F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, accepted={metrics['accepted']}"
        )

        detailed_rows = []
        predicted_label_series = pd.Series(df_noise_corrected["dnn_pred_label"].astype(str).values)
        original_label_series = df_noise["original_label"].astype(str)
        selected_mask = accept_mask.astype(bool)
        is_correct_pred = predicted_label_series.values == original_label_series.values

        label_names = sorted(set(original_label_series.unique()).union(set(predicted_label_series.unique())))
        for class_name in label_names:
            source_mask = (original_label_series == class_name).to_numpy()
            predicted_mask = (predicted_label_series == class_name).to_numpy()
            class_total = int(source_mask.sum())
            class_recovered = int(np.sum(selected_mask & source_mask))
            class_correct_recovered = int(np.sum(selected_mask & source_mask & is_correct_pred))
            detailed_rows.append(
                {
                    "class": class_name,
                    "noise_total": class_total,
                    "recovered_count": class_recovered,
                    "recovered_rate": float(class_recovered / class_total) if class_total > 0 else 0.0,
                    "correct_recovered_count": class_correct_recovered,
                    "correct_recovered_rate": float(class_correct_recovered / class_total)
                    if class_total > 0
                    else 0.0,
                    "precision_on_recovered": float(class_correct_recovered / class_recovered)
                    if class_recovered > 0
                    else 0.0,
                    "predicted_as_this_class": int(predicted_mask.sum()),
                }
            )

        detailed_recovery_df = pd.DataFrame(detailed_rows)
    else:
        detailed_recovery_df = pd.DataFrame(
            [
                {
                    "class": class_name,
                    "noise_total": 0,
                    "recovered_count": 0,
                    "recovered_rate": 0.0,
                    "correct_recovered_count": 0,
                    "correct_recovered_rate": 0.0,
                    "precision_on_recovered": 0.0,
                    "predicted_as_this_class": 0,
                }
                for class_name in class_names
            ]
        )

    # STEP 6: Export
    logger.info(f"\n{'=' * 80}")
    logger.info("[+] STEP 6: EXPORT ARTIFACTS")
    logger.info(f"{'=' * 80}")
    if args.output_dir is not None:
        report_dir = Path(args.output_dir)
    else:
        report_dir = Path(SCRIPT_DIR) / "reports" / Path(args.clean_in).stem
    report_dir.mkdir(parents=True, exist_ok=True)

    per_class_out = report_dir / "recovery_per_class.csv"
    detailed_recovery_df.to_csv(per_class_out, index=False)

    summary_report["per_class_report_csv"] = str(per_class_out)
    summary_out = report_dir / "recovery_summary.json"
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(summary_report, handle, indent=2)

    if has_source and len(df_recovered) > 0:
        df_recovered["__source__"] = source_col.loc[df_recovered.index].values
    if has_original_label and len(df_recovered) > 0:
        df_recovered["original_label"] = original_label_col.loc[df_recovered.index].values
    if has_is_noisy and len(df_recovered) > 0:
        df_recovered["is_noisy"] = is_noisy_col.loc[df_recovered.index].values

    os.makedirs(os.path.dirname(args.output_clean) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_noise) or ".", exist_ok=True)
    df_final_clean.to_csv(args.output_clean, index=False)
    df_remaining_noise.to_csv(args.output_noise, index=False)

    logger.info(f"[+] Final clean: {args.output_clean} (rows={len(df_final_clean)})")
    logger.info(f"[+] Remaining noise: {args.output_noise} (rows={len(df_remaining_noise)})")
    logger.info(f"[+] Recovery summary JSON -> {summary_out}")
    logger.info(f"[+] Recovery per-class CSV -> {per_class_out}")
    logger.info(f"[+] Recovered rows added: {int(accept_mask.sum())}")

    logger.info(f"\n{'=' * 80}")
    logger.info("EdgeIIoT DNN RECOVERY COMPLETED")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
