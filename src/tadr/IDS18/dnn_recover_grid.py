"""Recover high-confidence samples from a detected-noise set using a robust DNN.

Overall flow:
1) Read the clean and noise CSV files produced by the ablation pipeline.
2) Load the fitted CIC-IDS-2018 encoders and encode both datasets.
3) Train a fixed DNN on the clean set only.
4) Predict labels and confidence scores for the noise set.
5) Recover only current-Benign rows that pass the fixed confidence thresholds.
6) Limit recovered rows per predicted class using a simple cap ratio.
7) Export the final clean CSV, remaining noise CSV, and a JSON summary.

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

from tadr.configs.cic2018 import CIC2018Resources
from tadr.preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from tadr.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_CLEAN_INPUT = Path(CIC2018Resources.CLEAN_MERGED_DATA_FOLDER) / "cic2018_train_original_undersampled.csv"
DEFAULT_NOISE_INPUT = Path(CIC2018Resources.CLEAN_MERGED_DATA_FOLDER) / "cic2018_train_original_undersampled.csv"
DEFAULT_BENIGN_LABEL = "Benign"
DEFAULT_ATTACK_THRESHOLD = 0.9
DEFAULT_BENIGN_THRESHOLD = 0.9
DEFAULT_RECOVER_CAP_RATIO = 0.5


def _distribution_rows(values: pd.Series) -> list[dict]:
    """Convert a value series into label/count/rate rows for reporting."""
    counts = values.astype(str).value_counts(dropna=False)
    total = int(len(values))
    rows = []
    for label_name, count in counts.items():
        rows.append({
            "label": str(label_name),
            "count": int(count),
            "rate": float(count / total) if total > 0 else 0.0,
        })
    return rows


def encode_dataset(df: pd.DataFrame, preprocessor: CIC2018Preprocessor) -> pd.DataFrame:
    """Select CIC-IDS-2018 features and apply the fixed preprocessing pipeline."""
    logger.info("[+] Encoding dataset with minmax...")

    df_enc = preprocessor.select_features_and_label(df)
    df_enc = preprocessor.preprocess_encode_numerical_features_minmax(df_enc)
    df_enc = preprocessor.preprocess_encode_binary_features(df_enc)
    df_enc = preprocessor.preprocess_encode_label(df_enc)
    df_enc = preprocessor.preprocess_encode_categorical_features(df_enc)

    logger.info(f"[+] Encoding complete: {df_enc.shape}")
    return df_enc


class RobustMLP(nn.Module):
    """Small feed-forward classifier used for clean-set training and recovery."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list[int], dropout: float):
        """Create a multilayer perceptron with batch norm and dropout blocks."""
        super().__init__()
        layers = []
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
        """Run a forward pass through the MLP."""
        return self.net(x)


def train_fixed_model(
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
    """Train one fixed MLP configuration and return the best checkpoint."""
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
        # Train for one epoch on the clean set.
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Evaluate on the held-out validation split.
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
                break

    model.load_state_dict(best_state)
    return model, {"val_acc": float(best_val_acc)}


def predict_proba(model: RobustMLP, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """Predict class probabilities for a matrix of samples."""
    model.eval()
    probs = []
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i + batch_size].to(device)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def recovery_metrics(y_true: np.ndarray, y_pred: np.ndarray, accept_mask: np.ndarray) -> dict:
    """Compute precision, recall, and F1 for accepted recovery candidates."""
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


def log_label_distribution(name: str, df: pd.DataFrame, label_col: str = "Label") -> None:
    """Log the class distribution of a dataframe column for quick inspection."""
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
    """Summarize recovered rows and correctness per predicted class."""
    rows = []
    if len(df_recovered) == 0:
        for class_name in class_names:
            rows.append({
                "class": class_name,
                "recovered_count": 0,
                "correct_count": 0,
                "correct_rate": 0.0,
            })
        return pd.DataFrame(rows)

    if "dnn_pred_label" not in df_recovered.columns or "original_label" not in df_recovered.columns:
        return pd.DataFrame(rows)

    for class_name in class_names:
        class_mask = df_recovered["dnn_pred_label"].astype(str) == str(class_name)
        recovered_count = int(class_mask.sum())
        correct_count = int((class_mask & (df_recovered["original_label"].astype(str) == str(class_name))).sum())
        correct_rate = (correct_count / recovered_count) if recovered_count > 0 else 0.0
        rows.append({
            "class": class_name,
            "recovered_count": recovered_count,
            "correct_count": correct_count,
            "correct_rate": float(correct_rate),
        })

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
    """Select recoverable current-Benign rows using confidence thresholds and class caps."""
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
    """Run the full recovery workflow: load data, train DNN, recover, and export results."""
    parser = argparse.ArgumentParser(description="CIC-IDS-2018 DNN recovery for the ablation pipeline")
    parser.add_argument("--clean-in", type=str, default=str(DEFAULT_CLEAN_INPUT), help="Input clean CSV (after filtering)")
    parser.add_argument("--noise-in", type=str, default=str(DEFAULT_NOISE_INPUT), help="Input noise CSV (after filtering)")
    parser.add_argument("--output-clean", "-oc", type=str, default=None, help="Output final clean CSV")
    parser.add_argument("--output-noise", "-on", type=str, default=None, help="Output remaining noise CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for all recovery artifacts")
    parser.add_argument("--name", type=str, default=None, help="Output file prefix when --output-dir is specified")
    parser.add_argument("--resource", "-r", type=str, default="cic2018", choices=["cic2018"])
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Normalize input and output paths for the ablation runner.
    args.clean_in = str(Path(args.clean_in).expanduser())
    args.noise_in = str(Path(args.noise_in).expanduser())
    if args.output_clean is not None:
        args.output_clean = str(Path(args.output_clean).expanduser())
    if args.output_noise is not None:
        args.output_noise = str(Path(args.output_noise).expanduser())
    if args.output_dir is not None:
        args.output_dir = str(Path(args.output_dir).expanduser())
        os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.clean_in):
        raise FileNotFoundError(f"Clean input not found: {args.clean_in}")
    if not os.path.exists(args.noise_in):
        raise FileNotFoundError(f"Noise input not found: {args.noise_in}")

    # Resolve output paths.
    if args.output_dir is not None:
        clean_stem = args.name if args.name else Path(args.clean_in).stem
        noise_stem = args.name if args.name else Path(args.noise_in).stem
        args.output_clean = os.path.join(args.output_dir, f"{clean_stem}_recovered_final.csv")
        args.output_noise = os.path.join(args.output_dir, f"{noise_stem}_remaining_after_recover.csv")
    else:
        if args.output_clean is None:
            base, ext = os.path.splitext(args.clean_in)
            args.output_clean = f"{base}_recovered_final{ext}"
        if args.output_noise is None:
            base, ext = os.path.splitext(args.noise_in)
            args.output_noise = f"{base}_remaining_after_recover{ext}"

    # Load the CIC-IDS-2018 preprocessor and fitted encoders.
    preprocessor = CIC2018Preprocessor()
    logger.info("[+] Loading encoders for cic2018...")
    if not preprocessor.load_encoders():
        raise SystemExit("Encoders not found. Please fit/load encoders first.")

    # Read raw clean/noise data.
    df_clean = pd.read_csv(args.clean_in, low_memory=False)
    df_noise = pd.read_csv(args.noise_in, low_memory=False)

    logger.info(f"[+] Clean shape: {df_clean.shape}")
    logger.info(f"[+] Noise shape: {df_noise.shape}")
    log_label_distribution("clean_before", df_clean)
    log_label_distribution("noise_before", df_noise)

    # Evaluate only when the original labels are present.
    has_original_label = "original_label" in df_noise.columns
    if has_original_label:
        y_noise_true_str = df_noise["original_label"].astype(str).values
        y_noise_true = preprocessor.encoders["label"].transform(y_noise_true_str)
    else:
        y_noise_true = None

    # Remove metadata columns before feature selection and encoding.
    drop_meta = ["original_label", "is_noisy", "__source__", "detected_as_noise"]
    df_clean_model = df_clean.drop(columns=drop_meta, errors="ignore")
    df_noise_model = df_noise.drop(columns=drop_meta, errors="ignore")

    # Encode clean and noise datasets with the same fitted preprocessors.
    df_clean_enc = encode_dataset(df_clean_model, preprocessor)
    df_noise_enc = encode_dataset(df_noise_model, preprocessor)

    # Split encoded frames into features and current labels.
    X_clean = df_clean_enc.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_clean = df_clean_enc.iloc[:, -1].to_numpy(dtype=np.int64)
    X_noise = df_noise_enc.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_noise_current = df_noise_enc.iloc[:, -1].to_numpy(dtype=np.int64)

    # Resolve label-space metadata used throughout recovery.
    n_classes = len(preprocessor.encoders["label"].classes_)
    benign_encoded = int(preprocessor.encoders["label"].transform([DEFAULT_BENIGN_LABEL])[0])

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[+] Device: {device}")

    # Use a simple cap on how many rows can be recovered per predicted class.
    clean_class_counts = np.bincount(y_clean, minlength=n_classes)
    class_caps: dict[int, int] = {}
    for cls in range(n_classes):
        class_caps[cls] = int(max(0, np.floor(clean_class_counts[cls] * DEFAULT_RECOVER_CAP_RATIO)))

    attack_threshold = DEFAULT_ATTACK_THRESHOLD
    benign_threshold = DEFAULT_BENIGN_THRESHOLD
    logger.info(f"[+] attack_threshold={attack_threshold}")
    logger.info(f"[+] benign_threshold={benign_threshold}")
    logger.info(f"[+] recover_cap_ratio={DEFAULT_RECOVER_CAP_RATIO}")

    # Fixed DNN configuration for the ablation pipeline.
    hidden_dims = [256, 128]
    dropout = 0.3
    lr = 5e-4
    weight_decay = 1e-4
    label_smoothing = 0.0

    # Train the DNN once on the clean set; the pipeline no longer grid-searches.
    logger.info("[+] Training fixed DNN recovery model...")
    model, train_info = train_fixed_model(
        X=X_clean,
        y=y_clean,
        num_classes=n_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=512,
        max_epochs=60,
        patience=8,
        label_smoothing=label_smoothing,
        device=device,
        seed=42,
    )
    logger.info(f"[+] Validation accuracy: {train_info['val_acc']:.4f}")

    # Predict the noise set and build the final acceptance mask.
    probs = predict_proba(model, X_noise, batch_size=512, device=device)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    accept_mask = build_accept_mask(
        pred=pred,
        conf=conf,
        y_noise_current=y_noise_current,
        benign_encoded=benign_encoded,
        attack_threshold=attack_threshold,
        benign_threshold=benign_threshold,
        class_caps=class_caps,
    )

    # Decode model predictions back into human-readable labels.
    inv_labels = preprocessor.encoders["label"].inverse_transform(pred.astype(int))

    df_noise_corrected = df_noise.copy()
    df_noise_corrected["dnn_pred_label"] = inv_labels
    df_noise_corrected["dnn_confidence"] = conf
    df_noise_corrected["selected_for_recover"] = accept_mask

    # Move accepted rows into the recovered-clean set and relabel them.
    df_recovered = df_noise_corrected[accept_mask].copy()
    if len(df_recovered) > 0:
        df_recovered["Label"] = df_recovered["dnn_pred_label"]

    # Keep the rest as remaining noise.
    df_remaining_noise = df_noise_corrected[~accept_mask].copy()
    df_final_clean = pd.concat([df_clean, df_recovered], ignore_index=True)

    # Summarize how many recovered rows belong to each predicted class.
    class_names = [str(c) for c in preprocessor.encoders["label"].classes_]
    class_recovery_report = per_class_recovery_stats(df_recovered, class_names)

    # Optional evaluation when original labels are available.
    total_noise_rows = int(len(df_noise_corrected))
    total_recovered = int(accept_mask.sum())
    if y_noise_true is not None:
        predicted_label_series = df_noise_corrected["dnn_pred_label"].astype(str)
        original_label_series = df_noise["original_label"].astype(str)
        is_correct_pred = (predicted_label_series.values == original_label_series.values)
        total_correct_recovered = int(np.sum(accept_mask & is_correct_pred))
    else:
        total_correct_recovered = 0

    # Prepare the report directory and write the per-class summary.
    if args.output_dir is not None:
        report_dir = Path(args.output_dir)
    else:
        report_dir = Path(SCRIPT_DIR) / "reports" / Path(args.clean_in).stem
    report_dir.mkdir(parents=True, exist_ok=True)
    per_class_out = report_dir / "recovery_per_class.csv"
    class_recovery_report.to_csv(per_class_out, index=False)

    # Capture label distributions from the raw inputs for the JSON summary.
    clean_distribution = _distribution_rows(df_clean["Label"].astype(str)) if "Label" in df_clean.columns else []
    noise_distribution = _distribution_rows(df_noise["Label"].astype(str)) if "Label" in df_noise.columns else []

    # Assemble the final summary payload.
    summary_report = {
        "clean_input_csv": str(args.clean_in),
        "noise_input_csv": str(args.noise_in),
        "output_clean_csv": str(args.output_clean),
        "output_noise_csv": str(args.output_noise),
        "total_noise_rows": total_noise_rows,
        "total_recovered_count": total_recovered,
        "total_recovered_rate": float(total_recovered / total_noise_rows) if total_noise_rows > 0 else 0.0,
        "total_correct_recovered_count": total_correct_recovered,
        "total_correct_recovered_rate": float(total_correct_recovered / total_noise_rows) if total_noise_rows > 0 else 0.0,
        "precision_on_recovered_all_classes": float(total_correct_recovered / total_recovered) if total_recovered > 0 else 0.0,
        "clean_input_label_distribution": clean_distribution,
        "noise_input_label_distribution": noise_distribution,
        "per_class_report_csv": str(per_class_out),
        "fixed_model": {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "attack_threshold": attack_threshold,
            "benign_threshold": benign_threshold,
            "recover_cap_ratio": DEFAULT_RECOVER_CAP_RATIO,
        },
    }

    summary_out = report_dir / "recovery_summary.json"
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(summary_report, handle, indent=2)

    # Make sure output directories exist before saving CSVs.
    os.makedirs(os.path.dirname(args.output_clean) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_noise) or ".", exist_ok=True)

    df_final_clean.to_csv(args.output_clean, index=False)
    df_remaining_noise.to_csv(args.output_noise, index=False)

    # Log post-recovery distributions and class-level summaries.
    logger.info("\n" + "=" * 80)
    logger.info("DISTRIBUTION AFTER RECOVERY")
    logger.info("=" * 80)
    log_label_distribution("final_clean_after", df_final_clean)
    log_label_distribution("remaining_noise_after", df_remaining_noise)

    logger.info("\n" + "=" * 80)
    logger.info("PER-CLASS RECOVERY STATISTICS")
    logger.info("=" * 80)
    if len(class_recovery_report) > 0:
        for _, row in class_recovery_report.iterrows():
            logger.info(
                f"[CLASS] {row['class']}: recovered={int(row['recovered_count'])}, "
                f"correct={int(row['correct_count'])}, correct_rate={float(row['correct_rate']):.4f}"
            )

    logger.info("\n" + "=" * 80)
    logger.info("RECOVERY RATE SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"[ALL] recovered={total_recovered}/{total_noise_rows} "
        f"({summary_report['total_recovered_rate']:.4f}), "
        f"correct_recovered={total_correct_recovered}/{total_noise_rows} "
        f"({summary_report['total_correct_recovered_rate']:.4f}), "
        f"precision_on_recovered={summary_report['precision_on_recovered_all_classes']:.4f}"
    )
    logger.info(f"[+] Recovery summary JSON -> {summary_out}")
    logger.info(f"[+] Recovery per-class CSV -> {per_class_out}")

    logger.info("\n" + "=" * 80)
    logger.info("RECOVERY COMPLETED")
    logger.info("=" * 80)
    logger.info(f"[+] Final clean: {args.output_clean} (rows={len(df_final_clean)})")
    logger.info(f"[+] Remaining noise: {args.output_noise} (rows={len(df_remaining_noise)})")
    logger.info(f"[+] Recovered rows added: {int(accept_mask.sum())}")


if __name__ == "__main__":
    main()