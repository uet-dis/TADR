"""NSL-KDD Benign defense: DAE embedding -> KMeans core -> KNN filtering.

Overall flow:
1) Read the poisoned input CSV produced by the ablation runner.
2) Load the fitted NSL-KDD encoders and encode the dataset.
3) Train a DAE and extract latent embeddings.
4) Keep only representative Benign core samples with KMeans.
5) Run a weighted KNN detector on the Benign core to flag suspicious rows.
6) Reconstruct the data back to the original feature space.
7) Save clean/noise CSVs and a JSON report under the requested output folder.

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
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from apelid.configs.nslkdd import NSLKDDResources
from apelid.preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from apelid.resampling.undersampling.kmeans_reps import KMeansRepresentativeSelector
from apelid.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_INPUT = Path(NSLKDDResources.CLEAN_MERGED_DATA_FOLDER) / "nslkdd_train_clean_merged.csv"
DEFAULT_BENIGN_LABEL = "Benign"


def _distribution_rows(values: pd.Series) -> list[dict]:
    """Convert a series into label/count/rate rows for the JSON report."""
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


def _evaluate_detection_metrics(true_noise: np.ndarray, detected_noise: np.ndarray) -> dict:
    """Compute standard binary detection metrics for the final noise mask."""
    tp = int(np.sum((true_noise == True) & (detected_noise == True)))
    tn = int(np.sum((true_noise == False) & (detected_noise == False)))
    fp = int(np.sum((true_noise == False) & (detected_noise == True)))
    fn = int(np.sum((true_noise == True) & (detected_noise == False)))

    total = len(true_noise)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def encode_dataset(df: pd.DataFrame, preprocessor: NSLKDDPreprocessor) -> pd.DataFrame:
    """Select NSL-KDD features and apply the fixed preprocessing pipeline."""
    logger.info("[+] Encoding dataset with minmax...")

    df_enc = preprocessor.select_features_and_label(df)
    df_enc = preprocessor.preprocess_encode_numerical_features_minmax(df_enc)
    df_enc = preprocessor.preprocess_encode_binary_features(df_enc)
    df_enc = preprocessor.preprocess_encode_label(df_enc)
    df_enc = preprocessor.preprocess_encode_categorical_features(df_enc)

    logger.info(f"[+] Encoding complete: {df_enc.shape}")
    return df_enc


def inverse_transform_data(df_encoded: pd.DataFrame, preprocessor: NSLKDDPreprocessor) -> pd.DataFrame:
    """Invert the encoded dataframe back to the original NSL-KDD layout."""
    logger.info("[+] Inverse transforming entire dataset to original format...")
    df_inverted = preprocessor.inverse_transform(df_encoded, numerical_inverse="minmax")
    logger.info(f"[+] Inverse transform complete: {df_inverted.shape}")
    return df_inverted


class OptimizedDAE(nn.Module):
    """Denoising autoencoder used to learn a compact representation of encoded data."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: list[int] | None = None, dropout_rate: float = 0.2):
        """Build a symmetric encoder/decoder around a latent bottleneck."""
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        encoder_layers = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_features, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        encoder_layers.append(nn.Linear(in_features, latent_dim))
        encoder_layers.append(nn.BatchNorm1d(latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_features = latent_dim
        for hidden_dim in hidden_dims[::-1]:
            decoder_layers.append(nn.Linear(in_features, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(in_features, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Encode an input batch and reconstruct it."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_dae_optimized(
    X_train: np.ndarray,
    input_dim: int,
    latent_dim: int = 32,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    noise_factor: float = 0.2,
    patience: int = 5,
    device: str = "cpu",
):
    """Train the DAE with noisy-input reconstruction and early stopping."""
    logger.info(f"[DAE] Init model: In({input_dim}) -> Hidden(256-128) -> Latent({latent_dim})")

    X_tensor = torch.FloatTensor(X_train)
    dataset = TensorDataset(X_tensor)

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = max(1, len(dataset) - val_size)
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = OptimizedDAE(input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_model_state = model.state_dict()
    early_stop_counter = 0

    progress_bar = tqdm(range(epochs), desc="[DAE] Training", unit="epoch")
    for epoch in progress_bar:
        # Train on noisy inputs so the autoencoder learns a stable latent space.
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            noise = torch.randn_like(inputs) * noise_factor
            noisy_inputs = torch.clamp(inputs + noise, 0.0, 1.0)

            optimizer.zero_grad()
            _, outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Evaluate the reconstruction loss on the validation split.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                _, outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)
        progress_bar.set_postfix({"T_Loss": f"{avg_train_loss:.5f}", "V_Loss": f"{avg_val_loss:.5f}"})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"[DAE] Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state)
    return model


def extract_features(model: OptimizedDAE, X_input: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Extract latent embeddings from the trained DAE encoder."""
    model.eval()
    X_tensor = torch.FloatTensor(X_input).to(device)
    batch_size = 1024
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            encoded, _ = model(batch)
            embeddings.append(encoded.cpu().numpy())

    return np.vstack(embeddings)


def select_benign_core_indices_from_embedding(
    df_embedded: pd.DataFrame,
    benign_encoded_label: int,
    budget: int,
    n_clusters: int | None,
    batch_size: int,
    kmeans_algo: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select representative Benign core samples from the latent embedding."""
    y = df_embedded["Label"].values
    benign_mask = y == benign_encoded_label
    benign_positions = np.where(benign_mask)[0]

    if len(benign_positions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    X_emb = df_embedded.iloc[:, :-1].values
    X_benign = X_emb[benign_positions]

    n_benign = len(benign_positions)
    target_budget = int(min(max(1, budget), n_benign))
    clusters = int(n_clusters) if n_clusters is not None else target_budget
    clusters = max(1, min(clusters, n_benign))

    logger.info(f"[+] Benign samples: {n_benign}")
    logger.info(f"[+] KMeans core budget: {target_budget}")
    logger.info(f"[+] KMeans clusters: {clusters}")

    ksel = KMeansRepresentativeSelector(
        n_clusters=clusters,
        batch_size=int(batch_size),
        random_state=42,
        algorithm=kmeans_algo,
    )
    labels_arr = ksel.fit_predict(X_benign)
    centers = ksel.centers_
    if centers is None:
        raise SystemExit("KMeans centers not available after fit")

    core_local_idx = KMeansRepresentativeSelector.select_representatives(
        X=X_benign,
        labels=labels_arr,
        centers=centers,
    )

    # Keep strict core behavior: no random top-up.
    if len(core_local_idx) < target_budget:
        logger.warning(
            f"[!] core reps {len(core_local_idx)} < target budget {target_budget} (empty clusters may exist)."
        )

    benign_df_indices = df_embedded.index.to_numpy()[benign_positions]
    core_indices = benign_df_indices[core_local_idx]

    core_set = set(core_indices.tolist())
    removed_indices = np.array([idx for idx in benign_df_indices if idx not in core_set], dtype=np.int64)
    return core_indices.astype(np.int64), removed_indices


class CorruptionDetector:
    """KNN-based detector that flags suspicious Benign-core samples as noise."""

    def __init__(self, data: pd.DataFrame):
        """Store the embedded dataset used for KNN corruption detection."""
        self.original_data = data
        self.noise_index: list[int] = []

    def local_detection(
        self,
        k_neighbors: int,
        benign_encoded_label: int,
        candidate_indices: set[int] | None = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """Detect noise among candidate Benign rows using weighted KNN voting."""
        X = self.original_data.iloc[:, :-1].values
        y = self.original_data.iloc[:, -1].values
        all_idx = self.original_data.index.to_numpy()

        benign_pos_mask = y == benign_encoded_label
        benign_positions = np.where(benign_pos_mask)[0]

        if candidate_indices is not None:
            candidate_positions = [p for p in benign_positions if int(all_idx[p]) in candidate_indices]
            candidate_positions = np.array(candidate_positions, dtype=np.int64)
        else:
            candidate_positions = benign_positions

        n_eval = len(candidate_positions)
        n_total = len(y)

        if verbose:
            logger.info(f"[+] Benign candidates for KNN check: {n_eval} / {n_total}")
            logger.info(f"[+] Fitting KNN on full embedding (k={k_neighbors})...")

        if n_eval == 0:
            self.noise_index = []
            return np.array([], dtype=np.int64)

        knn_model = KNeighborsClassifier(n_neighbors=k_neighbors + 1)
        knn_model.fit(X, y)

        X_candidates = X[candidate_positions]
        distances_b, neighbor_indices_b = knn_model.kneighbors(X_candidates)

        filtered_indices_b = neighbor_indices_b[:, 1 : k_neighbors + 1]
        filtered_distances_b = distances_b[:, 1 : k_neighbors + 1]

        noise_idx = []
        iterator = tqdm(range(n_eval), desc="Detecting", unit="sample") if verbose else range(n_eval)

        for i in iterator:
            neighbor_labels = y[filtered_indices_b[i]]
            neighbor_distances = filtered_distances_b[i]

            weights = 1.0 / (neighbor_distances ** 2 + 1e-10)
            weight_sum = weights.sum()
            weighted_votes: dict[int, float] = {}
            for j in range(len(neighbor_labels)):
                label_id = int(neighbor_labels[j])
                weighted_votes[label_id] = weighted_votes.get(label_id, 0.0) + float(weights[j])

            if not weighted_votes:
                continue

            predicted_label = max(
                weighted_votes.items(),
                key=lambda kv: kv[1] / max(weight_sum, 1e-12),
            )[0]
            if predicted_label != benign_encoded_label:
                data_index = int(all_idx[candidate_positions[i]])
                noise_idx.append(data_index)

        self.noise_index = noise_idx
        return np.array(noise_idx, dtype=np.int64)


def main() -> None:
    """Run the NSL-KDD defense pipeline and export clean/noise outputs."""
    parser = argparse.ArgumentParser(description="NSL-KDD Benign defense: DAE embedding + KMeans core + KNN detection")
    parser.add_argument("--input", "-i", type=str, default=str(DEFAULT_INPUT), help="Input poisoned CSV path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for all DAE/KMeans/KNN artifacts")
    parser.add_argument("--name", type=str, default=None, help="Output file prefix when --output-dir is specified")
    parser.add_argument("--kmeans-budget", type=int, required=True, help="Target benign core size")
    parser.add_argument("--k-neighbors", "-k", type=int, default=15)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Normalize input paths and prepare the output folder.
    args.input = str(Path(args.input).expanduser())
    if args.output_dir is not None:
        args.output_dir = str(Path(args.output_dir).expanduser())
        os.makedirs(args.output_dir, exist_ok=True)

    if args.kmeans_budget <= 0:
        raise SystemExit("--kmeans-budget must be > 0")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Resolve output paths based on the pipeline's run tag.
    if args.output_dir is not None:
        input_stem = args.name if args.name else Path(args.input).stem
        args.output_clean = os.path.join(args.output_dir, f"{input_stem}_clean.csv")
        args.output_noise = os.path.join(args.output_dir, f"{input_stem}_noise.csv")
    else:
        base, ext = os.path.splitext(args.input)
        args.output_clean = f"{base}_dae_kmeans_knn_clean{ext}"
        args.output_noise = f"{base}_dae_kmeans_knn_noise{ext}"

    logger.info(f"\n{'=' * 70}")
    logger.info("NSL-KDD DEFENSE PIPELINE (DAE + KMEANS + KNN)")
    logger.info(f"{'=' * 70}")
    logger.info(f"[+] Input: {args.input}")

    # Load the NSL-KDD preprocessor and fitted encoders.
    preprocessor = NSLKDDPreprocessor()
    logger.info("[+] Loading encoders for nslkdd...")
    if not preprocessor.load_encoders():
        raise SystemExit("Encoders not found. Please fit/load encoders first.")

    # Read the poisoned dataset and capture optional tracking columns.
    df_in = pd.read_csv(args.input, low_memory=False)
    logger.info(f"[+] Dataset shape: {df_in.shape}")

    has_original_label = "original_label" in df_in.columns
    has_is_noisy = "is_noisy" in df_in.columns
    has_source = "__source__" in df_in.columns

    original_col = df_in["original_label"].copy() if has_original_label else None
    is_noisy_col = df_in["is_noisy"].copy() if has_is_noisy else None
    source_col = df_in["__source__"].copy() if has_source else None

    # This ablation path does not use any supervision columns during filtering.
    cols_to_drop = ["original_label", "is_noisy", "__source__", "detected_as_noise"]
    df_to_encode = df_in.drop(columns=cols_to_drop, errors="ignore")

    # STEP 1: Encode the data for the DAE and KNN stages.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 1: ENCODE DATA")
    logger.info(f"{'=' * 60}")
    df_encoded = encode_dataset(df_to_encode, preprocessor)

    # Separate the encoded feature matrix and the encoded label vector.
    X_train = df_encoded.iloc[:, :-1].values.astype(np.float32)
    y_train = df_encoded.iloc[:, -1].values.astype(int)
    input_dim = X_train.shape[1]

    # Resolve the compute device automatically for this simplified run.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[+] Device: {device}")

    # STEP 2: Train the DAE and extract latent embeddings.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 2: DAE EMBEDDING")
    logger.info(f"{'=' * 60}")

    logger.info("[+] Training new DAE model...")
    dae = train_dae_optimized(
        X_train,
        input_dim=input_dim,
        latent_dim=32,
        epochs=200,
        batch_size=256,
        lr=1e-3,
        noise_factor=0.2,
        patience=5,
        device=device,
    )

    X_emb = extract_features(dae, X_train, device=device)
    df_embedded = pd.DataFrame(X_emb, index=df_encoded.index)
    df_embedded["Label"] = y_train
    logger.info(f"[+] Embedding shape: {df_embedded.shape}")

    # Get the encoded label for Benign, which is the only class we core-filter.
    if "label" not in preprocessor.encoders:
        raise SystemExit("Label encoder not loaded.")
    benign_encoded = int(preprocessor.encoders["label"].transform([DEFAULT_BENIGN_LABEL])[0])

    # STEP 3: KMeans core selection on Benign only.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 3: KMEANS CORE (BENIGN ONLY)")
    logger.info(f"{'=' * 60}")

    benign_core_idx, benign_removed_idx = select_benign_core_indices_from_embedding(
        df_embedded=df_embedded,
        benign_encoded_label=benign_encoded,
        budget=int(args.kmeans_budget),
        n_clusters=int(args.kmeans_budget),
        batch_size=100000,
        kmeans_algo="minibatch",
    )
    core_set = set(benign_core_idx.tolist())
    kmeans_noise_set = set(benign_removed_idx.tolist())

    logger.info(f"[+] Benign core kept: {len(core_set)}")
    logger.info(f"[+] Benign removed by KMeans: {len(kmeans_noise_set)}")

    # STEP 4: KNN filtering on the Benign-core subset.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 4: KNN FILTER ON BENIGN-CORE")
    logger.info(f"{'=' * 60}")

    detector = CorruptionDetector(df_embedded)
    knn_noise_idx = detector.local_detection(
        k_neighbors=int(args.k_neighbors),
        benign_encoded_label=benign_encoded,
        candidate_indices=core_set,
        verbose=True,
    )

    final_noise_set = set(kmeans_noise_set)
    final_noise_set.update(knn_noise_idx.tolist())

    total_rows = len(df_embedded)
    kmeans_mask = np.zeros(total_rows, dtype=bool)
    if len(kmeans_noise_set) > 0:
        kmeans_mask[list(kmeans_noise_set)] = True
    knn_mask = np.zeros(total_rows, dtype=bool)
    if len(knn_noise_idx) > 0:
        knn_mask[np.asarray(knn_noise_idx, dtype=int)] = True
    final_mask = kmeans_mask | knn_mask

    logger.info(f"[+] Final removed by KMeans: {len(kmeans_noise_set)}")
    logger.info(f"[+] Final removed by KNN: {len(knn_noise_idx)}")
    logger.info(f"[+] Final total noise detected: {len(final_noise_set)}")

    # STEP 5: Inverse transform and export.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 5: INVERSE + EXPORT")
    logger.info(f"{'=' * 60}")

    df_inverted = inverse_transform_data(df_encoded, preprocessor)

    # Restore metadata columns if they existed in the input.
    if has_source:
        df_inverted["__source__"] = source_col
    if has_original_label:
        df_inverted["original_label"] = original_col
    if has_is_noisy:
        df_inverted["is_noisy"] = is_noisy_col

    # Mark detected rows for downstream inspection.
    df_inverted["detected_as_noise"] = False
    if len(final_noise_set) > 0:
        df_inverted.loc[list(final_noise_set), "detected_as_noise"] = True

    clean_indices = [i for i in df_inverted.index if i not in final_noise_set]
    noise_indices = sorted(list(final_noise_set))

    df_clean = df_inverted.loc[clean_indices].copy()
    df_noise = df_inverted.loc[noise_indices].copy()

    os.makedirs(os.path.dirname(args.output_clean) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_noise) or ".", exist_ok=True)

    df_clean.to_csv(args.output_clean, index=False)
    df_noise.to_csv(args.output_noise, index=False)

    logger.info(f"[+] Saved CLEAN: {args.output_clean} ({len(df_clean)} rows)")
    logger.info(f"[+] Saved NOISE: {args.output_noise} ({len(df_noise)} rows)")

    # Write the JSON report into the requested output directory or a script-local report folder.
    if args.output_dir is not None:
        report_dir = Path(args.output_dir)
    else:
        report_dir = Path(SCRIPT_DIR) / "reports" / Path(args.input).stem
    report_dir.mkdir(parents=True, exist_ok=True)

    original_label_series = original_col.astype(str) if has_original_label else None
    current_label_series = df_inverted["Label"].astype(str)
    original_distribution = _distribution_rows(original_label_series) if original_label_series is not None else []
    current_distribution = _distribution_rows(current_label_series)

    stage_summary = {
        "kmeans": {
            "removed_count": int(kmeans_mask.sum()),
            "removed_rate": float(kmeans_mask.mean()),
        },
        "knn": {
            "removed_count": int(knn_mask.sum()),
            "removed_rate": float(knn_mask.mean()),
        },
        "kmeans_only": {
            "removed_count": int(np.sum(kmeans_mask & ~knn_mask)),
            "removed_rate": float(np.mean(kmeans_mask & ~knn_mask)),
        },
        "knn_only": {
            "removed_count": int(np.sum(knn_mask & ~kmeans_mask)),
            "removed_rate": float(np.mean(knn_mask & ~kmeans_mask)),
        },
        "overlap_kmeans_knn": {
            "removed_count": int(np.sum(kmeans_mask & knn_mask)),
            "removed_rate": float(np.mean(kmeans_mask & knn_mask)),
        },
        "final_union": {
            "removed_count": int(final_mask.sum()),
            "removed_rate": float(final_mask.mean()),
        },
    }

    detection_report = {
        "input_csv": str(args.input),
        "output_clean_csv": str(args.output_clean),
        "output_noise_csv": str(args.output_noise),
        "total_samples": int(total_rows),
        "k_neighbors": int(args.k_neighbors),
        "kmeans_budget": int(args.kmeans_budget),
        "original_label_distribution": original_distribution,
        "current_input_label_distribution": current_distribution,
        "stage_summary": stage_summary,
    }

    if original_label_series is not None:
        true_noise_mask = df_in["Label"].astype(str).values != original_label_series.values
        detected_noise = df_inverted["detected_as_noise"].values.astype(bool)
        metrics = _evaluate_detection_metrics(true_noise_mask, detected_noise)
        logger.info(
            f"[+] Final Detection -> F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, Acc={metrics['accuracy']:.4f}"
        )
        detection_report["metrics_vs_true_noise"] = {
            "kmeans": _evaluate_detection_metrics(true_noise_mask, kmeans_mask),
            "knn": _evaluate_detection_metrics(true_noise_mask, knn_mask),
            "final_union": _evaluate_detection_metrics(true_noise_mask, final_mask),
        }

        per_label_rows = []
        label_names = sorted(set(original_label_series.unique()).union(set(current_label_series.unique())))
        for label_name in label_names:
            original_mask = (original_label_series == label_name).to_numpy()
            current_mask = (current_label_series == label_name).to_numpy()
            original_count = int(original_mask.sum())
            current_count = int(current_mask.sum())
            true_noise_count = int(np.sum(true_noise_mask & original_mask))
            removed_count = int(np.sum(final_mask & original_mask))
            removed_noise_count = int(np.sum(true_noise_mask & original_mask & final_mask))
            per_label_rows.append(
                {
                    "label": str(label_name),
                    "original_count": original_count,
                    "current_input_count": current_count,
                    "true_noise_count": true_noise_count,
                    "true_noise_rate_in_original": float(true_noise_count / original_count) if original_count > 0 else 0.0,
                    "removed_count": removed_count,
                    "removed_rate_in_original": float(removed_count / original_count) if original_count > 0 else 0.0,
                    "removed_noise_count": removed_noise_count,
                    "noise_precision_among_removed": float(removed_noise_count / removed_count) if removed_count > 0 else 0.0,
                    "noise_detection_rate": float(removed_noise_count / true_noise_count) if true_noise_count > 0 else 0.0,
                }
            )

        per_label_df = pd.DataFrame(per_label_rows)
        per_label_out = report_dir / "noise_detection_per_label.csv"
        per_label_df.to_csv(per_label_out, index=False)
        detection_report["per_label_report_csv"] = str(per_label_out)

    report_out = report_dir / "noise_detection_report.json"
    with open(report_out, "w", encoding="utf-8") as handle:
        json.dump(detection_report, handle, indent=2)
    logger.info(f"[+] Noise detection report -> {report_out}")

    logger.info(f"\n{'=' * 70}")
    logger.info("NSL-KDD DEFENSE PIPELINE COMPLETED")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
