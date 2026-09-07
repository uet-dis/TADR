"""CIC-IDS-2018 Benign defense: DAE embedding -> KMeans core -> KNN filtering.

Overall flow:
1) Read the poisoned input CSV produced by the ablation runner.
2) Load the fitted CIC-IDS-2018 encoders and encode the dataset.
3) Train or load a DAE and extract latent embeddings.
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

from tadr.configs.cic2018 import CIC2018Resources
from tadr.preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from tadr.resampling.undersampling.kmeans_reps import KMeansRepresentativeSelector
from tadr.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_INPUT = Path(CIC2018Resources.CLEAN_MERGED_DATA_FOLDER) / "cic2018_train_original_undersampled.csv"
DEFAULT_BENIGN_LABEL = "Benign"


def _distribution_rows(values: pd.Series) -> list[dict]:
    """Convert a series into label/count/rate rows for the JSON report."""
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
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

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
            batch = X_tensor[i:i + batch_size]
            encoded, _ = model(batch)
            embeddings.append(encoded.cpu().numpy())

    return np.vstack(embeddings)


def encode_dataset(df: pd.DataFrame, preprocessor: CIC2018Preprocessor) -> pd.DataFrame:
    """Select CIC-IDS-2018 features and apply the configured preprocessing pipeline."""
    logger.info("[+] Encoding dataset with minmax...")

    df_enc = preprocessor.select_features_and_label(df)
    df_enc = preprocessor.preprocess_encode_numerical_features_minmax(df_enc)
    df_enc = preprocessor.preprocess_encode_binary_features(df_enc)
    df_enc = preprocessor.preprocess_encode_label(df_enc)
    df_enc = preprocessor.preprocess_encode_categorical_features(df_enc)

    logger.info(f"[+] Encoding complete: {df_enc.shape}")
    return df_enc


def inverse_transform_data(df_encoded: pd.DataFrame, preprocessor: CIC2018Preprocessor) -> pd.DataFrame:
    """Invert the encoded dataframe back to the original CIC-IDS-2018 layout."""
    logger.info("[+] Inverse transforming entire dataset to original format...")
    df_inverted = preprocessor.inverse_transform(df_encoded, numerical_inverse="minmax")
    logger.info(f"[+] Inverse transform complete: {df_inverted.shape}")
    return df_inverted


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
    benign_mask = (y == benign_encoded_label)
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

        benign_pos_mask = (y == benign_encoded_label)
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

        filtered_indices_b = neighbor_indices_b[:, 1:k_neighbors + 1]
        filtered_distances_b = distances_b[:, 1:k_neighbors + 1]

        noise_idx = []
        iterator = tqdm(range(n_eval), desc="Detecting", unit="sample") if verbose else range(n_eval)

        for i in iterator:
            neighbor_labels = y[filtered_indices_b[i]]
            neighbor_distances = filtered_distances_b[i]

            weights = 1.0 / (neighbor_distances ** 2 + 1e-10)
            weight_sum = weights.sum()
            # Encoded labels can be sparse or non-contiguous, so use dict accumulation.
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
    """Run the CIC-IDS-2018 defense pipeline and export clean/noise outputs."""
    parser = argparse.ArgumentParser(
        description="CIC-IDS-2018 Benign defense: DAE embedding + KMeans core + KNN detection"
    )
    parser.add_argument("--input", "-i", type=str, default=str(DEFAULT_INPUT), help="Input poisoned CSV path")
    parser.add_argument("--output-clean", "-oc", type=str, default=None, help="Output path for clean samples")
    parser.add_argument("--output-noise", "-on", type=str, default=None, help="Output path for noise samples")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for all DAE/KMeans/KNN artifacts")
    parser.add_argument("--name", type=str, default=None, help="Output file prefix when --output-dir is specified")
    parser.add_argument("--resource", "-r", type=str, default="cic2018", choices=["cic2018"])
    parser.add_argument("--kmeans-budget", type=int, default=6000, help="Target benign core size")
    parser.add_argument("--kmeans-clusters", type=int, default=None, help="If None, equals budget")
    parser.add_argument("--kmeans-batch-size", type=int, default=100000)
    parser.add_argument("--kmeans-algo", type=str, default="minibatch", choices=["minibatch", "full"])
    parser.add_argument("--k-neighbors", "-k", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-factor", type=float, default=0.2)
    parser.add_argument("--dae-model-path", type=str, default=None, help="Load an existing DAE model if provided")
    parser.add_argument("--save-dae-model", type=str, default="models/cic2018/dae_model.pth")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Normalize user-provided paths and create the output directory when requested.
    args.input = str(Path(args.input).expanduser())
    if args.output_clean is not None:
        args.output_clean = str(Path(args.output_clean).expanduser())
    if args.output_noise is not None:
        args.output_noise = str(Path(args.output_noise).expanduser())
    if args.output_dir is not None:
        args.output_dir = str(Path(args.output_dir).expanduser())
        os.makedirs(args.output_dir, exist_ok=True)
    if args.dae_model_path is not None:
        args.dae_model_path = str(Path(args.dae_model_path).expanduser())
    args.save_dae_model = str(Path(args.save_dae_model).expanduser())

    if args.kmeans_budget <= 0:
        raise SystemExit("--kmeans-budget must be > 0")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Resolve output paths for the requested run layout.
    if args.output_dir is not None:
        input_stem = args.name if args.name else Path(args.input).stem
        args.output_clean = os.path.join(args.output_dir, f"{input_stem}_clean.csv")
        args.output_noise = os.path.join(args.output_dir, f"{input_stem}_noise.csv")
        args.save_dae_model = os.path.join(args.output_dir, "dae_model.pth")
    else:
        if args.output_clean is None:
            base, ext = os.path.splitext(args.input)
            args.output_clean = f"{base}_dae_kmeans_knn_clean{ext}"
        if args.output_noise is None:
            base, ext = os.path.splitext(args.input)
            args.output_noise = f"{base}_dae_kmeans_knn_noise{ext}"

    logger.info(f"\n{'=' * 70}")
    logger.info("CIC-IDS-2018 DEFENSE PIPELINE (DAE + KMEANS + KNN)")
    logger.info(f"{'=' * 70}")
    logger.info(f"[+] Input: {args.input}")

    # Load the CIC-IDS-2018 preprocessor and fitted encoders.
    preprocessor = CIC2018Preprocessor()
    logger.info("[+] Loading encoders for cic2018...")
    if not preprocessor.load_encoders():
        raise SystemExit("Encoders not found. Please fit/load encoders first.")

    # Read the poisoned dataset and keep a copy of the optional Benign-tracking metadata.
    df_in = pd.read_csv(args.input, low_memory=False)
    logger.info(f"[+] Dataset shape: {df_in.shape}")

    has_original_label = "original_label" in df_in.columns
    has_is_noisy = "is_noisy" in df_in.columns
    has_source = "__source__" in df_in.columns

    original_label_col = df_in["original_label"].copy() if has_original_label else None
    is_noisy_col = df_in["is_noisy"].copy() if has_is_noisy else None
    source_col = df_in["__source__"].copy() if has_source else None

    # The ablation path does not depend on ground-truth noise supervision.
    cols_to_drop = ["original_label", "is_noisy", "__source__", "detected_as_noise"]
    df_to_encode = df_in.drop(columns=cols_to_drop, errors="ignore")

    # STEP 1: Encode the dataset for the DAE and KNN stages.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 1: ENCODE DATA")
    logger.info(f"{'=' * 60}")
    df_encoded = encode_dataset(df_to_encode, preprocessor)

    # Separate the encoded feature matrix and the encoded label vector.
    X_train = df_encoded.iloc[:, :-1].values.astype(np.float32)
    y_train = df_encoded.iloc[:, -1].values.astype(int)
    input_dim = X_train.shape[1]

    # Resolve the compute device explicitly so the ablation command can force cpu/cuda.
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[+] Device: {device}")

    # STEP 2: Train or load the DAE and extract latent embeddings.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 2: DAE EMBEDDING")
    logger.info(f"{'=' * 60}")

    if args.dae_model_path and os.path.exists(args.dae_model_path):
        logger.info(f"[+] Loading pre-trained DAE: {args.dae_model_path}")
        dae = OptimizedDAE(input_dim, latent_dim=args.latent_dim).to(device)
        dae.load_state_dict(torch.load(args.dae_model_path, map_location=device))
        dae.eval()
    else:
        logger.info("[+] Training new DAE model...")
        dae = train_dae_optimized(
            X_train,
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            epochs=200,
            batch_size=args.batch_size,
            lr=args.lr,
            noise_factor=args.noise_factor,
            patience=5,
            device=device,
        )
        os.makedirs(os.path.dirname(args.save_dae_model) or ".", exist_ok=True)
        torch.save(dae.state_dict(), args.save_dae_model)
        logger.info(f"[+] Saved DAE model: {args.save_dae_model}")

    X_emb = extract_features(dae, X_train, device=device)
    df_embedded = pd.DataFrame(X_emb, index=df_encoded.index)
    df_embedded["Label"] = y_train
    logger.info(f"[+] Embedding shape: {df_embedded.shape}")

    # Obtain the encoded label for Benign, which is the only class we core-filter.
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
        n_clusters=args.kmeans_clusters,
        batch_size=int(args.kmeans_batch_size),
        kmeans_algo=args.kmeans_algo,
    )

    core_set = set(benign_core_idx.tolist())
    kmeans_noise_set = set(benign_removed_idx.tolist())

    logger.info(f"[+] Benign core kept: {len(core_set)}")
    logger.info(f"[+] Benign removed by KMeans: {len(kmeans_noise_set)}")

    # STEP 4: KNN detection on the Benign-core subset.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 4: KNN FILTER ON BENIGN-CORE")
    logger.info(f"{'=' * 60}")

    detector = CorruptionDetector(df_embedded)

    # Run the final KNN detector using the ablation's fixed k value.
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

    # STEP 5: Inverse-transform the encoded data and export results.
    logger.info(f"\n{'=' * 60}")
    logger.info("[+] STEP 5: INVERSE + EXPORT")
    logger.info(f"{'=' * 60}")

    df_inverted = inverse_transform_data(df_encoded, preprocessor)

    # Restore optional metadata columns if they existed in the input.
    if has_source:
        df_inverted["__source__"] = source_col
    if has_original_label:
        df_inverted["original_label"] = original_label_col
    if has_is_noisy:
        df_inverted["is_noisy"] = is_noisy_col

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

    per_label_rows: list[dict] = []
    if has_original_label and original_label_col is not None:
        original_labels = original_label_col.astype(str)
        detected_mask = pd.Series(final_mask, index=df_inverted.index)
        is_noisy_series = is_noisy_col.astype(bool) if has_is_noisy and is_noisy_col is not None else None

        for label_name in sorted(original_labels.unique().tolist()):
            label_mask = (original_labels == label_name)
            total_count = int(label_mask.sum())
            detected_count = int((label_mask & detected_mask).sum())
            true_noise_count = int((label_mask & is_noisy_series).sum()) if is_noisy_series is not None else None
            true_positive_count = int((label_mask & detected_mask & is_noisy_series).sum()) if is_noisy_series is not None else None

            row = {
                "label": label_name,
                "total_count": total_count,
                "detected_noise_count": detected_count,
                "detected_noise_rate": float(detected_count / total_count) if total_count > 0 else 0.0,
            }
            if true_noise_count is not None:
                row["true_noise_count"] = true_noise_count
                row["true_noise_rate"] = float(true_noise_count / total_count) if total_count > 0 else 0.0
            if true_positive_count is not None:
                row["true_positive_count"] = true_positive_count
                row["precision_on_detected"] = float(true_positive_count / detected_count) if detected_count > 0 else 0.0
            per_label_rows.append(row)

    per_label_report = pd.DataFrame(per_label_rows)
    per_label_out = report_dir / "noise_detection_per_label.csv"
    if not per_label_report.empty:
        per_label_report.to_csv(per_label_out, index=False)

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

    current_label_series = df_inverted["Label"].astype(str)
    current_distribution = _distribution_rows(current_label_series)

    detection_report = {
        "input_csv": str(args.input),
        "output_clean_csv": str(args.output_clean),
        "output_noise_csv": str(args.output_noise),
        "total_samples": int(total_rows),
        "k_neighbors": int(args.k_neighbors),
        "kmeans_budget": int(args.kmeans_budget),
        "current_input_label_distribution": current_distribution,
        "stage_summary": stage_summary,
        "per_label_report_csv": str(per_label_out) if not per_label_report.empty else None,
        "metrics_vs_true_noise": None,
    }

    report_out = report_dir / "noise_detection_report.json"
    with open(report_out, "w", encoding="utf-8") as handle:
        json.dump(detection_report, handle, indent=2)
    logger.info(f"[+] Noise detection report -> {report_out}")
    if not per_label_report.empty:
        logger.info(f"[+] Noise detection per-label CSV -> {per_label_out}")

    logger.info(f"\n{'=' * 70}")
    logger.info("CIC-IDS-2018 DEFENSE PIPELINE COMPLETED")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()