"""
Targeted Label Flipping Generator for CIC-IDS-2018 Dataset

Implements targeted label flipping for testing model robustness.
With flip rate x, for each non-Benign class, exactly x% of its samples are
randomly selected and flipped to Benign.

Usage:
    python symmetric_label_noise.py --input cic2018_merged_train_raw_processed_drop_wgan.csv --noise-rate 0.1 --seed 42
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List

import sys
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tadr.configs.cic2018 import CIC2018Resources
from tadr.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def apply_targeted_flip_to_benign(
    df: pd.DataFrame,
    label_column: str,
    flip_rate: float,
    benign_label: str,
    all_labels: List[str] | None = None,
    seed: int = 42,
    keep_tracking: bool = False
) -> pd.DataFrame:
    """
    Flip x% samples from each non-benign class to benign.
    
    Args:
        df: Input DataFrame
        label_column: Name of label column
        flip_rate: Proportion to flip per non-benign class (0.0 to 1.0)
        benign_label: Label name used as target benign class (e.g., 'Benign')
        all_labels: Optional list of labels (for logging order)
        seed: Random seed for reproducibility
        keep_tracking: If True, add 'original_label' and 'is_noisy' columns
        
    Returns:
        DataFrame with flipped labels
    """
    # This helper implements the core label-flipping logic used by the
    # ablation pipeline's symmetric noise generation step. It is intentionally
    # deterministic when provided a `seed` so the pipeline can reproduce runs.
    if not 0.0 <= flip_rate <= 1.0:
        raise ValueError(f"flip_rate must be between 0 and 1, got {flip_rate}")
    if label_column not in df.columns:
        raise ValueError(f"label column '{label_column}' not found in dataframe")
    
    df_noisy = df.copy()
    original_labels = df_noisy[label_column].copy()
    rng = np.random.RandomState(seed)

    labels_in_data = df_noisy[label_column].unique().tolist()
    if all_labels is None:
        all_labels = sorted(labels_in_data)

    if benign_label not in labels_in_data:
        raise ValueError(
            f"benign_label '{benign_label}' not found in data. "
            f"Available labels: {labels_in_data}"
        )

    non_benign_labels = [label for label in labels_in_data if label != benign_label]

    n_samples = len(df_noisy)
    logger.info(f"[+] Total samples: {n_samples}")
    logger.info(f"[+] Number of classes: {len(labels_in_data)}")
    logger.info(f"[+] Flip rate per non-benign class: {flip_rate:.2%}")
    logger.info(f"[+] Benign target label: {benign_label}")

    flip_stats = {label: 0 for label in non_benign_labels}
    for label in non_benign_labels:
        class_idx = df_noisy.index[df_noisy[label_column] == label].to_numpy()
        class_count = len(class_idx)
        n_flip = int(np.floor(class_count * flip_rate))

        if n_flip <= 0:
            continue

        selected_idx = rng.choice(class_idx, size=n_flip, replace=False)
        df_noisy.loc[selected_idx, label_column] = benign_label
        flip_stats[label] = n_flip

    total_flipped = int((original_labels != df_noisy[label_column]).sum())
    actual_noise_rate = total_flipped / n_samples if n_samples > 0 else 0.0

    logger.info(f"\n[+] Actual flip rate (global): {actual_noise_rate:.2%}")
    logger.info(f"[+] Total flipped: {total_flipped} / {n_samples}")
    logger.info("[+] Flip breakdown (non-benign -> Benign):")
    for label in non_benign_labels:
        original_count = int((original_labels == label).sum())
        count = flip_stats[label]
        pct = (count / original_count * 100) if original_count > 0 else 0.0
        logger.info(f"    {label} -> {benign_label}: {count}/{original_count} ({pct:.2f}%)")
    
    # Show label distribution comparison
    logger.info("\n[+] Label distribution comparison:")
    logger.info("Original:")
    for label in all_labels:
        count = (original_labels == label).sum()
        logger.info(f"    {label}: {count}")
    logger.info("After flipping:")
    for label in all_labels:
        count = (df_noisy[label_column] == label).sum()
        logger.info(f"    {label}: {count}")
    
    # Optionally add tracking columns
    if keep_tracking:
        df_noisy['original_label'] = original_labels
        df_noisy['is_noisy'] = (original_labels != df_noisy[label_column])
        logger.info("\n[+] Added tracking columns: 'original_label', 'is_noisy'")
    
    return df_noisy


def main():
    default_input = Path(CIC2018Resources.CLEAN_MERGED_DATA_FOLDER) / "cic2018_train_original_undersampled.csv"

    parser = argparse.ArgumentParser(
        description="Apply targeted label flipping (non-benign -> benign) to CIC-IDS-2018 dataset"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=str(default_input),
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV file path (default: input_noise_{rate}.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for the noisy CSV (overrides --output if specified)'
    )
    parser.add_argument(
        '--noise-rate', '-n',
        type=float,
        required=True,
        help='Noise rate (0.0 to 1.0), e.g., 0.1 for 10%% corruption'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default='Label',
        help='Name of label column (default: Label)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=['Benign', 'Bot', 'DDoS attacks-LOIC-HTTP', 'SSH-Bruteforce', 'DoS attacks-Hulk', 'DDOS attack-HOIC', 'DoS attacks-GoldenEye', 'Infilteration', 'DoS attacks-Slowloris', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'],
        help='All possible label values (default: CIC-IDS-2018 classes)'
    )
    parser.add_argument(
        '--benign-label',
        type=str,
        default='Benign',
        help="Target benign label to flip into (default: 'Benign')"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--keep-tracking',
        action='store_true',
        help='Keep tracking columns (original_label, is_noisy) in output'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)

    args.input = str(Path(args.input).expanduser())
    if args.output is not None:
        args.output = str(Path(args.output).expanduser())
    if args.output_dir is not None:
        args.output_dir = str(Path(args.output_dir).expanduser())
    
    # Validate input file
    if not os.path.exists(args.input):
        raise SystemExit(f"[!] Input file not found: {args.input}")
    
    # Determine output path
    if args.output_dir is not None:
        # Use output directory
        os.makedirs(args.output_dir, exist_ok=True)
        noise_pct = int(args.noise_rate * 100)
        input_stem = Path(args.input).stem
        args.output = os.path.join(args.output_dir, f"{input_stem}_noise_{noise_pct}.csv")
    elif args.output is None:
        base, ext = os.path.splitext(args.input)
        noise_pct = int(args.noise_rate * 100)
        args.output = f"{base}_noise_{noise_pct}{ext}"
    
    logger.info(f"[+] Loading dataset: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    logger.info(f"[+] Dataset shape: {df.shape}")
    
    # Validate label column
    if args.label_column not in df.columns:
        logger.error(f"[!] Label column '{args.label_column}' not found in dataset")
        logger.info(f"    Available columns: {list(df.columns)}")
        return
    
    # Validate labels
    unique_labels = df[args.label_column].unique().tolist()
    logger.info(f"[+] Unique labels in dataset: {unique_labels}")
    
    missing_labels = set(unique_labels) - set(args.labels)
    if missing_labels:
        logger.warning(f"[!] Found labels not in specified list: {missing_labels}")
        logger.info("[+] Using all unique labels from dataset")
        args.labels = unique_labels
    
    # Apply targeted flipping
    logger.info(f"\n{'='*60}")
    logger.info(f"APPLYING TARGETED LABEL FLIPPING (NON-BENIGN -> BENIGN)")
    logger.info(f"{'='*60}")
    
    df_noisy = apply_targeted_flip_to_benign(
        df=df,
        label_column=args.label_column,
        flip_rate=args.noise_rate,
        benign_label=args.benign_label,
        all_labels=args.labels,
        seed=args.seed,
        keep_tracking=args.keep_tracking
    )
    
    # Save output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df_noisy.to_csv(args.output, index=False)
    logger.info(f"\n[+] Saved noisy dataset: {args.output}")
    logger.info(f"[+] Output shape: {df_noisy.shape}")
    if args.keep_tracking:
        logger.info(f"[+] New columns added: 'original_label', 'is_noisy'")
    else:
        logger.info(f"[+] Label column directly modified (no tracking columns)")
    logger.info(f"\n{'='*60}")
    logger.info("DONE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()