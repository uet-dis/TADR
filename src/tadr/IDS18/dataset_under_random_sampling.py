"""
Undersample original IDS18 train/test splits by class with benign balancing.

Rules:
1) MINORITY_LABELS are kept unchanged in both train and test.
2) MAJORITY labels except Benign are capped:
   - train: 14000
   - test: 6000
3) Benign is sampled to match total attack samples in each split.

Input folders (default):
- resources/original/train_IDS18
- resources/original/test_IDS18

Outputs (default):
- resources/IDS18/clean_merged/cic2018_train_original_undersampled.csv
- resources/IDS18/clean_merged/cic2018_test_original_undersampled.csv
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from tadr.utils.logging import setup_logging, get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from tadr.configs import CIC2018Resources, NSLKDDResources


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': CIC2018Resources,
    'nslkdd': NSLKDDResources,
}


def _safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Safely concatenate dataframes with potentially different columns by unioning columns and filling missing ones with NaN."""
    if not dfs:
        return pd.DataFrame()
    cols = set()
    for d in dfs:
        cols.update(d.columns.tolist())
    ordered = sorted(cols)
    normed = [d.reindex(columns=ordered) for d in dfs]
    return pd.concat(normed, ignore_index=True)


def _find_label_file(split_dir: Path, resource_name: str, label_safe: str, split_name: str) -> Path | None:
    """Find a class CSV file for one label in train/test original folder."""
    # Prefer exact naming pattern.
    exact = split_dir / f"{resource_name}_{label_safe}_{split_name}_clean_merged.csv"
    if exact.exists():
        return exact

    # Fallback to fuzzy match if naming differs slightly.
    pattern = f"{resource_name}_{label_safe}_*.csv"
    candidates = sorted(split_dir.glob(pattern))
    if candidates:
        return candidates[0]
    return None


def _sample_one_split(
    split_name: str,
    split_dir: Path,
    out_path: Path,
    res,
    label_col: str,
    majority_cap: int,
    seed: int,
) -> None:
    labels = list(res.MAJORITY_LABELS + res.MINORITY_LABELS)
    minority_set = set(res.MINORITY_LABELS)
    benign_label = "Benign"

    if benign_label not in labels:
        raise SystemExit("'Benign' must exist in label list.")

    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    logger.info("=" * 90)
    logger.info(f"[+] UNDERSAMPLING {split_name.upper()} from: {split_dir}")
    logger.info("=" * 90)

    sampled_parts: list[pd.DataFrame] = []
    benign_df: pd.DataFrame | None = None
    attack_total = 0

    for label in tqdm(labels, desc=f"Undersample {split_name}", unit="label"):
        label_safe = res.get_label_name(label)
        file_path = _find_label_file(split_dir, res.resources_name, label_safe, split_name)
        if file_path is None:
            logger.warning(f"[!] Missing file for label '{label}' in {split_dir}")
            continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as exc:
            logger.warning(f"[!] Failed to read {file_path}: {exc}")
            continue

        if label_col not in df.columns:
            logger.warning(f"[!] Missing label column '{label_col}' in {file_path}; skip")
            continue

        n_before = len(df)
        if label == benign_label:
            benign_df = df
            logger.info(f"[+] {split_name}:{label}: defer benign balancing | rows={n_before}")
            continue

        if label in minority_set:
            df_sampled = df
        else:
            n_take = min(int(majority_cap), n_before) if majority_cap > 0 else n_before
            df_sampled = df.sample(n=n_take, random_state=seed).reset_index(drop=True) if n_take < n_before else df

        sampled_parts.append(df_sampled)
        attack_total += len(df_sampled)
        logger.info(f"[+] {split_name}:{label}: {n_before} -> {len(df_sampled)}")

    if benign_df is None:
        raise SystemExit(f"Benign file not found in {split_dir}")

    benign_before = len(benign_df)
    benign_take = min(int(attack_total), benign_before)
    benign_sampled = (
        benign_df.sample(n=benign_take, random_state=seed).reset_index(drop=True)
        if benign_take < benign_before else benign_df
    )
    sampled_parts.insert(0, benign_sampled)

    logger.info(f"[+] {split_name}:Benign balanced to total_attack={attack_total} -> {len(benign_sampled)}")

    out_df = _safe_concat(sampled_parts)

    if split_name.lower() == 'train' and len(out_df) > 0:
        out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        logger.info(f"[+] {split_name}: shuffled final dataset with seed={seed}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    logger.info(f"[+] Saved {split_name} undersampled file: {out_path}")
    logger.info(f"[+] {split_name} final rows: {len(out_df)}")


def main():
    parser = argparse.ArgumentParser(description="Undersample original IDS18 train/test with minority-preserving rules")
    parser.add_argument('--resource', '-r', type=str, default='cic2018', choices=list(REGISTRY.keys()))
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--label-col', type=str, default='Label', help='Label column name')

    default_train_dir = Path(SCRIPT_DIR) / 'resources' / 'original' / 'train_IDS18'
    default_test_dir = Path(SCRIPT_DIR) / 'resources' / 'original' / 'test_IDS18'
    default_out_dir = Path(SCRIPT_DIR) / 'resources' / 'IDS18' / 'clean_merged'

    parser.add_argument('--train-dir', type=str, default=str(default_train_dir), help='Original train folder path')
    parser.add_argument('--test-dir', type=str, default=str(default_test_dir), help='Original test folder path')
    parser.add_argument('--train-major-cap', type=int, default=14000, help='Cap for majority attack labels in train')
    parser.add_argument('--test-major-cap', type=int, default=6000, help='Cap for majority attack labels in test')
    parser.add_argument('--train-out', type=str, default=str(default_out_dir / 'cic2018_train_original_undersampled.csv'))
    parser.add_argument('--test-out', type=str, default=str(default_out_dir / 'cic2018_test_original_undersampled.csv'))
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResClass = REGISTRY[args.resource]
    res = ResClass

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    train_out = Path(args.train_out)
    test_out = Path(args.test_out)

    _sample_one_split(
        split_name='train',
        split_dir=train_dir,
        out_path=train_out,
        res=res,
        label_col=args.label_col,
        majority_cap=int(args.train_major_cap),
        seed=int(args.seed),
    )

    _sample_one_split(
        split_name='test',
        split_dir=test_dir,
        out_path=test_out,
        res=res,
        label_col=args.label_col,
        majority_cap=int(args.test_major_cap),
        seed=int(args.seed),
    )

    logger.info("=" * 90)
    logger.info("[DONE] Undersampling train/test completed")
    logger.info(f"[DONE] train_out={train_out}")
    logger.info(f"[DONE] test_out={test_out}")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()


