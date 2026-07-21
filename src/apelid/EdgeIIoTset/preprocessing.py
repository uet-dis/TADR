"""
================================================================================
EDGEIIOTSET DATASET PREPROCESSING PIPELINE
================================================================================
Module: src/apelid/EdgeIIoTset/preprocessing.py

Purpose:
    Process EdgeIIoT data through the following steps: load -> select features
    -> clean -> analyze categorical features -> train/test split -> setup encoders.

Processing flow:
    1. Load raw EdgeIIoT CSV data
    2. Select important features and label
    3. Cleaning: drop missing values, drop duplicate rows, shuffle
    4. Analyze categorical features
    5. Train/Test split (70/30, stratified, shuffled) -> save two files
    6. Fit and save encoders based on the training set

Output folders (all under src/apelid/EdgeIIoTset/):
    - resources/ML-EdgeIIoT-dataset.csv:          original raw data
    - resources/edgeiot_original.csv:              original data in CSV format
    - resources/clean_merged/edgeiot_clean_merged.csv:  cleaned merged data
    - resources/clean_merged/edgeiot_train_clean_merged.csv: train 70%
    - resources/clean_merged/edgeiot_test_clean_merged.csv:  test 30%
    - encoders/:                                  saved encoders (OneHotEncoder, MinMaxScaler, etc.)

Config:
    - Multiclass labels: Attack_type
    - Categorical features to encode: see EDGEIOTPreprocessor.encoded_categorical_features
    - Numerical scaling: MinMax / Standard / QuantileTransformer
================================================================================
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Setup path for importing from the package
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from apelid.preprocessing.edgeiot_preprocessor import EDGEIOTPreprocessor
from apelid.configs.edgeiot import EDGEIOTResources as edgeiot
from apelid.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================
RANDOM_STATE = 42
TRAIN_TEST_SPLIT_RATIO = 0.3  # 70% train, 30% test


def load_raw_data(preprocessor: EDGEIOTPreprocessor, edgeiot_config) -> pd.DataFrame:
    """STEP 1: Load raw EdgeIIoT CSV data."""
    logger.info("[STEP 1] Loading raw EdgeIIoT data...")
    logger.info(f"  -> Input: {edgeiot_config.EDGEIOT_TEXT_PATH}")

    df = pd.read_csv(edgeiot_config.EDGEIOT_TEXT_PATH)
    logger.info(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

    df.to_csv(edgeiot_config.EDGEIOT_ORIGINAL_CSV_PATH, index=False)
    logger.info(f"  ✓ Saved to CSV: {edgeiot_config.EDGEIOT_ORIGINAL_CSV_PATH}")

    return df


def select_features_and_label(preprocessor: EDGEIOTPreprocessor, df: pd.DataFrame) -> pd.DataFrame:
    """STEP 2: Keep only model features and the label column."""
    logger.info("[STEP 2] Selecting important features and label...")
    df = preprocessor.select_features_and_label(df)
    logger.info(f"  ✓ Selected {len(df.columns) - 1} features + 1 label column")
    logger.info(f"  ✓ Total columns: {len(df.columns)}")
    return df


def clean_data(preprocessor: EDGEIOTPreprocessor, df: pd.DataFrame, edgeiot_config) -> pd.DataFrame:
    """STEP 3: Clean dataset by dropping missing values, duplicates, and shuffling."""
    logger.info("[STEP 3] Cleaning data...")
    initial_rows = len(df)

    # Drop columns is effectively handled by feature selection above, then clean the selected frame.
    df = preprocessor.remove_missing_and_inf_values(df)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"  ✓ Removed {removed_rows:,} rows with missing/infinite values")

    initial_rows = len(df)
    df = preprocessor.fix_duplicates(df)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"  ✓ Removed {removed_rows:,} duplicate rows")

    clean_folder = edgeiot_config.CLEAN_MERGED_DATA_FOLDER
    os.makedirs(clean_folder, exist_ok=True)
    clean_path = os.path.join(clean_folder, "edgeiot_clean_merged.csv")
    df.to_csv(clean_path, index=False)
    logger.info(f"  ✓ Saved cleaned merged data: {clean_path}")

    preprocessor.info_dataset(df)
    return df


def analyze_categorical_features(preprocessor: EDGEIOTPreprocessor, df: pd.DataFrame) -> None:
    """STEP 4: Analyze categorical feature cardinalities before encoding."""
    logger.info("[STEP 4] Analyzing categorical features...")

    categorical_features = preprocessor.encoded_categorical_features
    logger.info(f"  Categorical features to encode: {categorical_features}")

    for feature in categorical_features:
        n_unique = df[feature].nunique(dropna=True)
        logger.info(f"    - {feature}: {n_unique} unique values")


def split_train_test(preprocessor: EDGEIOTPreprocessor, df: pd.DataFrame, edgeiot_config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """STEP 5: Stratified 70/30 train/test split."""
    logger.info("[STEP 5] Train/Test splitting (70/30, stratified, shuffled)...")

    label_counts = df[preprocessor.label_column].value_counts()
    if label_counts.min() < 2:
        raise ValueError(
            f"Cannot stratify split because at least one label has fewer than 2 samples: {label_counts.to_dict()}"
        )

    dst_folder = edgeiot_config.CLEAN_MERGED_DATA_FOLDER
    os.makedirs(dst_folder, exist_ok=True)

    df_train, df_test = train_test_split(
        df,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=df[preprocessor.label_column],
        shuffle=True,
    )

    train_path = os.path.join(dst_folder, "edgeiot_train_clean_merged.csv")
    test_path = os.path.join(dst_folder, "edgeiot_test_clean_merged.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.info(f"  ✓ Saved: {len(df_train):,} train + {len(df_test):,} test")
    logger.info(f"    - Train: {train_path}")
    logger.info(f"    - Test : {test_path}")
    logger.info(
        f"  ✓ Train label distribution:\n{df_train[preprocessor.label_column].value_counts().to_string()}"
    )
    logger.info(
        f"  ✓ Test label distribution:\n{df_test[preprocessor.label_column].value_counts().to_string()}"
    )

    return df_train, df_test


def setup_encoders(preprocessor: EDGEIOTPreprocessor, train_df: pd.DataFrame, edgeiot_config) -> None:
    """STEP 6: Fit and save encoders based on the training set."""
    logger.info("[STEP 6] Setting up & saving encoders (based on train set)...")
    logger.info(f"  ✓ Train data: {len(train_df):,} samples")
    preprocessor.encoders_dir = edgeiot_config.ENCODERS_FOLDER

    logger.info("  -> Fitting encoders...")
    preprocessor.setup_encoders(train_df)

    logger.info(f"  -> Saving encoders to {preprocessor.encoders_dir}/")
    preprocessor.save_encoders()
    logger.info("  ✓ Encoders saved successfully")

    categorical_features = preprocessor.encoded_categorical_features
    logger.info(f"  Categorical features encoded: {categorical_features}")
    for feature in categorical_features:
        n_unique_train = train_df[feature].nunique(dropna=True)
        logger.info(f"    - {feature}: {n_unique_train} unique values in train set")


def main():
    """Run the full EdgeIIoT preprocessing pipeline."""
    logger.info("=" * 80)
    logger.info("EDGEIIOTSET DATASET PREPROCESSING PIPELINE START")
    logger.info("=" * 80)

    preprocessor = EDGEIOTPreprocessor()

    # STEP 1: Load raw data
    df = load_raw_data(preprocessor, edgeiot)

    # STEP 2: Keep only important features and the label
    df = select_features_and_label(preprocessor, df)

    # STEP 3: Clean data
    df = clean_data(preprocessor, df, edgeiot)

    # STEP 4: Analyze categorical features
    analyze_categorical_features(preprocessor, df)

    # STEP 5: Train/Test split (70/30, stratified, shuffled)
    df_train, df_test = split_train_test(preprocessor, df, edgeiot)

    # STEP 6: Setup encoders from train set
    setup_encoders(preprocessor, df_train, edgeiot)

    logger.info("=" * 80)
    logger.info("EDGEIIOTSET DATASET PREPROCESSING PIPELINE COMPLETED ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging("INFO")
    main()