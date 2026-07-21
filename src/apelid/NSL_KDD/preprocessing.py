"""
================================================================================
NSL-KDD DATASET PREPROCESSING PIPELINE
================================================================================
Module: src/apelid/NSL_KDD/preprocessing.py

Purpose:
    Process NSL-KDD data through the following steps: load -> clean -> feature
    selection -> encode -> train/test split -> setup encoders

Processing flow:
    1. Load raw data (KDD+.txt format)
    2. Convert to standardized CSV format
    3. Map attack labels into main groups (DoS, Probe, U2R, R2L, Benign)
    4. Select important features
    5. Cleaning: remove missing values and duplicate rows
    6. Train/Test split (70/30, stratified, shuffled) → save two files
    7. Fit and save encoders based on the training set

Output folders:
    - resources/NSLKDD/nslkdd_original.csv: original data in CSV format
    - resources/NSLKDD/clean_merged/nslkdd_train_clean_merged.csv: train 70%
    - resources/NSLKDD/clean_merged/nslkdd_test_clean_merged.csv: test 30%
    - encoders/nslkdd/: saved encoders (OneHotEncoder, MinMaxScaler, LabelEncoder, etc.)

Config:
    - Multiclass labels: Benign, DoS, Probe, R2L, U2R
    - Categorical features to encode: protocol_type, service, flag
    - Numerical scaling: MinMax (0-1 range)
================================================================================
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup path for importing from the package
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from apelid.preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from apelid.utils.logging import setup_logging, get_logger
from apelid.configs import NSLKDDResources as nslkdd

logger = get_logger(__name__)

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================
RANDOM_STATE = 42  # For reproducibility
TRAIN_TEST_SPLIT_RATIO = 0.3  # 70% train, 30% test


def load_and_convert_raw_data(preprocessor: NSLKDDPreprocessor, nslkdd_config) -> pd.DataFrame:
    """
    STEP 1: Load raw NSL-KDD data from a text file and convert to CSV format

    Input: KDD+.txt (raw text format from UCI Machine Learning Repository)
    Output: nslkdd_original.csv (standard CSV format)

    Args:
        preprocessor: NSLKDDPreprocessor instance
        nslkdd_config: Config object containing paths

    Returns:
        DataFrame containing the raw data
    """
    logger.info("[STEP 1] Loading raw NSL-KDD data...")
    logger.info(f"  → Input: {nslkdd_config.KDD_TEXT_PATH}")
    
    # Load text file với column names từ preprocessor
    df = pd.read_csv(nslkdd_config.KDD_TEXT_PATH, names=preprocessor.nslkdd_columns)
    logger.info(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Save as CSV format cho lần sau có thể load nhanh hơn
    df.to_csv(nslkdd_config.NSLKDD_ORIGINAL_CSV_PATH, index=False)
    logger.info(f"  ✓ Saved to CSV: {nslkdd_config.NSLKDD_ORIGINAL_CSV_PATH}")
    
    return df


def map_labels_and_select_features(preprocessor: NSLKDDPreprocessor, df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 2: Map attack types into 5 main groups and select important features

    Attack type mapping examples:
        - DoS (Denial of Service): apache2, back, land, neptune, ...
        - Probe (Reconnaissance): ipsweep, mscan, nmap, portsweep, ...
        - U2R (User to Root): buffer_overflow, loadmodule, perl, ...
        - R2L (Remote to Local): ftp_write, guess_passwd, imap, ...
        - Benign (Normal traffic): all non-attack traffic

    Args:
        preprocessor: NSLKDDPreprocessor instance
        df: Raw DataFrame

    Returns:
        DataFrame with the label column mapped and only selected features kept
    """
    logger.info("[STEP 2] Mapping attack labels & selecting features...")
    
    # Map attack types thành 5 nhóm chính
    df = preprocessor.map_label(df)
    logger.info(f"  ✓ Attack labels mapped to 5 groups")
    logger.info(f"    - DoS, Probe, U2R, R2L, Benign")
    
    # Select features quan trọng (loại bỏ những features không cần thiết)
    df = preprocessor.select_features_and_label(df)
    logger.info(f"  ✓ Selected {len(df.columns)-1} features + 1 label column")
    logger.info(f"    Total columns: {len(df.columns)}")
    
    return df


def clean_data(preprocessor: NSLKDDPreprocessor, df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 3: Data cleaning - remove invalid/anomalous rows

    Cleaning tasks:
        - Remove missing values (NaN)
        - Remove infinite values (inf, -inf)
        - Remove duplicate rows

    Args:
        preprocessor: NSLKDDPreprocessor instance
        df: DataFrame to be cleaned

    Returns:
        Cleaned DataFrame
    """
    logger.info("[STEP 3] Cleaning data...")
    initial_rows = len(df)
    
    # Remove missing & infinite values
    df = preprocessor.remove_missing_and_inf_values(df)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"  ✓ Removed {removed_rows:,} rows with missing/infinite values")
    
    initial_rows = len(df)
    
    # Remove duplicates
    df = preprocessor.fix_duplicates(df)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"  ✓ Removed {removed_rows:,} duplicate rows")
    
    # Log dataset info sau cleaning
    preprocessor.info_dataset(df)
    
    return df


def analyze_categorical_features(preprocessor: NSLKDDPreprocessor, df: pd.DataFrame) -> None:
    """
    STEP 4: Analyze categorical features

    Log the number of unique values for each categorical feature. This is
    important to know the number of categories before encoding.

    Args:
        preprocessor: NSLKDDPreprocessor instance
        df: Cleaned DataFrame
    """
    logger.info("[STEP 4] Analyzing categorical features...")
    
    categorical_features = preprocessor.encoded_categorical_features
    logger.info(f"  Categorical features to encode: {categorical_features}")
    
    for feature in categorical_features:
        n_unique = df[feature].nunique()
        logger.info(f"    - {feature}: {n_unique} unique values")


def split_train_test(preprocessor: NSLKDDPreprocessor, df: pd.DataFrame, nslkdd_config) -> tuple:
    """
    STEP 5: Train/Test split for the entire dataset (70/30, stratified, shuffled)

    Split ratio: 70% train vs 30% test. Stratify by label to preserve class
    proportions. Data is shuffled before splitting.

    Output files:
        - resources/NSLKDD/clean_merged/nslkdd_train_clean_merged.csv
        - resources/NSLKDD/clean_merged/nslkdd_test_clean_merged.csv

    Args:
        preprocessor: NSLKDDPreprocessor instance
        df: Cleaned DataFrame
        nslkdd_config: Config object

    Returns:
        (df_train, df_test)
    """
    logger.info("[STEP 5] Train/Test splitting (70/30, stratified, shuffled)...")

    dst_folder = nslkdd_config.CLEAN_MERGED_DATA_FOLDER
    os.makedirs(dst_folder, exist_ok=True)

    df_train, df_test = train_test_split(
        df,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=df[preprocessor.label_column],
        shuffle=True
    )

    train_path = os.path.join(dst_folder, "nslkdd_train_clean_merged.csv")
    test_path = os.path.join(dst_folder, "nslkdd_test_clean_merged.csv")
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


def setup_encoders(preprocessor: NSLKDDPreprocessor, train_df: pd.DataFrame) -> None:
    """
    STEP 6: Fit & save encoders based on the training set

    Encoders are fit on the train set to avoid data leakage. Encoders include:
        - OneHotEncoder: encode categorical features (protocol_type, service, flag)
        - MinMaxScaler: scale numerical features to [0, 1]
        - StandardScaler: standardize numerical features (mean=0, std=1)
        - LabelEncoder: encode attack labels (DoS, Probe, U2R, R2L, Benign)
        - OrdinalEncoder: ordinal-encode categorical features

    Output (saved under encoders/nslkdd/):
        - categorical_encoder.pkl (OneHotEncoder)
        - minmax_scaler.pkl (MinMaxScaler)
        - label_encoder.pkl (LabelEncoder)
        - ...

    Args:
        preprocessor: NSLKDDPreprocessor instance
        train_df: Training DataFrame
    """
    logger.info("[STEP 6] Setting up & saving encoders (based on train set)...")
    logger.info(f"  ✓ Train data: {len(train_df):,} samples")

    # Fit encoders trên train set
    logger.info(f"  → Fitting encoders...")
    preprocessor.setup_encoders(train_df)

    # Lưu encoders
    logger.info(f"  → Saving encoders to {preprocessor.encoders_dir}/")
    preprocessor.save_encoders()
    logger.info(f"  ✓ Encoders saved successfully")

    # Log encoder info
    categorical_features = preprocessor.encoded_categorical_features
    logger.info(f"  Categorical features encoded: {categorical_features}")
    for feature in categorical_features:
        n_unique_train = train_df[feature].nunique()
        logger.info(f"    - {feature}: {n_unique_train} unique values in train set")


def main():
    """
    Main function: run the full NSL-KDD preprocessing pipeline
    """
    logger.info("=" * 80)
    logger.info("NSL-KDD DATASET PREPROCESSING PIPELINE START")
    logger.info("=" * 80)
    
    # Initialize preprocessor
    preprocessor = NSLKDDPreprocessor()
    
    # STEP 1: Load raw data
    df = load_and_convert_raw_data(preprocessor, nslkdd)
    
    # STEP 2: Map labels & select features
    df = map_labels_and_select_features(preprocessor, df)
    
    # STEP 3: Clean data
    df = clean_data(preprocessor, df)
    
    # STEP 4: Analyze categorical features
    analyze_categorical_features(preprocessor, df)
    
    # STEP 5: Train/Test split (70/30, stratified, shuffled)
    df_train, df_test = split_train_test(preprocessor, df, nslkdd)

    # STEP 6: Setup encoders from train set
    setup_encoders(preprocessor, df_train)
    
    logger.info("=" * 80)
    logger.info("NSL-KDD DATASET PREPROCESSING PIPELINE COMPLETED ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging("INFO")
    main()


    

    