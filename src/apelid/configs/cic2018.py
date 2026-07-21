from .resources import Resources
from pathlib import Path


class CIC2018Resources(Resources):
    resources_name = "cic2018"

    _IDS18_ROOT = Path(__file__).resolve().parents[1] / "IDS18"

    ORIGINAL_DATA_FOLDER = str(_IDS18_ROOT / "resources" / "original")
    DATA_FOLDER = str(_IDS18_ROOT / "resources" / "IDS18")
    REPORT_FOLDER = str(_IDS18_ROOT / "reports" / "cic2018")
    ENCODERS_FOLDER = str(_IDS18_ROOT / "encoders")
    
    CLEAN_MERGED_FOLDER = f"{DATA_FOLDER}/clean_merged"

    ENCODED_DATA_FOLDER = f"{DATA_FOLDER}/encoded"
    RAW_PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/raw_processed"
    CLEAN_MERGED_DATA_FOLDER = f"{DATA_FOLDER}/clean_merged"
    TEST_ALL_MERGED_FILE = f"{CLEAN_MERGED_DATA_FOLDER}/cic2018_test_all_merged.csv"
    TEST_RANDOM_SAMPLE_TAU = 14000
    TEST_RANDOM_SAMPLE_FILE = f"{CLEAN_MERGED_DATA_FOLDER}/cic2018_test_original_undersampled.csv"

    EMBEDDINGS_FOLDER = f"{DATA_FOLDER}/embeddings"
    PCA_CACHE_FOLDER = f"{EMBEDDINGS_FOLDER}/pca_cache"

    LABEL_COLUMN = "Label"

    MAJORITY_LABELS = [
        'Benign', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk', 'Bot', 
        'Infilteration', 'SSH-Bruteforce', 'DoS attacks-GoldenEye'
    ]

    MINORITY_LABELS = [
        'DoS attacks-Slowloris',
        # 'DDOS attack-LOIC-UDP',
        'Brute Force -Web',
        'Brute Force -XSS',
        'SQL Injection',
        # 'DoS attacks-SlowHTTPTest',
        # 'FTP-BruteForce'
    ]

    ACCEPT_RATE_MAP = {
        'DoS attacks-Slowloris': 0.30,
        'Brute Force -Web': 0.35,
        'Brute Force -XSS': 0.50,
        'SQL Injection': 0.3,
    }
    