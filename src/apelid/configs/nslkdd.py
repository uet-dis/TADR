from .resources import Resources


class NSLKDDResources(Resources):
    resources_name = "nslkdd"
    DATA_FOLDER = "resources/NSLKDD"
    
    CLEAN_MERGED_FOLDER = f"{DATA_FOLDER}/clean_merged"

    KDD_TEXT_PATH = f"{DATA_FOLDER}/KDD+.txt"
    NSLKDD_ORIGINAL_CSV_PATH = f"{DATA_FOLDER}/nslkdd_original.csv"

    CLEAN_MERGED_DATA_FOLDER = f"{DATA_FOLDER}/clean_merged"
    ENCODED_DATA_FOLDER = f"{DATA_FOLDER}/encoded"
    RAW_PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/raw_processed"

    ENCODERS_FOLDER = "encoders/nslkdd"
    REPORT_FOLDER = "reports/nslkdd"

    MAJORITY_LABELS = ['Benign', 'DoS']
    MINORITY_LABELS = ['Probe', 'R2L', 'U2R']