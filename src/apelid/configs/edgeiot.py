import os
from .resources import Resources


class EDGEIOTResources(Resources):
    resources_name = "edgeiot"
    BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "EdgeIIoTset"))
    RESOURCES_FOLDER = os.path.join(BASE_FOLDER, "resources")
    ENCODERS_FOLDER = os.path.join(BASE_FOLDER, "encoders")
    REPORT_FOLDER = os.path.join(BASE_FOLDER, "reports")

    DATA_FOLDER = os.path.join(RESOURCES_FOLDER, "edgeiot")
    CLEAN_MERGED_FOLDER = os.path.join(DATA_FOLDER, "clean_merged")

    EDGEIOT_TEXT_PATH = os.path.join(DATA_FOLDER, "ML-EdgeIIoT-dataset.csv")
    EDGEIOT_ORIGINAL_CSV_PATH = os.path.join(DATA_FOLDER, "edgeiot_original.csv")

    CLEAN_MERGED_DATA_FOLDER = CLEAN_MERGED_FOLDER
    ENCODED_DATA_FOLDER = os.path.join(DATA_FOLDER, "encoded")
    RAW_PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, "raw_processed")