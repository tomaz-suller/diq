from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Project specific paths
PROFILING_DATA_DIR = DATA_DIR / "profiling"
INTERIM_DATA_DIR = DATA_DIR / "interim"
CLEAN_DATA_DIR = DATA_DIR / "clean"

NEIGHBOURHOOD_SHAPEFILE_PATH = EXTERNAL_DATA_DIR / "analysis-neighborhoods"
ZIP_CODE_SHAPEFILE_PATH = EXTERNAL_DATA_DIR / "bay-area-zip-codes"
STREET_NAMES_PATH = EXTERNAL_DATA_DIR / "street-names.csv"
RAW_DATASET_FILENAME = "building_permits.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / RAW_DATASET_FILENAME
CLEAN_DATASET_PATH = CLEAN_DATA_DIR / "dataset.parquet"
INTERIM_DATASET_PATH = INTERIM_DATA_DIR / "dataset.parquet"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
