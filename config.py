import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Root
ROOT_DIR = Path(__file__).parent

# Data Paths
DATA_DIR        = ROOT_DIR / 'data'
RAW_DATA_DIR    = DATA_DIR / 'raw'
PROC_DATA_DIR   = DATA_DIR / 'processed'
EXPORT_DIR      = DATA_DIR / 'exports'
RAW_DATA_FILE   = RAW_DATA_DIR / 'telco_churn.csv'
CLEAN_DATA_FILE = PROC_DATA_DIR / 'telco_churn_clean.csv'

# Model Paths
MODELS_DIR = ROOT_DIR / 'models'
SAVED_DIR  = MODELS_DIR / 'saved'
EVAL_DIR   = MODELS_DIR / 'evaluation'

# Database Configuration
DB_CONFIG = {
    'host':     os.getenv('DB_HOST', 'localhost'),
    'port':     os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'churn_analytics'),
    'user':     os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# Model Parameters
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
TARGET_COLUMN = 'Churn'

# Feature Groups
CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod',
]

NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
]
