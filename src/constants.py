"""
Constants used throughout the first_model package.

This module contains all configuration constants, default values, and mappings
used across the entire project to ensure consistency and easy maintenance.
"""

from typing import List, Tuple, Dict, Any
from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE CONSTANTS
# ============================================================================

# Base directories - dynamically determined from file location
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
TESTS_DIR = PROJECT_ROOT / "tests"

# ============================================================================
# DATA PROCESSING CONSTANTS
# ============================================================================

# Default column configurations
DEFAULT_DATE_COLUMNS: List[str] = ['재무제표결산날짜', '사고접수날짜']
DEFAULT_DATE_FORMAT: str = '%Y-%m-%d'

# Default preprocessing parameters
DEFAULT_OUTLIER_METHOD: str = 'iqr'
DEFAULT_OUTLIER_THRESHOLD: float = 3.0
DEFAULT_ZSCORE_THRESHOLD: float = 3.0

# Default scaling and encoding
DEFAULT_SCALER_TYPE: str = 'standard'
DEFAULT_ENCODING_METHOD: str = 'onehot'

# File handling
SUPPORTED_DATA_FORMATS: List[str] = ['.csv', '.xlsx', '.xls']
DEFAULT_ENCODING: str = 'utf-8'
DEFAULT_CSV_SEPARATOR: str = ','

# Data validation
MIN_ROWS_THRESHOLD: int = 10
MAX_MISSING_RATIO: float = 0.5  # Maximum 50% missing values allowed
MIN_UNIQUE_VALUES: int = 2  # Minimum unique values for categorical features

# ============================================================================
# MODEL TRAINING CONSTANTS
# ============================================================================

# Default model parameters
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_CV_SPLITS: int = 5

# Default target column (Korean accident type code)
DEFAULT_TARGET_COLUMN: str = '사고유형코드'

# Model types available
AVAILABLE_MODEL_TYPES: List[str] = ['lr', 'rf', 'xgboost']
DEFAULT_MODEL_TYPE: str = 'all'

# Cross-validation settings
DEFAULT_CV_SCORING: str = 'f1_macro'
DEFAULT_CV_SHUFFLE: bool = True
DEFAULT_CV_STRATIFY: bool = True

# ============================================================================
# SEVERITY MAPPING
# ============================================================================

# Accident severity mapping (0: Low, 1: Medium, 2: High)
# Note: These should be updated based on actual accident codes in your data
ACCIDENT_SEVERITY_MAPPING: Dict[str, int] = {
    # Low severity examples
    '210': 0,
    
    # Medium severity examples  
    '701': 1,
    '703': 1,
    
    # High severity examples
    '705': 2,
    '707': 2
}

# Default severity level for unmapped codes
DEFAULT_SEVERITY_LEVEL: int = -1

# ============================================================================
# FEATURE ENGINEERING CONSTANTS
# ============================================================================

# Feature engineering defaults
DEFAULT_POLYNOMIAL_DEGREE: int = 2
DEFAULT_PCA_VARIANCE_THRESHOLD: float = 0.95
DEFAULT_PCA_COMPONENTS: int = 10

# Feature selection
DEFAULT_FEATURE_SELECTION_METHOD: str = 'f_test'
DEFAULT_FEATURE_SELECTION_K: str = 'all'  # Can be int or 'all'

# Feature importance
DEFAULT_TOP_N_FEATURES: int = 20
MIN_FEATURE_IMPORTANCE: float = 0.001

# ============================================================================
# IMBALANCE HANDLING CONSTANTS
# ============================================================================

# SMOTE and imbalance handling
SMOTE_SAMPLING_STRATEGY: str = 'auto'
DEFAULT_IMBALANCE_METHOD: str = 'smote'
AVAILABLE_IMBALANCE_METHODS: List[str] = ['smote', 'class_weight', 'none']

# Class weight calculation methods
CLASS_WEIGHT_METHODS: List[str] = ['balanced', 'manual']

# ============================================================================
# EVALUATION CONSTANTS
# ============================================================================

# Evaluation metrics
DEFAULT_CLASSIFICATION_METRICS: List[str] = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
]

DEFAULT_REGRESSION_METRICS: List[str] = [
    'mse', 'rmse', 'mae', 'r2', 'explained_variance', 'mape'
]

# Model evaluation settings
DEFAULT_CLASSIFICATION_AVERAGE: str = 'macro'
ZERO_DIVISION_HANDLING: int = 0
DEFAULT_PREDICTION_THRESHOLD: float = 0.5

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================

# Visualization settings
DEFAULT_VISUALIZATION_FIGSIZE: Tuple[int, int] = (10, 8)
DEFAULT_DPI: int = 300
DEFAULT_PLOT_STYLE: str = 'seaborn-v0_8'

# Korean font support
KOREAN_FONT_FAMILY: str = 'NanumGothic'
KOREAN_FONT_FALLBACKS: List[str] = [
    'NanumGothic', 'Malgun Gothic', 'AppleGothic', 
    'Noto Sans CJK KR', 'DejaVu Sans'
]

# Color schemes
DEFAULT_COLOR_PALETTE: List[str] = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Plot parameters
DEFAULT_FONT_SIZE: int = 12
DEFAULT_TITLE_SIZE: int = 14
DEFAULT_LABEL_SIZE: int = 10

# ============================================================================
# LOGGING CONSTANTS
# ============================================================================

# Logging configuration
DEFAULT_LOG_LEVEL: str = 'INFO'
DEFAULT_LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

# Log file settings
LOG_FILE_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT: int = 5
LOG_FILE_ENCODING: str = 'utf-8'

# ============================================================================
# PIPELINE CONSTANTS
# ============================================================================

# Pipeline settings
DEFAULT_PIPELINE_STEPS: List[str] = [
    'preprocessing', 'feature_engineering', 'model_training', 'evaluation'
]
DEFAULT_SAVE_INTERMEDIATE: bool = True

# Processing settings
DEFAULT_N_JOBS: int = -1  # Use all available cores
DEFAULT_VERBOSE: int = 0   # Minimal output from sklearn models
DEFAULT_MEMORY_LIMIT: str = '4GB'

# ============================================================================
# HYPERPARAMETER SEARCH CONSTANTS
# ============================================================================

# Hyperparameter optimization
DEFAULT_SEARCH_METHOD: str = 'grid'  # 'grid', 'random', 'bayesian'
DEFAULT_SEARCH_CV: int = 3
DEFAULT_SEARCH_N_ITER: int = 50

# Search spaces for models
HYPERPARAMETER_SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    'lr': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000]
    },
    'rf': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Data validation thresholds
MIN_CORRELATION_THRESHOLD: float = 0.01
MAX_CORRELATION_THRESHOLD: float = 0.95
MIN_VARIANCE_THRESHOLD: float = 0.01

# Model validation
MIN_ACCURACY_THRESHOLD: float = 0.5
MIN_F1_THRESHOLD: float = 0.3
MAX_OVERFITTING_RATIO: float = 0.1  # Max diff between train and validation

# ============================================================================
# PERFORMANCE CONSTANTS
# ============================================================================

# Performance monitoring
MEMORY_USAGE_THRESHOLD: float = 0.8  # 80% memory usage warning
PROCESSING_TIME_THRESHOLD: int = 300  # 5 minutes warning
BATCH_SIZE_DEFAULT: int = 1000

# Caching settings
ENABLE_CACHING: bool = True
CACHE_SIZE_LIMIT: int = 100  # Number of cached items
CACHE_TTL: int = 3600  # 1 hour in seconds

# ============================================================================
# FILE NAMING CONVENTIONS
# ============================================================================

# File naming patterns
MODEL_FILE_PATTERN: str = "{model_type}_{timestamp}.joblib"
DATA_FILE_PATTERN: str = "{step}_{timestamp}.csv"
LOG_FILE_PATTERN: str = "{module}_{date}.log"
REPORT_FILE_PATTERN: str = "{report_type}_{timestamp}.json"

# File extensions
MODEL_EXTENSION: str = '.joblib'
DATA_EXTENSION: str = '.csv'
CONFIG_EXTENSION: str = '.json'
LOG_EXTENSION: str = '.log'

# ============================================================================
# ENVIRONMENT-SPECIFIC CONSTANTS
# ============================================================================

# Environment detection
PRODUCTION_INDICATORS: List[str] = ['prod', 'production', 'live']
DEVELOPMENT_INDICATORS: List[str] = ['dev', 'development', 'test']

# Resource limits by environment
RESOURCE_LIMITS: Dict[str, Dict[str, Any]] = {
    'development': {
        'max_memory_gb': 4,
        'max_cpu_cores': 2,
        'max_processing_time': 600  # 10 minutes
    },
    'production': {
        'max_memory_gb': 16,
        'max_cpu_cores': -1,  # All available
        'max_processing_time': 3600  # 1 hour
    }
} 