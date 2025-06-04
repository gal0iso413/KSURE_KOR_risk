# First Model: Accident Severity Classification Pipeline

A comprehensive machine learning pipeline for predicting accident severity levels using various classification algorithms.

## 🏗️ Project Structure

```
first_model/
├── src/                          # Main source code
│   ├── main.py                   # Pipeline orchestration and CLI
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_engineering.py   # Feature creation and selection
│   ├── evaluate.py               # Model evaluation pipeline
│   ├── constants.py              # Project constants and configuration
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── logging_config.py     # Centralized logging
│       └── common.py             # Common utilities
├── models/                       # Model-related modules
│   ├── __init__.py
│   ├── models.py                 # Model definitions and factory
│   ├── model_training.py         # Training pipeline
│   └── model_utils.py            # Model utilities and evaluation
├── tests/                        # Test modules
│   ├── __init__.py
│   ├── test_model.py             # Comprehensive model tests
│   └── test_preprocessing.py     # Utility and processing tests
├── notebooks/                    # Jupyter notebooks (optional)
├── data/                         # Data directory
│   ├── raw/                      # Raw input data
│   └── processed/                # Processed data outputs
├── logs/                         # Log files (auto-created)
├── output/                       # Pipeline outputs (auto-created)
├── config.json                   # Configuration file (optional)
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
```bash
# Ensure your CSV data file is accessible
# Default expected column name: '사고유형코드' (accident type code)
```

### Basic Usage

#### **Complete Pipeline (Recommended)**
```bash
python src/main.py \
    --input-path data/raw/merged_data.csv \
    --output-dir ./output \
    --target-column "사고유형코드" \
    --mode full
```

#### **Step-by-Step Pipeline**
```bash
# 1. Preprocessing only
python src/main.py \
    --input-path data/raw/merged_data.csv \
    --output-dir ./output \
    --mode preprocess

# 2. Feature engineering only (requires preprocessed data)
python src/main.py \
    --input-path ./output/preprocessed_data.csv \
    --output-dir ./output \
    --mode feature

# 3. Model training only (requires featured data)
python src/main.py \
    --input-path ./output/featured_data.csv \
    --output-dir ./output \
    --mode train

# 4. Model evaluation only (requires trained models)
python src/main.py \
    --input-path ./output/featured_data.csv \
    --output-dir ./output \
    --mode evaluate
```

#### **Using Configuration File**
```bash
# Create config.json (see Configuration section for examples)
python src/main.py --config config.json
```

#### **Advanced Usage with Custom Settings**
```bash
python src/main.py \
    --input-path data/raw/merged_data.csv \
    --output-dir ./output \
    --target-column "사고유형코드" \
    --mode full \
    --model-type all \
    --handle-imbalance smote \
    --numeric-columns money asset liability sales \
    --categorical-columns region type category \
    --date-columns frdate accdate \
    --columns-to-drop 사고번호 기업번호 사고접수일자 \
    --test-size 0.2 \
    --random-state 42 \
    --n-splits 5
```

## 📋 Configuration

The pipeline supports multiple configuration methods:

### **1. Command Line Arguments (Primary Method)**

```bash
# Basic usage
python src/main.py \
    --input-path data/raw/merged_data.csv \
    --output-dir output \
    --target-column "사고유형코드" \
    --mode full

# Advanced usage with all options
python src/main.py \
    --input-path data/raw/merged_data.csv \
    --output-dir output \
    --target-column "사고유형코드" \
    --mode full \
    --model-type all \
    --handle-imbalance smote \
    --numeric-columns feature1 feature2 money \
    --categorical-columns type category \
    --date-columns frdate accdate \
    --columns-to-drop 사고번호 기업번호 \
    --test-size 0.2 \
    --random-state 42 \
    --n-splits 5
```

### **2. JSON Configuration File**

Create a JSON config file and use it with the `--config` option:

**config.json:**
```json
{
    "input-path": "data/raw/merged_data.csv",
    "output-dir": "output",
    "target-column": "사고유형코드",
    "mode": "full",
    "model-type": "all",
    "handle-imbalance": "smote",
    "numeric-columns": ["money", "asset", "liability", "sales", "profit"],
    "categorical-columns": ["region", "industry_type"],
    "date-columns": ["frdate", "accdate"],
    "columns-to-drop": ["사고번호", "기업번호", "사고접수일자"],
    "handle-outliers-cols": ["money", "asset", "liability", "sales", "profit"],
    "test-size": 0.2,
    "random-state": 42,
    "n-splits": 5,
    "interaction-features": ["asset", "liability", "sales", "profit"],
    "polynomial-features": ["money", "asset", "sales"],
    "polynomial-degree": 2,
    "pca-components": null,
    "n-select-features": 20
}
```

**Usage:**
```bash
python src/main.py --config config.json
```

### **3. Programmatic Usage (Advanced)**

For integration into other Python projects:

```python
# Import pipeline functions directly
from models.model_training import train_pipeline
from src.evaluate import evaluate_multiple_models_pipeline

# Train models
results = train_pipeline(
    input_path="data/raw/merged_data.csv",
    output_dir="./output",
    target_column="사고유형코드",
    model_types=['lr', 'rf', 'xgboost'],
    handle_imbalance='smote',
    test_size=0.2,
    random_state=42
)

# Evaluate models  
eval_results = evaluate_multiple_models_pipeline(
    model_dir="./output",
    test_data_path="data/test_data.csv",
    target_column="사고유형코드",
    output_dir="./evaluation"
)
```

### **Available Command Line Options**

#### **Required Arguments:**
- `--input-path`: Path to input CSV data file
- `--output-dir`: Directory for saving results

#### **Pipeline Configuration:**
- `--target-column`: Target column name for prediction (default: "사고유형코드")
- `--mode`: Pipeline mode - `preprocess`, `feature`, `train`, `evaluate`, or `full` (default: `full`)
- `--config`: Path to JSON configuration file
- `--random-state`: Random seed for reproducibility (default: 42)

#### **Data Preprocessing:**
- `--numeric-columns`: List of numeric column names
- `--categorical-columns`: List of categorical column names  
- `--date-columns`: List of date column names
- `--columns-to-drop`: Columns to exclude from analysis
- `--handle-outliers-cols`: Columns for outlier detection
- `--test-size`: Train/test split ratio (default: 0.2)

#### **Feature Engineering:**
- `--interaction-features`: Features for creating interactions (pairs)
- `--polynomial-features`: Columns for polynomial feature generation
- `--polynomial-degree`: Degree for polynomial features (default: 2)
- `--pca-components`: Number of PCA components (optional)
- `--n-select-features`: Number of top features to select

#### **Model Training:**
- `--model-type`: Model types to train - `lr`, `rf`, `xgboost`, or `all` (default: `all`)
- `--handle-imbalance`: Imbalance handling method - `smote`, `class_weight`, or `none` (default: `smote`)
- `--n-splits`: Number of cross-validation folds (default: 5)

## 🛠️ Features

### Data Preprocessing
- Automatic data type detection and conversion
- Missing value imputation (median for numeric, mode for categorical)
- Outlier detection and handling (IQR and Z-score methods)
- Feature scaling (StandardScaler, MinMaxScaler)
- Categorical encoding (One-hot, Label encoding)

### Feature Engineering
- Interaction features between numeric variables
- Polynomial features
- Date feature extraction (year, month, quarter, day of week)
- Principal Component Analysis (PCA)
- Statistical feature selection

### Model Training
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost
- Cross-validation with stratified splits
- Imbalanced data handling (SMOTE, class weighting)
- Hyperparameter optimization
- Model persistence and loading

### Evaluation
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Visualization (confusion matrix, ROC curves, feature importance)
- Model comparison and selection
- Performance reporting

## 📊 Output Structure

```
output/
├── logs/                         # Execution logs
├── models/                       # Trained models
│   ├── best_model.joblib
│   ├── logistic_regression/
│   ├── random_forest/
│   └── xgboost/
├── evaluation/                   # Evaluation results
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── evaluation_results.json
├── preprocessed_data.csv         # Cleaned data
└── featured_data.csv             # Engineered features
```

## 🔧 Advanced Usage

### Custom Model Registration

```python
from models.models import ModelFactory
from sklearn.svm import SVC

# Register custom model
factory = ModelFactory()
factory.register_model('svm', SVC)
```

### Programmatic Usage

```python
from src.main import run_preprocessing, run_feature_engineering
from src.config import Config

# Load configuration
config = Config('config_local.json')

# Run specific pipeline steps
preprocessed_path = run_preprocessing(args)
featured_path = run_feature_engineering(preprocessed_path, args)
```

## 🧪 Testing

Run tests with:
```bash
python -m pytest tests/ -v
```

## 📝 Logging

The pipeline provides comprehensive logging:
- Console output for real-time monitoring
- File logs saved to `output/logs/`
- Configurable log levels
- Structured log messages

## ⚠️ Error Handling

- Custom exception classes for different error types
- Graceful error handling with detailed messages
- Input validation and file existence checks
- Safe file operations with automatic directory creation

## 🤝 Contributing

1. Follow the existing code structure and naming conventions
2. Add appropriate type hints and documentation
3. Include tests for new functionality
4. Update this README for significant changes

## 📄 License

[Add your license information here] 