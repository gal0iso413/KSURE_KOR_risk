# Accident Severity Classification Pipeline

A machine learning pipeline for predicting accident severity levels using classification algorithms.

## 🏗️ Project Structure

```
first_model/
├── src/                          # Main source code
│   ├── main.py                   # Pipeline orchestration and CLI
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_engineering.py   # Feature creation and selection
│   ├── evaluate.py               # Model evaluation
│   ├── constants.py              # Project constants
│   └── utils/                    # Utility modules
│       ├── logging_config.py     # Logging configuration
│       └── common.py             # Common utilities
├── models/                       # Model-related modules
│   ├── models.py                 # Model definitions
│   ├── model_training.py         # Training pipeline
│   └── model_utils.py            # Model utilities and evaluation
├── tests/                        # Test modules
│   ├── test_model.py             # Model tests
│   └── test_preprocessing.py     # Processing tests
├── data/                         # Data directory
│   ├── raw/                      # Raw input data
│   └── processed/                # Processed outputs
├── config.json                   # Example configuration
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

#### **Complete Pipeline (Recommended)**
```bash
python src/main.py \
    --input-path data/raw/your_data.csv \
    --output-dir ./output \
    --target-column "사고유형코드" \
    --mode full
```

#### **Using Configuration File**
```bash
# Edit config.json with your settings, then:
python src/main.py --config config.json
```

#### **Step-by-Step Pipeline**
```bash
# 1. Preprocessing only
python src/main.py --input-path data/raw/your_data.csv --output-dir ./output --mode preprocess

# 2. Feature engineering (requires preprocessed data)
python src/main.py --input-path ./output/preprocessed_data.csv --output-dir ./output --mode feature

# 3. Model training (requires featured data)
python src/main.py --input-path ./output/featured_data.csv --output-dir ./output --mode train

# 4. Model evaluation (requires trained models)
python src/main.py --input-path ./output/featured_data.csv --output-dir ./output --mode evaluate
```

## 📋 Configuration

### **Command Line Options**

**Required:**
- `--input-path`: Path to input CSV file
- `--output-dir`: Directory for output files

**Common Options:**
- `--target-column`: Target column for prediction (default: "사고유형코드")
- `--mode`: Pipeline mode - `preprocess`, `feature`, `train`, `evaluate`, `full` (default: `full`)
- `--config`: Path to JSON configuration file
- `--random-state`: Random seed (default: 42)
- `--test-size`: Train/test split ratio (default: 0.2)

**Data Processing:**
- `--numeric-columns`: List of numeric column names
- `--categorical-columns`: List of categorical column names
- `--date-columns`: List of date column names
- `--identifier-columns`: Columns to keep as identifiers (excluded from processing)
- `--handle-outliers-cols`: Columns for outlier detection

**Model Training:**
- `--model-type`: Model types - `lr`, `rf`, `xgboost`, `all` (default: `all`)
- `--handle-imbalance`: Imbalance handling - `smote`, `class_weight`, `none` (default: `smote`)
- `--n-splits`: Cross-validation folds (default: 5)

**Hyperparameter Optimization:**
- `--optimize-hyperparameters`: Enable hyperparameter search
- `--search-method`: Search method - `grid`, `random` (default: `grid`)
- `--search-cv`: CV folds for search (default: 3)
- `--search-n-iter`: Iterations for random search (default: 50)

### **JSON Configuration Example**

```json
{
    "input-path": "data/raw/merged_data.csv",
    "output-dir": "output",
    "target-column": "사고유형코드",
    "mode": "full",
    "model-type": "all",
    "handle-imbalance": "smote",
    "numeric-columns": ["asset", "liability", "sales", "profit", "money"],
    "categorical-columns": ["industry_code", "region_code", "company_size"],
    "date-columns": ["frdate", "accdate"],
    "identifier-columns": ["사고번호", "기업번호", "사고접수일자"],
    "test-size": 0.2,
    "random-state": 42,
    "n-splits": 5,
    "optimize-hyperparameters": true,
    "search-method": "grid"
}
```

## 🛠️ Features

**Data Preprocessing:**
- Automatic data type detection
- Missing value imputation
- Outlier detection and handling
- Feature scaling and encoding

**Feature Engineering:**
- Interaction features
- Polynomial features
- Date feature extraction
- PCA dimensionality reduction
- Feature selection

**Model Training:**
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost
- Cross-validation
- Imbalanced data handling (SMOTE, class weighting)
- Hyperparameter optimization

**Evaluation:**
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Visualizations (confusion matrix, ROC curves, feature importance)
- Model comparison

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

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 📝 Example Usage

### Simple Classification Task
```bash
python src/main.py \
    --input-path data/accidents.csv \
    --output-dir results \
    --target-column severity \
    --numeric-columns age income \
    --categorical-columns region type
```

### Advanced Usage with Hyperparameter Optimization
```bash
python src/main.py \
    --input-path data/accidents.csv \
    --output-dir results \
    --target-column severity \
    --optimize-hyperparameters \
    --search-method random \
    --search-n-iter 25 \
    --model-type rf xgboost
```

### Programmatic Usage
```python
from models.model_training import train_pipeline

results = train_pipeline(
    input_path="data/accidents.csv",
    output_dir="results",
    target_column="severity",
    model_types=['rf', 'xgboost'],
    optimize_hyperparameters=True
)
```

## ⚠️ Notes

- Default target column is "사고유형코드" (Korean accident type code)
- Identifier columns are preserved in output but excluded from model training
- Hyperparameter optimization can significantly increase runtime
- All models are saved with joblib for later use

## 🤝 Contributing

1. Follow existing code structure and conventions
2. Add appropriate tests for new functionality
3. Update this README for significant changes
