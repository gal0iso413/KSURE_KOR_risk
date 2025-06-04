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
    "n-select-features": 20,
    "optimize-hyperparameters": true,
    "search-method": "grid",
    "search-cv": 5,
    "search-n-iter": 50
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

#### **Hyperparameter Optimization:**
- `--optimize-hyperparameters`: Enable hyperparameter search (flag)
- `--search-method`: Search method - `grid` or `random` (default: `grid`)
- `--search-cv`: Number of CV folds for hyperparameter search (default: 3)
- `--search-n-iter`: Number of iterations for random search (default: 50)

## 🔍 Hyperparameter Optimization

The pipeline includes comprehensive hyperparameter optimization capabilities using predefined search spaces for optimal model performance.

### **Quick Start with Hyperparameter Search**

```bash
# Basic hyperparameter optimization (Grid Search)
python src/main.py \
    --input-path data/raw/merged_data.csv \
    --output-dir output \
    --optimize-hyperparameters

# Fast optimization using Random Search
python src/main.py \
    --input-path data/raw/merged_data.csv \
    --output-dir output \
    --optimize-hyperparameters \
    --search-method random \
    --search-n-iter 25

# Comprehensive optimization with more CV folds
python src/main.py \
    --input-path data/raw/merged_data.csv \
    --output-dir output \
    --optimize-hyperparameters \
    --search-method grid \
    --search-cv 5
```

### **Predefined Search Spaces**

The following hyperparameter search spaces are automatically used when optimization is enabled:

**Logistic Regression (`lr`):**
- `C`: [0.1, 1.0, 10.0, 100.0] - Regularization strength
- `solver`: ['lbfgs', 'liblinear'] - Optimization algorithm
- `max_iter`: [1000, 2000] - Maximum iterations

**Random Forest (`rf`):**
- `n_estimators`: [50, 100, 200] - Number of trees
- `max_depth`: [None, 10, 20, 30] - Maximum tree depth
- `min_samples_split`: [2, 5, 10] - Minimum samples to split
- `min_samples_leaf`: [1, 2, 4] - Minimum samples in leaf

**XGBoost (`xgboost`):**
- `n_estimators`: [50, 100, 200] - Number of boosting rounds
- `max_depth`: [3, 6, 9] - Maximum tree depth
- `learning_rate`: [0.01, 0.1, 0.2] - Learning rate
- `subsample`: [0.8, 0.9, 1.0] - Subsample ratio

### **Search Methods**

**Grid Search (`--search-method grid`):**
- **Exhaustive**: Tests all parameter combinations
- **Thorough**: Guarantees finding the best combination in the search space
- **Slower**: Can be time-consuming with large search spaces
- **Best for**: Small to medium parameter spaces, when computational time is not critical

**Random Search (`--search-method random`):**
- **Sampling**: Randomly samples parameter combinations
- **Faster**: More efficient for large search spaces
- **Configurable**: Set number of iterations with `--search-n-iter`
- **Best for**: Large parameter spaces, when time is limited

### **Configuration Examples**

#### **JSON Configuration with Hyperparameter Optimization**
```json
{
    "input-path": "data/raw/merged_data.csv",
    "output-dir": "output",
    "target-column": "사고유형코드",
    "mode": "full",
    "model-type": ["rf", "xgboost"],
    "optimize-hyperparameters": true,
    "search-method": "grid",
    "search-cv": 5,
    "search-n-iter": 100
}
```

#### **Programmatic Usage**
```python
from models.model_training import train_pipeline

# Train with hyperparameter optimization
results = train_pipeline(
    input_path="data/raw/merged_data.csv",
    output_dir="output",
    target_column="사고유형코드",
    model_types=['rf', 'xgboost'],
    optimize_hyperparameters=True,
    search_method='grid',
    search_cv=5
)

# Access optimization results
if 'hyperparameter_search' in results:
    for model_type, search_results in results['hyperparameter_search'].items():
        print(f"\n{model_type.upper()} Optimization Results:")
        print(f"Best Score: {search_results['best_score']:.4f}")
        print(f"Best Parameters: {search_results['best_params']}")
        print(f"Total Combinations Tested: {search_results['n_candidates']}")
```

### **Performance Considerations**

**Execution Time Estimates:**
- **Grid Search**: 10-60 minutes depending on parameter combinations and data size
- **Random Search**: 5-30 minutes with default settings (50 iterations)
- **CV Folds**: More folds = more accurate but slower (3-5 folds recommended)

**Resource Usage:**
- **Memory**: May require additional RAM for multiple model training
- **CPU**: Utilizes all available cores (`n_jobs=-1`)
- **Disk**: Saves best models and detailed search results

### **Output Structure with Hyperparameter Search**

```
output/
├── models/
│   ├── rf/
│   │   ├── best_model.joblib          # Optimized Random Forest
│   │   └── hyperparameter_results.json
│   ├── xgboost/
│   │   ├── best_model.joblib          # Optimized XGBoost
│   │   └── hyperparameter_results.json
│   └── best_model.joblib              # Overall best model
├── hyperparameter_search_results.json # Complete optimization results
└── logs/
    └── hyperparameter_search.log      # Detailed search logs
```

### **Best Practices**

1. **Start with Random Search**: For initial exploration, especially with large parameter spaces
2. **Use Grid Search for Fine-tuning**: Once you've identified promising regions
3. **Monitor Resource Usage**: Large searches can be memory and time intensive
4. **Adjust CV Folds**: Balance between accuracy (more folds) and speed (fewer folds)
5. **Check Search Results**: Review the `hyperparameter_search` section in results for insights

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
- **Automated hyperparameter optimization** (Grid Search & Random Search)
- **Predefined search spaces** for optimal performance
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

### Custom Hyperparameter Spaces

```python
from models.model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(output_dir="output")

# Custom parameter grid for Random Forest
custom_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10]
}

# Run hyperparameter search with custom parameters
best_model, search_results = trainer.hyperparameter_search(
    model_type='rf',
    X_train=X_train,
    y_train=y_train,
    custom_param_grid=custom_params,
    search_method='random',
    n_iter=50
)
```

### Custom Model Registration

```python
from models.models import ModelFactory
from sklearn.svm import SVC

# Register custom model
factory = ModelFactory()
factory.register_model('svm', SVC)
```

### Detailed Pipeline Control

```python
from models.model_training import train_pipeline

# Advanced configuration with hyperparameter optimization
results = train_pipeline(
    input_path="data/raw/merged_data.csv",
    output_dir="output",
    target_column="사고유형코드",
    model_types=['lr', 'rf', 'xgboost'],
    handle_imbalance='smote',
    test_size=0.2,
    random_state=42,
    optimize_hyperparameters=True,
    search_method='grid',
    search_cv=5,
    perform_cv=True,
    save_models=True
)

# Extract and analyze results
print("Model Performance Summary:")
for model_type, metrics in results['models'].items():
    if 'error' not in metrics:
        print(f"{model_type}: F1={metrics.get('f1', 'N/A'):.3f}")

if 'hyperparameter_search' in results:
    print("\nHyperparameter Optimization Results:")
    for model_type, search_info in results['hyperparameter_search'].items():
        print(f"{model_type}: Best Score={search_info['best_score']:.3f}")
        print(f"  Best Params: {search_info['best_params']}")
```

### Batch Processing with Optimization

```python
import pandas as pd
from pathlib import Path

# Process multiple datasets with hyperparameter optimization
datasets = ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']

for dataset in datasets:
    print(f"Processing {dataset}...")
    
    results = train_pipeline(
        input_path=f"data/{dataset}",
        output_dir=f"output/{dataset.stem}",
        target_column="사고유형코드",
        optimize_hyperparameters=True,
        search_method='random',
        search_n_iter=25  # Faster for batch processing
    )
    
    # Save results summary
    summary = {
        'dataset': dataset,
        'best_model': max(results['models'].items(), 
                         key=lambda x: x[1].get('f1', 0))[0],
        'hyperparameter_results': results.get('hyperparameter_search', {})
    }
    
    with open(f"output/{dataset.stem}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
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
