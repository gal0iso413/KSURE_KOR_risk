#!/usr/bin/env python3
"""
Example script demonstrating hyperparameter search functionality.

This script shows how to run hyperparameter optimization using the 
HYPERPARAMETER_SEARCH_SPACES defined in constants.py.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from src.main import main

def demonstrate_hyperparameter_search():
    """
    Demonstrate how to run hyperparameter search.
    
    This function shows different ways to use the hyperparameter optimization.
    """
    
    print("=== Hyperparameter Search Demonstration ===\n")
    
    print("1. Basic Usage (Grid Search):")
    print("python src/main.py --input-path data/your_data.csv --output-dir output --optimize-hyperparameters")
    print()
    
    print("2. Using Random Search:")
    print("python src/main.py --input-path data/your_data.csv --output-dir output --optimize-hyperparameters --search-method random --search-n-iter 25")
    print()
    
    print("3. With specific model types:")
    print("python src/main.py --input-path data/your_data.csv --output-dir output --optimize-hyperparameters --model-type rf xgboost")
    print()
    
    print("4. Full pipeline with hyperparameter optimization:")
    print("python src/main.py --input-path data/your_data.csv --output-dir output --mode full --optimize-hyperparameters --search-method grid --search-cv 5")
    print()
    
    print("5. Programmatic usage:")
    print("""
from models.model_training import train_pipeline

results = train_pipeline(
    input_path='data/your_data.csv',
    output_dir='output',
    target_column='사고유형코드',
    model_types=['rf', 'xgboost'],
    optimize_hyperparameters=True,
    search_method='grid',
    search_cv=5
)

# Access hyperparameter search results
if 'hyperparameter_search' in results:
    for model_type, search_results in results['hyperparameter_search'].items():
        print(f"{model_type} best score: {search_results['best_score']}")
        print(f"{model_type} best params: {search_results['best_params']}")
    """)
    
    print("\n=== Available Hyperparameter Search Spaces ===")
    print("The following hyperparameter spaces are defined in constants.py:")
    
    from src.constants import HYPERPARAMETER_SEARCH_SPACES
    
    for model_type, param_space in HYPERPARAMETER_SEARCH_SPACES.items():
        print(f"\n{model_type.upper()}:")
        for param, values in param_space.items():
            print(f"  {param}: {values}")
    
    print("\n=== Configuration Options ===")
    print("--optimize-hyperparameters: Enable hyperparameter optimization")
    print("--search-method: 'grid' (exhaustive) or 'random' (sampling)")
    print("--search-cv: Number of cross-validation folds (default: 3)")
    print("--search-n-iter: Number of iterations for random search (default: 50)")
    
    print("\n=== Results ===")
    print("When hyperparameter search is enabled, results will include:")
    print("- 'hyperparameter_search' section with best parameters and scores")
    print("- Detailed CV results for each parameter combination")
    print("- Models trained with optimized hyperparameters")

if __name__ == "__main__":
    demonstrate_hyperparameter_search() 