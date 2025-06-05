#!/usr/bin/env python3
"""
Simple utility to analyze existing model evaluation results.
Works with saved predictions and probabilities to show top N data points for any class.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

def analyze_existing_predictions(
    model_path: str,
    test_data_path: str, 
    target_class: int,
    top_n: int = 10,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze existing model to show top N predictions for a specific class.
    
    Args:
        model_path: Path to saved model (.joblib)
        test_data_path: Path to test data (same data used for evaluation)
        target_class: Class to analyze (0=Low, 1=Medium, 2=High severity)
        top_n: Number of top predictions to show
        output_path: Optional path to save results CSV
        
    Returns:
        DataFrame with top N predictions and all original data
    """
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading test data from: {test_data_path}")
    df = pd.read_csv(test_data_path)
    
    # Separate features from identifier/target columns
    # Assume target column is named 'severity' or similar, and identifier columns are non-numeric
    target_col = 'severity'  # Adjust based on your data
    if target_col not in df.columns:
        # Try to find target column
        possible_targets = ['severity', 'target', 'label', 'class']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError("Could not find target column. Please specify it.")
    
    # Get feature columns (exclude target and likely identifier columns)
    identifier_cols = []
    feature_cols = []
    
    for col in df.columns:
        if col == target_col:
            continue
        # Keep non-numeric columns as identifiers, numeric as features
        if df[col].dtype in ['object', 'string']:
            identifier_cols.append(col)
        else:
            feature_cols.append(col)
    
    print(f"Found {len(feature_cols)} feature columns and {len(identifier_cols)} identifier columns")
    
    # Prepare features for prediction
    X = df[feature_cols].values
    y_true = df[target_col].values
    
    # Get predictions and probabilities
    print("Making predictions...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Create result DataFrame with all original data
    result_df = df.copy()
    result_df['predicted_class'] = y_pred
    result_df['true_class'] = y_true
    
    # Add probability columns
    class_names = ['Low_Severity_Prob', 'Medium_Severity_Prob', 'High_Severity_Prob']
    for i, prob_col in enumerate(class_names):
        if i < y_proba.shape[1]:
            result_df[prob_col] = y_proba[:, i]
    
    # Get top N for target class
    target_prob_col = class_names[target_class]
    print(f"Finding top {top_n} predictions for class {target_class} ({target_prob_col})")
    
    top_predictions = result_df.nlargest(top_n, target_prob_col)
    
    # Add ranking
    top_predictions = top_predictions.copy()
    top_predictions['rank'] = range(1, len(top_predictions) + 1)
    
    # Reorder columns to show most important info first
    priority_cols = ['rank', target_prob_col, 'predicted_class', 'true_class']
    other_cols = [col for col in top_predictions.columns if col not in priority_cols]
    final_cols = priority_cols + other_cols
    
    result = top_predictions[final_cols]
    
    # Print summary
    print(f"\nTop {top_n} predictions for class {target_class}:")
    print(f"Probability range: {result[target_prob_col].min():.4f} - {result[target_prob_col].max():.4f}")
    print(f"Correct predictions: {(result['predicted_class'] == result['true_class']).sum()}/{len(result)}")
    
    # Save if requested
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Analyze existing model predictions')
    parser.add_argument('--model-path', required=True, help='Path to saved model (.joblib)')
    parser.add_argument('--test-data', required=True, help='Path to test data CSV')
    parser.add_argument('--target-class', type=int, required=True, 
                       help='Class to analyze (0=Low, 1=Medium, 2=High)')
    parser.add_argument('--top-n', type=int, default=10, 
                       help='Number of top predictions to show')
    parser.add_argument('--output', help='Output CSV path (optional)')
    
    args = parser.parse_args()
    
    try:
        result = analyze_existing_predictions(
            model_path=args.model_path,
            test_data_path=args.test_data,
            target_class=args.target_class,
            top_n=args.top_n,
            output_path=args.output
        )
        
        print(f"\nShowing first 5 rows of top {args.top_n} predictions:")
        print(result.head().to_string())
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 