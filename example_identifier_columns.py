#!/usr/bin/env python3
"""
Example script demonstrating how to use the identifier columns functionality.

This allows you to keep certain columns in your dataset as row identifiers
without using them in model processing (preprocessing, feature engineering, training).
"""

import subprocess
import json

def run_with_identifier_columns():
    """
    Example: Keep ID columns as identifiers but exclude from processing
    """
    
    # Your identifier columns (keep but don't process)
    identifier_columns = [
        "row_id",           # Database primary key  
        "customer_id",      # Customer identifier
        "transaction_id",   # Transaction identifier
        "timestamp",        # Raw timestamp for reference
        "record_source"     # Source system identifier
    ]
    
    # Your processing columns
    numeric_columns = [
        "amount", "balance", "credit_score", "income"
    ]
    
    categorical_columns = [
        "product_type", "region", "status"  
    ]
    
    # Run pipeline
    cmd = [
        "python", "src/main.py",
        "--input-path", "data/your_dataset.csv",
        "--output-dir", "output/with_identifiers", 
        "--target-column", "risk_level",
        
        # NEW: Specify identifier columns (kept but not processed)
        "--identifier-columns"] + identifier_columns + [
        
        # Specify columns for processing
        "--numeric-columns"] + numeric_columns + [
        "--categorical-columns"] + categorical_columns + [
        
        # Pipeline settings
        "--mode", "full",
        "--model-type", "all"
    ]
    
    print("Running pipeline with identifier columns...")
    print(f"Identifier columns (kept but not processed): {identifier_columns}")
    print(f"Processing columns: {numeric_columns + categorical_columns}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Pipeline completed successfully!")
        print("Your identifier columns are preserved in all output files")
        print("But they were excluded from preprocessing, feature engineering, and training")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        print("Error:", e.stderr)

def run_with_config_file():
    """
    Alternative: Use JSON config file
    """
    config = {
        "input-path": "data/your_dataset.csv",
        "output-dir": "output/config_identifiers",
        "target-column": "risk_level",
        "mode": "full",
        
        # Identifier columns (kept as row identifiers, not processed)
        "identifier-columns": [
            "row_id", "customer_id", "transaction_id", 
            "timestamp", "record_source"
        ],
        
        # Processing columns
        "numeric-columns": ["amount", "balance", "credit_score", "income"],
        "categorical-columns": ["product_type", "region", "status"],
        
        # Model settings
        "model-type": "all",
        "handle-imbalance": "smote"
    }
    
    # Save config
    with open("config_identifiers.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run with config
    cmd = ["python", "src/main.py", "--config", "config_identifiers.json"]
    
    print("Running with identifier config file...")
    print("Config saved to: config_identifiers.json")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Pipeline completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")

def show_concept():
    """
    Explain the identifier columns concept
    """
    print("=== Identifier Columns Concept ===")
    print()
    print("--identifier-columns row_id customer_id timestamp")
    print("  ↓")
    print("• Columns are KEPT in all output files")
    print("• Columns are EXCLUDED from preprocessing")  
    print("• Columns are EXCLUDED from feature engineering")
    print("• Columns are EXCLUDED from model training")
    print()
    print("Perfect for: IDs, timestamps, reference data, tracking info")

if __name__ == "__main__":
    print("=== Identifier Columns Example ===")
    print()
    print("This shows how to keep certain columns as row identifiers")
    print("while excluding them from all processing steps.")
    print()
    
    # Show the concept
    show_concept()
    print()
    
    print("Uncomment one of these to try:")
    print("# run_with_identifier_columns()")  
    print("# run_with_config_file()")
    print()
    print("Update the file paths and column names for your dataset.") 