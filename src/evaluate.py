"""
Model evaluation pipeline for the accident severity classification project.

This module provides pipeline functions for comprehensive model evaluation,
loading saved models, and generating evaluation reports. The actual evaluation
classes and functionality are in models.model_utils to avoid duplication.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import joblib
import json

# Import project utilities and constants
from utils.logging_config import get_logger
from utils.common import (
    safe_load_csv, safe_save_csv, create_directory_if_not_exists,
    standardize_error_handling, validate_file_exists
)
from constants import DEFAULT_ENCODING, DEFAULT_TOP_N_FEATURES

# Import evaluation components from models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_utils import ModelEvaluator, DataProcessor
from models.models import ModelFactory

# Initialize logger
logger = get_logger(__name__)


class ModelLoader:
    """Class for loading and managing saved models."""
    
    def __init__(self):
        """Initialize model loader."""
        self.model_factory = ModelFactory()
    
    @standardize_error_handling
    def load_model(self, model_path: Union[str, Path]) -> Any:
        """
        Load a trained model from file with comprehensive error handling.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        model_path = Path(model_path)
        
        if not validate_file_exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Try loading with joblib (our format)
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict) and 'model' in model_data:
                # New format with metadata
                model = model_data['model']
                model_name = model_data.get('model_name', 'unknown')
                logger.info(f"Loaded {model_name} model with metadata from {model_path}")
            else:
                # Old format or direct model
                model = model_data
                logger.info(f"Loaded model from {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    @standardize_error_handling
    def load_multiple_models(self, model_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Load multiple models from a directory.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Dictionary mapping model names to loaded models
        """
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        models = {}
        model_files = list(model_dir.glob("*.joblib"))
        
        if not model_files:
            logger.warning(f"No .joblib model files found in {model_dir}")
            return models
        
        for model_file in model_files:
            try:
                model_name = model_file.stem
                model = self.load_model(model_file)
                models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
        
        logger.info(f"Successfully loaded {len(models)} models from {model_dir}")
        return models


class EvaluationPipeline:
    """Pipeline for comprehensive model evaluation."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize evaluation pipeline.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = create_directory_if_not_exists(output_dir, "Evaluation results")
        self.model_loader = ModelLoader()
        self.data_processor = DataProcessor()
        
        logger.info(f"EvaluationPipeline initialized with output_dir: {self.output_dir}")
    
    @standardize_error_handling
    def evaluate_single_model(
        self,
        model_path: Union[str, Path],
        test_data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.
        
        Args:
            model_path: Path to saved model
            test_data: Test dataset
            target_column: Name of target column
            feature_columns: List of feature columns (None = all except target)
            model_name: Custom name for the model (None = derive from path)
            
        Returns:
            Evaluation results dictionary
        """
        # Load model
        model = self.model_loader.load_model(model_path)
        
        # Determine model name
        if model_name is None:
            model_name = Path(model_path).stem
        
        logger.info(f"Evaluating model '{model_name}' on {len(test_data)} test samples")
        
        # Prepare data
        if feature_columns is None:
            feature_columns = [col for col in test_data.columns if col != target_column]
        
        # Validate columns exist
        missing_features = [col for col in feature_columns if col not in test_data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in test data: {missing_features}")
        
        if target_column not in test_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in test data")
        
        # Extract features and target
        X_test = test_data[feature_columns]
        y_test_raw = test_data[target_column]
        
        # Convert target to severity levels
        y_test = self.data_processor.convert_to_severity_levels(y_test_raw)
        
        # Validate data
        X_test, y_test = self.data_processor.validate_features_and_target(X_test, y_test)
        
        # Create evaluator and evaluate
        evaluator = ModelEvaluator(self.output_dir)
        results = evaluator.evaluate_classification_model(
            model, X_test, y_test, feature_columns, model_name
        )
        
        logger.info(f"Evaluation completed for model '{model_name}'")
        return results
    
    @standardize_error_handling
    def evaluate_multiple_models(
        self,
        model_dir: Union[str, Path],
        test_data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple models from a directory.
        
        Args:
            model_dir: Directory containing saved models
            test_data: Test dataset
            target_column: Name of target column
            feature_columns: List of feature columns
            
        Returns:
            Comprehensive evaluation results for all models
        """
        logger.info(f"Evaluating multiple models from {model_dir}")
        
        # Load all models
        models = self.model_loader.load_multiple_models(model_dir)
        
        if not models:
            raise ValueError(f"No models found in directory: {model_dir}")
        
        # Prepare common data once
        if feature_columns is None:
            feature_columns = [col for col in test_data.columns if col != target_column]
        
        X_test = test_data[feature_columns]
        y_test_raw = test_data[target_column]
        y_test = self.data_processor.convert_to_severity_levels(y_test_raw)
        X_test, y_test = self.data_processor.validate_features_and_target(X_test, y_test)
        
        # Evaluate each model
        all_results = {}
        evaluator = ModelEvaluator(self.output_dir)
        
        for model_name, model in models.items():
            try:
                logger.info(f"Evaluating model: {model_name}")
                
                results = evaluator.evaluate_classification_model(
                    model, X_test, y_test, feature_columns, model_name
                )
                all_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        # Generate model comparison
        comparison_df = evaluator.compare_models(all_results)
        
        # Compile comprehensive results
        final_results = {
            'evaluation_summary': {
                'total_models': len(models),
                'successful_evaluations': len([r for r in all_results.values() if 'error' not in r]),
                'test_samples': len(test_data),
                'n_features': len(feature_columns),
                'target_column': target_column
            },
            'individual_results': all_results,
            'model_comparison': comparison_df.to_dict() if not comparison_df.empty else {}
        }
        
        logger.info(f"Completed evaluation of {len(models)} models")
        return final_results
    
    @standardize_error_handling
    def generate_evaluation_report(
        self,
        results: Dict[str, Any],
        report_name: str = "evaluation_report"
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            report_name: Name for the report file
            
        Returns:
            Path to generated report
        """
        report_path = self.output_dir / f"{report_name}.json"
        
        # Add metadata to results
        enhanced_results = {
            'metadata': {
                'report_generated_at': pd.Timestamp.now().isoformat(),
                'evaluation_output_dir': str(self.output_dir)
            },
            **results
        }
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        logger.info(f"Generated evaluation report: {report_path}")
        return str(report_path)


@standardize_error_handling
def evaluate_model_pipeline(
    model_path: Union[str, Path],
    test_data_path: Union[str, Path],
    target_column: str,
    output_dir: Union[str, Path],
    feature_columns: Optional[List[str]] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete pipeline for evaluating a single model.
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data CSV
        target_column: Name of target column
        output_dir: Directory for saving evaluation results
        feature_columns: List of feature columns to use
        model_name: Custom name for the model
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Starting single model evaluation pipeline")
    logger.info(f"Model: {model_path}, Data: {test_data_path}")
    
    # Load test data
    test_data = safe_load_csv(test_data_path, encoding=DEFAULT_ENCODING)
    logger.info(f"Loaded test data with shape {test_data.shape}")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(output_dir)
    
    # Evaluate model
    results = pipeline.evaluate_single_model(
        model_path, test_data, target_column, feature_columns, model_name
    )
    
    # Generate report
    report_path = pipeline.generate_evaluation_report(results, "single_model_evaluation")
    
    logger.info(f"Single model evaluation pipeline completed. Report: {report_path}")
    return results


@standardize_error_handling
def evaluate_multiple_models_pipeline(
    model_dir: Union[str, Path],
    test_data_path: Union[str, Path],
    target_column: str,
    output_dir: Union[str, Path],
    feature_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Complete pipeline for evaluating multiple models.
    
    Args:
        model_dir: Directory containing saved models
        test_data_path: Path to test data CSV
        target_column: Name of target column
        output_dir: Directory for saving evaluation results
        feature_columns: List of feature columns to use
        
    Returns:
        Comprehensive evaluation results for all models
    """
    logger.info(f"Starting multiple models evaluation pipeline")
    logger.info(f"Models dir: {model_dir}, Data: {test_data_path}")
    
    # Load test data
    test_data = safe_load_csv(test_data_path, encoding=DEFAULT_ENCODING)
    logger.info(f"Loaded test data with shape {test_data.shape}")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(output_dir)
    
    # Evaluate all models
    results = pipeline.evaluate_multiple_models(
        model_dir, test_data, target_column, feature_columns
    )
    
    # Generate comprehensive report
    report_path = pipeline.generate_evaluation_report(results, "multiple_models_evaluation")
    
    logger.info(f"Multiple models evaluation pipeline completed. Report: {report_path}")
    return results


@standardize_error_handling
def compare_models_performance(
    evaluation_results: Dict[str, Any],
    metrics: List[str] = ['accuracy', 'f1_macro', 'f1_weighted'],
    output_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Create a detailed comparison of model performance.
    
    Args:
        evaluation_results: Results from evaluate_multiple_models_pipeline
        metrics: List of metrics to compare
        output_dir: Optional directory to save comparison CSV
        
    Returns:
        DataFrame with model performance comparison
    """
    logger.info("Creating detailed model performance comparison")
    
    # Extract individual results
    individual_results = evaluation_results.get('individual_results', {})
    
    comparison_data = []
    for model_name, results in individual_results.items():
        if 'error' not in results:
            row = {'Model': model_name}
            
            # Add requested metrics
            for metric in metrics:
                row[metric.title()] = results.get(metric, 'N/A')
            
            # Add additional useful info
            row['Sample_Count'] = results.get('sample_count', 'N/A')
            
            comparison_data.append(row)
    
    if not comparison_data:
        logger.warning("No valid results found for comparison")
        return pd.DataFrame()
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by primary metric (first in list)
    if metrics and len(comparison_df) > 1:
        primary_metric = metrics[0].title()
        if primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        comparison_path = output_dir / "detailed_model_comparison.csv"
        safe_save_csv(comparison_df, comparison_path, "Detailed model comparison")
        logger.info(f"Saved detailed comparison: {comparison_path}")
    
    return comparison_df
