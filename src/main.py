#!/usr/bin/env python3
"""
Main pipeline orchestration for the accident severity classification project.

This module provides the command-line interface and coordinates the execution
of preprocessing, feature engineering, training, and evaluation pipelines.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the parent directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from constants import (
    DEFAULT_TARGET_COLUMN, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE,
    DEFAULT_CV_SPLITS, AVAILABLE_MODEL_TYPES, DEFAULT_DATE_COLUMNS,
    AVAILABLE_SEARCH_METHODS
)
from utils.logging_config import get_logger
from utils.common import (
    validate_file_exists, create_directory_if_not_exists,
    standardize_error_handling, safe_load_csv, safe_save_csv
)
from data_preprocessing import preprocess_pipeline
from feature_engineering import engineer_features
from models.model_training import train_pipeline
from evaluate import evaluate_model_pipeline

# Initialize logger
logger = get_logger(__name__)


class PipelineConfig:
    """Configuration class for pipeline parameters with validation."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize configuration from parsed arguments."""
        self.input_path = args.input_path
        self.output_dir = args.output_dir
        self.target_column = args.target_column
        self.mode = args.mode
        self.random_state = args.random_state
        self.test_size = args.test_size
        
        # Data processing parameters
        self.numeric_columns = args.numeric_columns or []
        self.categorical_columns = args.categorical_columns or []
        self.date_columns = args.date_columns or DEFAULT_DATE_COLUMNS
        self.identifier_columns = args.identifier_columns or []
        self.handle_outliers_cols = args.handle_outliers_cols or []
        
        # Feature engineering parameters
        self.interaction_features = self._parse_interaction_features(args.interaction_features)
        self.polynomial_features = args.polynomial_features or []
        self.polynomial_degree = args.polynomial_degree
        self.pca_components = args.pca_components
        self.n_select_features = args.n_select_features
        
        # Model parameters
        self.model_type = args.model_type
        self.handle_imbalance = args.handle_imbalance
        self.n_splits = args.n_splits
        self.perform_cv = args.n_splits >= 2  # Enable CV only if n_splits >= 2
        
        # Hyperparameter optimization parameters
        self.optimize_hyperparameters = args.optimize_hyperparameters
        self.search_method = args.search_method
        self.search_cv = args.search_cv
        self.search_n_iter = args.search_n_iter
        
        self._validate_config()
    
    def _parse_interaction_features(self, interaction_features: Optional[list]) -> Optional[list]:
        """Parse interaction features into feature pairs."""
        if not interaction_features:
            return None
        
        feature_pairs = []
        for i in range(0, len(interaction_features), 2):
            if i + 1 < len(interaction_features):
                feature_pairs.append((interaction_features[i], interaction_features[i+1]))
            else:
                logger.warning(f"Odd number of interaction features, ignoring last one: {interaction_features[i]}")
        
        return feature_pairs if feature_pairs else None
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate required paths
        if not self.input_path:
            raise ValueError("Input path is required")
        if not self.output_dir:
            raise ValueError("Output directory is required")
        
        # Validate file exists
        validate_file_exists(self.input_path, "Input data file")
        
        # Create output directory
        create_directory_if_not_exists(self.output_dir, "Output directory")
        
        # Validate numeric parameters
        if not (0 < self.test_size < 1):
            raise ValueError(f"Test size must be between 0 and 1, got {self.test_size}")
        
        if self.n_splits < 0:
            raise ValueError(f"Number of CV splits must be >= 0, got {self.n_splits}")
        
        if self.polynomial_degree < 1:
            raise ValueError(f"Polynomial degree must be >= 1, got {self.polynomial_degree}")
        
        # Validate hyperparameter search settings
        if self.optimize_hyperparameters:
            if self.search_method not in AVAILABLE_SEARCH_METHODS:
                raise ValueError(f"Invalid search method: {self.search_method}. "
                               f"Available methods: {AVAILABLE_SEARCH_METHODS}")
            
            if self.search_cv < 2:
                raise ValueError(f"Search CV folds must be >= 2, got {self.search_cv}")
            
            if self.search_method == 'random' and self.search_n_iter < 1:
                raise ValueError(f"Number of search iterations must be >= 1, got {self.search_n_iter}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='Accident Severity Classification Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add argument groups for better organization
    _add_required_arguments(parser)
    _add_pipeline_arguments(parser)
    _add_data_arguments(parser)
    _add_feature_arguments(parser)
    _add_model_arguments(parser)
    
    return parser


def _add_required_arguments(parser: argparse.ArgumentParser) -> None:
    """Add required arguments to parser."""
    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
        '--input-path',
        required=True,
        help='Path to input data file'
    )
    required.add_argument(
        '--output-dir',
        required=True,
        help='Directory for output files'
    )


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Add pipeline configuration arguments."""
    pipeline = parser.add_argument_group('Pipeline Configuration')
    pipeline.add_argument(
        '--target-column',
        default=DEFAULT_TARGET_COLUMN,
        help='Target column for prediction'
    )
    pipeline.add_argument(
        '--mode',
        choices=['preprocess', 'feature', 'train', 'evaluate', 'full'],
        default='full',
        help='Pipeline mode to run'
    )
    pipeline.add_argument(
        '--config',
        help='Path to JSON config file'
    )
    pipeline.add_argument(
        '--random-state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help='Random seed for reproducibility'
    )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add data preprocessing arguments."""
    data = parser.add_argument_group('Data Preprocessing')
    data.add_argument(
        '--numeric-columns',
        nargs='+',
        help='List of numeric columns'
    )
    data.add_argument(
        '--categorical-columns',
        nargs='+',
        help='List of categorical columns'
    )
    data.add_argument(
        '--date-columns',
        nargs='+',
        help='List of date columns'
    )
    data.add_argument(
        '--identifier-columns',
        nargs='+',
        help='Columns to keep as row identifiers but exclude from processing'
    )
    data.add_argument(
        '--handle-outliers-cols',
        nargs='+',
        help='List of columns to handle outliers for'
    )


def _add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    """Add feature engineering arguments."""
    features = parser.add_argument_group('Feature Engineering')
    features.add_argument(
        '--interaction-features',
        nargs='+',
        help='Pairs of columns for interaction features'
    )
    features.add_argument(
        '--polynomial-features',
        nargs='+',
        help='Columns for polynomial features'
    )
    features.add_argument(
        '--polynomial-degree',
        type=int,
        default=2,
        help='Degree for polynomial features'
    )
    features.add_argument(
        '--pca-components',
        type=int,
        help='Number of PCA components'
    )
    features.add_argument(
        '--n-select-features',
        type=int,
        help='Number of features to select using feature selection'
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model training arguments."""
    models = parser.add_argument_group('Model Training')
    models.add_argument(
        '--test-size',
        type=float,
        default=DEFAULT_TEST_SIZE,
        help='Test set size'
    )
    models.add_argument(
        '--model-type',
        choices=AVAILABLE_MODEL_TYPES + ['all'],
        default='all',
        help='Type of model to train'
    )
    models.add_argument(
        '--handle-imbalance',
        choices=['smote', 'class_weight', 'none'],
        default='smote',
        help='Method to handle imbalanced data'
    )
    models.add_argument(
        '--n-splits',
        type=int,
        default=DEFAULT_CV_SPLITS,
        help='Number of cross-validation splits'
    )
    models.add_argument(
        '--optimize-hyperparameters',
        action='store_true',
        help='Perform hyperparameter optimization'
    )
    models.add_argument(
        '--search-method',
        choices=['grid', 'random'],
        default='grid',
        help='Hyperparameter search method'
    )
    models.add_argument(
        '--search-cv',
        type=int,
        default=3,
        help='Number of CV folds for hyperparameter search'
    )
    models.add_argument(
        '--search-n-iter',
        type=int,
        default=50,
        help='Number of iterations for random search'
    )


def load_config_from_file(args: argparse.Namespace) -> argparse.Namespace:
    """Load configuration from JSON file and override command line arguments."""
    if not args.config:
        return args
    
    # Validate config file exists
    validate_file_exists(args.config, "Configuration file")
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Loading configuration from {args.config}")
        
        # Override args with config values
        for key, value in config.items():
            attr_name = key.replace('-', '_')
            if hasattr(args, attr_name):
                # Handle boolean conversion for store_true arguments
                if isinstance(getattr(args, attr_name), bool) and isinstance(value, str):
                    value = value.lower() in ['true', '1', 'yes']
                
                setattr(args, attr_name, value)
                logger.info(f"Config override: {attr_name} = {value}")
            else:
                logger.warning(f"Unknown config key ignored: {key}")
        
        return args
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {args.config}: {e}")
    except Exception as e:
        logger.error(f"Failed to load config file {args.config}: {e}")
        raise


@standardize_error_handling
def run_preprocessing_step(config: PipelineConfig) -> str:
    """Execute the data preprocessing step."""
    preprocessed_path = os.path.join(config.output_dir, 'preprocessed_data.csv')
    
    preprocess_pipeline(
        input_path=config.input_path,
        output_path=preprocessed_path,
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        date_columns=config.date_columns,
        identifier_columns=config.identifier_columns,
        handle_outliers_cols=config.handle_outliers_cols
    )
    
    return preprocessed_path


@standardize_error_handling
def run_feature_engineering_step(input_path: str, config: PipelineConfig) -> str:
    """Execute the feature engineering step."""
    featured_path = os.path.join(config.output_dir, 'featured_data.csv')
    
    # Read data to handle identifier columns
    df = safe_load_csv(input_path, encoding='utf-8')
    
    # Separate identifier columns from processing columns
    identifier_columns_to_preserve = []
    if config.identifier_columns:
        identifier_columns_existing = [col for col in config.identifier_columns if col in df.columns]
        if identifier_columns_existing:
            # Store identifier columns data
            identifier_data = df[identifier_columns_existing].copy()
            identifier_columns_to_preserve = identifier_columns_existing
            logger.info(f"Preserving {len(identifier_columns_existing)} identifier columns during feature engineering")
    
    engineer_features(
        input_path=input_path,
        output_path=featured_path,
        numeric_features=config.numeric_columns,
        categorical_features=config.categorical_columns,
        date_features=config.date_columns,
        target_column=config.target_column,
        feature_pairs=config.interaction_features,
        polynomial_features=config.polynomial_features,
        n_components=config.pca_components,
        n_select_features=config.n_select_features,
        identifier_columns=config.identifier_columns  # Pass identifier columns to exclude from processing
    )
    
    # Re-add identifier columns if they were preserved
    if identifier_columns_to_preserve:
        # Read the feature-engineered data
        df_featured = safe_load_csv(featured_path, encoding='utf-8')
        
        # Add back identifier columns
        for col in identifier_columns_to_preserve:
            df_featured[col] = identifier_data[col]
        
        # Save with identifier columns included
        safe_save_csv(df_featured, featured_path, "feature-engineered data with identifiers")
        logger.info(f"Added back {len(identifier_columns_to_preserve)} identifier columns to featured data")
    
    return featured_path


@standardize_error_handling
def run_training_step(input_path: str, config: PipelineConfig) -> Dict[str, Any]:
    """Execute the model training step."""
    model_dir = create_directory_if_not_exists(
        os.path.join(config.output_dir, 'models'),
        "Models directory"
    )
    
    # Calculate feature columns (exclude identifier columns and target)
    df = safe_load_csv(input_path, encoding='utf-8')
    all_columns = df.columns.tolist()
    
    # Remove target column and identifier columns from feature list
    feature_columns = [col for col in all_columns if col != config.target_column]
    if config.identifier_columns:
        identifier_columns_existing = [col for col in config.identifier_columns if col in df.columns]
        feature_columns = [col for col in feature_columns if col not in identifier_columns_existing]
        logger.info(f"Excluding {len(identifier_columns_existing)} identifier columns from training: {identifier_columns_existing}")
    
    logger.info(f"Using {len(feature_columns)} feature columns for training")
    
    results = train_pipeline(
        input_path=input_path,
        output_dir=str(model_dir),
        target_column=config.target_column,
        model_types=config.model_type,
        feature_columns=feature_columns,  # Explicitly exclude identifiers
        handle_imbalance=config.handle_imbalance,
        test_size=config.test_size,
        random_state=config.random_state,
        optimize_hyperparameters=config.optimize_hyperparameters,
        search_method=config.search_method,
        search_cv=config.search_cv,
        search_n_iter=config.search_n_iter,
        identifier_columns=config.identifier_columns,  # Pass identifier columns for analysis
        n_splits=config.n_splits,  # Pass n_splits parameter
        perform_cv=config.perform_cv
    )
    
    return results


@standardize_error_handling
def run_evaluation_step(config: PipelineConfig) -> Dict[str, Any]:
    """Execute the model evaluation step."""
    model_path = os.path.join(config.output_dir, 'models', 'best_model.joblib')
    eval_dir = create_directory_if_not_exists(
        os.path.join(config.output_dir, 'evaluation'),
        "Evaluation directory"
    )
    
    # Validate model file exists
    validate_file_exists(model_path, "Trained model file")
    
    # Calculate feature columns (same logic as training step)
    df = safe_load_csv(config.input_path, encoding='utf-8')
    all_columns = df.columns.tolist()
    
    # Remove target column and identifier columns from feature list
    feature_columns = [col for col in all_columns if col != config.target_column]
    if config.identifier_columns:
        identifier_columns_existing = [col for col in config.identifier_columns if col in df.columns]
        feature_columns = [col for col in feature_columns if col not in identifier_columns_existing]
        logger.info(f"Excluding {len(identifier_columns_existing)} identifier columns from evaluation: {identifier_columns_existing}")
    
    logger.info(f"Using {len(feature_columns)} feature columns for evaluation")
    
    results = evaluate_model_pipeline(
        model_path=model_path,
        test_data_path=config.input_path,
        target_column=config.target_column,
        output_dir=str(eval_dir),
        feature_columns=feature_columns,  # Pass the calculated feature columns
        identifier_columns=config.identifier_columns  # Pass identifier columns from config
    )
    
    return results


def execute_pipeline(config: PipelineConfig) -> None:
    """Execute the complete pipeline based on configuration."""
    logger.info(f"Starting pipeline in '{config.mode}' mode")
    
    current_input = config.input_path
    
    # Execute steps based on mode
    if config.mode in ['preprocess', 'full']:
        current_input = run_preprocessing_step(config)
    
    if config.mode in ['feature', 'full']:
        current_input = run_feature_engineering_step(current_input, config)
    
    if config.mode in ['train', 'full']:
        training_results = run_training_step(current_input, config)
        logger.info("Training completed successfully")
    
    if config.mode in ['evaluate', 'full']:
        evaluation_results = run_evaluation_step(config)
        logger.info("Evaluation completed successfully")
    
    logger.info("Pipeline execution completed successfully!")


def main() -> None:
    """Main entry point for the pipeline."""
    try:
        # Parse arguments and load configuration
        parser = create_argument_parser()
        args = parser.parse_args()
        args = load_config_from_file(args)
        
        # Create configuration object with validation
        config = PipelineConfig(args)
        
        # Execute pipeline
        execute_pipeline(config)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 