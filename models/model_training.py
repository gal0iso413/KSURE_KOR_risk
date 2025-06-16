"""
Model training module for the accident severity classification project.

This module provides comprehensive model training functionality including
data preparation, model training, cross-validation, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

# Import project utilities and constants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import get_logger
from src.utils.common import (
    safe_load_csv, create_directory_if_not_exists, standardize_error_handling
)
from src.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, DEFAULT_CV_SPLITS, 
    DEFAULT_CV_SCORING, DEFAULT_CV_SHUFFLE, DEFAULT_IMBALANCE_METHOD,
    AVAILABLE_MODEL_TYPES, DEFAULT_ENCODING, HYPERPARAMETER_SEARCH_SPACES,
    DEFAULT_SEARCH_METHOD, DEFAULT_SEARCH_CV, DEFAULT_SEARCH_N_ITER
)

# Import model components
from .models import ModelFactory
from .model_utils import DataProcessor, ImbalanceHandler, ModelEvaluator

# Initialize logger
logger = get_logger(__name__)


class ModelTrainer:
    """Comprehensive model training pipeline with proper validation and logging."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        random_state: int = DEFAULT_RANDOM_STATE,
        test_size: float = DEFAULT_TEST_SIZE
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            output_dir: Directory for saving models and results
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for testing
        """
        self.output_dir = create_directory_if_not_exists(output_dir, "Model training output")
        self.random_state = random_state
        self.test_size = test_size
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.data_processor = DataProcessor()
        self.imbalance_handler = ImbalanceHandler(random_state)
        
        # Training state
        self.is_data_prepared = False
        self.feature_names = None
        self.target_column = None
        self.original_test_data = None  # Store original test data for analysis
        self.identifier_columns = None  # Store identifier columns
        
        logger.info(f"ModelTrainer initialized with output_dir: {self.output_dir}")
    
    @standardize_error_handling
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        handle_imbalance: str = DEFAULT_IMBALANCE_METHOD
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for model training with comprehensive validation.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            feature_columns: List of feature columns to use (None = all except target)
            handle_imbalance: Method to handle class imbalance
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Validate inputs
        if data.empty:
            raise ValueError("Input data is empty")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data. "
                           f"Available columns: {list(data.columns)}")
        
        logger.info(f"Preparing data with shape {data.shape} for training")
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
            logger.info(f"Using all columns except target as features: {len(feature_columns)} features")
        else:
            # Validate feature columns exist
            missing_features = [col for col in feature_columns if col not in data.columns]
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")
            logger.info(f"Using specified feature columns: {len(feature_columns)} features")
        
        # Extract features and target
        X = data[feature_columns].copy()
        y_raw = data[target_column].copy()
        
        # Convert target to severity levels
        y = self.data_processor.convert_to_severity_levels(y_raw)
        
        # Validate data quality
        X, y = self.data_processor.validate_features_and_target(X, y, feature_columns)
        
        # Split data before handling imbalance (to avoid data leakage)
        X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(
            X, y, data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Store original test data for analysis
        self.original_test_data = data_test
        
        logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Handle imbalanced data (only on training set)
        if handle_imbalance and handle_imbalance != 'none':
            result = self.imbalance_handler.handle_imbalanced_data(
                X_train, y_train, method=handle_imbalance, feature_columns=feature_columns
            )
            
            if handle_imbalance == 'smote':
                X_train, y_train = result
            elif handle_imbalance == 'class_weight':
                X_train, y_train, self.class_weights = result
            
            logger.info(f"Applied {handle_imbalance} for imbalance handling")
        
        # Store metadata
        self.feature_names = feature_columns
        self.target_column = target_column
        self.is_data_prepared = True
        
        logger.info("Data preparation completed successfully")
        return X_train, X_test, y_train, y_test, feature_columns
    
    @standardize_error_handling
    def train_single_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None,
        class_weights: Optional[Dict[int, float]] = None
    ) -> Any:
        """
        Train a single model with optional hyperparameters.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training target
            hyperparameters: Optional custom hyperparameters
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Trained model instance
        """
        if model_type not in AVAILABLE_MODEL_TYPES:
            raise ValueError(f"Unknown model type '{model_type}'. "
                           f"Available types: {AVAILABLE_MODEL_TYPES}")
        
        logger.info(f"Training {model_type} model with {len(X_train)} training samples")
        
        # Get model instance
        model = self.model_factory.get_model(model_type, self.random_state)
        
        # Set custom hyperparameters if provided
        if hyperparameters:
            model.set_params(**hyperparameters)
            logger.info(f"Applied custom hyperparameters: {hyperparameters}")
        
        # Set class weights if provided and supported
        if class_weights and hasattr(model.model, 'set_params'):
            try:
                model.set_params(class_weight=class_weights)
                logger.info(f"Applied class weights: {class_weights}")
            except Exception as e:
                logger.warning(f"Could not set class weights for {model_type}: {e}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        return model
    
    @standardize_error_handling
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model and save results.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name for saving evaluation results
            
        Returns:
            Evaluation results dictionary
        """
        # Create model-specific evaluation directory
        eval_dir = self.output_dir / f"{model_name}_evaluation"
        evaluator = ModelEvaluator(eval_dir)
        
        # Perform comprehensive evaluation
        results = evaluator.evaluate_classification_model(
            model.model if hasattr(model, 'model') else model,
            X_test,
            y_test,
            self.feature_names,
            model_name,
            self.original_test_data,
            self.identifier_columns
        )
        
        return results
    
    @standardize_error_handling
    def cross_validate_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: int = DEFAULT_CV_SPLITS,
        scoring: str = DEFAULT_CV_SCORING,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_type: Type of model to validate
            X: Features
            y: Target
            cv_splits: Number of cross-validation splits
            scoring: Scoring metric for CV
            hyperparameters: Optional custom hyperparameters
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_splits}-fold cross-validation for {model_type}")
        
        # Get model instance
        model = self.model_factory.get_model(model_type, self.random_state)
        
        # Set custom hyperparameters if provided
        if hyperparameters:
            model.set_params(**hyperparameters)
        
        # Create the actual sklearn model for cross-validation
        sklearn_model = model.create_model()
        
        # Perform cross-validation
        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=DEFAULT_CV_SHUFFLE,
            random_state=self.random_state
        )
        
        scores = cross_val_score(
            sklearn_model,
            X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cv_results = {
            'model_type': model_type,
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_splits': cv_splits,
            'scoring': scoring
        }
        
        logger.info(f"{model_type} CV Results - Mean: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return cv_results
    
    @standardize_error_handling
    def hyperparameter_search(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search_method: str = DEFAULT_SEARCH_METHOD,
        cv_splits: int = DEFAULT_SEARCH_CV,
        n_iter: int = DEFAULT_SEARCH_N_ITER,
        scoring: str = DEFAULT_CV_SCORING,
        custom_param_grid: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter search for a model.
        
        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training target
            search_method: Search method ('grid', 'random', 'bayesian')
            cv_splits: Number of cross-validation splits
            n_iter: Number of iterations for random/bayesian search
            scoring: Scoring metric for optimization
            custom_param_grid: Custom parameter grid (overrides default)
            
        Returns:
            best_model, search_results
        """
        if model_type not in AVAILABLE_MODEL_TYPES:
            raise ValueError(f"Unknown model type '{model_type}'. "
                           f"Available types: {AVAILABLE_MODEL_TYPES}")
        
        logger.info(f"Starting {search_method} hyperparameter search for {model_type}")
        
        # Get base model
        base_model = self.model_factory.get_model(model_type, self.random_state)
        sklearn_model = base_model.create_model()
        
        # Get parameter grid
        if custom_param_grid:
            param_grid = custom_param_grid
            logger.info(f"Using custom parameter grid: {param_grid}")
        else:
            param_grid = HYPERPARAMETER_SEARCH_SPACES.get(model_type, {})
            if not param_grid:
                logger.warning(f"No default parameter grid found for {model_type}")
                return base_model, {'warning': 'No parameter grid available'}
            logger.info(f"Using default parameter grid with {len(param_grid)} parameters")
        
        # Create cross-validation strategy
        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Perform hyperparameter search
        if search_method.lower() == 'grid':
            search = GridSearchCV(
                estimator=sklearn_model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
        elif search_method.lower() == 'random':
            search = RandomizedSearchCV(
                estimator=sklearn_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unsupported search method: {search_method}. "
                           f"Supported methods: 'grid', 'random'")
        
        # Fit the search
        logger.info(f"Fitting {search_method} search with {cv_splits}-fold CV...")
        search.fit(X_train, y_train)
        
        # Extract results
        best_model = search.best_estimator_
        
        search_results = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': {
                'mean_test_scores': search.cv_results_['mean_test_score'].tolist(),
                'std_test_scores': search.cv_results_['std_test_score'].tolist(),
                'mean_train_scores': search.cv_results_['mean_train_score'].tolist(),
                'std_train_scores': search.cv_results_['std_train_score'].tolist(),
                'params': search.cv_results_['params']
            },
            'search_method': search_method,
            'cv_splits': cv_splits,
            'scoring': scoring,
            'n_candidates': len(search.cv_results_['params'])
        }
        
        logger.info(f"Best {model_type} score: {search.best_score_:.4f}")
        logger.info(f"Best {model_type} params: {search.best_params_}")
        
        return best_model, search_results
    
    @standardize_error_handling
    def train_multiple_models(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_types: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        handle_imbalance: str = DEFAULT_IMBALANCE_METHOD,
        perform_cv: bool = True,
        save_models: bool = True,
        optimize_hyperparameters: bool = False,
        search_method: str = DEFAULT_SEARCH_METHOD,
        search_cv: int = DEFAULT_SEARCH_CV,
        search_n_iter: int = DEFAULT_SEARCH_N_ITER,
        identifier_columns: Optional[List[str]] = None,
        n_splits: int = DEFAULT_CV_SPLITS
    ) -> Dict[str, Any]:
        """
        Train and evaluate multiple models.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            model_types: List of model types to train (None = all available)
            feature_columns: List of feature columns to use
            handle_imbalance: Method to handle class imbalance
            perform_cv: Whether to perform cross-validation
            save_models: Whether to save trained models
            optimize_hyperparameters: Whether to perform hyperparameter search
            search_method: Hyperparameter search method ('grid', 'random')
            search_cv: Number of CV folds for hyperparameter search
            search_n_iter: Number of iterations for random search
            identifier_columns: Optional list of identifier columns
            n_splits: Number of cross-validation splits
            
        Returns:
            Comprehensive results for all models
        """
        # Default to all available models if none specified or if 'all' is specified
        if model_types is None or (isinstance(model_types, list) and len(model_types) == 1 and model_types[0] == 'all'):
            model_types = AVAILABLE_MODEL_TYPES
        elif isinstance(model_types, str):
            if model_types == 'all':
                model_types = AVAILABLE_MODEL_TYPES
            else:
                model_types = [model_types]
        
        # Validate model types
        invalid_types = [mt for mt in model_types if mt not in AVAILABLE_MODEL_TYPES]
        if invalid_types:
            raise ValueError(f"Invalid model types: {invalid_types}. "
                           f"Available types: {AVAILABLE_MODEL_TYPES}")
        
        logger.info(f"Training {len(model_types)} model types: {model_types}")
        
        # Store identifier columns for analysis
        self.identifier_columns = identifier_columns
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            data, target_column, feature_columns, handle_imbalance
        )
        
        results = {
            'data_info': {
                'total_samples': len(data),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': len(feature_names),
                'target_column': target_column,
                'imbalance_method': handle_imbalance
            },
            'models': {}
        }
        
        trained_models = {}
        evaluation_results = {}
        cv_results = {}
        hyperparameter_results = {}
        
        # Train each model type
        for model_type in model_types:
            try:
                logger.info(f"Processing {model_type} model...")
                
                # Create model directory
                model_dir = self.output_dir / model_type
                create_directory_if_not_exists(model_dir, f"{model_type} model directory")
                
                # Step 1: Hyperparameter optimization if requested
                if optimize_hyperparameters:
                    logger.info(f"Performing hyperparameter search for {model_type}...")
                    best_model, search_results = self.hyperparameter_search(
                        model_type=model_type,
                        X_train=X_train,
                        y_train=y_train,
                        search_method=search_method,
                        cv_splits=search_cv,
                        n_iter=search_n_iter
                    )
                    hyperparameter_results[model_type] = search_results
                    
                    # Create and fit model with best parameters
                    model = self.model_factory.get_model(model_type, self.random_state)
                    model.model = best_model
                    model.fit(X_train, y_train)  # Ensure model is fitted
                else:
                    # Train with default hyperparameters
                    class_weights = getattr(self, 'class_weights', None)
                    model = self.train_single_model(model_type, X_train, y_train, class_weights=class_weights)
                
                trained_models[model_type] = model
                
                # Step 2: Cross-validation if requested (only on training data)
                if perform_cv:
                    cv_result = self.cross_validate_model(
                        model_type=model_type,
                        X=X_train,
                        y=y_train,
                        cv_splits=n_splits
                    )
                    cv_results[model_type] = cv_result
                
                # Step 3: Save model if requested
                if save_models:
                    model_path = model.save(model_dir)
                    logger.info(f"Saved {model_type} model to {model_path}")
                
                # Step 4: Final evaluation on test set
                eval_results = self.evaluate_model(model, X_test, y_test, model_type)
                evaluation_results[model_type] = eval_results
                
                logger.info(f"Completed {model_type} model processing")
                
            except Exception as e:
                logger.error(f"Failed to process {model_type} model: {e}")
                evaluation_results[model_type] = {'error': str(e)}
        
        # Compile results
        results['models'] = evaluation_results
        if cv_results:
            results['cross_validation'] = cv_results
        if hyperparameter_results:
            results['hyperparameter_search'] = hyperparameter_results
        
        # Generate model comparison
        if len(evaluation_results) > 1:
            evaluator = ModelEvaluator(self.output_dir)
            comparison_df = evaluator.compare_models(evaluation_results)
            results['model_comparison'] = comparison_df.to_dict() if not comparison_df.empty else {}
        
        logger.info(f"Training pipeline completed for {len(model_types)} models")
        return results


@standardize_error_handling
def train_pipeline(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    target_column: str,
    model_types: Union[str, List[str]] = AVAILABLE_MODEL_TYPES,
    feature_columns: Optional[List[str]] = None,
    handle_imbalance: str = DEFAULT_IMBALANCE_METHOD,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    perform_cv: bool = True,
    save_models: bool = True,
    optimize_hyperparameters: bool = False,
    search_method: str = DEFAULT_SEARCH_METHOD,
    search_cv: int = DEFAULT_SEARCH_CV,
    search_n_iter: int = DEFAULT_SEARCH_N_ITER,
    identifier_columns: Optional[List[str]] = None,
    n_splits: int = DEFAULT_CV_SPLITS
) -> Dict[str, Any]:
    """
    Complete model training pipeline function.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory for saving models and results
        target_column: Name of target column
        model_types: Model type(s) to train
        feature_columns: List of feature columns to use
        handle_imbalance: Method to handle class imbalance
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        perform_cv: Whether to perform cross-validation
        save_models: Whether to save trained models
        optimize_hyperparameters: Whether to perform hyperparameter search
        search_method: Hyperparameter search method
        search_cv: Number of CV folds for hyperparameter search
        search_n_iter: Number of iterations for random search
        identifier_columns: Optional list of identifier columns
        n_splits: Number of cross-validation splits
        
    Returns:
        Comprehensive training results
    """
    logger.info(f"Starting model training pipeline")
    logger.info(f"Input: {input_path}, Output: {output_dir}")
    logger.info(f"Target: {target_column}, Models: {model_types}")
    
    # Load data
    data = safe_load_csv(input_path, encoding=DEFAULT_ENCODING)
    logger.info(f"Loaded data with shape {data.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        output_dir=output_dir,
        random_state=random_state,
        test_size=test_size
    )
    
    # Run training pipeline
    results = trainer.train_multiple_models(
        data=data,
        target_column=target_column,
        model_types=model_types,
        feature_columns=feature_columns,
        handle_imbalance=handle_imbalance,
        perform_cv=perform_cv,
        save_models=save_models,
        optimize_hyperparameters=optimize_hyperparameters,
        search_method=search_method,
        search_cv=search_cv,
        search_n_iter=search_n_iter,
        identifier_columns=identifier_columns,
        n_splits=n_splits
    )
    
    logger.info("Model training pipeline completed successfully")
    return results 