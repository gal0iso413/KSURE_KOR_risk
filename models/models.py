"""
Model definitions for the accident severity classification project.

This module provides base model classes and specific implementations for
logistic regression, random forest, and XGBoost classifiers.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Type, Union, Optional, List
from pathlib import Path
import joblib

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import project utilities and constants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import get_logger
from src.utils.common import create_directory_if_not_exists, standardize_error_handling
from src.constants import DEFAULT_RANDOM_STATE, AVAILABLE_MODEL_TYPES

# Initialize logger
logger = get_logger(__name__)


class BaseModel(ABC):
    """Base class for all models with common functionality."""
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        """
        Initialize base model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model: Optional[BaseEstimator] = None
        self.model_name: Optional[str] = None
        self.is_fitted: bool = False
    
    @abstractmethod
    def create_model(self) -> BaseEstimator:
        """Create and return the actual model instance."""
        pass
    
    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return default hyperparameters for the model."""
        pass
    
    @standardize_error_handling
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments for model fitting
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot fit model with empty dataset")
        
        if X.shape[0] != len(y):
            raise ValueError(f"Feature matrix and target have different lengths: {X.shape[0]} vs {len(y)}")
        
        if self.model is None:
            self.model = self.create_model()
        
        logger.info(f"Training {self.model_name} model with {X.shape[0]} samples and {X.shape[1]} features")
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        logger.info(f"{self.model_name} model training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_name} must be fitted before making predictions")
        
        if X.shape[0] == 0:
            raise ValueError("Cannot make predictions with empty dataset")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_name} must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {self.model_name} does not support probability predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None if not available
        """
        if not self.is_fitted:
            logger.warning(f"Model {self.model_name} is not fitted, cannot get feature importance")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            logger.info(f"Model {self.model_name} does not have feature importance")
            return None
    
    @standardize_error_handling
    def save(self, model_dir: Union[str, Path]) -> str:
        """
        Save the model to disk.
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            Path to saved model file
        """
        if not self.is_fitted:
            raise ValueError(f"Cannot save unfitted model {self.model_name}")
        
        # Create directory if it doesn't exist
        model_dir = create_directory_if_not_exists(model_dir, f"Model directory for {self.model_name}")
        
        model_path = model_dir / f"{self.model_name}.joblib"
        
        # Save model metadata along with the model
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'hyperparameters': self.get_default_hyperparameters()
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Saved {self.model_name} model to {model_path}")
        
        return str(model_path)
    
    @standardize_error_handling
    def load(self, model_path: Union[str, Path]) -> None:
        """
        Load the model from disk.
        
        Args:
            model_path: Path to the saved model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        # Handle both old format (just model) and new format (with metadata)
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.model_name = model_data.get('model_name', self.model_name)
            self.random_state = model_data.get('random_state', self.random_state)
            self.is_fitted = model_data.get('is_fitted', True)
        else:
            # Old format - just the model
            self.model = model_data
            self.is_fitted = True
        
        logger.info(f"Loaded {self.model_name} model from {model_path}")
    
    def set_params(self, **params) -> None:
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        if self.model is None:
            self.model = self.create_model()
        
        self.model.set_params(**params)
        logger.info(f"Updated parameters for {self.model_name}: {params}")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            return self.get_default_hyperparameters()
        
        return self.model.get_params()


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for classification."""
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        super().__init__(random_state)
        self.model_name = "logistic_regression"
    
    def create_model(self) -> BaseEstimator:
        """Create and return a logistic regression model."""
        return LogisticRegression(
            **self.get_default_hyperparameters(),
            random_state=self.random_state
        )
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return default hyperparameters for logistic regression."""
        return {
            'max_iter': 1000,
            'C': 1.0,
            'class_weight': None,
            'solver': 'lbfgs',
            'multi_class': 'auto'
        }


class RandomForestModel(BaseModel):
    """Random Forest model for classification."""
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        super().__init__(random_state)
        self.model_name = "random_forest"
    
    def create_model(self) -> BaseEstimator:
        """Create and return a random forest model."""
        return RandomForestClassifier(
            **self.get_default_hyperparameters(),
            random_state=self.random_state
        )
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return default hyperparameters for random forest."""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': None,
            'bootstrap': True
        }


class XGBoostModel(BaseModel):
    """XGBoost model for classification."""
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        super().__init__(random_state)
        self.model_name = "xgboost"
    
    def create_model(self) -> BaseEstimator:
        """Create and return an XGBoost model."""
        return XGBClassifier(
            **self.get_default_hyperparameters(),
            random_state=self.random_state
        )
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return default hyperparameters for XGBoost."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'verbosity': 0  # Reduce XGBoost output
        }


class ModelFactory:
    """Factory class for creating model instances."""
    
    def __init__(self):
        """Initialize the model factory with available models."""
        self._models: Dict[str, Type[BaseModel]] = {
            'lr': LogisticRegressionModel,
            'rf': RandomForestModel,
            'xgboost': XGBoostModel
        }
        
        # Validate that our models match the constants
        available_models = set(AVAILABLE_MODEL_TYPES)
        factory_models = set(self._models.keys())
        
        if available_models != factory_models:
            logger.warning(f"Model factory models {factory_models} don't match constants {available_models}")
    
    def get_model(self, model_type: str, random_state: int = DEFAULT_RANDOM_STATE) -> BaseModel:
        """
        Create and return a model instance of the specified type.
        
        Args:
            model_type: Type of model to create
            random_state: Random seed for reproducibility
            
        Returns:
            Model instance
        """
        if model_type not in self._models:
            available = list(self._models.keys())
            raise ValueError(f"Unknown model type '{model_type}'. Available types: {available}")
        
        logger.info(f"Creating {model_type} model with random_state={random_state}")
        return self._models[model_type](random_state=random_state)
    
    def register_model(self, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Name for the model type
            model_class: Model class that inherits from BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel")
        
        self._models[model_type] = model_class
        logger.info(f"Registered new model type: {model_type}")
    
    def get_available_models(self) -> List[str]:
        """
        Return list of available model types.
        
        Returns:
            List of available model type names
        """
        return list(self._models.keys())
    
    def create_all_models(self, random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, BaseModel]:
        """
        Create instances of all available models.
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary mapping model types to model instances
        """
        models = {}
        for model_type in self._models.keys():
            models[model_type] = self.get_model(model_type, random_state)
        
        logger.info(f"Created {len(models)} model instances: {list(models.keys())}")
        return models 