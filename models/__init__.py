"""
Models package for the accident severity classification project.

This package provides comprehensive model functionality including:
- Model definitions (BaseModel, LogisticRegressionModel, RandomForestModel, XGBoostModel)
- Model factory for creating model instances
- Model training pipeline with cross-validation and evaluation
- Model utilities for data processing, imbalance handling, and evaluation
- Model evaluation pipeline functions

Example usage:
    from models import ModelFactory, ModelTrainer, train_pipeline
    
    # Create models
    factory = ModelFactory()
    model = factory.get_model('rf')
    
    # Train models
    trainer = ModelTrainer(output_dir='./models')
    results = trainer.train_multiple_models(data, target_column='target')
    
    # Use pipeline function
    results = train_pipeline(
        input_path='data.csv',
        output_dir='./models',
        target_column='target'
    )
"""

# Core model components
from .models import (
    BaseModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    ModelFactory
)

# Model utilities
from .model_utils import (
    DataProcessor,
    ImbalanceHandler,
    ModelEvaluator
)

# Training pipeline
from .model_training import (
    ModelTrainer,
    train_pipeline
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Accident Severity Classification Team"

# All available exports
__all__ = [
    # Core models
    'BaseModel',
    'LogisticRegressionModel',
    'RandomForestModel',
    'XGBoostModel',
    'ModelFactory',
    
    # Utilities
    'DataProcessor',
    'ImbalanceHandler',
    'ModelEvaluator',
    
    # Training
    'ModelTrainer',
    'train_pipeline',
] 