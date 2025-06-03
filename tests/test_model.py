"""
Comprehensive test suite for model functionality.

This module tests all model-related components including model classes,
training pipeline, evaluation, and utilities.
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import our model components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import ModelFactory, LogisticRegressionModel, RandomForestModel, XGBoostModel
from models.model_utils import DataProcessor, ImbalanceHandler, ModelEvaluator
from models.model_training import ModelTrainer, train_pipeline
from src.evaluate import ModelLoader, EvaluationPipeline
from src.constants import DEFAULT_RANDOM_STATE, AVAILABLE_MODEL_TYPES


class TestModelComponents(unittest.TestCase):
    """Test individual model components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across tests."""
        # Create synthetic classification data
        np.random.seed(DEFAULT_RANDOM_STATE)
        n_samples = 200
        n_features = 5
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with 3 classes (severity levels)
        probabilities = np.array([0.5, 0.3, 0.2])  # Low, Medium, High severity
        y = np.random.choice([0, 1, 2], size=n_samples, p=probabilities)
        
        # Create DataFrame
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        cls.sample_data = pd.DataFrame(X, columns=feature_columns)
        cls.sample_data['사고유형코드'] = y  # Korean target column name
        
        # Create temporary directory for tests
        cls.temp_dir = Path(tempfile.mkdtemp())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        self.output_dir = self.temp_dir / f"test_{self._testMethodName}"
        self.output_dir.mkdir(exist_ok=True)
    
    def test_model_factory(self):
        """Test ModelFactory functionality."""
        factory = ModelFactory()
        
        # Test getting available models
        available = factory.get_available_models()
        self.assertEqual(set(available), set(AVAILABLE_MODEL_TYPES))
        
        # Test creating individual models
        for model_type in AVAILABLE_MODEL_TYPES:
            model = factory.get_model(model_type)
            self.assertIsNotNone(model)
            self.assertEqual(model.model_name, model_type)
            self.assertFalse(model.is_fitted)
        
        # Test creating all models
        all_models = factory.create_all_models()
        self.assertEqual(len(all_models), len(AVAILABLE_MODEL_TYPES))
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            factory.get_model('invalid_model')
    
    def test_individual_models(self):
        """Test individual model classes."""
        models_to_test = [
            ('lr', LogisticRegressionModel),
            ('rf', RandomForestModel),
            ('xgboost', XGBoostModel)
        ]
        
        for model_name, model_class in models_to_test:
            with self.subTest(model=model_name):
                # Create model
                model = model_class()
                self.assertEqual(model.model_name, model_name)
                self.assertFalse(model.is_fitted)
                
                # Test hyperparameters
                params = model.get_default_hyperparameters()
                self.assertIsInstance(params, dict)
                self.assertGreater(len(params), 0)
                
                # Test model creation
                sklearn_model = model.create_model()
                self.assertIsNotNone(sklearn_model)
    
    def test_data_processor(self):
        """Test DataProcessor functionality."""
        processor = DataProcessor()
        
        # Test severity conversion
        test_codes = pd.Series(['MINOR', 'SEVERE', 'MODERATE', 'UNKNOWN'])
        severity_levels = processor.convert_to_severity_levels(test_codes)
        
        self.assertEqual(len(severity_levels), 4)
        self.assertIn(0, severity_levels.values)  # MINOR -> 0
        self.assertIn(2, severity_levels.values)  # SEVERE -> 2
        self.assertIn(1, severity_levels.values)  # MODERATE -> 1
        
        # Test data validation
        X = self.sample_data.drop('사고유형코드', axis=1)
        y = self.sample_data['사고유형코드']
        
        X_val, y_val = processor.validate_features_and_target(X, y)
        self.assertEqual(X_val.shape, X.shape)
        self.assertEqual(len(y_val), len(y))
    
    def test_imbalance_handler(self):
        """Test ImbalanceHandler functionality."""
        handler = ImbalanceHandler()
        
        X = self.sample_data.drop('사고유형코드', axis=1).values
        y = self.sample_data['사고유형코드'].values
        
        # Test SMOTE
        X_smote, y_smote = handler.handle_imbalanced_data(X, y, method='smote')
        self.assertGreaterEqual(len(y_smote), len(y))  # SMOTE should increase size
        
        # Test class weights
        X_weight, y_weight, weights = handler.handle_imbalanced_data(X, y, method='class_weight')
        self.assertEqual(len(X_weight), len(X))  # Size should remain same
        self.assertIsInstance(weights, dict)
    
    def test_model_evaluator(self):
        """Test ModelEvaluator functionality."""
        evaluator = ModelEvaluator(self.output_dir)
        
        # Create a simple trained model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=DEFAULT_RANDOM_STATE)
        
        X = self.sample_data.drop('사고유형코드', axis=1).values
        y = self.sample_data['사고유형코드'].values
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        results = evaluator.evaluate_classification_model(
            model, X_test, y_test, feature_names, 'test_model'
        )
        
        # Check results
        self.assertIn('accuracy', results)
        self.assertIn('f1_macro', results)
        self.assertIn('confusion_matrix', results)
        self.assertGreater(results['accuracy'], 0)


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create larger synthetic dataset for training
        np.random.seed(DEFAULT_RANDOM_STATE)
        n_samples = 500
        n_features = 8
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])
        
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        cls.training_data = pd.DataFrame(X, columns=feature_columns)
        cls.training_data['사고유형코드'] = y
        
        cls.temp_dir = Path(tempfile.mkdtemp())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        self.output_dir = self.temp_dir / f"training_{self._testMethodName}"
        self.output_dir.mkdir(exist_ok=True)
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(self.output_dir)
        
        self.assertEqual(trainer.output_dir, self.output_dir)
        self.assertEqual(trainer.random_state, DEFAULT_RANDOM_STATE)
        self.assertFalse(trainer.is_data_prepared)
    
    def test_data_preparation(self):
        """Test data preparation functionality."""
        trainer = ModelTrainer(self.output_dir)
        
        X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
            self.training_data, '사고유형코드'
        )
        
        # Check data splits
        self.assertGreater(len(X_train), len(X_test))
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertTrue(trainer.is_data_prepared)
        
        # Check feature names
        expected_features = [col for col in self.training_data.columns if col != '사고유형코드']
        self.assertEqual(set(feature_names), set(expected_features))
    
    def test_single_model_training(self):
        """Test training a single model."""
        trainer = ModelTrainer(self.output_dir)
        
        # Prepare data
        X_train, X_test, y_train, y_test, _ = trainer.prepare_data(
            self.training_data, '사고유형코드'
        )
        
        # Train model
        model = trainer.train_single_model('rf', X_train, y_train)
        
        self.assertIsNotNone(model)
        self.assertTrue(model.is_fitted)
        
        # Test predictions
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        trainer = ModelTrainer(self.output_dir)
        
        # Prepare and train
        X_train, X_test, y_train, y_test, _ = trainer.prepare_data(
            self.training_data, '사고유형코드'
        )
        model = trainer.train_single_model('rf', X_train, y_train)
        
        # Evaluate
        results = trainer.evaluate_model(model, X_test, y_test, 'test_rf')
        
        self.assertIn('accuracy', results)
        self.assertIn('f1_macro', results)
        self.assertGreater(results['accuracy'], 0)
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        trainer = ModelTrainer(self.output_dir)
        
        # Prepare data
        X_train, _, y_train, _, _ = trainer.prepare_data(
            self.training_data, '사고유형코드'
        )
        
        # Cross-validate
        cv_results = trainer.cross_validate_model('rf', X_train, y_train, cv_splits=3)
        
        self.assertIn('cv_scores', cv_results)
        self.assertIn('cv_mean', cv_results)
        self.assertIn('cv_std', cv_results)
        self.assertEqual(len(cv_results['cv_scores']), 3)
    
    @patch('models.model_training.safe_load_csv')
    def test_train_pipeline_function(self, mock_load_csv):
        """Test the training pipeline function."""
        # Mock CSV loading to return our test data
        mock_load_csv.return_value = self.training_data
        
        # Save test data to temporary file
        test_file = self.temp_dir / "test_data.csv"
        self.training_data.to_csv(test_file, index=False)
        
        # Use actual file loading instead of mock for this test
        mock_load_csv.side_effect = None
        
        # Run pipeline with single model for faster testing
        results = train_pipeline(
            input_path=test_file,
            output_dir=self.output_dir,
            target_column='사고유형코드',
            model_types=['lr'],  # Test with single model
            perform_cv=False,  # Skip CV for faster testing
            save_models=True
        )
        
        # Check results structure
        self.assertIn('data_info', results)
        self.assertIn('models', results)
        self.assertIn('lr', results['models'])
        
        # Check model was saved
        model_files = list(self.output_dir.glob('**/*.joblib'))
        self.assertGreater(len(model_files), 0)


class TestEvaluation(unittest.TestCase):
    """Test evaluation pipeline functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        np.random.seed(DEFAULT_RANDOM_STATE)
        n_samples = 100
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], size=n_samples)
        
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        cls.test_data = pd.DataFrame(X, columns=feature_columns)
        cls.test_data['사고유형코드'] = y
        
        cls.temp_dir = Path(tempfile.mkdtemp())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        self.output_dir = self.temp_dir / f"eval_{self._testMethodName}"
        self.output_dir.mkdir(exist_ok=True)
    
    def test_model_loader(self):
        """Test ModelLoader functionality."""
        loader = ModelLoader()
        
        # Create and save a test model
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        model = RandomForestClassifier(n_estimators=10, random_state=DEFAULT_RANDOM_STATE)
        X = self.test_data.drop('사고유형코드', axis=1).values
        y = self.test_data['사고유형코드'].values
        model.fit(X, y)
        
        model_path = self.output_dir / "test_model.joblib"
        joblib.dump(model, model_path)
        
        # Test loading
        loaded_model = loader.load_model(model_path)
        self.assertIsNotNone(loaded_model)
        
        # Test predictions match
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_evaluation_pipeline(self):
        """Test EvaluationPipeline functionality."""
        pipeline = EvaluationPipeline(self.output_dir)
        
        # Create and save a test model
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        model = RandomForestClassifier(n_estimators=10, random_state=DEFAULT_RANDOM_STATE)
        X = self.test_data.drop('사고유형코드', axis=1).values
        y = self.test_data['사고유형코드'].values
        
        # Use part of data for training, part for testing
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model.fit(X_train, y_train)
        
        model_path = self.output_dir / "test_model.joblib"
        joblib.dump(model, model_path)
        
        # Create test data for evaluation
        test_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        test_data['사고유형코드'] = y_test
        
        # Test evaluation
        results = pipeline.evaluate_single_model(
            model_path, test_data, '사고유형코드', model_name='test_rf'
        )
        
        self.assertIn('accuracy', results)
        self.assertIn('model_name', results)
        self.assertEqual(results['model_name'], 'test_rf')


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2) 