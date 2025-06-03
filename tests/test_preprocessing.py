"""
Comprehensive test suite for data processing and utility functions.

This module tests data preprocessing, utility functions, logging,
and other common functionality.
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging

# Import our utilities and components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import (
    validate_file_exists, validate_directory_path, create_directory_if_not_exists,
    validate_file_format, safe_load_csv, safe_save_csv, validate_dataframe,
    validate_columns_exist, clean_column_names, get_memory_usage_summary,
    standardize_error_handling, safe_execute
)
from src.utils.logging_config import (
    get_logger, setup_logger, ColoredFormatter, PerformanceFilter
)
from src.constants import (
    DEFAULT_ENCODING, SUPPORTED_DATA_FORMATS, MIN_ROWS_THRESHOLD, 
    DEFAULT_RANDOM_STATE, ACCIDENT_SEVERITY_MAPPING
)
from models.model_utils import DataProcessor, ImbalanceHandler


class TestFileOperations(unittest.TestCase):
    """Test file operation utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test CSV data
        np.random.seed(DEFAULT_RANDOM_STATE)
        cls.test_data = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randint(0, 10, 50),
            'feature_3': np.random.choice(['A', 'B', 'C'], 50),
            'target': np.random.choice([0, 1, 2], 50)
        })
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        self.test_output_dir = self.temp_dir / f"test_{self._testMethodName}"
        self.test_output_dir.mkdir(exist_ok=True)
    
    def test_validate_file_exists(self):
        """Test file existence validation."""
        # Create test file
        test_file = self.test_output_dir / "test_file.csv"
        self.test_data.to_csv(test_file, index=False)
        
        # Test valid file
        validated_path = validate_file_exists(test_file)
        self.assertEqual(validated_path, test_file)
        
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            validate_file_exists(self.test_output_dir / "nonexistent.csv")
        
        # Test directory instead of file
        with self.assertRaises(ValueError):
            validate_file_exists(self.test_output_dir)
    
    def test_validate_directory_path(self):
        """Test directory path validation."""
        # Test existing directory
        validated_path = validate_directory_path(self.test_output_dir)
        self.assertEqual(validated_path, self.test_output_dir)
        
        # Test non-existent directory (should not raise error)
        new_dir = self.test_output_dir / "new_directory"
        validated_path = validate_directory_path(new_dir)
        self.assertEqual(validated_path, new_dir)
        
        # Test file instead of directory
        test_file = self.test_output_dir / "test_file.csv"
        test_file.touch()
        with self.assertRaises(ValueError):
            validate_directory_path(test_file)
    
    def test_create_directory_if_not_exists(self):
        """Test directory creation."""
        new_dir = self.test_output_dir / "new_test_dir"
        
        # Directory should not exist initially
        self.assertFalse(new_dir.exists())
        
        # Create directory
        created_dir = create_directory_if_not_exists(new_dir)
        
        # Check directory was created
        self.assertTrue(new_dir.exists())
        self.assertTrue(new_dir.is_dir())
        self.assertEqual(created_dir, new_dir)
        
        # Test creating existing directory (should not fail)
        created_again = create_directory_if_not_exists(new_dir)
        self.assertEqual(created_again, new_dir)
    
    def test_validate_file_format(self):
        """Test file format validation."""
        # Test valid CSV file
        csv_file = self.test_output_dir / "test.csv"
        self.assertTrue(validate_file_format(csv_file, ['.csv']))
        
        # Test invalid format
        with self.assertRaises(ValueError):
            validate_file_format(self.test_output_dir / "test.txt", ['.csv'])
        
        # Test default supported formats
        for ext in SUPPORTED_DATA_FORMATS:
            test_file = self.test_output_dir / f"test{ext}"
            self.assertTrue(validate_file_format(test_file))


class TestDataOperations(unittest.TestCase):
    """Test data loading, saving, and validation utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Create various test datasets
        np.random.seed(DEFAULT_RANDOM_STATE)
        
        # Normal dataset
        cls.normal_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randint(0, 10, 100),
            'feature_3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1, 2], 100)
        })
        
        # Dataset with missing values
        cls.missing_data = cls.normal_data.copy()
        cls.missing_data.loc[10:20, 'feature_1'] = np.nan
        cls.missing_data.loc[30:35, 'feature_3'] = np.nan
        
        # Small dataset (below threshold)
        cls.small_data = cls.normal_data.head(5)
        
        # Dataset with duplicate columns
        cls.duplicate_cols_data = cls.normal_data.copy()
        cls.duplicate_cols_data['feature_1_duplicate'] = cls.duplicate_cols_data['feature_1']
        cls.duplicate_cols_data.columns = ['feature_1', 'feature_2', 'feature_3', 'target', 'feature_1']
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        self.test_output_dir = self.temp_dir / f"test_{self._testMethodName}"
        self.test_output_dir.mkdir(exist_ok=True)
    
    def test_safe_load_csv(self):
        """Test safe CSV loading."""
        # Save test data
        test_file = self.test_output_dir / "test_data.csv"
        self.normal_data.to_csv(test_file, index=False)
        
        # Test loading
        loaded_data = safe_load_csv(test_file)
        pd.testing.assert_frame_equal(loaded_data, self.normal_data)
        
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            safe_load_csv(self.test_output_dir / "nonexistent.csv")
        
        # Test empty file
        empty_file = self.test_output_dir / "empty.csv"
        empty_file.touch()
        with self.assertRaises(ValueError):
            safe_load_csv(empty_file)
        
        # Test loading without validation
        loaded_no_validation = safe_load_csv(test_file, validate_data=False)
        pd.testing.assert_frame_equal(loaded_no_validation, self.normal_data)
    
    def test_safe_save_csv(self):
        """Test safe CSV saving."""
        output_file = self.test_output_dir / "output_data.csv"
        
        # Test saving
        safe_save_csv(self.normal_data, output_file)
        
        # Verify file was created and data is correct
        self.assertTrue(output_file.exists())
        loaded_data = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(loaded_data, self.normal_data)
        
        # Test saving with custom parameters
        custom_file = self.test_output_dir / "custom_data.csv"
        safe_save_csv(self.normal_data, custom_file, description="custom test data", sep=';')
        
        self.assertTrue(custom_file.exists())
        loaded_custom = pd.read_csv(custom_file, sep=';')
        pd.testing.assert_frame_equal(loaded_custom, self.normal_data)
    
    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        # Test valid DataFrame
        validate_dataframe(self.normal_data)  # Should not raise
        
        # Test empty DataFrame
        with self.assertRaises(ValueError):
            validate_dataframe(pd.DataFrame())
        
        # Test DataFrame below minimum rows
        with self.assertRaises(ValueError):
            validate_dataframe(self.small_data, min_rows=10)
        
        # Test DataFrame with too many missing values
        with self.assertRaises(ValueError):
            validate_dataframe(self.missing_data, max_missing_ratio=0.05)
        
        # Test DataFrame with duplicate columns
        with self.assertRaises(ValueError):
            validate_dataframe(self.duplicate_cols_data)
    
    def test_validate_columns_exist(self):
        """Test column existence validation."""
        # Test existing columns
        existing_cols = validate_columns_exist(
            self.normal_data, 
            ['feature_1', 'feature_2'], 
            'test'
        )
        self.assertEqual(existing_cols, ['feature_1', 'feature_2'])
        
        # Test mix of existing and non-existing columns
        mixed_cols = validate_columns_exist(
            self.normal_data,
            ['feature_1', 'nonexistent_col', 'target'],
            'mixed'
        )
        self.assertEqual(set(mixed_cols), {'feature_1', 'target'})
        
        # Test empty column list
        empty_result = validate_columns_exist(self.normal_data, [], 'empty')
        self.assertEqual(empty_result, [])


class TestDataProcessing(unittest.TestCase):
    """Test data processing utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test data with messy column names
        cls.messy_data = pd.DataFrame({
            'Feature 1!@#': [1, 2, 3, 4, 5],
            '  Feature 2  ': [6, 7, 8, 9, 10],
            'Feature-3 (Special)': [11, 12, 13, 14, 15],
            'Normal_Feature': [16, 17, 18, 19, 20]
        })
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def test_clean_column_names(self):
        """Test column name cleaning."""
        # Test cleaning without modifying original
        cleaned_data = clean_column_names(self.messy_data, inplace=False)
        
        # Original should be unchanged
        self.assertNotEqual(list(cleaned_data.columns), list(self.messy_data.columns))
        
        # Cleaned columns should be normalized
        expected_cols = ['feature_1', 'feature_2', 'feature3_special', 'normal_feature']
        self.assertEqual(len(cleaned_data.columns), len(expected_cols))
        
        # Test in-place cleaning
        data_copy = self.messy_data.copy()
        clean_column_names(data_copy, inplace=True)
        self.assertNotEqual(list(data_copy.columns), list(self.messy_data.columns))
    
    def test_get_memory_usage_summary(self):
        """Test memory usage summary."""
        summary = get_memory_usage_summary(self.messy_data)
        
        # Check required keys
        required_keys = ['total_memory_mb', 'memory_per_row_kb', 'memory_by_dtype', 'shape']
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check values are reasonable
        self.assertGreater(summary['total_memory_mb'], 0)
        self.assertGreater(summary['memory_per_row_kb'], 0)
        self.assertEqual(summary['shape'], self.messy_data.shape)
        self.assertIsInstance(summary['memory_by_dtype'], dict)


class TestModelUtils(unittest.TestCase):
    """Test model utility functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test data for model utilities
        np.random.seed(DEFAULT_RANDOM_STATE)
        cls.test_data = pd.DataFrame({
            'feature_1': np.random.randn(200),
            'feature_2': np.random.randint(0, 10, 200),
            'feature_3': np.random.choice(['A', 'B', 'C'], 200),
            'accident_code': np.random.choice(['MINOR', 'MODERATE', 'SEVERE', 'UNKNOWN'], 200)
        })
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def test_data_processor_severity_conversion(self):
        """Test accident code to severity level conversion."""
        processor = DataProcessor()
        
        # Test with known codes
        test_codes = pd.Series(['MINOR', 'MODERATE', 'SEVERE', 'UNKNOWN', 'INVALID'])
        severity_levels = processor.convert_to_severity_levels(test_codes)
        
        # Check conversions
        expected_mapping = {
            'MINOR': 0,
            'MODERATE': 1, 
            'SEVERE': 2,
            'UNKNOWN': 0,  # Should default to 0
            'INVALID': 0   # Should default to 0
        }
        
        for i, code in enumerate(test_codes):
            self.assertEqual(severity_levels.iloc[i], expected_mapping[code])
        
        # Test empty series
        with self.assertRaises(ValueError):
            processor.convert_to_severity_levels(pd.Series([]))
    
    def test_data_processor_validation(self):
        """Test data validation in DataProcessor."""
        processor = DataProcessor()
        
        X = self.test_data[['feature_1', 'feature_2']].values
        y = np.random.choice([0, 1, 2], len(X))
        
        # Test valid data
        X_val, y_val = processor.validate_features_and_target(X, y)
        self.assertEqual(X_val.shape, X.shape)
        self.assertEqual(len(y_val), len(y))
        
        # Test mismatched lengths
        with self.assertRaises(ValueError):
            processor.validate_features_and_target(X, y[:-10])
        
        # Test empty data
        with self.assertRaises(ValueError):
            processor.validate_features_and_target(np.array([]), np.array([]))
    
    def test_imbalance_handler(self):
        """Test imbalance handling functionality."""
        handler = ImbalanceHandler(random_state=DEFAULT_RANDOM_STATE)
        
        # Create imbalanced data
        X = np.random.randn(100, 3)
        y = np.array([0] * 80 + [1] * 15 + [2] * 5)  # Imbalanced classes
        
        # Test SMOTE
        try:
            X_smote, y_smote = handler.handle_imbalanced_data(X, y, method='smote')
            self.assertGreaterEqual(len(y_smote), len(y))
        except ImportError:
            # SMOTE might not be available in test environment
            self.skipTest("SMOTE not available")
        
        # Test class weights
        X_weight, y_weight, weights = handler.handle_imbalanced_data(X, y, method='class_weight')
        
        self.assertEqual(len(X_weight), len(X))
        self.assertEqual(len(y_weight), len(y))
        self.assertIsInstance(weights, dict)
        self.assertEqual(set(weights.keys()), {0, 1, 2})


class TestLogging(unittest.TestCase):
    """Test logging functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = Path(tempfile.mkdtemp())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger('test_logger', log_dir=self.temp_dir)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test_logger')
        
        # Test logging
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")
    
    def test_setup_logger_with_options(self):
        """Test logger setup with various options."""
        # Test with file output only
        file_logger = setup_logger(
            'test_file_logger',
            log_dir=self.temp_dir,
            console_output=False,
            file_output=True
        )
        
        self.assertIsInstance(file_logger, logging.Logger)
        
        # Test with console output only
        console_logger = setup_logger(
            'test_console_logger',
            console_output=True,
            file_output=False
        )
        
        self.assertIsInstance(console_logger, logging.Logger)
    
    def test_colored_formatter(self):
        """Test colored formatter."""
        formatter = ColoredFormatter()
        
        # Create test log record
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        self.assertIsInstance(formatted, str)
        self.assertIn('Test message', formatted)
    
    def test_performance_filter(self):
        """Test performance filter."""
        perf_filter = PerformanceFilter()
        
        # Create test log record
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        result = perf_filter.filter(record)
        self.assertTrue(result)
        
        # Check if performance attributes were added
        self.assertTrue(hasattr(record, 'memory_mb'))
        self.assertTrue(hasattr(record, 'cpu_time'))


class TestErrorHandling(unittest.TestCase):
    """Test error handling utilities."""
    
    def test_standardize_error_handling_decorator(self):
        """Test standardized error handling decorator."""
        
        @standardize_error_handling
        def test_function_success():
            return "success"
        
        @standardize_error_handling
        def test_function_error():
            raise ValueError("Test error")
        
        # Test successful function
        result = test_function_success()
        self.assertEqual(result, "success")
        
        # Test function with error
        with self.assertRaises(ValueError):
            test_function_error()
    
    def test_safe_execute(self):
        """Test safe execution utility."""
        
        def success_function():
            return "success"
        
        def error_function():
            raise ValueError("Test error")
        
        # Test successful execution
        result = safe_execute(success_function)
        self.assertEqual(result, "success")
        
        # Test error handling with default return
        result = safe_execute(error_function, default_return="default")
        self.assertEqual(result, "default")
        
        # Test error handling without logging
        result = safe_execute(error_function, default_return="quiet", log_errors=False)
        self.assertEqual(result, "quiet")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2) 