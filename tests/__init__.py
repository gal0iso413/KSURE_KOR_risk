"""
Test package for the first_model project.

This package contains comprehensive tests for all project components including
models, utilities, data processing, and pipeline functionality.

Usage:
    python -m pytest tests/
    python -m unittest discover tests/
    python tests/run_all_tests.py
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from .test_model import TestModelComponents, TestModelTraining, TestEvaluation
from .test_preprocessing import (
    TestFileOperations, TestDataOperations, TestDataProcessing,
    TestModelUtils, TestLogging, TestErrorHandling
)

# Test suite configuration
__version__ = "1.0.0"
__author__ = "First Model Test Team"

# All test classes
ALL_TEST_CLASSES = [
    # Model tests
    TestModelComponents,
    TestModelTraining, 
    TestEvaluation,
    
    # Utility tests
    TestFileOperations,
    TestDataOperations,
    TestDataProcessing,
    TestModelUtils,
    TestLogging,
    TestErrorHandling
]


def create_test_suite():
    """
    Create a test suite containing all test cases.
    
    Returns:
        unittest.TestSuite: Complete test suite
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in ALL_TEST_CLASSES:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_all_tests(verbosity=2):
    """
    Run all tests with specified verbosity.
    
    Args:
        verbosity: Test output verbosity level (0-2)
        
    Returns:
        unittest.TestResult: Test results
    """
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def run_specific_tests(test_pattern="test_*", verbosity=2):
    """
    Run tests matching a specific pattern.
    
    Args:
        test_pattern: Pattern to match test methods
        verbosity: Test output verbosity level
        
    Returns:
        unittest.TestResult: Test results
    """
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(Path(__file__).parent),
        pattern=test_pattern,
        top_level_dir=str(project_root)
    )
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == "__main__":
    # Run all tests when this module is executed directly
    print("Running all first_model tests...")
    result = run_all_tests()
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1) 