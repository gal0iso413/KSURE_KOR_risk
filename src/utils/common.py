"""
Common utility functions used across the first_model package.

This module provides essential utility functions for file operations, data validation,
error handling, and other common tasks used throughout the project.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Union, Any, Callable
from functools import wraps
import pandas as pd

from .logging_config import get_logger
from constants import (
    DEFAULT_ENCODING, SUPPORTED_DATA_FORMATS, MIN_ROWS_THRESHOLD,
    MAX_MISSING_RATIO, MEMORY_USAGE_THRESHOLD, PROCESSING_TIME_THRESHOLD
)

logger = get_logger(__name__)


# ============================================================================
# FILE VALIDATION AND OPERATIONS
# ============================================================================

def validate_file_exists(file_path: Union[str, Path], description: str = "File") -> Path:
    """
    Validate that a file exists and is readable.
    
    Args:
        file_path: Path to the file
        description: Description of the file for error messages
        
    Returns:
        Path object of the validated file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is not a file
        PermissionError: If file is not readable
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"{description} is not a file: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"{description} is not readable: {file_path}")
    
    return file_path


def validate_directory_path(dir_path: Union[str, Path], description: str = "Directory") -> Path:
    """
    Validate that a directory path is valid and accessible.
    
    Args:
        dir_path: Path to the directory
        description: Description for error messages
        
    Returns:
        Path object of the validated directory
        
    Raises:
        ValueError: If path exists but is not a directory
        PermissionError: If directory is not accessible
    """
    dir_path = Path(dir_path)
    
    if dir_path.exists() and not dir_path.is_dir():
        raise ValueError(f"{description} exists but is not a directory: {dir_path}")
    
    # Check parent directory permissions if directory doesn't exist
    parent_dir = dir_path.parent
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"Cannot create {description.lower()} - no write permission in parent: {parent_dir}")
    
    return dir_path


def create_directory_if_not_exists(dir_path: Union[str, Path], description: str = "Directory") -> Path:
    """
    Create directory if it doesn't exist with proper validation.
    
    Args:
        dir_path: Path to the directory
        description: Description for logging
        
    Returns:
        Path object of the created/existing directory
    """
    dir_path = validate_directory_path(dir_path, description)
    
    if not dir_path.exists():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created {description.lower()}: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create {description.lower()} {dir_path}: {e}")
            raise
    
    return dir_path


def validate_file_format(file_path: Union[str, Path], supported_formats: List[str] = None) -> bool:
    """
    Validate file format is supported.
    
    Args:
        file_path: Path to the file
        supported_formats: List of supported file extensions
        
    Returns:
        True if format is supported
        
    Raises:
        ValueError: If file format is not supported
    """
    if supported_formats is None:
        supported_formats = SUPPORTED_DATA_FORMATS
    
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    
    if file_extension not in supported_formats:
        raise ValueError(f"Unsupported file format '{file_extension}'. "
                        f"Supported formats: {supported_formats}")
    
    return True


# ============================================================================
# DATA LOADING AND SAVING
# ============================================================================

def safe_load_csv(
    file_path: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
    validate_data: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Safely load CSV file with comprehensive error handling and validation.
    
    Args:
        file_path: Path to CSV file
        encoding: File encoding
        validate_data: Whether to perform data validation
        **kwargs: Additional arguments for pd.read_csv()
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty, invalid, or fails validation
    """
    # Validate file exists and format
    file_path = validate_file_exists(file_path, "CSV file")
    validate_file_format(file_path, ['.csv'])
    
    start_time = time.time()
    
    try:
        # Load CSV with progress logging for large files
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # Log for files > 100MB
            logger.info(f"Loading large CSV file ({file_size_mb:.1f}MB): {file_path}")
        
        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        
        load_time = time.time() - start_time
        logger.info(f"Successfully loaded CSV with shape {df.shape} in {load_time:.2f}s: {file_path}")
        
        # Validate data if requested
        if validate_data:
            validate_dataframe(df, "loaded CSV data")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty or has no valid data: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file {file_path}: {str(e)}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error reading {file_path} with {encoding}: {str(e)}")
    except MemoryError:
        raise ValueError(f"CSV file {file_path} is too large to load into memory")


def safe_save_csv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    description: str = "processed data",
    validate_data: bool = True,
    **kwargs
) -> None:
    """
    Safely save DataFrame to CSV with directory creation and validation.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        description: Description for logging
        validate_data: Whether to validate data before saving
        **kwargs: Additional arguments for df.to_csv()
    """
    if validate_data:
        validate_dataframe(df, description)
    
    file_path = Path(file_path)
    
    # Create parent directory if needed
    create_directory_if_not_exists(file_path.parent, f"{description} directory")
    
    # Default arguments
    save_kwargs = {'index': False, 'encoding': DEFAULT_ENCODING}
    save_kwargs.update(kwargs)
    
    start_time = time.time()
    
    try:
        df.to_csv(file_path, **save_kwargs)
        save_time = time.time() - start_time
        
        # Log file size info
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"Successfully saved {description} to {file_path} "
                   f"(shape: {df.shape}, size: {file_size_mb:.1f}MB, time: {save_time:.2f}s)")
        
    except Exception as e:
        logger.error(f"Failed to save {description} to {file_path}: {str(e)}")
        raise


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame, 
    description: str = "DataFrame",
    min_rows: int = MIN_ROWS_THRESHOLD,
    max_missing_ratio: float = MAX_MISSING_RATIO
) -> None:
    """
    Comprehensive DataFrame validation with detailed reporting.
    
    Args:
        df: DataFrame to validate
        description: Description for error messages
        min_rows: Minimum number of rows required
        max_missing_ratio: Maximum ratio of missing values allowed
        
    Raises:
        ValueError: If validation fails
    """
    # Basic structure validation
    if df.empty:
        raise ValueError(f"{description} is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"{description} has only {len(df)} rows, minimum {min_rows} required")
    
    # Missing values validation
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_ratio > max_missing_ratio:
        raise ValueError(f"{description} has {missing_ratio:.1%} missing values, "
                        f"maximum {max_missing_ratio:.1%} allowed")
    
    # Duplicate columns check
    if df.columns.duplicated().any():
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{description} has duplicate column names: {duplicated_cols}")
    
    # Log validation summary
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Validated {description}: {df.shape} shape, "
               f"{missing_ratio:.1%} missing, {memory_usage_mb:.1f}MB memory")


def validate_columns_exist(df: pd.DataFrame, columns: List[str], column_type: str = "specified") -> List[str]:
    """
    Validate that specified columns exist in DataFrame.
    
    Args:
        df: DataFrame to check
        columns: List of column names to validate
        column_type: Type description for logging
        
    Returns:
        List of existing columns
    """
    if not columns:
        return []
    
    missing_cols = [col for col in columns if col not in df.columns]
    existing_cols = [col for col in columns if col in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing {column_type} columns: {missing_cols}")
        if existing_cols:
            logger.info(f"Using existing {column_type} columns: {existing_cols}")
    
    return existing_cols


# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================

def clean_column_names(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Clean DataFrame column names by stripping whitespace and normalizing.
    
    Args:
        df: Input DataFrame
        inplace: Whether to modify DataFrame in place
        
    Returns:
        DataFrame with cleaned column names
    """
    if not inplace:
        df = df.copy()
    
    original_columns = df.columns.tolist()
    
    # Clean column names
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters
                  .str.replace(' ', '_')  # Replace spaces with underscores
                  .str.lower())  # Convert to lowercase
    
    cleaned_columns = df.columns.tolist()
    
    # Log changes
    changed_columns = [
        (orig, clean) for orig, clean in zip(original_columns, cleaned_columns)
        if orig != clean
    ]
    
    if changed_columns:
        logger.info(f"Cleaned {len(changed_columns)} column names:")
        for orig, clean in changed_columns[:5]:  # Show first 5 changes
            logger.info(f"  '{orig}' -> '{clean}'")
        if len(changed_columns) > 5:
            logger.info(f"  ... and {len(changed_columns) - 5} more")
    
    return df


def get_memory_usage_summary(df: pd.DataFrame) -> dict:
    """
    Get basic memory usage summary for DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory_mb = memory_usage.sum() / (1024 * 1024)
    
    return {
        'total_memory_mb': total_memory_mb,
        'memory_per_row_kb': (total_memory_mb * 1024) / len(df) if len(df) > 0 else 0,
        'shape': df.shape
    }


# ============================================================================
# ERROR HANDLING
# ============================================================================

def standardize_error_handling(func: Callable) -> Callable:
    """
    Decorator for standardized error handling and logging with performance monitoring.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with comprehensive error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        
        logger.info(f"Starting {func_name}")
        
        try:
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            logger.info(f"Successfully completed {func_name} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in {func_name} after {execution_time:.2f}s: {str(e)}", exc_info=True)
            raise
    
    return wrapper


def safe_execute(func: Callable, *args, default_return=None, log_errors: bool = True, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Value to return if function fails
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default_return 