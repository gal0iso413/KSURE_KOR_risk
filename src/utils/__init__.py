"""
Utilities package for the first_model project.

This package provides essential utilities for logging, file operations,
data validation, performance monitoring, and error handling.
"""

# Logging utilities
from .logging_config import (
    get_logger,
    setup_logger,
    configure_root_logger,
    set_logging_level,
    create_logger_hierarchy,
    disable_third_party_loggers,
    ColoredFormatter,
    _create_console_formatter,
    _create_file_formatter,
    _setup_log_directory,
    _generate_log_filename
)

# Common utilities
from .common import (
    # File operations
    validate_file_exists,
    validate_directory_path,
    create_directory_if_not_exists,
    validate_file_format,
    
    # Data loading/saving
    safe_load_csv,
    safe_save_csv,
    
    # Data validation
    validate_dataframe,
    validate_columns_exist,
    
    # Data processing
    clean_column_names,
    get_memory_usage_summary,
    
    # Error handling
    standardize_error_handling,
    safe_execute
)

# Package version and metadata
__version__ = "1.0.0"
__author__ = "First Model Team"

# All available exports
__all__ = [
    # Logging
    'get_logger',
    'setup_logger',
    'configure_root_logger',
    'set_logging_level',
    'create_logger_hierarchy',
    'disable_third_party_loggers',
    'ColoredFormatter',
    '_create_console_formatter',
    '_create_file_formatter',
    '_setup_log_directory',
    '_generate_log_filename',
    
    # File operations
    'validate_file_exists',
    'validate_directory_path', 
    'create_directory_if_not_exists',
    'validate_file_format',
    
    # Data operations
    'safe_load_csv',
    'safe_save_csv',
    'validate_dataframe',
    'validate_columns_exist',
    
    # Data processing
    'clean_column_names',
    'get_memory_usage_summary',
    
    # Error handling
    'standardize_error_handling',
    'safe_execute'
] 