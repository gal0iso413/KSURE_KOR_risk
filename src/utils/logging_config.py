"""
Centralized logging configuration for the first_model package.

This module provides comprehensive logging setup with file rotation, performance
monitoring, and configurable output options for different environments.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

from constants import (
    DEFAULT_LOG_LEVEL, DEFAULT_LOG_FORMAT, DEFAULT_LOG_DATE_FORMAT,
    LOG_FILE_MAX_SIZE, LOG_BACKUP_COUNT, LOG_FILE_ENCODING, LOGS_DIR
)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to log level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: str = DEFAULT_LOG_LEVEL,
    console_output: bool = True,
    file_output: bool = True,
    colored_console: bool = True,
    max_file_size: int = LOG_FILE_MAX_SIZE,
    backup_count: int = LOG_BACKUP_COUNT
) -> logging.Logger:
    """
    Set up a comprehensive logger with file rotation and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory for log files (default: project_root/logs)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
        file_output: Whether to output to file
        colored_console: Whether to use colored output in console
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    
    # Create formatters
    console_formatter = _create_console_formatter(colored_console)
    file_formatter = _create_file_formatter()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        log_dir = _setup_log_directory(log_dir)
        log_file = _generate_log_filename(log_dir, name)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding=LOG_FILE_ENCODING
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Log the log file creation
        logger.info(f"Log file created: {log_file}")
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Args:
        name: Logger name (usually __name__)
        **kwargs: Additional arguments for setup_logger
        
    Returns:
        Configured logger
    """
    return setup_logger(name, **kwargs)


def configure_root_logger(
    level: str = DEFAULT_LOG_LEVEL,
    format_string: str = DEFAULT_LOG_FORMAT
) -> None:
    """
    Configure the root logger with basic settings.
    
    Args:
        level: Logging level
        format_string: Log message format
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        datefmt=DEFAULT_LOG_DATE_FORMAT
    )


def set_logging_level(logger_name: str, level: str) -> None:
    """
    Set logging level for a specific logger.
    
    Args:
        logger_name: Name of the logger
        level: New logging level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def create_logger_hierarchy(
    base_name: str,
    modules: list,
    **logger_kwargs
) -> Dict[str, logging.Logger]:
    """
    Create a hierarchy of loggers for different modules.
    
    Args:
        base_name: Base name for the logger hierarchy
        modules: List of module names
        **logger_kwargs: Arguments for setup_logger
        
    Returns:
        Dictionary mapping module names to loggers
    """
    loggers = {}
    
    for module in modules:
        logger_name = f"{base_name}.{module}"
        loggers[module] = setup_logger(logger_name, **logger_kwargs)
    
    return loggers


def disable_third_party_loggers(level: str = 'WARNING') -> None:
    """
    Disable or reduce verbosity of third-party library loggers.
    
    Args:
        level: Minimum level for third-party loggers
    """
    third_party_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'sklearn',
        'pandas',
        'numpy'
    ]
    
    log_level = getattr(logging, level.upper(), logging.WARNING)
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(log_level)


def _create_console_formatter(colored: bool) -> logging.Formatter:
    """Create formatter for console output."""
    format_string = DEFAULT_LOG_FORMAT
    
    if colored:
        return ColoredFormatter(format_string, datefmt=DEFAULT_LOG_DATE_FORMAT)
    else:
        return logging.Formatter(format_string, datefmt=DEFAULT_LOG_DATE_FORMAT)


def _create_file_formatter() -> logging.Formatter:
    """Create formatter for file output with additional information."""
    detailed_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )
    
    return logging.Formatter(detailed_format, datefmt=DEFAULT_LOG_DATE_FORMAT)


def _setup_log_directory(log_dir: Optional[str]) -> Path:
    """Set up log directory and return Path object."""
    if log_dir is None:
        log_dir = LOGS_DIR
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return log_dir


def _generate_log_filename(log_dir: Path, logger_name: str) -> Path:
    """Generate log filename based on logger name and current date."""
    # Clean logger name for filename
    clean_name = logger_name.replace('.', '_').replace('/', '_')
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{clean_name}_{date_str}.log"
    
    return log_dir / filename


# Configure third-party loggers on import
disable_third_party_loggers()

# Set up a package-level logger
package_logger = get_logger('first_model') 