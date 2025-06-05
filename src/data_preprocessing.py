"""
Data preprocessing module for the accident severity classification project.

This module provides comprehensive data cleaning, transformation, and preprocessing
functionality including missing value handling, outlier detection, scaling, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path
import re

# Import project utilities and constants
from utils.logging_config import get_logger
from utils.common import (
    safe_load_csv, safe_save_csv, clean_column_names, 
    standardize_error_handling, create_directory_if_not_exists
)
from constants import (
    DEFAULT_DATE_COLUMNS, DEFAULT_DATE_FORMAT, DEFAULT_OUTLIER_METHOD,
    DEFAULT_OUTLIER_THRESHOLD, DEFAULT_ZSCORE_THRESHOLD, DEFAULT_SCALER_TYPE,
    DEFAULT_ENCODING_METHOD, DEFAULT_ENCODING
)

# Initialize logger
logger = get_logger(__name__)


# ============================================================================
# UTILITY FUNCTIONS FOR COLUMN NAME CLEANING
# ============================================================================

def clean_column_name_for_matching(column_name: str) -> str:
    """
    Apply the same column cleaning logic as clean_column_names() for consistent matching.
    
    This ensures that column names provided in configuration match the cleaned DataFrame columns.
    """
    cleaned = (column_name
               .strip()
               .replace(' ', '_'))  # First replace spaces with underscores
    
    # Remove special characters (including (*), !, @, etc.) - same as clean_column_names
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    
    # Replace any remaining spaces with underscores and convert to lowercase
    cleaned = cleaned.replace(' ', '_').lower()
    
    return cleaned


def clean_column_list_for_matching(columns: List[str]) -> List[str]:
    """Clean a list of column names using the same logic as clean_column_names()."""
    return [clean_column_name_for_matching(col) for col in columns] if columns else []


class DataValidator:
    """Class for validating data integrity and structure."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> None:
        """Validate basic DataFrame requirements."""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if len(df) < min_rows:
            raise ValueError(f"DataFrame has only {len(df)} rows, minimum {min_rows} required")
        
        if df.columns.duplicated().any():
            duplicated_cols = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names found: {duplicated_cols}")
    
    @staticmethod
    def validate_columns_exist(df: pd.DataFrame, columns: List[str], column_type: str = "specified") -> List[str]:
        """Validate that specified columns exist in DataFrame."""
        if not columns:
            return []
        
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing {column_type} columns: {missing_cols}")
            # Return only existing columns
            existing_cols = [col for col in columns if col in df.columns]
            logger.info(f"Using existing {column_type} columns: {existing_cols}")
            return existing_cols
        
        return columns
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
        """Validate and filter numeric columns."""
        valid_numeric = []
        for col in columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    valid_numeric.append(col)
                else:
                    logger.warning(f"Column '{col}' is not numeric, skipping")
        
        return valid_numeric


class DataCleaner:
    """Class for data cleaning operations."""
    
    def __init__(self):
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
    
    @standardize_error_handling
    def clean_data(
        self, 
        df: pd.DataFrame, 
        date_columns: List[str] = None,
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Clean the data by handling missing values, duplicates, and data type conversions.
        
        Args:
            df: Input DataFrame
            date_columns: List of column names containing dates
            numeric_columns: List of columns to be treated as numeric
            
        Returns:
            Cleaned DataFrame
        """
        DataValidator.validate_dataframe(df)
        df = df.copy()
        
        # Clean column names first
        df = clean_column_names(df, inplace=True)
        
        # Convert date columns
        if date_columns:
            df = self._convert_date_columns(df, date_columns)
        
        # Handle missing values
        df = self._handle_missing_values(df, numeric_columns)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        return df
    
    def _convert_date_columns(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Convert date columns from integer format to datetime."""
        valid_date_cols = DataValidator.validate_columns_exist(df, date_columns, "date")
        
        for col in valid_date_cols:
            try:
                # Handle different date formats
                if df[col].dtype == 'object':
                    df[col] = pd.to_datetime(df[col], format=DEFAULT_DATE_FORMAT, errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col].astype(str), format=DEFAULT_DATE_FORMAT, errors='coerce')
                
                # Check for conversion failures
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Failed to convert {null_count} values in date column '{col}'")
                
                logger.info(f"Converted {col} to datetime")
                
            except Exception as e:
                logger.error(f"Failed to convert date column '{col}': {e}")
                # Keep original column if conversion fails
                continue
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, numeric_columns: Optional[List[str]]) -> pd.DataFrame:
        """Handle missing values using appropriate strategies."""
        logger.info("Starting missing value handling...")
        
        if numeric_columns:
            # Use ONLY the specified columns (no auto-detection)
            cleaned_numeric_columns = clean_column_list_for_matching(numeric_columns)
            valid_numeric = DataValidator.validate_numeric_columns(df, cleaned_numeric_columns)
            if valid_numeric:
                logger.info(f"Processing additional specified numeric columns: {valid_numeric}")
                df = self._fill_numeric_missing(df, valid_numeric)
        else:
            # Auto-detect ONLY when no columns specified
            all_numeric_cols = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
            numeric_with_missing = [col for col in all_numeric_cols if df[col].isnull().any()]
            if numeric_with_missing:
                logger.info(f"Auto-detected {len(numeric_with_missing)} numeric columns with missing values: {numeric_with_missing}")
                df = self._fill_numeric_missing(df, numeric_with_missing)
        
        # Handle categorical missing values
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_with_missing = [col for col in categorical_columns if df[col].isnull().any()]
        if categorical_with_missing:
            logger.info(f"Found {len(categorical_with_missing)} categorical columns with missing values: {categorical_with_missing}")
            df = self._fill_categorical_missing(df, categorical_with_missing.tolist())
        
        return df
    
    def _fill_numeric_missing(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values in numeric columns with median."""
        for col in columns:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                original_nulls = df[col].isnull().sum()
                df[col] = df[col].fillna(median_value)
                logger.info(f"Filled {original_nulls} missing values in '{col}' with median ({median_value:.2f})")
        
        return df
    
    def _fill_categorical_missing(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values in categorical columns with mode."""
        for col in columns:
            if col in df.columns and df[col].isnull().any():
                mode_values = df[col].mode()
                if len(mode_values) > 0:
                    mode_value = mode_values[0]
                    original_nulls = df[col].isnull().sum()
                    df[col] = df[col].fillna(mode_value)
                    logger.info(f"Filled {original_nulls} missing values in '{col}' with mode ('{mode_value}')")
                else:
                    # If no mode (all values are unique), fill with a placeholder
                    df[col] = df[col].fillna('Unknown')
                    logger.info(f"Filled missing values in '{col}' with 'Unknown' (no clear mode)")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from DataFrame."""
        original_len = len(df)
        df = df.drop_duplicates()
        removed_count = original_len - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows ({removed_count/original_len*100:.1f}%)")
        
        return df


class OutlierHandler:
    """Class for handling outliers in data."""
    
    @staticmethod
    @standardize_error_handling
    def handle_outliers(
        df: pd.DataFrame, 
        columns: List[str], 
        method: str = DEFAULT_OUTLIER_METHOD,
        threshold: float = DEFAULT_OUTLIER_THRESHOLD
    ) -> pd.DataFrame:
        """
        Handle outliers in specified numeric columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with handled outliers
        """
        DataValidator.validate_dataframe(df)
        df = df.copy()
        
        logger.info("Starting outlier handling...")
        
        if columns:  # If columns specified, use ONLY those
            cleaned_provided_columns = clean_column_list_for_matching(columns)
            columns_to_handle = [col for col in cleaned_provided_columns if col in df.columns]
        else:  # If no columns specified, auto-detect ALL numeric
            columns_to_handle = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        
        # Validate that they are actually numeric
        valid_columns = DataValidator.validate_numeric_columns(df, columns_to_handle)
        
        if not valid_columns:
            logger.warning("No valid numeric columns found for outlier handling")
            return df
        
        logger.info(f"Handling outliers in {len(valid_columns)} numeric columns: {valid_columns}")
        
        for col in valid_columns:
            if method == 'iqr':
                df = OutlierHandler._handle_iqr_outliers(df, col, threshold)
            elif method == 'zscore':
                df = OutlierHandler._handle_zscore_outliers(df, col, threshold)
            else:
                logger.warning(f"Unknown outlier method '{method}', skipping column '{col}'")
        
        return df
    
    @staticmethod
    def _handle_iqr_outliers(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Count outliers before clipping
        outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        
        # Cap the outliers
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        if outliers_count > 0:
            logger.info(f"Handled {outliers_count} outliers in '{column}' using IQR method "
                       f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        
        return df
    
    @staticmethod
    def _handle_zscore_outliers(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
        """Handle outliers using Z-score method."""
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        # Count outliers before clipping
        outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        
        # Cap the outliers
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        if outliers_count > 0:
            logger.info(f"Handled {outliers_count} outliers in '{column}' using Z-score method "
                       f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        
        return df


class DataTransformer:
    """Class for data transformation operations."""
    
    def __init__(self):
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}

    def _clean_column_list(self, columns: List[str]) -> List[str]:
        """Clean a list of column names using the global utility function."""
        return clean_column_list_for_matching(columns)
    
    @standardize_error_handling
    def transform_data(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        scaler_type: str = DEFAULT_SCALER_TYPE,
        encoding_method: str = DEFAULT_ENCODING_METHOD
    ) -> pd.DataFrame:
        """
        Transform the data by scaling numeric features and encoding categorical features.
        
        Args:
            df: Input DataFrame
            numeric_columns: List of numeric columns to scale
            categorical_columns: List of categorical columns to encode
            scaler_type: Type of scaler ('standard' or 'minmax')
            encoding_method: Method for encoding categoricals ('onehot' or 'label')
            
        Returns:
            Transformed DataFrame
        """
        DataValidator.validate_dataframe(df)
        df = df.copy()
        
        # Scale numeric features
        if numeric_columns:
            df = self._scale_numeric_features(df, numeric_columns, scaler_type)
        
        # Encode categorical features
        if categorical_columns:
            df = self._encode_categorical_features(df, categorical_columns, encoding_method)
        
        return df
    
    def _scale_numeric_features(self, df: pd.DataFrame, columns: List[str], scaler_type: str) -> pd.DataFrame:
        """Scale numeric features using specified scaler."""
        logger.info("Starting numeric feature scaling...")
        
        if columns:  # If columns specified, use ONLY those
            cleaned_provided_columns = self._clean_column_list(columns)
            columns_to_scale = [col for col in cleaned_provided_columns if col in df.columns]
        else:  # If no columns specified, auto-detect ALL numeric
            columns_to_scale = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        
        # Validate that they are actually numeric
        valid_columns = DataValidator.validate_numeric_columns(df, columns_to_scale)
        
        if not valid_columns:
            logger.warning("No valid numeric columns found for scaling")
            return df
        
        logger.info(f"Scaling {len(valid_columns)} numeric columns: {valid_columns}")
        
        # Create scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaler type '{scaler_type}', using StandardScaler")
            scaler = StandardScaler()
        
        # Fit and transform
        df[valid_columns] = scaler.fit_transform(df[valid_columns])
        self.scalers['numeric'] = scaler
        
        logger.info(f"Successfully scaled {len(valid_columns)} numeric columns using {scaler_type} scaler")
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, columns: List[str], encoding_method: str) -> pd.DataFrame:
        """Encode categorical features using specified method."""
        logger.info("Starting categorical feature encoding...")
        
        # Auto-detect all categorical columns
        all_categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Clean the provided column names to match the cleaned DataFrame columns
        cleaned_provided_columns = self._clean_column_list(columns)
        
        # Combine auto-detected with cleaned provided columns (remove duplicates)
        columns_to_encode = list(set(all_categorical_cols + [col for col in cleaned_provided_columns if col in df.columns]))
        
        # Validate that specified columns actually exist
        valid_columns = DataValidator.validate_columns_exist(df, columns_to_encode, "categorical")
        
        if not valid_columns:
            logger.warning("No valid categorical columns found for encoding")
            return df
        
        logger.info(f"Encoding {len(valid_columns)} categorical columns: {valid_columns}")
        
        if encoding_method == 'onehot':
            # Get original column count
            original_cols = len(df.columns)
            df = pd.get_dummies(df, columns=valid_columns, prefix=valid_columns)
            new_cols = len(df.columns)
            
            logger.info(f"Applied one-hot encoding to {len(valid_columns)} categorical columns "
                       f"(created {new_cols - original_cols} new columns)")
        
        elif encoding_method == 'label':
            for col in valid_columns:
                unique_values = df[col].nunique()
                df[col] = pd.factorize(df[col])[0]
                logger.info(f"Applied label encoding to '{col}' ({unique_values} unique values)")
        
        else:
            logger.warning(f"Unknown encoding method '{encoding_method}', skipping categorical encoding")
        
        return df


@standardize_error_handling
def preprocess_pipeline(
    input_path: str, 
    output_path: str,
    numeric_columns: List[str],
    categorical_columns: List[str],
    date_columns: List[str] = None,
    identifier_columns: Optional[List[str]] = None,
    handle_outliers_cols: Optional[List[str]] = None,
    outlier_method: str = DEFAULT_OUTLIER_METHOD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    scaler_type: str = DEFAULT_SCALER_TYPE,
    encoding_method: str = DEFAULT_ENCODING_METHOD
) -> None:
    """
    Complete preprocessing pipeline that reads data, cleans it, and saves the result.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed CSV file
        numeric_columns: List of numeric columns
        categorical_columns: List of categorical columns
        date_columns: List of date columns
        identifier_columns: List of columns to keep as identifiers but exclude from processing
        handle_outliers_cols: List of columns to handle outliers for
        outlier_method: Method for outlier handling
        outlier_threshold: Threshold for outlier detection
        scaler_type: Type of scaler to use
        encoding_method: Method for categorical encoding
    """
    # Read the data
    df = safe_load_csv(input_path, encoding=DEFAULT_ENCODING)
    logger.info(f"Loaded data with shape {df.shape}")
    
    # Keep identifier columns but exclude them from processing
    if identifier_columns:
        identifier_columns_existing = [col for col in identifier_columns if col in df.columns]
        if identifier_columns_existing:
            logger.info(f"Keeping {len(identifier_columns_existing)} columns as identifiers (excluded from processing): {identifier_columns_existing}")
        
        # Remove identifier columns from processing lists
        numeric_columns = [col for col in numeric_columns if col not in identifier_columns_existing]
        categorical_columns = [col for col in categorical_columns if col not in identifier_columns_existing]
        if date_columns:
            date_columns = [col for col in date_columns if col not in identifier_columns_existing]
        if handle_outliers_cols:
            handle_outliers_cols = [col for col in handle_outliers_cols if col not in identifier_columns_existing]
    
    # Initialize processors
    cleaner = DataCleaner()
    outlier_handler = OutlierHandler()
    transformer = DataTransformer()
    
    # Clean the data (excluding identifier columns)
    df = cleaner.clean_data(df, date_columns or DEFAULT_DATE_COLUMNS, numeric_columns)
    
    # Handle outliers if specified (excluding identifier columns)
    if handle_outliers_cols:
        df = outlier_handler.handle_outliers(df, handle_outliers_cols, outlier_method, outlier_threshold)
    
    # Transform the data (excluding identifier columns)
    df = transformer.transform_data(df, numeric_columns, categorical_columns, scaler_type, encoding_method)
    
    # Save the processed data (identifier columns will still be present)
    safe_save_csv(df, output_path, "preprocessed data")
    logger.info(f"Preprocessing pipeline completed successfully. Final shape: {df.shape}")
    if identifier_columns:
        logger.info(f"Identifier columns preserved: {[col for col in identifier_columns if col in df.columns]}")
