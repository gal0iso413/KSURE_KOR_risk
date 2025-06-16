"""
Feature engineering module for the accident severity classification project.

This module provides comprehensive feature engineering functionality including
interaction features, polynomial features, date feature extraction, dimensionality
reduction, and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path

# Import project utilities and constants
from utils.logging_config import get_logger
from utils.common import (
    safe_load_csv, safe_save_csv, standardize_error_handling,
    create_directory_if_not_exists
)
from constants import (
    DEFAULT_POLYNOMIAL_DEGREE, DEFAULT_PCA_VARIANCE_THRESHOLD, DEFAULT_ENCODING
)

# Initialize logger
logger = get_logger(__name__)


class FeatureValidator:
    """Class for validating feature engineering inputs."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> None:
        """Validate basic DataFrame requirements."""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if len(df) < min_rows:
            raise ValueError(f"DataFrame has only {len(df)} rows, minimum {min_rows} required")
    
    @staticmethod
    def validate_columns_exist(df: pd.DataFrame, columns: List[str], column_type: str = "specified") -> List[str]:
        """Validate that specified columns exist in DataFrame."""
        if not columns:
            return []
        
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing {column_type} columns: {missing_cols}")
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
    
    @staticmethod
    def validate_feature_pairs(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Validate feature pairs exist and are numeric."""
        valid_pairs = []
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                if (pd.api.types.is_numeric_dtype(df[feat1]) and 
                    pd.api.types.is_numeric_dtype(df[feat2])):
                    valid_pairs.append((feat1, feat2))
                else:
                    logger.warning(f"Feature pair ({feat1}, {feat2}) contains non-numeric columns, skipping")
            else:
                missing = [f for f in [feat1, feat2] if f not in df.columns]
                logger.warning(f"Feature pair ({feat1}, {feat2}) missing columns: {missing}")
        
        return valid_pairs


class InteractionFeatureCreator:
    """Class for creating interaction features between variables."""
    
    @staticmethod
    @standardize_error_handling
    def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified pairs of features.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs to interact
            
        Returns:
            DataFrame with added interaction features
        """
        FeatureValidator.validate_dataframe(df)
        df = df.copy()
        
        valid_pairs = FeatureValidator.validate_feature_pairs(df, feature_pairs)
        
        if not valid_pairs:
            logger.warning("No valid feature pairs found for interaction features")
            return df
        
        features_created = 0
        
        for feat1, feat2 in valid_pairs:
            # Multiplication interaction
            interaction_name = f"{feat1}_{feat2}_interaction"
            df[interaction_name] = df[feat1] * df[feat2]
            features_created += 1
            logger.info(f"Created interaction feature: {interaction_name}")
            
            # Ratio interaction (if applicable and safe)
            if InteractionFeatureCreator._is_safe_division(df[feat2]):
                ratio_name = f"{feat1}_to_{feat2}_ratio"
                df[ratio_name] = df[feat1] / df[feat2]
                features_created += 1
                logger.info(f"Created ratio feature: {ratio_name}")
            else:
                logger.warning(f"Skipping ratio feature for {feat1}/{feat2} due to zero or near-zero values")
        
        logger.info(f"Created {features_created} interaction features from {len(valid_pairs)} feature pairs")
        return df
    
    @staticmethod
    def _is_safe_division(series: pd.Series, min_threshold: float = 1e-10) -> bool:
        """Check if division is safe (no zeros or near-zero values)."""
        return (series.abs() > min_threshold).all() and not series.isnull().any()


class PolynomialFeatureCreator:
    """Class for creating polynomial features."""
    
    def __init__(self):
        self.poly_features = None
    
    @standardize_error_handling
    def create_polynomial_features(
        self, 
        df: pd.DataFrame,
        features: List[str],
        degree: int = DEFAULT_POLYNOMIAL_DEGREE,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features up to specified degree.
        
        Args:
            df: Input DataFrame
            features: List of features to create polynomials from
            degree: Maximum polynomial degree
            include_bias: Whether to include bias term
            
        Returns:
            DataFrame with added polynomial features
        """
        FeatureValidator.validate_dataframe(df)
        df = df.copy()
        
        # Validate input parameters
        if degree < 1:
            raise ValueError(f"Polynomial degree must be >= 1, got {degree}")
        
        valid_features = FeatureValidator.validate_numeric_columns(df, features)
        
        if not valid_features:
            logger.warning("No valid numeric features found for polynomial creation")
            return df
        
        # Create polynomial features
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X = df[valid_features]
        
        try:
            poly_features_array = self.poly_features.fit_transform(X)
            
            # Get feature names
            feature_names = self.poly_features.get_feature_names_out(valid_features)
            
            # Create new dataframe with polynomial features (excluding original features)
            start_idx = len(valid_features)
            poly_df = pd.DataFrame(
                poly_features_array[:, start_idx:],
                columns=feature_names[start_idx:],
                index=df.index
            )
            
            # Concatenate with original dataframe
            df = pd.concat([df, poly_df], axis=1)
            
            logger.info(f"Created {len(poly_df.columns)} polynomial features (degree {degree}) "
                       f"from {len(valid_features)} input features")
            
        except Exception as e:
            logger.error(f"Failed to create polynomial features: {e}")
            raise
        
        return df


class DateFeatureExtractor:
    """Class for extracting features from date columns."""
    
    @staticmethod
    @standardize_error_handling
    def extract_date_features(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """
        Extract features from date columns.
        
        Args:
            df: Input DataFrame
            date_columns: List of datetime columns
            
        Returns:
            DataFrame with extracted date features
        """
        FeatureValidator.validate_dataframe(df)
        df = df.copy()
        
        valid_date_cols = FeatureValidator.validate_columns_exist(df, date_columns, "date")
        
        if not valid_date_cols:
            logger.warning("No valid date columns found for feature extraction")
            return df
        
        features_created = 0
        
        for col in valid_date_cols:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.info(f"Converted '{col}' to datetime for feature extraction")
                except Exception as e:
                    logger.error(f"Failed to convert '{col}' to datetime: {e}")
                    continue
            
            # Extract date features
            date_features = DateFeatureExtractor._extract_date_components(df[col], col)
            
            # Add features to DataFrame
            for feature_name, feature_values in date_features.items():
                df[feature_name] = feature_values
                features_created += 1
            
            # Remove original date column
            df.drop(col, axis=1, inplace=True)
            logger.info(f"Extracted date features from '{col}' and removed original column")
        
        logger.info(f"Created {features_created} date features from {len(valid_date_cols)} date columns")
        return df
    
    @staticmethod
    def _extract_date_components(date_series: pd.Series, column_name: str) -> Dict[str, pd.Series]:
        """Extract individual date components."""
        return {
            f"{column_name}_year": date_series.dt.year,
            f"{column_name}_month": date_series.dt.month,
            f"{column_name}_day": date_series.dt.day,
            f"{column_name}_dayofweek": date_series.dt.dayofweek,
            f"{column_name}_quarter": date_series.dt.quarter,
            f"{column_name}_is_weekend": (date_series.dt.dayofweek >= 5).astype(int)
        }


class DimensionalityReducer:
    """Class for dimensionality reduction operations."""
    
    def __init__(self):
        self.pca = None
    
    @standardize_error_handling
    def reduce_dimensions(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_components: int,
        method: str = 'pca',
        variance_threshold: float = DEFAULT_PCA_VARIANCE_THRESHOLD
    ) -> pd.DataFrame:
        """
        Reduce dimensionality of the dataset.
        
        Args:
            df: Input DataFrame
            features: List of features to use for dimension reduction
            n_components: Number of components to keep
            method: Dimension reduction method ('pca' only for now)
            variance_threshold: Minimum variance to retain
            
        Returns:
            DataFrame with reduced dimensions
        """
        FeatureValidator.validate_dataframe(df)
        df = df.copy()
        
        valid_features = FeatureValidator.validate_numeric_columns(df, features)
        
        if not valid_features:
            logger.warning("No valid numeric features found for dimensionality reduction")
            return df
        
        if len(valid_features) <= n_components:
            logger.warning(f"Number of features ({len(valid_features)}) <= n_components ({n_components}), "
                          "skipping dimensionality reduction")
            return df
        
        if method == 'pca':
            return self._apply_pca(df, valid_features, n_components, variance_threshold)
        else:
            logger.warning(f"Unknown dimensionality reduction method '{method}', skipping")
            return df
    
    def _apply_pca(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_components: int,
        variance_threshold: float
    ) -> pd.DataFrame:
        """Apply PCA dimensionality reduction."""
        self.pca = PCA(n_components=n_components)
        
        try:
            X = df[features]
            components = self.pca.fit_transform(X)
            
            # Create component column names
            component_names = [f"PC{i+1}" for i in range(n_components)]
            
            # Add components to dataframe
            for i, name in enumerate(component_names):
                df[name] = components[:, i]
            
            # Log explained variance information
            explained_var = self.pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            logger.info(f"PCA explained variance by component: {explained_var[:5]}")  # Show first 5
            logger.info(f"Cumulative explained variance: {cumulative_var[-1]:.3f}")
            
            if cumulative_var[-1] < variance_threshold:
                logger.warning(f"PCA components explain only {cumulative_var[-1]:.3f} of variance, "
                              f"below threshold {variance_threshold}")
            
            # Remove original features
            df = df.drop(columns=features)
            
            logger.info(f"Applied PCA: reduced {len(features)} features to {n_components} components")
            
        except Exception as e:
            logger.error(f"Failed to apply PCA: {e}")
            raise
        
        return df


class FeatureSelector:
    """Class for feature selection operations."""
    
    def __init__(self):
        self.feature_selector = None
        self.selected_features = None
    
    @standardize_error_handling
    def select_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        n_features: int,
        method: str = 'f_test'
    ) -> pd.DataFrame:
        """
        Select most important features using statistical tests.
        
        Args:
            df: Input DataFrame
            features: List of features to select from
            target: Target variable name
            n_features: Number of features to select
            method: Feature selection method ('f_test' or 'mutual_info')
            
        Returns:
            DataFrame with selected features
        """
        FeatureValidator.validate_dataframe(df)
        df = df.copy()
        
        # Validate target column exists
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        valid_features = FeatureValidator.validate_numeric_columns(df, features)
        
        if not valid_features:
            logger.warning("No valid numeric features found for feature selection")
            return df
        
        if len(valid_features) <= n_features:
            logger.info(f"Number of features ({len(valid_features)}) <= n_features ({n_features}), "
                       "returning all features")
            return df[valid_features + [target]]
        
        # Create feature selector
        if method == 'f_test':
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        else:
            logger.warning(f"Unknown feature selection method '{method}', using f_test")
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        
        try:
            # Fit and transform
            X = df[valid_features]
            y = df[target]
            
            selected_features = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            selected_names = np.array(valid_features)[selected_mask]
            
            # Get feature scores for logging
            scores = self.feature_selector.scores_
            selected_scores = scores[selected_mask]
            
            # Create new dataframe with selected features
            df_selected = pd.DataFrame(
                selected_features,
                columns=selected_names,
                index=df.index
            )
            
            # Add target variable
            df_selected[target] = y
            
            # Store selected feature names
            self.selected_features = selected_names.tolist()
            
            logger.info(f"Selected {len(selected_names)} features using {method}:")
            for name, score in zip(selected_names[:5], selected_scores[:5]):  # Show top 5
                logger.info(f"  {name}: {score:.3f}")
            
            return df_selected
            
        except Exception as e:
            logger.error(f"Failed to perform feature selection: {e}")
            raise


class FeatureEngineeringPipeline:
    """Main pipeline class for feature engineering operations."""
    
    def __init__(self):
        self.interaction_creator = InteractionFeatureCreator()
        self.polynomial_creator = PolynomialFeatureCreator()
        self.date_extractor = DateFeatureExtractor()
        self.dimensionality_reducer = DimensionalityReducer()
        self.feature_selector = FeatureSelector()
    
    @standardize_error_handling
    def run_pipeline(
        self,
        df: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: List[str],
        date_features: List[str],
        target_column: Optional[str] = None,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        polynomial_features: Optional[List[str]] = None,
        polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
        n_components: Optional[int] = None,
        n_select_features: Optional[int] = None,
        identifier_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            date_features: List of date feature names
            target_column: Name of target column
            feature_pairs: List of feature pairs for interaction
            polynomial_features: List of features for polynomial creation
            polynomial_degree: Degree for polynomial features
            n_components: Number of PCA components
            n_select_features: Number of features to select
            identifier_columns: List of identifier columns to exclude from processing
            
        Returns:
            Transformed DataFrame
        """
        logger.info(f"Starting feature engineering pipeline with input shape: {df.shape}")
        
        # Handle identifier columns
        if identifier_columns:
            identifier_columns_existing = [col for col in identifier_columns if col in df.columns]
            if identifier_columns_existing:
                logger.info(f"Excluding {len(identifier_columns_existing)} identifier columns from feature engineering: {identifier_columns_existing}")
                
                # Remove identifier columns from processing lists
                numeric_features = [col for col in numeric_features if col not in identifier_columns_existing]
                categorical_features = [col for col in categorical_features if col not in identifier_columns_existing]
                if date_features:
                    date_features = [col for col in date_features if col not in identifier_columns_existing]
                if feature_pairs:
                    feature_pairs = [(f1, f2) for f1, f2 in feature_pairs 
                                   if f1 not in identifier_columns_existing and f2 not in identifier_columns_existing]
                if polynomial_features:
                    polynomial_features = [col for col in polynomial_features if col not in identifier_columns_existing]
        
        # Create interaction features
        if feature_pairs:
            df = self.interaction_creator.create_interaction_features(df, feature_pairs)
        
        # Create polynomial features
        if polynomial_features:
            df = self.polynomial_creator.create_polynomial_features(
                df, polynomial_features, polynomial_degree
            )
        
        # Extract date features
        if date_features:
            df = self.date_extractor.extract_date_features(df, date_features)
        
        # Apply dimensionality reduction
        if n_components:
            feature_list = numeric_features + categorical_features
            # Remove target column if present
            if target_column and target_column in feature_list:
                feature_list.remove(target_column)
            
            df = self.dimensionality_reducer.reduce_dimensions(df, feature_list, n_components)
        
        # Perform feature selection
        if n_select_features and target_column:
            # Get all feature columns (excluding target and identifiers)
            feature_cols = [col for col in df.columns if col != target_column]
            if identifier_columns:
                feature_cols = [col for col in feature_cols if col not in identifier_columns_existing]
            df = self.feature_selector.select_features(df, feature_cols, target_column, n_select_features)
        
        logger.info(f"Feature engineering pipeline completed. Final shape: {df.shape}")
        return df


@standardize_error_handling
def engineer_features(
    input_path: str,
    output_path: str,
    numeric_features: List[str],
    categorical_features: List[str],
    date_features: List[str],
    target_column: Optional[str] = None,
    feature_pairs: Optional[List[Tuple[str, str]]] = None,
    polynomial_features: Optional[List[str]] = None,
    polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    n_components: Optional[int] = None,
    n_select_features: Optional[int] = None,
    identifier_columns: Optional[List[str]] = None
) -> None:
    """
    Complete feature engineering pipeline function.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save engineered CSV file
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        date_features: List of date feature names
        target_column: Name of target column
        feature_pairs: List of feature pairs for interaction
        polynomial_features: List of features for polynomial creation
        polynomial_degree: Degree for polynomial features
        n_components: Number of PCA components
        n_select_features: Number of features to select
        identifier_columns: List of identifier columns to exclude from processing
    """
    # Read the data
    df = safe_load_csv(input_path, encoding=DEFAULT_ENCODING)
    logger.info(f"Loaded data for feature engineering with shape {df.shape}")
    
    # Initialize and run pipeline
    pipeline = FeatureEngineeringPipeline()
    df_transformed = pipeline.run_pipeline(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        date_features=date_features,
        target_column=target_column,
        feature_pairs=feature_pairs,
        polynomial_features=polynomial_features,
        polynomial_degree=polynomial_degree,
        n_components=n_components,
        n_select_features=n_select_features,
        identifier_columns=identifier_columns
    )
    
    # Save the engineered data
    safe_save_csv(df_transformed, output_path, "feature-engineered data")
    logger.info(f"Feature engineering completed successfully. Final shape: {df_transformed.shape}")
