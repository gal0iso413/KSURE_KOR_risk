"""
Model utilities for the accident severity classification project.

This module provides utilities for data processing, handling imbalanced data,
and comprehensive model evaluation with visualization capabilities.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE

# Import project utilities and constants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import get_logger
from src.utils.common import (
    create_directory_if_not_exists, safe_save_csv, standardize_error_handling
)
from src.constants import (
    ACCIDENT_SEVERITY_MAPPING, DEFAULT_RANDOM_STATE, SMOTE_SAMPLING_STRATEGY,
    DEFAULT_VISUALIZATION_FIGSIZE, DEFAULT_TOP_N_FEATURES, KOREAN_FONT_FAMILY
)

# Initialize logger
logger = get_logger(__name__)

# Configure Korean font support
plt.rcParams['font.family'] = KOREAN_FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue


class DataProcessor:
    """Class for handling data preprocessing tasks specific to model training."""
    
    @staticmethod
    @standardize_error_handling
    def convert_to_severity_levels(accident_codes: pd.Series) -> pd.Series:
        """
        Convert accident codes into severity levels using predefined mapping.
        
        Args:
            accident_codes: Series containing accident codes
            
        Returns:
            Series with severity levels (0: Low, 1: Medium, 2: High)
        """
        if accident_codes.empty:
            raise ValueError("Cannot process empty accident codes series")
        
        # Convert to string for consistent mapping
        accident_codes_str = accident_codes.astype(str)
        
        # Apply severity mapping
        severity_levels = accident_codes_str.map(ACCIDENT_SEVERITY_MAPPING)
        
        # Handle unmapped codes - assign to low severity (0) and log warning
        unmapped_count = severity_levels.isnull().sum()
        if unmapped_count > 0:
            unique_unmapped = accident_codes_str[severity_levels.isnull()].unique()
            logger.warning(f"Found {unmapped_count} unmapped accident codes: {unique_unmapped[:10]}... "
                          f"Assigning to low severity (0)")
            severity_levels = severity_levels.fillna(0)
        
        # Convert to integer
        severity_levels = severity_levels.astype(int)
        
        # Log severity distribution
        severity_counts = severity_levels.value_counts().sort_index()
        logger.info(f"Severity level distribution: {severity_counts.to_dict()}")
        
        return severity_levels
    
    @staticmethod
    @standardize_error_handling
    def validate_features_and_target(
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate features and target arrays.
        
        Args:
            X: Feature matrix or full DataFrame
            y: Target vector
            feature_columns: List of specific columns to use as features (if X is DataFrame)
            
        Returns:
            Validated X and y as numpy arrays
        """
        # Convert to numpy arrays, using only specified feature columns if provided
        if isinstance(X, pd.DataFrame):
            if feature_columns is not None:
                # Use only the specified feature columns
                available_features = [col for col in feature_columns if col in X.columns]
                if len(available_features) != len(feature_columns):
                    missing = set(feature_columns) - set(available_features)
                    logger.warning(f"Missing feature columns: {missing}")
                logger.info(f"Using {len(available_features)} feature columns for training")
                X = X[available_features].values
            else:
                X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Validation checks
        if X.shape[0] == 0:
            raise ValueError("Feature matrix is empty")
        
        if len(y) == 0:
            raise ValueError("Target vector is empty")
        
        if X.shape[0] != len(y):
            raise ValueError(f"Feature matrix ({X.shape[0]}) and target ({len(y)}) have different lengths")
        
        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            logger.warning(f"Found {nan_count} NaN values in features")
        
        if np.isnan(y).any():
            nan_count = np.isnan(y).sum()
            raise ValueError(f"Found {nan_count} NaN values in target - cannot proceed")
        
        return X, y


class ImbalanceHandler:
    """Class for handling imbalanced datasets."""
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        """
        Initialize imbalance handler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
    
    @standardize_error_handling
    def handle_imbalanced_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        method: str = 'smote',
        sampling_strategy: str = SMOTE_SAMPLING_STRATEGY,
        feature_columns: Optional[List[str]] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[int, float]]]:
        """
        Handle imbalanced dataset using specified method.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Method to handle imbalance ('smote' or 'class_weight')
            sampling_strategy: Sampling strategy for SMOTE
            feature_columns: List of feature column names (if X is DataFrame)
            
        Returns:
            For SMOTE: (X_resampled, y_resampled)
            For class_weight: (X, y, class_weights)
        """
        # Validate inputs
        X, y = DataProcessor.validate_features_and_target(X, y, feature_columns)
        
        # Log original class distribution
        unique, counts = np.unique(y, return_counts=True)
        original_dist = dict(zip(unique, counts))
        logger.info(f"Original class distribution: {original_dist}")
        
        if method == 'smote':
            return self._apply_smote(X, y, sampling_strategy)
        elif method == 'class_weight':
            return self._calculate_class_weights(X, y)
        else:
            raise ValueError(f"Unknown imbalance handling method: {method}. "
                           f"Available methods: ['smote', 'class_weight']")
    
    def _apply_smote(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sampling_strategy: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling."""
        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Log new class distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            new_dist = dict(zip(unique, counts))
            logger.info(f"After SMOTE class distribution: {new_dist}")
            logger.info(f"Dataset size changed from {len(y)} to {len(y_resampled)} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            logger.info("Falling back to original data without resampling")
            return X, y
    
    def _calculate_class_weights(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
        """Calculate class weights for handling imbalance."""
        unique_classes, counts = np.unique(y, return_counts=True)
        
        # Calculate balanced class weights
        total_samples = len(y)
        n_classes = len(unique_classes)
        
        class_weights = {}
        for class_label, count in zip(unique_classes, counts):
            class_weights[class_label] = total_samples / (n_classes * count)
        
        logger.info(f"Calculated class weights: {class_weights}")
        
        return X, y, class_weights


class ModelEvaluator:
    """Class for comprehensive model evaluation and visualization."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize ModelEvaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = create_directory_if_not_exists(output_dir, "Model evaluation results")
        self.evaluation_results: Dict[str, Any] = {}
    
    @standardize_error_handling
    def evaluate_classification_model(
        self,
        model: Any,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        model_name: str = "model",
        original_test_data: Optional[pd.DataFrame] = None,
        identifier_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive classification model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            feature_names: List of feature names
            model_name: Name of the model for saving files
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Validate inputs - for evaluation, we assume X_test only contains feature columns
        X_test, y_test = DataProcessor.validate_features_and_target(X_test, y_test, feature_columns=None)
        
        logger.info(f"Evaluating {model_name} model on {len(y_test)} test samples")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results = self._calculate_classification_metrics(y_test, y_pred, model_name)
        
        # Generate and save visualizations
        self._create_classification_visualizations(
            model, X_test, y_test, y_pred, feature_names, model_name
        )
        
        # Save detailed results
        self._save_evaluation_results(results, model_name)
        
        # Analyze top predictions for each class (if model supports probabilities)
        if hasattr(model, 'predict_proba'):
            self._analyze_top_predictions(model, X_test, y_test, model_name, feature_names, original_test_data, identifier_columns)
        
        self.evaluation_results[model_name] = results
        logger.info(f"Evaluation completed for {model_name}")
        
        return results
    
    def _calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
                'sample_count': len(y_true)
            }
            
            # Log key metrics
            logger.info(f"{model_name} Metrics - Accuracy: {accuracy:.4f}, "
                       f"F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            return {'model_name': model_name, 'error': str(e)}
    
    def _create_classification_visualizations(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]],
        model_name: str
    ) -> None:
        """Create and save classification visualizations."""
        try:
            # Confusion Matrix
            self._plot_confusion_matrix(y_test, y_pred, model_name)
            
            # Feature Importance (if available)
            if feature_names:
                self._plot_feature_importance(model, feature_names, model_name)
            
            # ROC Curves (if model supports probability prediction)
            if hasattr(model, 'predict_proba'):
                self._plot_roc_curves(model, X_test, y_test, model_name)
                self._plot_precision_recall_curves(model, X_test, y_test, model_name)
            
        except Exception as e:
            logger.error(f"Error creating visualizations for {model_name}: {e}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
        """Plot and save confusion matrix."""
        plt.figure(figsize=DEFAULT_VISUALIZATION_FIGSIZE)
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add class labels if available
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        plt.xticks(np.arange(len(unique_labels)) + 0.5, 
                  [f'Class {label}' for label in unique_labels])
        plt.yticks(np.arange(len(unique_labels)) + 0.5, 
                  [f'Class {label}' for label in unique_labels])
        
        plt.tight_layout()
        save_path = self.output_dir / f'{model_name}_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix: {save_path}")
    
    def _plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str,
        top_n: int = DEFAULT_TOP_N_FEATURES
    ) -> None:
        """Plot and save feature importance."""
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.info(f"Model {model_name} does not support feature importance")
            return
        
        if len(importances) != len(feature_names):
            logger.warning(f"Feature importance length ({len(importances)}) doesn't match "
                          f"feature names length ({len(feature_names)})")
            return
        
        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[indices]
        top_features = [feature_names[i] for i in indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(indices)), top_importances)
        plt.title(f'Top {len(indices)} Feature Importances - {model_name}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        
        # Rotate labels for better readability
        plt.xticks(range(len(indices)), top_features, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, importance in zip(bars, top_importances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / f'{model_name}_feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot: {save_path}")
    
    def _plot_roc_curves(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> None:
        """Plot ROC curves for multi-class classification."""
        try:
            y_score = model.predict_proba(X_test)
            unique_classes = sorted(np.unique(y_test))
            
            plt.figure(figsize=DEFAULT_VISUALIZATION_FIGSIZE)
            
            # Plot ROC curve for each class
            for i, class_label in enumerate(unique_classes):
                if i < y_score.shape[1]:  # Ensure we don't go out of bounds
                    y_binary = (y_test == class_label).astype(int)
                    fpr, tpr, _ = roc_curve(y_binary, y_score[:, i])
                    auc_score = roc_auc_score(y_binary, y_score[:, i])
                    
                    plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {auc_score:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.output_dir / f'{model_name}_roc_curves.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved ROC curves: {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting ROC curves for {model_name}: {e}")
    
    def _plot_precision_recall_curves(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        model_name: str
    ) -> None:
        """Plot precision-recall curves for multi-class classification."""
        try:
            y_score = model.predict_proba(X_test)
            unique_classes = sorted(np.unique(y_test))
            
            plt.figure(figsize=DEFAULT_VISUALIZATION_FIGSIZE)
            
            # Plot precision-recall curve for each class
            for i, class_label in enumerate(unique_classes):
                if i < y_score.shape[1]:  # Ensure we don't go out of bounds
                    y_binary = (y_test == class_label).astype(int)
                    precision, recall, _ = precision_recall_curve(y_binary, y_score[:, i])
                    
                    plt.plot(recall, precision, label=f'Class {class_label}')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.output_dir / f'{model_name}_precision_recall_curves.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved precision-recall curves: {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting precision-recall curves for {model_name}: {e}")
    
    def _save_evaluation_results(self, results: Dict[str, Any], model_name: str) -> None:
        """Save evaluation results to files."""
        try:
            # Save classification report as text
            if 'classification_report' in results:
                report_path = self.output_dir / f'{model_name}_classification_report.txt'
                with open(report_path, 'w') as f:
                    f.write(f"Classification Report for {model_name}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Write summary metrics
                    f.write(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}\n")
                    f.write(f"F1-Score (Macro): {results.get('f1_macro', 'N/A'):.4f}\n")
                    f.write(f"F1-Score (Weighted): {results.get('f1_weighted', 'N/A'):.4f}\n")
                    f.write(f"Precision (Macro): {results.get('precision_macro', 'N/A'):.4f}\n")
                    f.write(f"Recall (Macro): {results.get('recall_macro', 'N/A'):.4f}\n\n")
                    
                    # Write detailed classification report
                    class_report = results['classification_report']
                    if isinstance(class_report, dict):
                        f.write("Detailed Classification Report:\n")
                        f.write("-" * 30 + "\n")
                        for class_name, metrics in class_report.items():
                            if isinstance(metrics, dict):
                                f.write(f"\n{class_name}:\n")
                                for metric, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        f.write(f"  {metric}: {value:.4f}\n")
                
                logger.info(f"Saved classification report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results for {model_name}: {e}")
    
    def _analyze_top_predictions(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        model_name: str,
        feature_names: Optional[List[str]] = None,
        original_test_data: Optional[pd.DataFrame] = None,
        identifier_columns: Optional[List[str]] = None,
        top_n: int = 10
    ) -> None:
        """
        Analyze and save top predictions for each class (flexible for any target column).
        
        Args:
            model: Trained model with probability prediction
            X_test: Test features
            y_test: True labels (already converted to numeric classes)
            model_name: Name of the model
            top_n: Number of top predictions to analyze per class
        """
        try:
            logger.info(f"Analyzing top {top_n} predictions for each class - {model_name}")
            
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Determine unique classes and their labels
            unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
            n_classes = len(unique_classes)
            
            # Create flexible class names based on actual classes found
            class_names = []
            class_labels = []
            for i, class_num in enumerate(unique_classes):
                class_names.append(f'Class_{class_num}_Prob')
                class_labels.append(f'Class_{class_num}')
            
            logger.info(f"Found {n_classes} classes: {unique_classes}")
            
            # Create results DataFrame with configured identifier columns only
            results_df = pd.DataFrame()
            
            if original_test_data is not None and identifier_columns:
                # Use only the specified identifier columns from configuration
                available_identifier_cols = []
                for col in identifier_columns:
                    if col in original_test_data.columns:
                        results_df[col] = original_test_data[col].values
                        available_identifier_cols.append(col)
                
                if available_identifier_cols:
                    logger.info(f"Including {len(available_identifier_cols)} configured identifier columns: {available_identifier_cols}")
                else:
                    logger.warning(f"None of the configured identifier columns found in data: {identifier_columns}")
            elif original_test_data is not None:
                logger.info("No identifier columns specified - only showing predictions and probabilities")
            else:
                logger.info("No original test data available - creating minimal DataFrame")
            
            # Add prediction results
            results_df['predicted_class'] = y_pred
            results_df['true_class'] = y_test
            
            # Add probability columns for each class
            for i, class_name in enumerate(class_names):
                if i < y_proba.shape[1]:
                    results_df[class_name] = y_proba[:, i]
            
            # Analyze each class
            for i, (target_class, prob_col, class_label) in enumerate(zip(unique_classes, class_names, class_labels)):
                if i < y_proba.shape[1]:
                    # Get top N predictions for this class
                    top_predictions = results_df.nlargest(top_n, prob_col).copy()
                    top_predictions['rank'] = range(1, len(top_predictions) + 1)
                    
                    # Reorder columns
                    priority_cols = ['rank', prob_col, 'predicted_class', 'true_class']
                    other_cols = [col for col in top_predictions.columns if col not in priority_cols]
                    final_cols = priority_cols + other_cols
                    top_predictions = top_predictions[final_cols]
                    
                    # Save top predictions for this class
                    output_path = self.output_dir / f'{model_name}_top_{top_n}_{class_label.lower()}.csv'
                    top_predictions.to_csv(output_path, index=False, encoding='utf-8-sig')
                    
                    # Calculate summary statistics
                    correct_predictions = (top_predictions['predicted_class'] == top_predictions['true_class']).sum()
                    prob_range = f"{top_predictions[prob_col].min():.4f} - {top_predictions[prob_col].max():.4f}"
                    
                    logger.info(f"{model_name} - Top {top_n} {class_label}:")
                    logger.info(f"  Probability range: {prob_range}")
                    logger.info(f"  Correct predictions: {correct_predictions}/{top_n}")
                    logger.info(f"  Results saved to: {output_path}")
            
            # Create summary report
            self._create_prediction_analysis_report(results_df, model_name, top_n, unique_classes, class_names, class_labels)
            
        except Exception as e:
            logger.error(f"Error analyzing top predictions for {model_name}: {e}")
    
    def _create_prediction_analysis_report(
        self, 
        results_df: pd.DataFrame, 
        model_name: str, 
        top_n: int,
        unique_classes: List[int],
        class_names: List[str],
        class_labels: List[str]
    ) -> None:
        """Create a comprehensive prediction analysis report (flexible for any target column)."""
        try:
            report_path = self.output_dir / f'{model_name}_prediction_analysis_report.txt'
            
            with open(report_path, 'w') as f:
                f.write(f"Prediction Analysis Report for {model_name}\n")
                f.write("=" * 60 + "\n\n")
                
                # Overall statistics
                f.write("OVERALL STATISTICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total samples analyzed: {len(results_df)}\n")
                overall_accuracy = (results_df['predicted_class'] == results_df['true_class']).mean()
                f.write(f"Overall accuracy: {overall_accuracy:.4f}\n")
                f.write(f"Number of classes: {len(unique_classes)}\n")
                f.write(f"Classes found: {unique_classes}\n\n")
                
                # Class distribution
                f.write("CLASS DISTRIBUTION:\n")
                f.write("-" * 18 + "\n")
                for class_num in unique_classes:
                    true_count = (results_df['true_class'] == class_num).sum()
                    pred_count = (results_df['predicted_class'] == class_num).sum()
                    f.write(f"Class {class_num}:\n")
                    f.write(f"  True labels: {true_count}\n")
                    f.write(f"  Predicted: {pred_count}\n\n")
                
                # Top predictions analysis for each class
                f.write(f"TOP {top_n} PREDICTIONS ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                for target_class, prob_col, class_label in zip(unique_classes, class_names, class_labels):
                    if prob_col in results_df.columns:
                        # Get top N for this class
                        top_class = results_df.nlargest(top_n, prob_col)
                        correct_in_top = (top_class['predicted_class'] == top_class['true_class']).sum()
                        correct_class_in_top = (top_class['true_class'] == target_class).sum()
                        
                        f.write(f"\n{class_label}:\n")
                        f.write(f"  Top {top_n} probability range: {top_class[prob_col].min():.4f} - {top_class[prob_col].max():.4f}\n")
                        f.write(f"  Correct predictions in top {top_n}: {correct_in_top}/{top_n} ({correct_in_top/top_n:.2%})\n")
                        f.write(f"  Actual class {target_class} cases in top {top_n}: {correct_class_in_top}/{top_n} ({correct_class_in_top/top_n:.2%})\n")
                        
                        # Show confidence levels
                        high_conf = (top_class[prob_col] >= 0.8).sum()
                        med_conf = ((top_class[prob_col] >= 0.6) & (top_class[prob_col] < 0.8)).sum()
                        low_conf = (top_class[prob_col] < 0.6).sum()
                        
                        f.write(f"  Confidence levels in top {top_n}:\n")
                        f.write(f"    High (≥0.8): {high_conf}\n")
                        f.write(f"    Medium (0.6-0.8): {med_conf}\n")
                        f.write(f"    Low (<0.6): {low_conf}\n")
                
                f.write(f"\nDetailed CSV files saved for each class's top {top_n} predictions.\n")
                f.write("Use these files to examine specific cases and improve model performance.\n")
                f.write("\nNOTE: Classes are automatically detected from your target column data.\n")
                f.write("The analysis works with any target column - Korean accident codes, severity levels, or custom classifications.\n")
            
            logger.info(f"Saved prediction analysis report: {report_path}")
            
        except Exception as e:
            logger.error(f"Error creating prediction analysis report for {model_name}: {e}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            if 'error' not in results:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results.get('accuracy', 0),
                    'F1-Macro': results.get('f1_macro', 0),
                    'F1-Weighted': results.get('f1_weighted', 0),
                    'Precision-Macro': results.get('precision_macro', 0),
                    'Recall-Macro': results.get('recall_macro', 0),
                    'Sample Count': results.get('sample_count', 0)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.round(4)
            
            # Save comparison
            comparison_path = self.output_dir / 'model_comparison.csv'
            safe_save_csv(comparison_df, comparison_path, "Model comparison results")
            
            logger.info(f"Saved model comparison: {comparison_path}")
            return comparison_df
        else:
            logger.warning("No valid model results to compare")
            return pd.DataFrame() 