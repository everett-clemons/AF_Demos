#Charting 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities / Data Processing
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import pickle
from dataclasses import dataclass
from enum import Enum

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna

# OpenAI API
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import time
from datetime import datetime
import joblib

def create_correlation_heatmap(data: pd.DataFrame, 
                             export_path: Path,
                             filename: str = 'correlation_heatmap.png',
                             figsize: tuple = (20, 16),
                             dpi: int = 300) -> None:
    """
    Generate and save a correlation heatmap visualization for the dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing numerical data for correlation
        export_path (Path): Directory path where the heatmap will be saved
        filename (str, optional): Name of the output file. Defaults to 'correlation_heatmap.png'
        figsize (tuple, optional): Figure dimensions (width, height). Defaults to (20, 16)
        dpi (int, optional): DPI for the saved image. Defaults to 300
    
    Raises:
        ValueError: If the DataFrame is empty or contains non-numeric data
        IOError: If there's an error saving the figure
    """
    # Input validation
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for non-numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        raise ValueError("No numeric columns found in the DataFrame")
    
    try:
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Set up the matplotlib figure
        plt.figure(figsize=figsize)
        
        # Create heatmap with customized appearance
        sns.heatmap(correlation_matrix,
                   annot=True,          # Show correlation values
                   fmt='.2f',           # Format to 2 decimal places
                   cmap='coolwarm',     # Color scheme for correlation values
                   center=0,            # Center colormap at 0
                   square=True,         # Make cells square-shaped
                   annot_kws={
                       'size': 8,       # Annotation text size
                       'weight': 'bold' # Make text more readable
                   },
                   cbar_kws={
                       'shrink': .8,    # Adjust colorbar size
                       'label': 'Correlation Coefficient'  # Add colorbar label
                   })
        
        # Customize axis labels
        plt.xticks(rotation=90, ha='center')
        plt.yticks(rotation=0, va='center')
        
        # Add title with styling
        plt.title('Correlation Heatmap of Cereal Nutritional Attributes', 
                 pad=20, 
                 size=16, 
                 weight='bold')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Ensure export path exists
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Save figure with high quality settings
        output_path = export_path / filename
        plt.savefig(output_path,
                   dpi=dpi,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        
        # Clean up by closing the figure
        plt.close()
        
        print(f"Correlation heatmap saved successfully to: {output_path}")
        
    except Exception as e:
        plt.close()  # Ensure figure is closed even if there's an error
        raise IOError(f"Error creating correlation heatmap: {str(e)}")
    
def analyze_correlations(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Identify and return strong correlations in the dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        threshold (float): Minimum absolute correlation value to consider (default: 0.5)
    
    Returns:
        pd.DataFrame: DataFrame containing pairs of features with strong correlations
    """
    # Calculate correlation matrix
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    
    # Create a DataFrame of strong correlations
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) >= threshold:
                strong_corrs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i,j]
                })
    
    return pd.DataFrame(strong_corrs).sort_values('Correlation', key=abs, ascending=False)

def setup_environment():
    """
    Initialize the environment by loading API keys and creating necessary directories.
    
    Returns:
        tuple: (api_key_status: bool, export_path: Path)
        
    Raises:
        ValueError: If no API key is found in the environment
        IOError: If export directory cannot be created
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Configure OpenAI API key
    # Note: This can be replaced with any other LLM API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError(
            "No OpenAI API key found. Please check your .env file.\n"
            "Note: This demo can be modified to use any LLM API by changing "
            "the API configuration and corresponding API calls."
        )
    
    # Create export directory for results
    export_dir = Path('./exports')
    try:
        export_dir.mkdir(exist_ok=True)
    except Exception as e:
        raise IOError(f"Failed to create export directory: {e}")
        
    return True, export_dir

def load_data(file_path='G:\\AF_Demos\\AF_Demos\\2024\\cereal_data.csv'):
    """
    Load and validate the cereal dataset.
    
    Args:
        file_path (str): Path to the CSV file containing cereal data
        
    Returns:
        pandas.DataFrame: Loaded and validated cereal data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the data format is invalid
    """
    try:
        data = pd.read_csv(file_path)
        
        # Basic validation of required columns could be added here
        # required_columns = ['column1', 'column2']
        # if not all(col in data.columns for col in required_columns):
        #     raise ValueError(f"Missing required columns. Expected: {required_columns}")
            
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cereal data file not found at {file_path}. "
            "Please ensure the data file is in the correct location."
        )
    except pd.errors.EmptyDataError:
        raise ValueError("The data file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Unable to parse the data file. Please ensure it's a valid CSV.")

def get_top_kpi_correlations(
    correlation_matrix: pd.DataFrame,
    kpi_cols: List[str],
    n_top: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Identify the strongest correlations for each KPI.
    
    Args:
        correlation_matrix (pd.DataFrame): Full correlation matrix
        kpi_cols (List[str]): List of KPI column names
        n_top (int, optional): Number of top correlations to return. Defaults to 5
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping each KPI to its top correlations
    """
    top_correlations = {}
    
    for kpi in kpi_cols:
        # Get correlations and sort by absolute value
        correlations = correlation_matrix[kpi].sort_values(key=abs, ascending=False)
        
        # Remove self-correlation and get top n
        top_corr = correlations[correlations.index != kpi][:n_top]
        
        # Add to dictionary with formatted DataFrame
        top_correlations[kpi] = pd.DataFrame(
            top_corr,
            columns=[f'Correlation with {kpi}']
        )
    
    return top_correlations

def create_process_kpi_heatmap(
    correlation_data: pd.DataFrame,
    export_dir: Path,
    filename: str,
    figsize: tuple = (12, 16)
) -> None:
    """
    Create and save a heatmap visualization of process parameters vs KPIs correlations.
    
    Args:
        correlation_data (pd.DataFrame): Correlation matrix
        export_dir (Path): Directory to save the visualization
        filename (str): Name of the output file
        figsize (tuple, optional): Figure dimensions. Defaults to (12, 16)
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        correlation_data,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        annot_kws={'size': 10},
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    # Customize appearance
    plt.title('Process Parameters vs KPIs Correlations', 
             pad=20, 
             size=14, 
             weight='bold')
    
    # Rotate axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Ensure export directory exists
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    output_path = export_dir / filename
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()
    
def analyze_kpi_correlations(
    data: pd.DataFrame,
    kpi_cols: List[str],
    export_path: Path,
    min_correlation: float = 0.3  # Minimum correlation threshold
) -> None:
    """
    Analyze and report correlations between KPIs and process parameters.
    
    Args:
        data (pd.DataFrame): Input DataFrame with process data
        kpi_cols (List[str]): List of KPI column names
        export_path (Path): Path for saving outputs
        min_correlation (float): Minimum absolute correlation to report
    """
    # Calculate correlation matrix
    correlation_matrix = data.corr()
    
    # Get process parameter columns (excluding KPIs)
    process_cols = [col for col in data.columns if col not in kpi_cols]
    
    # Create visualization of process vs KPI correlations
    plt.figure(figsize=(12, 16))
    process_kpi_corr = correlation_matrix.loc[process_cols, kpi_cols]
    
    sns.heatmap(
        process_kpi_corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        annot_kws={'size': 10}
    )
    
    plt.title('Process Parameters vs KPIs Correlations', pad=20, size=14)
    plt.tight_layout()
    plt.savefig(export_path / 'process_kpi_correlations.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

    # Print analysis for each KPI
    print("\nDetailed KPI Correlation Analysis:")
    print("=" * 50)
    
    for kpi in kpi_cols:
        print(f"\nSignificant correlations for {kpi}:")
        print("-" * 40)
        
        # Get correlations and filter out self-correlation
        correlations = correlation_matrix[kpi].copy()
        correlations = correlations[correlations.index != kpi]
        
        # Filter by minimum correlation threshold
        significant_corr = correlations[abs(correlations) >= min_correlation]
        
        # Sort by absolute correlation value
        significant_corr = significant_corr.reindex(
            significant_corr.abs().sort_values(ascending=False).index
        )
        
        if len(significant_corr) > 0:
            # Create a formatted DataFrame for display
            corr_df = pd.DataFrame({
                'Parameter': significant_corr.index,
                'Correlation': significant_corr.values
            })
            
            # Add correlation strength description
            corr_df['Strength'] = corr_df['Correlation'].apply(
                lambda x: 'Strong' if abs(x) >= 0.7 else 
                         'Moderate' if abs(x) >= 0.5 else 'Weak'
            )
            
            # Format correlation values
            corr_df['Correlation'] = corr_df['Correlation'].map('{:.3f}'.format)
            
            print(corr_df.to_string(index=False))
        else:
            print(f"No significant correlations found (threshold: {min_correlation})")
        
        print()  # Add blank line between KPIs
        
"""
Feature Engineering and Data Preparation Module
--------------------------------------------
This module handles feature engineering, data splitting, and scaling for the 
manufacturing process optimization model. It creates interaction terms between
related process parameters and prepares the data for model training.

Features created:
- Temperature-time interactions for cooking and drying processes
- Flaking roll parameters interaction
- Coating process interaction

"""
class FeatureEngineer:
    """
    Handles feature engineering and data preparation for the manufacturing process model.
    """
    
    def __init__(self, 
                 target_cols: List[str] = ['throughput', 'first_pass_quality', 'scrap'],
                 random_state: int = 42,
                 test_size: float = 0.2):
        """
        Initialize the feature engineer.
        
        Args:
            target_cols (List[str]): Names of target variables
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data to use for testing
        """
        self.target_cols = target_cols
        self.random_state = random_state
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.feature_definitions = self._get_feature_definitions()
        
    def _get_feature_definitions(self) -> Dict[str, Tuple[str, str]]:
        """
        Define the interaction features to be created.
        
        Returns:
            Dict mapping new feature names to tuples of constituent features
        """
        return {
            'cook_temp_time_interaction': ('cook_temp', 'cook_time'),
            'drying_temp_time_interaction_1': ('drying_temp_1', 'drying_time_1'),
            'drying_temp_time_interaction_2': ('drying_temp_2', 'drying_time_2'),
            'flaking_interaction': ('flaking_rolls_gap', 'flaking_rolls_speed'),
            'coating_interaction': ('coating_flow', 'conveyor_speed')
        }
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features from the original dataset.
        
        Args:
            data (pd.DataFrame): Original dataset
            
        Returns:
            pd.DataFrame: Dataset with additional interaction features
            
        Raises:
            ValueError: If required columns are missing
        """
        # Verify all required columns are present
        required_cols = set()
        for col1, col2 in self.feature_definitions.values():
            required_cols.add(col1)
            required_cols.add(col2)
            
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Create a copy to avoid modifying the original data
        enhanced_data = data.copy()
        
        # Create interaction features
        for new_col, (col1, col2) in self.feature_definitions.items():
            enhanced_data[new_col] = enhanced_data[col1] * enhanced_data[col2]
            
        return enhanced_data
    
    def prepare_data(self, 
                    data: pd.DataFrame,
                    export_path: Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training including feature engineering, splitting, and scaling.
        
        Args:
            data (pd.DataFrame): Raw input data
            export_path (Path, optional): Path to export preprocessing objects
            
        Returns:
            Tuple containing:
                - X_train_scaled: Scaled training features
                - X_test_scaled: Scaled test features
                - y_train: Training targets
                - y_test: Test targets
        """
        # Create interaction features
        enhanced_data = self.create_interaction_features(data)
        
        # Split features and targets
        X = enhanced_data.drop(columns=self.target_cols)
        y = enhanced_data[self.target_cols]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Export preprocessing objects if path provided
        if export_path:
            self.export_preprocessor(export_path)
            
        # Print preparation summary
        self._print_preparation_summary(X_train_scaled, X_test_scaled, y_train, y_test)
        
        return X, X_test, X_train, X_train_scaled, X_test_scaled, y_train, y_test
    
    def export_preprocessor(self, export_path: Path) -> None:
        """
        Export the preprocessing objects for later use.
        
        Args:
            export_path (Path): Directory to save preprocessing objects
        """
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        with open(export_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save feature names
        with open(export_path / 'feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
            
        # Save feature definitions
        with open(export_path / 'feature_definitions.pkl', 'wb') as f:
            pickle.dump(self.feature_definitions, f)
    
    def _print_preparation_summary(self,
                                 X_train: np.ndarray,
                                 X_test: np.ndarray,
                                 y_train: pd.DataFrame,
                                 y_test: pd.DataFrame) -> None:
        """
        Print a summary of the data preparation process.
        """
        print("\nData Preparation Summary")
        print("=" * 50)
        print(f"\nFeatures created:")
        for feature, (col1, col2) in self.feature_definitions.items():
            print(f"- {feature}: {col1} × {col2}")
            
        print(f"\nDataset splits:")
        print(f"- Training set: {X_train.shape[0]} samples")
        print(f"- Test set: {X_test.shape[0]} samples")
        print(f"- Number of features: {X_train.shape[1]}")
        
        print(f"\nTarget variables:")
        for target in self.target_cols:
            print(f"- {target}")
            print(f"  Train range: [{y_train[target].min():.2f}, {y_train[target].max():.2f}]")
            print(f"  Test range: [{y_test[target].min():.2f}, {y_test[target].max():.2f}]")        
        
"""
Model Tuning and Training Module with Target Direction Awareness
------------------------------------------------------------
Handles optimization direction (minimize/maximize) for different targets:
- Throughput: Maximize
- First Pass Quality: Maximize
- Scrap: Minimize
"""

"""
Model Tuning and Training Module with Target Direction Awareness
------------------------------------------------------------
Handles optimization direction (minimize/maximize) for different targets:
- Throughput: Maximize
- First Pass Quality: Maximize
- Scrap: Minimize

Dependencies:
    - optuna
    - xgboost
    - sklearn
    - numpy
    - pandas
    - matplotlib
    - seaborn
"""
class OptimizationDirection(Enum):
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'

@dataclass
class TargetConfig:
    """Configuration for each target variable"""
    name: str
    direction: OptimizationDirection
    
    @property
    def score_multiplier(self) -> float:
        """Return multiplier for converting predictions to optimization score"""
        return 1.0 if self.direction == OptimizationDirection.MAXIMIZE else -1.0

class ModelTrainer:
    def __init__(self,
                 export_dir: Path,
                 n_trials: int = 50,
                 random_state: int = 42,
                 n_jobs: int = 1):
        """Initialize model trainer with target configurations."""
        self.export_dir = Path(export_dir)
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Define target configurations
        self.target_configs = {
            'throughput': TargetConfig(
                name='throughput',
                direction=OptimizationDirection.MAXIMIZE
            ),
            'first_pass_quality': TargetConfig(
                name='first_pass_quality',
                direction=OptimizationDirection.MAXIMIZE
            ),
            'scrap': TargetConfig(
                name='scrap',
                direction=OptimizationDirection.MINIMIZE
            )
        }

    def _get_param_ranges(self, target_name: str) -> Dict[str, Tuple[float, float]]:
        """Define parameter ranges based on target."""
        if target_name == 'throughput':
            return {
                'n_estimators': (100, 500),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.2),
                'min_child_weight': (1, 5),
                'subsample': (0.7, 0.9),
                'colsample_bytree': (0.7, 0.9),
                'gamma': (0.01, 0.5),
                'reg_alpha': (0.01, 0.5),
                'reg_lambda': (0.01, 0.5)
            }
        else:  # first_pass_quality and scrap
            return {
                'n_estimators': (100, 500),
                'max_depth': (3, 7),
                'learning_rate': (0.01, 0.15),
                'min_child_weight': (1, 4),
                'subsample': (0.7, 0.9),
                'colsample_bytree': (0.7, 0.9),
                'gamma': (0.01, 0.3),
                'reg_alpha': (0.01, 0.3),
                'reg_lambda': (0.01, 0.3)
            }

    def _suggest_parameters(self, 
                          trial: optuna.Trial, 
                          param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Suggest parameters for current trial."""
        params = {}
        for param_name, (low, high) in param_ranges.items():
            if param_name in ['n_estimators', 'max_depth', 'min_child_weight']:
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                params[param_name] = trial.suggest_float(param_name, low, high)
        return params

    def objective(self,
                 trial: optuna.Trial,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 target_config: TargetConfig) -> float:
        """Modified objective function that considers optimization direction."""
        try:
            param_ranges = self._get_param_ranges(target_config.name)
            params = self._suggest_parameters(trial, param_ranges)
            
            model = xgb.XGBRegressor(
                **params,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=self.n_jobs
            )
            
            rmse_scores = np.sqrt(-scores)
            adjusted_score = target_config.score_multiplier * rmse_scores.mean()
            
            trial.set_user_attr('raw_rmse', rmse_scores.mean())
            trial.set_user_attr('direction', target_config.direction.value)
            
            return adjusted_score
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            return float('inf' if target_config.direction == OptimizationDirection.MINIMIZE else '-inf')

    def _train_final_model(self,
                          best_params: Dict[str, Any],
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          target_config: TargetConfig) -> Tuple[xgb.XGBRegressor, Dict[str, float], np.ndarray]:
        """Train and evaluate final model."""
        model = xgb.XGBRegressor(
            **best_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(root_mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'prediction_mean': y_pred.mean(),
            'prediction_std': y_pred.std()
        }
        
        return model, metrics, y_pred

    def _create_optimization_plots(self,
                                 study: optuna.Study,
                                 target_name: str) -> None:
        """Create and save optimization visualization plots."""
        try:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title('Optimization History')
            
            plt.subplot(1, 2, 2)
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title('Parameter Importances')
            
            plt.tight_layout()
            plt.savefig(self.export_dir / f'{target_name}_optimization.png')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create optimization plots: {str(e)}")

    def _create_model_analysis_plots(self,
                                   model: xgb.XGBRegressor,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   y_pred: np.ndarray,
                                   target_name: str,
                                   feature_names: List[str]) -> None:
        """Create and save feature importance and prediction analysis plots."""
        try:
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            sns.barplot(x='importance', y='feature', 
                       data=feature_importance.head(10))
            plt.title(f'Top 10 Feature Importances\n{target_name}')
            
            plt.subplot(1, 2, 2)
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            
            plt.tight_layout()
            plt.savefig(self.export_dir / f'{target_name}_analysis.png')
            plt.close()
            
            feature_importance.to_csv(
                self.export_dir / f'{target_name}_feature_importance.csv',
                index=False
            )
                
        except Exception as e:
            print(f"Warning: Could not create analysis plots for {target_name}: {str(e)}")

    def tune_and_train_model(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           target_name: str,
                           feature_names: List[str]) -> Tuple[Any, Dict[str, Any], optuna.Study]:
        """Tune hyperparameters, train model, and create visualizations."""
        try:
            target_config = self.target_configs[target_name]
            print(f"\nTuning model for {target_name} ({target_config.direction.value})")
            
            study = optuna.create_study(
                direction='minimize' if target_config.direction == OptimizationDirection.MINIMIZE else 'maximize',
                study_name=f"{target_name}_optimization"
            )
            
            study.optimize(
                lambda trial: self.objective(
                    trial, X_train, y_train, target_config
                ),
                n_trials=self.n_trials,
                show_progress_bar=True,
                catch=(Exception,)
            )
            
            best_model, metrics, predictions = self._train_final_model(
                study.best_params,
                X_train, y_train,
                X_test, y_test,
                target_config
            )
            
            self._create_optimization_plots(study, target_name)
            self._create_model_analysis_plots(
                best_model,
                X_test,
                y_test,
                predictions,
                target_name,
                feature_names
            )
            
            return best_model, metrics, study
            
        except Exception as e:
            print(f"Error in tune_and_train_model for {target_name}: {str(e)}")
            return None, None, None

def train_models(X_train_scaled: np.ndarray,
                X_test_scaled: np.ndarray,
                y_train: pd.DataFrame,
                y_test: pd.DataFrame,
                feature_names: list,
                export_dir: str = './exports',
                n_trials: int = 50) -> dict:
    """
    Train models for all targets with minimal output.
    Full metrics analysis will be done separately.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training target variables
        y_test: Test target variables
        feature_names: Names of features
        export_dir: Directory for saving outputs
        n_trials: Number of optimization trials
    
    Returns:
        dict: Trained models and their metrics
    """
    # Initialize export directory
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = ModelTrainer(
        export_dir=export_dir,
        n_trials=n_trials,
        random_state=42,
        n_jobs=1
    )

    # Initialize results storage
    models = {}
    training_summary = []
    start_time = time.time()

    print("Starting Model Training Process")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of trials per model: {trainer.n_trials}")

    try:
        for target_name in ['throughput', 'first_pass_quality', 'scrap']:
            print(f"\nTraining model for {target_name}...")
            
            # Get target data
            y_target_train = y_train[target_name]
            y_target_test = y_test[target_name]
            
            # Validate data
            if y_target_train.isnull().any() or y_target_test.isnull().any():
                raise ValueError(f"Found null values in {target_name} data")
            
            # Tune and train model
            model, metrics, study = trainer.tune_and_train_model(
                X_train_scaled,
                y_target_train,
                X_test_scaled,
                y_target_test,
                target_name,
                feature_names=feature_names
            )
            
            if model is not None:
                models[target_name] = {
                    'model': model,
                    'metrics': metrics,
                    'study': study,
                    'training_time': time.time() - start_time
                }
                print(f"✓ Model trained successfully")
            else:
                print(f"✗ Failed to train model")

    except Exception as e:
        print(f"\nError in training process: {str(e)}")
        
    finally:
        # Save models and studies
        if models:
            models_dir = export_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            
            for target_name, model_data in models.items():
                # Save model
                joblib.dump(
                    model_data['model'],
                    models_dir / f'{target_name}_model.joblib'
                )
                
                # Save study
                joblib.dump(
                    model_data['study'],
                    models_dir / f'{target_name}_study.joblib'
                )
        
        print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
        print(f"Models saved to: {export_dir}")

        return models
def get_direction_str(study: optuna.study.Study) -> str:
    """Convert Optuna study direction to string."""
    return 'MINIMIZE' if study.direction == optuna.study.StudyDirection.MINIMIZE else 'MAXIMIZE'

def print_metrics_summary(models: dict, X) -> None:
    """Print formatted performance metrics for all models."""
    
    print("\nMODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Header for metrics table
    header = f"{'Target':<20} {'RMSE':>10} {'R²':>10} {'Mean Pred':>12} {'Std Dev':>10} {'Direction':>12}"
    print("\n" + header)
    print("-" * 80)
    
    # Print metrics for each model
    for target_name, model_data in models.items():
        metrics = model_data['metrics']
        direction = get_direction_str(model_data['study'])
        
        print(
            f"{target_name:<20} "
            f"{metrics['rmse']:>10.4f} "
            f"{metrics['r2']:>10.4f} "
            f"{metrics['prediction_mean']:>12.4f} "
            f"{metrics['prediction_std']:>10.4f} "
            f"{direction:>12}"
        )
    
    print("\nDETAILED ANALYSIS BY TARGET")
    print("=" * 80)
    
    for target_name, model_data in models.items():
        metrics = model_data['metrics']
        study = model_data['study']
        model = model_data['model']
        direction = get_direction_str(study)
        
        print(f"\n{target_name.upper()}")
        print("-" * len(target_name))
        
        # Model performance metrics
        print(f"\nPerformance Metrics:")
        print(f"- RMSE: {metrics['rmse']:.4f}")
        print(f"- R² Score: {metrics['r2']:.4f}")
        print(f"- Mean Prediction: {metrics['prediction_mean']:.4f}")
        print(f"- Prediction Std Dev: {metrics['prediction_std']:.4f}")
        
        # Optimization details
        print(f"\nOptimization Details:")
        print(f"- Best Trial: #{study.best_trial.number}")
        print(f"- Optimization Direction: {direction}")
        print(f"- Training Time: {model_data['training_time']:.2f} seconds")
        
        # Best parameters
        print(f"\nBest Parameters:")
        for param, value in study.best_params.items():
            print(f"- {param}: {value}")
        
        # Top feature importances
        try:
            feature_importance = pd.DataFrame({
                'Feature': model_data.get('feature_names', X.columns),  # Fallback to X.columns if feature_names not stored
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 5 Important Features:")
            for _, row in feature_importance.head().iterrows():
                print(f"- {row['Feature']:<30} {row['Importance']:.4f}")
        except Exception as e:
            print("\nCould not generate feature importance rankings:", str(e))
            
        print("\n" + "=" * 80)

def export_feature_importances(models: dict, 
                             export_dir: str = './exports',
                             feature_names: list = None) -> None:
    """
    Export feature importance rankings for each model to CSV files.
    
    Args:
        models: Dictionary containing trained models and their data
        export_dir: Directory to save the CSV files
        feature_names: List of feature names (optional)
    """
    # Create export directory if it doesn't exist
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nExporting feature importances...")
    
    for target_name, model_data in models.items():
        try:
            # Get feature names from model data or use provided names
            names = (model_data.get('feature_names') or 
                    feature_names or 
                    [f'feature_{i}' for i in range(len(model_data['model'].feature_importances_))])
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': names,
                'Importance': model_data['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Add percentage contribution
            feature_importance['Percentage'] = (
                feature_importance['Importance'] / feature_importance['Importance'].sum() * 100
            )
            
            # Add cumulative percentage
            feature_importance['Cumulative_Percentage'] = (
                feature_importance['Percentage'].cumsum()
            )
            
            # Format percentages
            feature_importance['Percentage'] = feature_importance['Percentage'].map('{:.2f}%'.format)
            feature_importance['Cumulative_Percentage'] = feature_importance['Cumulative_Percentage'].map('{:.2f}%'.format)
            
            # Save to CSV
            output_path = export_dir / f'{target_name}_feature_importance.csv'
            feature_importance.to_csv(output_path, index=False)
            
            print(f"✓ Saved importance rankings for {target_name} to {output_path}")
            
        except Exception as e:
            print(f"✗ Error saving feature importance for {target_name}: {str(e)}")

def analyze_prediction_correlations(models: dict,
                                  y_test: pd.DataFrame,
                                  X_test_scaled: np.ndarray,
                                  export_dir: str = './exports') -> pd.DataFrame:
    """
    Analyze and visualize correlations between actual and predicted values.
    
    Args:
        models: Dictionary containing trained models
        y_test: Test set target values
        X_test_scaled: Scaled test features
        export_dir: Directory to save outputs
        
    Returns:
        DataFrame containing actual and predicted values
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nAnalyzing prediction correlations...")
    
    try:
        # Create predictions DataFrame
        predictions = pd.DataFrame()
        
        for target_name in y_test.columns:
            # Add actual values
            predictions[f'{target_name}_actual'] = y_test[target_name]
            
            # Generate and add predictions
            predictions[f'{target_name}_predicted'] = models[target_name]['model'].predict(X_test_scaled)
            
        # Calculate correlations
        correlations = predictions.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        
        # Create heatmap with custom formatting
        mask = np.triu(np.ones_like(correlations), k=0)
        
        sns.heatmap(correlations,
                   annot=True,
                   fmt='.3f',
                   cmap='coolwarm',
                   center=0,
                   vmin=-1,
                   vmax=1,
                   square=True,
                   mask=mask,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Correlation Matrix of Actual vs Predicted Values',
                 pad=20,
                 size=14,
                 weight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(export_dir / 'prediction_correlations.png',
                   dpi=300,
                   bbox_inches='tight')
        plt.close()
        
        # Print correlation analysis
        print("\nCorrelation Analysis:")
        print("-" * 50)
        
        for target_name in y_test.columns:
            actual_col = f'{target_name}_actual'
            pred_col = f'{target_name}_predicted'
            correlation = correlations.loc[actual_col, pred_col]
            
            print(f"\n{target_name}:")
            print(f"- Actual vs Predicted Correlation: {correlation:.4f}")
            
            # Calculate prediction metrics
            actual = predictions[actual_col]
            predicted = predictions[pred_col]
            
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            print(f"- Mean Absolute Error: {mae:.4f}")
            print(f"- Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Save predictions to CSV
        predictions.to_csv(export_dir / 'predictions.csv', index=False)
        
        return predictions
        
    except Exception as e:
        print(f"Error in prediction correlation analysis: {str(e)}")
        return None

def print_performance_summary(models: dict) -> None:
    """
    Print comprehensive performance summary for all models.
    
    Args:
        models: Dictionary containing trained models and their metrics
    """
    print("\nModel Performance Summary:")
    print("=" * 50)
    
    for target_name, model_data in models.items():
        metrics = model_data['metrics']
        study = model_data['study']
        
        print(f"\n{target_name.upper()}:")
        print("-" * (len(target_name) + 1))
        
        # Performance metrics
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        
        # Additional metrics
        print(f"Mean Prediction: {metrics['prediction_mean']:.4f}")
        print(f"Prediction Std Dev: {metrics['prediction_std']:.4f}")
        print(f"Optimization Direction: {get_direction_str(study)}")
        print(f"Best Trial: #{study.best_trial.number}")
        print(f"Training Time: {model_data['training_time']:.2f} seconds")


"""
Process Parameter Optimization Module
----------------------------------
Optimizes manufacturing process parameters with prioritized objectives:
1. First Pass Quality >= 95%
2. Throughput ≈ 900
3. Minimum Scrap

Uses simulated annealing-like approach with adaptive step sizes.
"""
@dataclass
class OptimizationTargets:
    """Target values and constraints for optimization"""
    QUALITY_TARGET: float = 95.0
    THROUGHPUT_TARGET: float = 900.0
    THROUGHPUT_TOLERANCE: float = 10.0
    MAX_SCRAP: float = 70.0
    
    # Optimization weights
    QUALITY_WEIGHT: float = 2.0
    THROUGHPUT_WEIGHT: float = 1.5
    SCRAP_WEIGHT: float = 1.0

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    params: np.ndarray
    predictions: Dict[str, float]
    score: float
    meets_criteria: bool

class ParameterOptimizer:
    """Handles optimization of process parameters."""
    
    def __init__(self,
                 models: Dict[str, Any],
                 scaler: Any,
                 feature_names: List[str],
                 targets: OptimizationTargets = None,
                 export_dir: str = './exports'):
        """
        Initialize optimizer.
        
        Args:
            models: Dictionary of trained models
            scaler: Fitted scaler for parameter scaling
            feature_names: List of parameter names
            targets: Optimization targets and constraints
            export_dir: Directory for saving results
        """
        self.models = models
        self.scaler = scaler
        self.feature_names = feature_names
        self.targets = targets or OptimizationTargets()
        self.export_dir = Path(export_dir)
        
    def predict(self, params_scaled: np.ndarray) -> Dict[str, float]:
        """Generate predictions for scaled parameters."""
        return {
            target: model['model'].predict(params_scaled)[0]
            for target, model in self.models.items()
        }
    
    def calculate_scores(self, prediction: Dict[str, float]) -> Tuple[float, bool]:
        """
        Calculate objective scores and check criteria.
        
        Returns:
            Tuple of (combined score, meets criteria flag)
        """
        # Calculate individual scores
        quality_score = prediction['first_pass_quality'] - self.targets.QUALITY_TARGET
        throughput_score = -abs(prediction['throughput'] - self.targets.THROUGHPUT_TARGET)
        scrap_score = -prediction['scrap']
        
        # Combined weighted score
        combined_score = (
            quality_score * self.targets.QUALITY_WEIGHT +
            throughput_score * self.targets.THROUGHPUT_WEIGHT +
            scrap_score * self.targets.SCRAP_WEIGHT
        )
        
        # Check if solution meets all criteria
        meets_criteria = (
            prediction['first_pass_quality'] >= self.targets.QUALITY_TARGET and
            abs(prediction['throughput'] - self.targets.THROUGHPUT_TARGET) <= self.targets.THROUGHPUT_TOLERANCE and
            prediction['scrap'] <= self.targets.MAX_SCRAP
        )
        
        return combined_score, meets_criteria
    
    def optimize(self,
                current_params: np.ndarray,
                n_iterations: int = 100000,
                initial_step_size: float = 0.1) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize process parameters using adaptive random search.
        
        Args:
            current_params: Starting parameter values
            n_iterations: Number of optimization iterations
            initial_step_size: Initial parameter adjustment size
            
        Returns:
            Tuple of (optimized parameters, predicted values)
        """
        print("\nStarting parameter optimization...")
        print(f"Target criteria:")
        print(f"- First Pass Quality >= {self.targets.QUALITY_TARGET}%")
        print(f"- Throughput = {self.targets.THROUGHPUT_TARGET} ± {self.targets.THROUGHPUT_TOLERANCE}")
        print(f"- Scrap <= {self.targets.MAX_SCRAP}")
        
        # Initialize tracking
        current_params_scaled = self.scaler.transform([current_params])
        current_prediction = self.predict(current_params_scaled)
        
        best_result = OptimizationResult(
            params=current_params.copy(),
            predictions=current_prediction.copy(),
            score=float('-inf'),
            meets_criteria=False
        )
        
        valid_solutions: List[OptimizationResult] = []
        
        try:
            for i in range(n_iterations):
                # Adaptive step size
                step_size = initial_step_size * (1 - i/n_iterations)
                
                # Generate new parameters
                new_params = current_params + np.random.normal(0, step_size, size=current_params.shape)
                new_params_scaled = self.scaler.transform([new_params])
                new_prediction = self.predict(new_params_scaled)
                
                # Evaluate solution
                score, meets_criteria = self.calculate_scores(new_prediction)
                
                result = OptimizationResult(
                    params=new_params.copy(),
                    predictions=new_prediction.copy(),
                    score=score,
                    meets_criteria=meets_criteria
                )
                
                # Track valid solutions
                if meets_criteria:
                    valid_solutions.append(result)
                
                # Update best solution
                if (meets_criteria and score > best_result.score) or \
                   (not best_result.meets_criteria and score > best_result.score):
                    best_result = result
                
                # Progress update
                if (i + 1) % (n_iterations // 10) == 0:
                    print(f"Progress: {(i + 1) / n_iterations * 100:.1f}% complete")
            
            # Print optimization results
            self._print_optimization_summary(valid_solutions, best_result)
            
            # Save results
            self._save_optimization_results(valid_solutions, best_result)
            
            return best_result.params, best_result.predictions
            
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return current_params, current_prediction
    
    def _print_optimization_summary(self,
                                  valid_solutions: List[OptimizationResult],
                                  best_result: OptimizationResult) -> None:
        """Print summary of optimization results."""
        print(f"\nOptimization Summary:")
        print(f"{'=' * 50}")
        print(f"Valid solutions found: {len(valid_solutions)}")
        
        if valid_solutions:
            qualities = [sol.predictions['first_pass_quality'] for sol in valid_solutions]
            throughputs = [sol.predictions['throughput'] for sol in valid_solutions]
            scraps = [sol.predictions['scrap'] for sol in valid_solutions]
            
            print("\nValid Solutions Statistics:")
            print(f"Quality     - Avg: {np.mean(qualities):.2f}, "
                  f"Min: {np.min(qualities):.2f}, "
                  f"Max: {np.max(qualities):.2f}")
            print(f"Throughput  - Avg: {np.mean(throughputs):.2f}, "
                  f"Min: {np.min(throughputs):.2f}, "
                  f"Max: {np.max(throughputs):.2f}")
            print(f"Scrap       - Avg: {np.mean(scraps):.2f}, "
                  f"Min: {np.min(scraps):.2f}, "
                  f"Max: {np.max(scraps):.2f}")
        
        print("\nBest Solution Found:")
        print(f"{'=' * 50}")
        print(f"Meets all criteria: {best_result.meets_criteria}")
        print(f"Predicted outcomes:")
        print(f"- First Pass Quality: {best_result.predictions['first_pass_quality']:.2f}%")
        print(f"- Throughput: {best_result.predictions['throughput']:.2f}")
        print(f"- Scrap: {best_result.predictions['scrap']:.2f}")
    
    def _save_optimization_results(self,
                                 valid_solutions: List[OptimizationResult],
                                 best_result: OptimizationResult) -> None:
        """Save optimization results to files."""
        try:
            # Create results directory
            results_dir = self.export_dir / 'optimization_results'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best parameters
            best_params_df = pd.DataFrame(
                [best_result.params],
                columns=self.feature_names
            )
            best_params_df.to_csv(
                results_dir / 'best_parameters.csv',
                index=False
            )
            
            # Save valid solutions if any exist
            if valid_solutions:
                valid_solutions_df = pd.DataFrame([
                    {
                        **dict(zip(self.feature_names, sol.params)),
                        **sol.predictions,
                        'score': sol.score
                    }
                    for sol in valid_solutions
                ])
                valid_solutions_df.to_csv(
                    results_dir / 'valid_solutions.csv',
                    index=False
                )
            
        except Exception as e:
            print(f"Error saving optimization results: {str(e)}")
            
def analyze_optimization_results(current_params: np.ndarray,
                               optimized_params: np.ndarray,
                               current_predictions: dict,
                               optimized_predictions: dict,
                               feature_names: list,
                               export_dir: str = './exports') -> None:
    """
    Analyze and report optimization results with detailed comparisons.
    
    Args:
        current_params: Original parameter values
        optimized_params: Optimized parameter values
        current_predictions: Predictions with current parameters
        optimized_predictions: Predictions with optimized parameters
        feature_names: Names of features/parameters
        export_dir: Directory to save results
    """
    export_dir = Path(export_dir)
    results_dir = export_dir / 'optimization_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate changes
    param_changes = optimized_params - current_params
    prediction_changes = {
        key: optimized_predictions[key] - current_predictions[key]
        for key in current_predictions
    }
    
    # Create parameter comparison DataFrame
    param_df = pd.DataFrame({
        'Parameter': feature_names,
        'Current': current_params,
        'Optimized': optimized_params,
        'Absolute_Change': param_changes,
        'Percent_Change': (param_changes / np.abs(current_params)) * 100
    })
    
    # Create predictions comparison DataFrame
    pred_df = pd.DataFrame({
        'Metric': list(current_predictions.keys()),
        'Current': list(current_predictions.values()),
        'Optimized': list(optimized_predictions.values()),
        'Absolute_Change': list(prediction_changes.values()),
        'Percent_Change': [
            (optimized_predictions[k] - current_predictions[k]) / abs(current_predictions[k]) * 100
            for k in current_predictions
        ]
    })
    
    # Print console report
    print("\nOPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    
    print("\nPARAMETER CHANGES:")
    print("-" * 80)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(param_df.to_string(index=False))
    
    print("\nPERFORMANCE IMPROVEMENTS:")
    print("-" * 80)
    print(pred_df.to_string(index=False))
    
    # Generate detailed report file
    report_path = results_dir / 'optimization_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("OPTIMIZATION RESULTS REPORT\n")
        f.write("=========================\n\n")
        
        # Process Parameters Section
        f.write("PROCESS PARAMETERS\n")
        f.write("-----------------\n")
        f.write("\nParameter Changes:\n")
        f.write(param_df.to_string())
        
        # Key Metrics Section
        f.write("\n\nPERFORMANCE METRICS\n")
        f.write("-----------------\n")
        f.write("\nMetric Changes:\n")
        f.write(pred_df.to_string())
        
        # Optimization Summary
        f.write("\n\nOPTIMIZATION SUMMARY\n")
        f.write("-----------------\n")
        f.write("\nKey Improvements:\n")
        for metric, row in pred_df.iterrows():
            f.write(f"\n{row['Metric']}:\n")
            f.write(f"- Changed from {row['Current']:.2f} to {row['Optimized']:.2f}\n")
            f.write(f"- Absolute improvement: {row['Absolute_Change']:.2f}\n")
            f.write(f"- Relative improvement: {row['Percent_Change']:.1f}%\n")
        
        # Parameter Recommendations
        f.write("\n\nPARAMETER RECOMMENDATIONS\n")
        f.write("-----------------------\n")
        significant_changes = param_df[abs(param_df['Percent_Change']) > 5].sort_values(
            'Percent_Change', key=abs, ascending=False
        )
        
        f.write("\nMost Significant Parameter Changes (>5% change):\n")
        for _, row in significant_changes.iterrows():
            f.write(f"\n{row['Parameter']}:\n")
            f.write(f"- Current: {row['Current']:.2f}\n")
            f.write(f"- Recommended: {row['Optimized']:.2f}\n")
            f.write(f"- Change: {row['Absolute_Change']:.2f} ({row['Percent_Change']:.1f}%)\n")
    
    # Save DataFrames to CSV
    param_df.to_csv(results_dir / 'parameter_changes.csv', index=False)
    pred_df.to_csv(results_dir / 'performance_improvements.csv', index=False)
    
    # Create visualization of changes
    plt.figure(figsize=(15, 10))
    
    # Parameter changes plot
    plt.subplot(2, 1, 1)
    significant_params = param_df[abs(param_df['Percent_Change']) > 1]  # Show only changes > 1%
    plt.bar(significant_params['Parameter'], significant_params['Percent_Change'])
    plt.title('Significant Parameter Changes')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Change (%)')
    
    # Performance improvements plot
    plt.subplot(2, 1, 2)
    colors = ['g' if x > 0 else 'r' for x in pred_df['Percent_Change']]
    plt.bar(pred_df['Metric'], pred_df['Percent_Change'], color=colors)
    plt.title('Performance Metric Changes')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Change (%)')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'optimization_changes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDetailed results saved to: {results_dir}")

"""
LLM-based Manufacturing Process Analysis
--------------------------------------
Generates insights and optimization suggestions using GPT-4 or GPT-3.5
for manufacturing process optimization.
"""
class ManufacturingAnalyst:
    """Handles LLM-based analysis of manufacturing processes."""
    
    def __init__(self,
                 export_dir: str = './exports',
                 model_name: str = 'gpt-4',
                 temperature: float = 0.7):
        """
        Initialize the manufacturing analyst.
        
        Args:
            export_dir: Directory for saving analysis results
            model_name: GPT model to use
            temperature: Temperature for GPT responses
        """
        self.export_dir = Path(export_dir)
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI()
        
    def _create_parameter_comparison(self,
                                   current_params: np.ndarray,
                                   optimized_params: np.ndarray,
                                   feature_names: List[str]) -> str:
        """Create formatted parameter comparison text."""
        comparison = "\nParameter Comparisons:\n" + "-" * 50
        
        for name, curr, opt in zip(feature_names, current_params, optimized_params):
            change = ((opt - curr) / abs(curr)) * 100
            comparison += f"\n\n{name}:"
            comparison += f"\n- Current: {curr:.2f}"
            comparison += f"\n- Optimized: {opt:.2f}"
            comparison += f"\n- Change: {change:+.1f}%"
        
        return comparison
    
    def _create_performance_comparison(self,
                                     current_pred: Dict[str, float],
                                     optimized_pred: Dict[str, float]) -> str:
        """Create formatted performance comparison text."""
        comparison = "\n\nPerformance Results:\n" + "-" * 50
        
        for metric in current_pred.keys():
            curr = current_pred[metric]
            opt = optimized_pred[metric]
            change = ((opt - curr) / abs(curr)) * 100
            comparison += f"\n\n{metric.replace('_', ' ').title()}:"
            comparison += f"\n- Current: {curr:.2f}"
            comparison += f"\n- Optimized: {opt:.2f}"
            comparison += f"\n- Change: {change:+.1f}%"
        
        return comparison
    
    def _create_feature_importance_summary(self,
                                         models: Dict[str, Any],
                                         feature_names: List[str]) -> str:
        """Create feature importance summary for each target."""
        summary = "\nFeature Importance Analysis:\n" + "-" * 50
        
        for target, model_data in models.items():
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': model_data['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            summary += f"\n\nKey Drivers for {target.replace('_', ' ').title()}:"
            for _, row in importances.head(5).iterrows():
                summary += f"\n- {row['feature']}: {row['importance']:.4f}"
        
        return summary
    
    def generate_insights(self,
                         current_params: np.ndarray,
                         optimized_params: np.ndarray,
                         current_pred: Dict[str, float],
                         optimized_pred: Dict[str, float],
                         feature_names: List[str]) -> str:
        """Generate manufacturing process insights."""
        try:
            prompt = f"""
As a manufacturing process expert, analyze these cereal production optimization results:

{self._create_parameter_comparison(current_params, optimized_params, feature_names)}

{self._create_performance_comparison(current_pred, optimized_pred)}

Please provide:
1. Analysis of the most significant parameter changes and their impacts
2. Potential risks or trade-offs in implementing these changes
3. Step-by-step implementation recommendations
4. Additional optimization opportunities
5. Quality control considerations
6. Safety and compliance considerations
7. Cost-benefit analysis
8. Maintenance and monitoring recommendations
"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": 
                     "You are an expert manufacturing process engineer specializing in "
                     "cereal production. Provide detailed, technical, and practical insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating manufacturing insights: {str(e)}"
    
    def generate_recipe_suggestions(self,
                                  models: Dict[str, Any],
                                  feature_names: List[str]) -> str:
        """Generate recipe optimization suggestions."""
        try:
            prompt = f"""
Based on detailed analysis of our cereal manufacturing process:

{self._create_feature_importance_summary(models, feature_names)}

Please provide:
1. Recipe optimization recommendations based on the most influential parameters
2. Specific suggestions for improving each KPI (throughput, quality, and scrap)
3. Potential recipe modifications to enhance product consistency
4. Process control strategies for critical parameters
5. Experimental trial recommendations
6. Quality assurance procedures
7. Cost optimization opportunities
8. Sustainability considerations
"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": 
                     "You are an expert food scientist specializing in cereal manufacturing. "
                     "Provide detailed, technical, and practical recipe optimization advice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating recipe suggestions: {str(e)}"
    
    def save_analysis(self,
                     manufacturing_insights: str,
                     recipe_suggestions: str,
                     timestamp: bool = True) -> None:
        """Save analysis results to files."""
        try:
            # Create analysis directory
            analysis_dir = self.export_dir / 'llm_analysis'
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp if requested
            time_str = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""
            
            # Save manufacturing insights
            insights_path = analysis_dir / f'manufacturing_insights{time_str}.txt'
            with open(insights_path, 'w') as f:
                f.write("=== Manufacturing Process Insights ===\n")
                f.write("Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                f.write("\n" + "=" * 50 + "\n\n")
                f.write(manufacturing_insights)
            
            # Save recipe suggestions
            suggestions_path = analysis_dir / f'recipe_suggestions{time_str}.txt'
            with open(suggestions_path, 'w') as f:
                f.write("=== Recipe Optimization Suggestions ===\n")
                f.write("Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                f.write("\n" + "=" * 50 + "\n\n")
                f.write(recipe_suggestions)
            
            print(f"\nAnalysis results saved to:")
            print(f"- {insights_path}")
            print(f"- {suggestions_path}")
            
        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")


def main():
    api_ready, export_path = setup_environment()
    data = load_data()
    print("Setup completed successfully!")
    print(f"Export directory: {export_path}")
    print(f"Data shape: {data.shape}")
    # Create and save heatmap
    create_correlation_heatmap(data, export_path)

    # Analyze strong correlations
    strong_correlations = analyze_correlations(data)
    print("\nStrong correlations found:")
    print(strong_correlations)
    
    
    # Perform analysis
    # Sample KPIs (replace with your actual KPIs)

    # Define KPIs
    kpi_cols = ['throughput', 'first_pass_quality', 'scrap']

    # Create analysis
    analyze_kpi_correlations(
        data=data,  # Your DataFrame
        kpi_cols=kpi_cols,
        export_path=Path('./exports'),
        min_correlation=0.3  # Show correlations >= 0.3
    )
    # Initialize feature engineer
    engineer = FeatureEngineer(
    target_cols=['throughput', 'first_pass_quality', 'scrap'],
    random_state=42
    )

    # Prepare data
    X, X_test, X_train, X_train_scaled, X_test_scaled, y_train, y_test = engineer.prepare_data(
    data,
    export_path=Path('./exports/preprocessing')
    )
    
    # Train models
    models = train_models(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=X.columns.tolist(),
        export_dir='./exports',
        n_trials=5 #increase as needed. Diminishing returns. Used 5 for demo speed purposes.
    )
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            'Target': target_name,
            'RMSE': data['metrics']['rmse'],
            'R2': data['metrics']['r2'],
            'Mean_Prediction': data['metrics']['prediction_mean'],
            'Std_Prediction': data['metrics']['prediction_std'],
            'Direction': get_direction_str(data['study']),
            'Best_Trial': data['study'].best_trial.number,
            'Training_Time': data['training_time']
        }
        for target_name, data in models.items()
    ])

    # Print metrics summary
    print_metrics_summary(models, X)

    # Save summary to CSV
    summary_df.to_csv('./exports/model_performance_summary.csv', index=False)

    # Return summary DataFrame
    print(summary_df)
    
    # Export feature importances
    export_feature_importances(
        models=models,
        export_dir='./exports',
        feature_names=X.columns.tolist()  # Pass your feature names here
    )        
    
    
    # Example usage
    predictions = analyze_prediction_correlations(
        models=models,
        y_test=y_test,
        X_test_scaled=X_test_scaled,
        export_dir=export_path
    )

    # Print performance summary
    print_performance_summary(models)


    optimizer = ParameterOptimizer(
    models=models,  # Your trained models dictionary
    scaler=engineer.scaler,  # Use the scaler from your FeatureEngineer instance
    feature_names=X.columns.tolist(),
    export_dir='./exports'
    )

    # Get current parameters from first test row
    current_params = X_test.iloc[0].values

    # Run optimization
    print("\nOptimizing parameters...")
    optimized_params, optimized_predictions = optimizer.optimize(
        current_params=current_params,
        n_iterations=100000,
        initial_step_size=0.1
    )

    # Print parameter changes
    params_df = pd.DataFrame({
        'Parameter': X.columns,
        'Current': current_params,
        'Optimized': optimized_params,
        'Change': optimized_params - current_params
    })   
    
    print("\nParameter Changes:")
    print("=" * 80)
    print(params_df.to_string(index=False))

    # Calculate percentage improvements
    current_predictions = {
        target: model['model'].predict(engineer.scaler.transform([current_params]))[0]
        for target, model in models.items()
    }

    improvements = pd.DataFrame({
        'Metric': list(current_predictions.keys()),
        'Current': [current_predictions[k] for k in current_predictions],
        'Optimized': [optimized_predictions[k] for k in current_predictions],
    })

    improvements['Change'] = improvements['Optimized'] - improvements['Current']
    improvements['Change %'] = (improvements['Change'] / improvements['Current'] * 100)

    print("\nPredicted Improvements:")
    print("=" * 80)
    print(improvements.to_string(index=False))
    
    # Use the function
    current_predictions = {
        target: model['model'].predict(engineer.scaler.transform([current_params]))[0]
        for target, model in models.items()
    }

    # Now run the analysis
    analyze_optimization_results(
        current_params=current_params,
        optimized_params=optimized_params,
        current_predictions=current_predictions,
        optimized_predictions=optimized_predictions,  # From your optimizer
        feature_names=X.columns.tolist(),
        export_dir='./exports'
    )    
    
    # Example usage
    analyst = ManufacturingAnalyst(
        export_dir='./exports',
        model_name='gpt-4o',  # or 'gpt-3.5-turbo'
        temperature=0.7
    )

    print("\nGenerating Manufacturing Insights...")
    manufacturing_insights = analyst.generate_insights(
        current_params=current_params,
        optimized_params=optimized_params,
        current_pred=current_predictions,
        optimized_pred=optimized_predictions,
        feature_names=X.columns.tolist()
    )

    print("\nGenerating Recipe Optimization Suggestions...")
    recipe_suggestions = analyst.generate_recipe_suggestions(
        models=models,
        feature_names=X.columns.tolist()
    )

    # Save results
    analyst.save_analysis(manufacturing_insights, recipe_suggestions)

    # Display results
    print("\n=== Manufacturing Process Insights ===")
    print(manufacturing_insights)
    print("\n=== Recipe Optimization Suggestions ===")
    print(recipe_suggestions)
    
    
# demo.py

if __name__ == "__main__":
    main()