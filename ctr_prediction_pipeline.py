import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

class CTRPredictionPipeline:
    """
    A comprehensive pipeline for CTR prediction including:
    - Automated feature engineering
    - Multiple model comparison
    - Hyperparameter optimization
    - Model explainability
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the pipeline with data path
        
        Args:
            data_path: Path to the locations data file
        """
        self.data_path = data_path
        self.logger = self._setup_logger()
        self.models = {
            'xgboost': xgb.XGBRegressor(),
            'lightgbm': lgb.LGBMRegressor(),
            'catboost': cb.CatBoostRegressor(verbose=False)
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the locations data
        
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Feature engineering
        df = self._create_features(df)
        
        return df
        
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create intelligent features from the location data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Location-based features
        df['location_cluster'] = self._cluster_locations(df[['latitude', 'longitude']])
        
        # Interaction features
        df['time_location_interaction'] = df['hour'] * df['location_cluster']
        
        # Historical CTR features
        df['location_mean_ctr'] = df.groupby('location_id')['ctr'].transform('mean')
        df['location_std_ctr'] = df.groupby('location_id')['ctr'].transform('std')
        
        # Distance-based features
        df['distance_to_center'] = self._calculate_distance_to_center(df)
        
        return df
        
    def _cluster_locations(self, coords: pd.DataFrame, n_clusters: int = 10) -> np.ndarray:
        """Cluster locations using KMeans"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(coords)
        
    def _calculate_distance_to_center(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance to city center"""
        from sklearn.metrics.pairwise import haversine_distances
        center = df[['latitude', 'longitude']].mean()
        return haversine_distances(
            df[['latitude', 'longitude']],
            center.values.reshape(1, -1)
        ).flatten()
        
    def train_and_evaluate_models(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train and evaluate multiple models
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary of model performances
        """
        self.logger.info("Training and evaluating models...")
        
        # Prepare data
        X = df.drop(['ctr', 'timestamp'], axis=1)
        y = df['ctr']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        performances = {}
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            performances[name] = {
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
            
        # Select best model
        best_model_name = min(performances.items(), key=lambda x: x[1]['rmse'])[0]
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        return performances
        
    def optimize_hyperparameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the best model
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Best hyperparameters
        """
        self.logger.info(f"Optimizing hyperparameters for {self.best_model_name}...")
        
        # Define hyperparameter space based on best model
        if self.best_model_name == 'xgboost':
            space = {
                'max_depth': hp.choice('max_depth', range(3, 10)),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                'n_estimators': hp.choice('n_estimators', range(100, 1000, 100))
            }
        # Add spaces for other models...
        
        # Optimization objective
        def objective(params):
            model = self.models[self.best_model_name]
            model.set_params(**params)
            
            X = df.drop(['ctr', 'timestamp'], axis=1)
            y = df['ctr']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            return {'loss': rmse, 'status': STATUS_OK}
            
        # Run optimization
        trials = Trials()
        best = fmin(fn=objective,
                   space=space,
                   algo=tpe.suggest,
                   max_evals=100,
                   trials=trials)
                   
        return best
        
    def explain_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate model explanations using SHAP and LIME
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary containing feature importance and explanations
        """
        self.logger.info("Generating model explanations...")
        
        X = df.drop(['ctr', 'timestamp'], axis=1)
        
        # SHAP explanations
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X)
        
        # LIME explanations
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=X.columns,
            class_names=['CTR'],
            mode='regression'
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': self.feature_importance,
            'lime_explainer': lime_explainer
        }
        
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Returns:
            Dictionary containing all results
        """
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Train and evaluate models
        model_performances = self.train_and_evaluate_models(df)
        
        # Optimize best model
        best_params = self.optimize_hyperparameters(df)
        
        # Generate explanations
        explanations = self.explain_model(df)
        
        return {
            'model_performances': model_performances,
            'best_model': self.best_model_name,
            'best_params': best_params,
            'feature_importance': self.feature_importance
        }

if __name__ == "__main__":
    # Example usage
    pipeline = CTRPredictionPipeline("path_to_your_locations_file.csv")
    results = pipeline.run_pipeline()
    
    # Print results
    print("\nModel Performances:")
    print(results['model_performances'])
    print("\nBest Model:", results['best_model'])
    print("\nOptimal Parameters:")
    print(results['best_params'])
    print("\nTop 10 Important Features:")
    print(results['feature_importance'].head(10)) 