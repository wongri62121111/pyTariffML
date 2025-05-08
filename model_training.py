import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

class TariffModelTrainer:
    def __init__(self):
        self.features = [
            'year', 'country_code', 'gdp', 
            'inflation_rate', 'tariff_intensity',
            'region', 'trade_balance'
        ]
        self.target = 'inflation_adjusted_tariffs'
        
    def create_pipeline(self, model):
        """Create a complete ML pipeline"""
        numeric_features = ['gdp', 'inflation_rate', 'tariff_intensity']
        categorical_features = ['country_code', 'region']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
    
    def train_models(self, data):
        """Train and evaluate multiple models"""
        X = data[self.features]
        y = data[self.target]
        
        # Time-series aware cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ),
            'xgboost': XGBRegressor(
                n_estimators=150,
                learning_rate=0.1,
                early_stopping_rounds=10
            )
        }
        
        results = {}
        for name, model in models.items():
            pipeline = self.create_pipeline(model)
            
            # Cross-validate
            cv_scores = cross_val_score(
                pipeline, X, y, 
                cv=tscv,
                scoring='neg_mean_absolute_error'
            )
            
            # Final training
            pipeline.fit(X, y)
            joblib.dump(pipeline, f'models/trained/{name}_pipeline.pkl')
            
            results[name] = {
                'mae': -cv_scores.mean(),
                'model': pipeline
            }
        
        return results