import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)

def evaluate_regression_model(model, X_test, y_test, model_name):
    """Comprehensive evaluation for regression models"""
    predictions = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'R2': r2_score(y_test, predictions),
        'Explained Variance': explained_variance_score(y_test, predictions),
        'Mean Actual': np.mean(y_test),
        'Mean Predicted': np.mean(predictions)
    }
    
    print(f"\n{model_name} Evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics

def evaluate_all_models():
    """Evaluate all trained models"""
    # Load test data (assuming same split as training)
    df = pd.read_csv('data/processed_tariff_data.csv')
    df['year_num'] = pd.to_datetime(df['year']).dt.year
    X = df[['year_num', 'country_code']]
    y = df['import_tariffs']
    X = pd.get_dummies(X, columns=['country_code'])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = joblib.load('models/scaler.pkl')
    X_test = scaler.transform(X_test)
    
    # Evaluate each model
    all_metrics = {}
    
    print("=== Regression Models Evaluation ===")
    models = {
        'Linear Regression': 'linear_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    for name, file in models.items():
        model = joblib.load(f'models/{file}')
        all_metrics[name] = evaluate_regression_model(model, X_test, y_test, name)
    
    # Time Series evaluation
    print("\n=== Time Series Model Evaluation ===")
    from time_series_model import evaluate_arima
    arima_model = joblib.load('models/arima_model.pkl')
    all_metrics['ARIMA'] = {'MAE': evaluate_arima(arima_model)}
    
    # Save all metrics
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv('data/all_model_metrics.csv')
    
    return metrics_df

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    evaluate_all_models()