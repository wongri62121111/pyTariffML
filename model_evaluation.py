import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)
from sklearn.model_selection import train_test_split
import os

def create_visualizations_dir():
    """Create visualizations directory if it doesn't exist"""
    os.makedirs('visualizations', exist_ok=True)

def load_test_data():
    """Load data and prepare test set with correct column names"""
    try:
        df = pd.read_csv('data/processed_tariff_data.csv')
        
        # Verify required columns exist
        required_cols = ['year', 'country_code', 'annual_import_tariffs']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Processed data is missing required columns")
        
        # Prepare features
        df['year_num'] = pd.to_datetime(df['year']).dt.year
        X = df[['year_num', 'country_code']]
        y = df['annual_import_tariffs']  # Corrected column name
        
        # One-hot encode country_code
        X = pd.get_dummies(X, columns=['country_code'])
        
        # Use same split as training (80% train, 20% test)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features using saved scaler
        scaler = joblib.load('models/scaler.pkl')
        X_test = scaler.transform(X_test)
        
        return X_test, y_test
    
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure you've run data_preprocessing.py and model_training.py first")
        return None, None
    except Exception as e:
        print(f"Error preparing test data: {str(e)}")
        return None, None

def evaluate_regression_model(model, X_test, y_test, model_name):
    """Comprehensive evaluation for regression models"""
    try:
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
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None

def evaluate_all_models():
    """Evaluate all trained models with error handling"""
    create_visualizations_dir()
    
    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None or y_test is None:
        return
    
    # Evaluate each model
    all_metrics = {}
    models_to_evaluate = {
        'Linear Regression': 'linear_regression.pkl',
        'Random Forest': 'random_forest.pkl'
    }
    
    print("=== Model Evaluation ===")
    for name, file in models_to_evaluate.items():
        try:
            model = joblib.load(f'models/{file}')
            metrics = evaluate_regression_model(model, X_test, y_test, name)
            if metrics:
                all_metrics[name] = metrics
        except FileNotFoundError:
            print(f"Model {name} not found at models/{file}")
        except Exception as e:
            print(f"Error loading {name} model: {str(e)}")
    
    # Save all metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.to_csv('data/model_metrics.csv')
        print("\nMetrics saved to data/model_metrics.csv")
    else:
        print("\nNo models were successfully evaluated")

if __name__ == "__main__":
    evaluate_all_models()