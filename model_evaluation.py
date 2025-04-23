import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def load_test_data():
    """Load data and prepare test set"""
    df = pd.read_csv('data/processed_tariff_data.csv')
    df['year_num'] = pd.to_datetime(df['year']).dt.year
    X = df[['year_num', 'country_code']]
    y = df['import_tariffs']
    X = pd.get_dummies(X, columns=['country_code'])
    
    # Use same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features using saved scaler
    scaler = joblib.load('models/scaler.pkl')
    X_test = scaler.transform(X_test)
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and print metrics"""
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"\n{model_name} Evaluation:")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Tariffs')
    plt.ylabel('Predicted Tariffs')
    plt.title(f'{model_name} - Actual vs Predicted Tariffs')
    plt.savefig(f'visualizations/{model_name}_evaluation.png')
    plt.close()
    
    return mae, rmse

def evaluate_all_models():
    """Evaluate all trained models"""
    X_test, y_test = load_test_data()
    
    print("Loading trained models...")
    lr_model = joblib.load('models/linear_regression.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    
    print("\nEvaluating models...")
    lr_mae, lr_rmse = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    rf_mae, rf_rmse = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Save metrics to file
    metrics = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'MAE': [lr_mae, rf_mae],
        'RMSE': [lr_rmse, rf_rmse]
    })
    metrics.to_csv('data/model_metrics.csv', index=False)
    
    return metrics

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split  # Needed for load_test_data
    evaluate_all_models()