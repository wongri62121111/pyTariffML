import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def prepare_xgboost_data():
    """Prepare data for XGBoost"""
    df = pd.read_csv('data/processed_tariff_data.csv')
    
    # Feature engineering
    df['year_num'] = pd.to_datetime(df['year']).dt.year
    df['tariff_ratio'] = df['import_tariffs'] / (df['export_tariffs'] + 1)  # Add 1 to avoid divide by zero
    
    # Features and target
    X = df[['year_num', 'country_code', 'tariff_ratio']]
    y = df['import_tariffs']
    
    # One-hot encoding
    X = pd.get_dummies(X, columns=['country_code'])
    
    return X, y

def train_xgboost_model():
    """Train and save XGBoost model"""
    X, y = prepare_xgboost_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'models/xgboost_model.pkl')
    
    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"XGBoost MAE: ${mae:,.2f}")
    
    return model, mae

if __name__ == "__main__":
    model, mae = train_xgboost_model()