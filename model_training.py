import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    """Load preprocessed data"""
    return pd.read_csv('data/processed_tariff_data.csv')

def prepare_data(df):
    """
    Prepare data for modeling
    Returns X_train, X_test, y_train, y_test
    """
    # Convert year to numeric feature
    df['year_num'] = pd.to_datetime(df['year']).dt.year
    
    # For simplicity, we'll predict import tariffs based on year and country code
    X = df[['year_num', 'country_code']]
    y = df['import_tariffs']
    
    # One-hot encode country_code
    X = pd.get_dummies(X, columns=['country_code'])
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """Train and save a linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/linear_regression.pkl')
    return model

def train_random_forest(X_train, y_train):
    """Train and save a random forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/random_forest.pkl')
    return model

def train_models():
    """Main function to train all models"""
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print("Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("Model training complete!")
    return lr_model, rf_model

if __name__ == "__main__":
    train_models()