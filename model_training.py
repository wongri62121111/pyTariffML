import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_models_dir():
    """Create models directory if it doesn't exist"""
    os.makedirs('models', exist_ok=True)

def load_data():
    """Load preprocessed data with error handling"""
    try:
        df = pd.read_csv('data/processed_tariff_data.csv')
        # Verify required columns exist
        required_cols = ['year', 'country_code', 'annual_import_tariffs']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Processed data is missing required columns")
        return df
    except FileNotFoundError:
        print("Error: Processed data file not found. Run data_preprocessing.py first.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def prepare_data(df):
    """
    Prepare data for modeling
    Returns X_train, X_test, y_train, y_test
    """
    # Convert year to numeric feature
    df['year_num'] = pd.to_datetime(df['year']).dt.year
    
    # Use the correct column name from your data
    X = df[['year_num', 'country_code']]
    y = df['annual_import_tariffs']  # Changed from 'import_tariffs'
    
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
    create_models_dir()
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
    create_models_dir()
    
    df = load_data()
    if df is None:
        return None, None
    
    try:
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        print("Training Linear Regression...")
        lr_model = train_linear_regression(X_train, y_train)
        
        print("Training Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        
        print("Model training complete!")
        return lr_model, rf_model
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return None, None

if __name__ == "__main__":
    train_models()