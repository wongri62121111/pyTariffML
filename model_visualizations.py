import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os

def create_visualizations_dir():
    """Create visualizations directory if it doesn't exist"""
    os.makedirs('visualizations', exist_ok=True)

def load_processed_data():
    """Load processed data with correct column names"""
    try:
        df = pd.read_csv('data/processed_tariff_data.csv')
        
        # Verify required columns exist
        required_cols = ['year', 'country_code', 'annual_import_tariffs']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Processed data is missing required columns")
        
        # Convert year to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['year']):
            df['year'] = pd.to_datetime(df['year'])
            
        return df
    except FileNotFoundError:
        print("Error: Processed data file not found. Run data_preprocessing.py first.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def prepare_test_data(df):
    """Prepare test data for visualization"""
    try:
        df['year_num'] = df['year'].dt.year
        X = df[['year_num', 'country_code']]
        y = df['annual_import_tariffs']  # Corrected column name
        
        # One-hot encode country_code
        X = pd.get_dummies(X, columns=['country_code'])
        
        # Get same test split as used in training
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features using saved scaler
        scaler = joblib.load('models/scaler.pkl')
        X_test = scaler.transform(X_test)
        
        return X_test, y_test
    except Exception as e:
        print(f"Error preparing test data: {str(e)}")
        return None, None

def plot_regression_results(y_true, y_pred, model_name):
    """Visualization for regression models"""
    create_visualizations_dir()
    
    plt.figure(figsize=(12, 6))
    
    # Residual plot
    plt.subplot(1, 2, 1)
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name} Residuals')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    # Actual vs Predicted
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_regression.png')
    plt.close()

def visualize_clusters():
    """Visualization for clustering results"""
    try:
        from clustering_model import prepare_clustering_data
        X, countries, _ = prepare_clustering_data()
        kmeans = joblib.load('models/kmeans_model.pkl')
        
        create_visualizations_dir()
        
        plt.figure(figsize=(14, 8))
        scatter = plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='viridis', s=100)
        
        # Annotate cluster centers
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')
        
        # Annotate some countries
        for i, country in enumerate(countries):
            if i % 5 == 0:  # Label every 5th country
                plt.annotate(country, (X[i,0], X[i,1]), fontsize=8)
        
        plt.title('Country Clusters by Tariff Patterns')
        plt.xlabel('Scaled Import Tariffs')
        plt.ylabel('Scaled Export Tariffs')
        plt.colorbar(scatter, label='Cluster')
        plt.savefig('visualizations/cluster_results.png')
        plt.close()
    except Exception as e:
        print(f"Error visualizing clusters: {str(e)}")

def visualize_time_series():
    """Visualization for time series results"""
    try:
        from time_series_model import prepare_time_series_data
        ts_data = prepare_time_series_data()
        model = joblib.load('models/arima_model.pkl')
        
        create_visualizations_dir()
        
        plt.figure(figsize=(12, 6))
        plt.plot(ts_data, label='Actual')
        plt.plot(model.fittedvalues, label='Fitted', alpha=0.7)
        plt.title('ARIMA Time Series Model')
        plt.xlabel('Year')
        plt.ylabel('Import Tariffs')
        plt.legend()
        plt.savefig('visualizations/time_series_results.png')
        plt.close()
    except Exception as e:
        print(f"Error visualizing time series: {str(e)}")

def visualize_feature_importance():
    """Visualization for feature importance"""
    try:
        import xgboost
        xgb_model = joblib.load('models/xgboost_model.pkl')
        
        create_visualizations_dir()
        
        plt.figure(figsize=(10, 6))
        xgboost.plot_importance(xgb_model)
        plt.title('XGBoost Feature Importance')
        plt.savefig('visualizations/xgboost_feature_importance.png')
        plt.close()
    except Exception as e:
        print(f"Error visualizing feature importance: {str(e)}")

def visualize_all_models():
    """Generate all visualizations with error handling"""
    df = load_processed_data()
    if df is None:
        return
    
    X_test, y_test = prepare_test_data(df)
    if X_test is None or y_test is None:
        return
    
    # Regression model visuals
    models_to_visualize = {
        'Linear Regression': 'linear_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    for name, file in models_to_visualize.items():
        try:
            model = joblib.load(f'models/{file}')
            preds = model.predict(X_test)
            plot_regression_results(y_test, preds, name)
        except FileNotFoundError:
            print(f"Model {name} not found at models/{file}")
        except Exception as e:
            print(f"Error visualizing {name}: {str(e)}")
    
    # Other visuals
    visualize_clusters()
    visualize_time_series()
    visualize_feature_importance()
    
    print("Visualization process completed (check for any errors above)")

if __name__ == "__main__":
    visualize_all_models()