import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

def plot_regression_results(y_true, y_pred, model_name):
    """Visualization for regression models"""
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
    kmeans = joblib.load('models/kmeans_model.pkl')
    X, countries, _ = prepare_clustering_data()
    
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

def visualize_time_series():
    """Visualization for time series results"""
    ts_data = prepare_time_series_data()
    model = joblib.load('models/arima_model.pkl')
    
    plt.figure(figsize=(12, 6))
    plt.plot(ts_data, label='Actual')
    plt.plot(model.fittedvalues, label='Fitted', alpha=0.7)
    plt.title('ARIMA Time Series Model')
    plt.xlabel('Year')
    plt.ylabel('Import Tariffs')
    plt.legend()
    plt.savefig('visualizations/time_series_results.png')
    plt.close()

def visualize_feature_importance():
    """Visualization for feature importance"""
    # XGBoost feature importance
    xgb_model = joblib.load('models/xgboost_model.pkl')
    
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model)
    plt.title('XGBoost Feature Importance')
    plt.savefig('visualizations/xgboost_feature_importance.png')
    plt.close()

def visualize_all_models():
    """Generate all visualizations"""
    # Load test data for regression models
    df = pd.read_csv('data/processed_tariff_data.csv')
    df['year_num'] = pd.to_datetime(df['year']).dt.year
    X = df[['year_num', 'country_code']]
    y = df['import_tariffs']
    X = pd.get_dummies(X, columns=['country_code'])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Regression model visuals
    models = {
        'Linear Regression': 'linear_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    for name, file in models.items():
        model = joblib.load(f'models/{file}')
        preds = model.predict(X_test)
        plot_regression_results(y_test, preds, name)
    
    # Other visuals
    visualize_clusters()
    visualize_time_series()
    visualize_feature_importance()

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from clustering_model import prepare_clustering_data
    from time_series_model import prepare_time_series_data
    visualize_all_models()