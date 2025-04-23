import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import joblib

def prepare_time_series_data():
    """Prepare data for time series analysis"""
    df = pd.read_csv('data/processed_tariff_data.csv')
    df['year'] = pd.to_datetime(df['year'])
    
    # Aggregate by year for time series
    ts_data = df.groupby('year')['import_tariffs'].sum().reset_index()
    ts_data.set_index('year', inplace=True)
    
    return ts_data

def train_arima_model():
    """Train and save ARIMA time series model"""
    ts_data = prepare_time_series_data()
    
    # Fit ARIMA(1,1,1) model - simple configuration
    model = ARIMA(ts_data, order=(1,1,1))
    model_fit = model.fit()
    
    # Save model
    joblib.dump(model_fit, 'models/arima_model.pkl')
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(ts_data, label='Actual')
    plt.plot(model_fit.fittedvalues, color='red', label='Fitted')
    plt.title('ARIMA Model Fit')
    plt.legend()
    plt.savefig('visualizations/arima_fit.png')
    plt.close()
    
    return model_fit

def evaluate_arima(model):
    """Evaluate ARIMA model"""
    ts_data = prepare_time_series_data()
    predictions = model.predict(start=1, end=len(ts_data))
    mae = mean_absolute_error(ts_data, predictions)
    print(f"ARIMA MAE: ${mae:,.2f}")
    return mae

if __name__ == "__main__":
    model = train_arima_model()
    evaluate_arima(model)