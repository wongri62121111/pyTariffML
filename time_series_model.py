import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import joblib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

def add_time_series_visualizations(ts_data, model_fit):
    """Add diagnostic visualizations for time series"""
    # 1. Time series decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(ts_data, model='additive', period=1)
    
    plt.figure(figsize=(12,8))
    decomposition.plot()
    plt.savefig('visualizations/ts_decomposition.png')
    plt.close()
    
    # 2. ACF/PACF plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
    plot_acf(ts_data, ax=ax1)
    plot_pacf(ts_data, ax=ax2)
    plt.savefig('visualizations/acf_pacf.png')
    plt.close()
    
    # 3. Forecast vs actual with confidence intervals
    forecast = model_fit.get_forecast(steps=5)
    ci = forecast.conf_int()
    
    plt.figure(figsize=(12,6))
    plt.plot(ts_data, label='Historical')
    plt.plot(forecast.predicted_mean, label='Forecast')
    plt.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='k', alpha=0.1)
    plt.title('5-Year Tariff Forecast with Confidence Intervals')
    plt.legend()
    plt.savefig('visualizations/forecast_ci.png')
    plt.close()

if __name__ == "__main__":
    model = train_arima_model()
    evaluate_arima(model)
    add_time_series_visualizations(prepare_time_series_data(), model)
    print("ARIMA model training and evaluation complete!")