import pandas as pd
import requests
from io import BytesIO

def download_census_data():
    """Download US Census Bureau trade data"""
    url = "https://www.census.gov/foreign-trade/balance/country.xlsx"
    response = requests.get(url)
    return pd.read_excel(BytesIO(response.content))

def download_seasonal_data():
    """Download seasonal adjusted trade data"""
    url = "https://www.census.gov/foreign-trade/statistics/country/ctyseasonal.xlsx"
    response = requests.get(url)
    return pd.read_excel(BytesIO(response.content))

def save_raw_data():
    """Download and save all raw data files"""
    census_data = download_census_data()
    seasonal_data = download_seasonal_data()
    
    census_data.to_csv('data/raw_census_data.csv', index=False)
    seasonal_data.to_csv('data/raw_seasonal_data.csv', index=False)
    
    return census_data, seasonal_data

if __name__ == "__main__":
    save_raw_data()