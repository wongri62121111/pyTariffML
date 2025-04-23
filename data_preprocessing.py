import pandas as pd
from datetime import datetime

def clean_census_data(df):
    """Clean and preprocess the census trade data"""
    # Select relevant columns
    cols = ['cty_code', 'cty_name', 'IYR', 'EYR', 'time']
    df = df[cols].copy()
    
    # Rename columns
    df.columns = ['country_code', 'country_name', 'import_tariffs', 'export_tariffs', 'year']
    
    # Convert year to datetime
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    
    # Handle missing values
    df['import_tariffs'] = df['import_tariffs'].fillna(0)
    df['export_tariffs'] = df['export_tariffs'].fillna(0)
    
    return df

def add_regional_data(df):
    """Add regional groupings to the data"""
    # This would be expanded with actual regional mappings
    regions = {
        'Canada': 'North America',
        'Mexico': 'North America',
        'China': 'Asia',
        # Add more country-region mappings
    }
    
    df['region'] = df['country_name'].map(regions)
    return df

def preprocess_data():
    """Run all preprocessing steps"""
    raw_data = pd.read_csv('data/raw_census_data.csv')
    cleaned_data = clean_census_data(raw_data)
    final_data = add_regional_data(cleaned_data)
    
    final_data.to_csv('data/processed_tariff_data.csv', index=False)
    return final_data

if __name__ == "__main__":
    preprocess_data()