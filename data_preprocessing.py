import pandas as pd
import os
from datetime import datetime

def create_data_directory():
    """Create data directory if it doesn't exist"""
    os.makedirs('data', exist_ok=True)

def clean_census_data(df):
    """Clean and preprocess the census trade data"""
    # Select relevant columns - using the actual column names from the data
    cols = ['year', 'CTY_CODE', 'CTYNAME', 'IYR', 'EYR']
    df = df[cols].copy()
    
    # Rename columns to be more descriptive
    df.columns = ['year', 'country_code', 'country_name', 'annual_import_tariffs', 'annual_export_tariffs']
    
    # Convert year to datetime (data shows just years)
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    
    # Handle missing values - fill with 0 for tariffs
    df['annual_import_tariffs'] = df['annual_import_tariffs'].fillna(0)
    df['annual_export_tariffs'] = df['annual_export_tariffs'].fillna(0)
    
    return df

def clean_seasonal_data(df):
    """Clean and preprocess the seasonal adjusted trade data"""
    # Select relevant columns from seasonal data
    cols = ['year', 'cty_code', 'cty_desc', 'IQ1', 'IQ2', 'IQ3', 'IQ4', 'EQ1', 'EQ2', 'EQ3', 'EQ4']
    df = df[cols].copy()
    
    # Rename columns
    df.columns = ['year', 'country_code', 'country_name', 
                 'imports_q1', 'imports_q2', 'imports_q3', 'imports_q4',
                 'exports_q1', 'exports_q2', 'exports_q3', 'exports_q4']
    
    # Convert year to datetime
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    
    # Fill missing values with 0
    for col in df.columns[3:]:  # Skip first 3 columns
        df[col] = df[col].fillna(0)
    
    return df

def merge_datasets(census_df, seasonal_df):
    """Merge census and seasonal data"""
    # Convert year to string for merging
    census_df['year_str'] = census_df['year'].dt.year.astype(str)
    seasonal_df['year_str'] = seasonal_df['year'].dt.year.astype(str)
    
    # Merge on year and country code
    merged_df = pd.merge(
        census_df,
        seasonal_df,
        how='left',
        on=['year_str', 'country_code'],
        suffixes=('', '_seasonal')
    )
    
    # Clean up merged columns
    merged_df.drop(columns=['year_str', 'country_name_seasonal'], inplace=True, errors='ignore')
    
    return merged_df

def add_regional_data(df):
    """Add regional groupings to the data"""
    # Example regional mapping - would need to be expanded
    regions = {
        'European Union': 'Europe',
        'Canada': 'North America',
        'Mexico': 'North America',
        'China': 'Asia',
        'Japan': 'Asia',
        'Brazil': 'South America'
    }
    
    df['region'] = df['country_name'].map(regions)
    return df

def preprocess_data():
    """Main function to run all preprocessing steps"""
    create_data_directory()
    
    try:
        # Load raw data
        census_df = pd.read_csv('data/raw_census_data.csv')
        seasonal_df = pd.read_csv('data/raw_seasonal_data.csv')
        
        # Clean data
        cleaned_census = clean_census_data(census_df)
        cleaned_seasonal = clean_seasonal_data(seasonal_df)
        
        # Merge datasets
        merged_data = merge_datasets(cleaned_census, cleaned_seasonal)
        
        # Add additional features
        final_data = add_regional_data(merged_data)
        
        # Save processed data
        final_data.to_csv('data/processed_tariff_data.csv', index=False)
        print("Data preprocessing completed successfully!")
        return final_data
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None

if __name__ == "__main__":
    preprocess_data()