import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations_dir():
    """Create visualizations directory if it doesn't exist"""
    os.makedirs('visualizations', exist_ok=True)

def load_processed_data():
    """Load the preprocessed data with error handling"""
    try:
        df = pd.read_csv('data/processed_tariff_data.csv')
        # Convert year column to datetime if it's not already
        if 'year' in df.columns:
            df['year'] = pd.to_datetime(df['year'])
        return df
    except FileNotFoundError:
        print("Error: Processed data file not found. Run data_preprocessing.py first.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def plot_tariff_trends(df):
    """Plot import/export tariff trends over time"""
    create_visualizations_dir()
    
    plt.figure(figsize=(12, 6))
    
    # Use the actual column names from your data
    if 'annual_import_tariffs' in df.columns:
        sns.lineplot(data=df, x='year', y='annual_import_tariffs', label='Annual Import Tariffs')
    if 'annual_export_tariffs' in df.columns:
        sns.lineplot(data=df, x='year', y='annual_export_tariffs', label='Annual Export Tariffs')
    
    plt.title('US Annual Import and Export Tariffs Over Time')
    plt.ylabel('Tariff Amount ($)')
    plt.xlabel('Year')
    plt.legend()
    plt.savefig('visualizations/tariff_trends.png')
    plt.close()

def plot_top_countries(df, year=2020):
    """Plot top countries by import tariffs for a given year"""
    create_visualizations_dir()
    
    # Filter by year - first convert year column if needed
    if not pd.api.types.is_datetime64_any_dtype(df['year']):
        df['year'] = pd.to_datetime(df['year'])
    
    yearly_data = df[df['year'].dt.year == year]
    
    if 'annual_import_tariffs' in yearly_data.columns:
        top_imports = yearly_data.nlargest(10, 'annual_import_tariffs')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_imports, x='country_name', y='annual_import_tariffs')
        plt.title(f'Top 10 Countries by Annual Import Tariffs in {year}')
        plt.ylabel('Import Tariffs ($)')
        plt.xlabel('Country')
        plt.xticks(rotation=45)
        plt.savefig(f'visualizations/top_imports_{year}.png')
        plt.close()

def run_eda():
    """Run all exploratory data analysis with error handling"""
    df = load_processed_data()
    if df is None:
        return
    
    print("Running exploratory data analysis...")
    
    # Basic data inspection
    print("\nData Summary:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Generate plots
    plot_tariff_trends(df)
    plot_top_countries(df)
    
    # Generate summary statistics
    summary = df.describe()
    summary.to_csv('data/summary_statistics.csv')
    print("\nSummary statistics saved to data/summary_statistics.csv")
    
    print("EDA completed successfully!")

if __name__ == "__main__":
    run_eda()