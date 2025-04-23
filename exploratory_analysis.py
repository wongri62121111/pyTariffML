import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data():
    """Load the preprocessed data"""
    return pd.read_csv('data/processed_tariff_data.csv')

def plot_tariff_trends(df):
    """Plot import/export tariff trends over time"""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y='import_tariffs', label='Import Tariffs')
    sns.lineplot(data=df, x='year', y='export_tariffs', label='Export Tariffs')
    plt.title('US Import and Export Tariffs Over Time')
    plt.ylabel('Tariff Amount ($)')
    plt.xlabel('Year')
    plt.legend()
    plt.savefig('visualizations/tariff_trends.png')
    plt.close()

def plot_top_countries(df, year=2020):
    """Plot top countries by import tariffs for a given year"""
    yearly_data = df[df['year'].dt.year == year]
    top_imports = yearly_data.nlargest(10, 'import_tariffs')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_imports, x='country_name', y='import_tariffs')
    plt.title(f'Top 10 Countries by Import Tariffs in {year}')
    plt.ylabel('Import Tariffs ($)')
    plt.xlabel('Country')
    plt.xticks(rotation=45)
    plt.savefig(f'visualizations/top_imports_{year}.png')
    plt.close()

def run_eda():
    """Run all exploratory data analysis"""
    df = load_processed_data()
    plot_tariff_trends(df)
    plot_top_countries(df)
    
    # Generate summary statistics
    summary = df.describe()
    summary.to_csv('data/summary_statistics.csv')

if __name__ == "__main__":
    run_eda()