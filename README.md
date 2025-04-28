# US Tariffs Analysis Machine Learning Project

## Project Overview
This machine learning project analyzes US import and export tariffs using data from the US Census Bureau and Customs and Border Protection. The system provides insights into tariff trends, predicts future tariff values, and clusters countries based on trade patterns.

**Developers**: Abdallah Salem and Richard Wong

## Project Structure
```
pyTariffML/
├── data/                   # Processed and raw data files
├── models/                 # Trained model files
├── visualizations/         # Generated plots and charts
├── data_collection.py      # Data downloading and collection
├── data_preprocessing.py   # Data cleaning and preparation
├── exploratory_analysis.py # Initial data exploration
├── model_training.py       # Model training scripts
├── model_evaluation.py     # Model performance evaluation
├── model_visualization.py  # Visualization generation
├── time_series_model.py    # ARIMA time series modeling
├── clustering_model.py     # K-means clustering implementation
├── xgboost_model.py        # XGBoost regression model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Key Features
- **Data Collection**: Automated download of tariff data from government sources
- **Data Processing**: Cleaning and transformation of raw tariff data
- **Machine Learning Models**:
  - Linear Regression
  - Random Forest
  - XGBoost
  - ARIMA Time Series
  - K-Means Clustering
- **Evaluation Metrics**: MAE, RMSE, R², Explained Variance
- **Visualizations**: Trend analysis, cluster plots, residual plots

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/pyTariffML.git
cd pyTariffML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the pipeline in order:

1. Data collection and preprocessing:
```bash
python data_collection.py
python data_preprocessing.py
```

2. Exploratory analysis:
```bash
python exploratory_analysis.py
```

3. Model training:
```bash
python model_training.py
python time_series_model.py
python clustering_model.py
python xgboost_model.py
```

4. Model evaluation:
```bash
python model_evaluation.py
```

5. Generate visualizations:
```bash
python model_visualization.py
```

## Web Application
The deployed web application allows users to:
- View tariff trends and predictions
- Interact with model outputs
- Generate custom visualizations

**Access**: [Web Application URL]

## Project Report

### Objectives
- Analyze historical US tariff data
- Predict future tariff trends
- Identify patterns in international trade relationships
- Provide actionable insights for trade policy analysis

### Data Collection
- Sources: US Census Bureau, Customs and Border Protection
- Data Types: Monthly/Annual import/export tariffs by country
- Time Period: 1997-present
- Total Records: [X] countries over [Y] years

### Data Processing
1. Data cleaning:
   - Handling missing values
   - Standardizing country names
   - Converting date formats

2. Feature engineering:
   - Time-based features
   - Tariff ratios
   - Regional groupings

### Model Development
| Model | Type | Input | Output | Key Metrics |
|-------|------|-------|--------|-------------|
| Linear Regression | Regression | Year, Country | Tariff Value | MAE: $X, R²: Y |
| Random Forest | Regression | Year, Country | Tariff Value | MAE: $X, R²: Y |
| XGBoost | Regression | Year, Country | Tariff Value | MAE: $X, R²: Y |
| ARIMA | Time Series | Historical Tariffs | Future Tariffs | MAE: $X |
| K-Means | Clustering | Import/Export Patterns | Country Clusters | Silhouette: X |

### Results
- Best performing model: [Model Name] with [Metric] of [Value]
- Key findings: [Summary of insights]
- Limitations: [Data quality issues, model constraints]

## Contribution
Equal contribution by both team members:
- **Abdallah Salem**: Data processing, model development, evaluation
- **Richard Wong**: Web implementation, visualization, documentation

## Future Work
- Expand data sources
- Incorporate additional economic indicators
- Develop real-time prediction system
- Enhance web interface with more interactive features
