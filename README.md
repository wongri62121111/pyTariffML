# US Tariffs Analysis Machine Learning Project

## Project Overview
This machine learning project analyzes US import and export tariffs using data from the US Census Bureau and Customs and Border Protection. The system provides insights into tariff trends, predicts future tariff values, and clusters countries based on trade patterns.

**Developers**: Abdallah Salem and Richard Wong


![image](https://github.com/user-attachments/assets/c8947692-206f-4694-8a06-eeed123ad0a1)
![image](https://github.com/user-attachments/assets/9969989e-05e7-4bd1-8e51-1bd3fbcfa693)
![image](https://github.com/user-attachments/assets/f8663104-b6fc-4924-a962-01f82b252651)
![image](https://github.com/user-attachments/assets/93e5f935-3374-4134-b61b-d017c98998bb)


## Project Structure
```
pyTariffML/
├── data/                   # Processed and raw data files
├── models/                 # Trained model files
├── visualizations/         # Generated plots and charts
├── WebPage/                # Web interface for the project
│   ├── frontend/           # React-based user interface
│   │   ├── src/            # React source code
│   │   └── public/         # Static web assets
│   └── backend/            # Flask API server
│       └── app.py          # Flask application serving ML models
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
- **Interactive Web Interface**: User-friendly application for exploring tariff data and model predictions

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

3. For the web interface, install additional dependencies:
```bash
# Backend dependencies
cd WebPage/backend
pip install flask pandas numpy

# Frontend dependencies
cd ../frontend
npm install
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
The interactive web application allows users to:
- View tariff trends and predictions
- Interact with model outputs
- Generate custom visualizations
- Explore country clusters based on trade patterns
- Compare model performance metrics

To run the web application:

1. Start the backend server:
```bash
cd WebPage/backend
python app.py
```

2. In a separate terminal, start the frontend:
```bash
cd WebPage/frontend
npm start
```

3. Access the application at http://localhost:3000

## Web Interface Features

### Tariff Analysis
- Interactive visualization of historical tariff trends
- Country-specific tariff data exploration
- Customizable year range selection
- Tariff category distribution breakdown

### Predictions
- Future tariff value predictions using various ML models
- Model selection (Linear Regression, Random Forest, XGBoost, ARIMA)
- Prediction quality metrics (MAE, RMSE, R²)
- Visual comparison between historical and predicted values

### Country Clustering
- Visual representation of country clusters based on trade patterns
- Scatter plot visualization of import/export relationships
- Cluster distribution analysis
- Detailed country listings with cluster characteristics

### Model Metrics
- Performance comparison across all models
- Detailed metrics (MAE, RMSE, R², Explained Variance)
- Visual representation of model performance

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
- Deploy web application to cloud platform for broader access
