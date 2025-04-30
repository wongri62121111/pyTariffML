// Frontend implementation for pyTariffML project
// app.js - Main Application File

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ScatterChart, Scatter, ZAxis, Label,
  PieChart, Pie, Cell
} from 'recharts';
import Select from 'react-select';
import './App.css';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  // State management
  const [loading, setLoading] = useState(false);
  const [countries, setCountries] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [yearRange, setYearRange] = useState([2000, 2024]);
  const [tariffData, setTariffData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [clusterData, setClusterData] = useState(null);
  const [modelType, setModelType] = useState('linear');
  const [activeTab, setActiveTab] = useState('analysis');
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  // Fetch countries on component mount
  useEffect(() => {
    const fetchCountries = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/countries`);
        const countryOptions = response.data.map(country => ({
          value: country,
          label: country
        }));
        setCountries(countryOptions);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch countries. Please try again later.');
        setLoading(false);
      }
    };

    fetchCountries();
    fetchMetrics();
  }, []);

  // Fetch model metrics
  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/metrics`);
      setMetrics(response.data);
    } catch (err) {
      console.error('Failed to fetch metrics', err);
    }
  };

  // Fetch tariff data for selected country and year range
  const fetchTariffData = async () => {
    if (!selectedCountry) {
      setError('Please select a country first');
      return;
    }

    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/tariff-data`, {
        params: {
          country: selectedCountry.value,
          startYear: yearRange[0],
          endYear: yearRange[1]
        }
      });
      setTariffData(response.data);
      setError(null);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch tariff data. Please try again later.');
      setLoading(false);
    }
  };

  // Generate predictions based on selected model
  const generatePredictions = async () => {
    if (!selectedCountry) {
      setError('Please select a country first');
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        country: selectedCountry.value,
        model: modelType,
        years: 5 // Predict 5 years into the future
      });
      setPredictions(response.data);
      setError(null);
      setLoading(false);
    } catch (err) {
      setError('Failed to generate predictions. Please try again later.');
      setLoading(false);
    }
  };

  // Fetch clustering data
  const fetchClusterData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/clusters`);
      setClusterData(response.data);
      setError(null);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch cluster data. Please try again later.');
      setLoading(false);
    }
  };

  // Color array for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>US Tariffs Analysis System</h1>
        <div className="tabs">
          <button 
            className={activeTab === 'analysis' ? 'active' : ''} 
            onClick={() => setActiveTab('analysis')}
          >
            Tariff Analysis
          </button>
          <button 
            className={activeTab === 'prediction' ? 'active' : ''} 
            onClick={() => setActiveTab('prediction')}
          >
            Predictions
          </button>
          <button 
            className={activeTab === 'clustering' ? 'active' : ''} 
            onClick={() => {
              setActiveTab('clustering');
              fetchClusterData();
            }}
          >
            Country Clustering
          </button>
          <button 
            className={activeTab === 'metrics' ? 'active' : ''} 
            onClick={() => setActiveTab('metrics')}
          >
            Model Metrics
          </button>
        </div>
      </header>

      <main className="app-main">
        {error && <div className="error-message">{error}</div>}
        
        {/* Tariff Analysis View */}
        {activeTab === 'analysis' && (
          <div className="analysis-container">
            <div className="control-panel">
              <h2>Tariff Data Explorer</h2>
              <div className="form-group">
                <label>Select Country:</label>
                <Select
                  options={countries}
                  value={selectedCountry}
                  onChange={setSelectedCountry}
                  placeholder="Select a country"
                  isSearchable
                />
              </div>
              <div className="form-group">
                <label>Year Range: {yearRange[0]} - {yearRange[1]}</label>
                <div className="range-slider">
                  <input
                    type="range"
                    min="1997"
                    max="2024"
                    value={yearRange[0]}
                    onChange={(e) => setYearRange([parseInt(e.target.value), yearRange[1]])}
                  />
                  <input
                    type="range"
                    min="1997"
                    max="2024"
                    value={yearRange[1]}
                    onChange={(e) => setYearRange([yearRange[0], parseInt(e.target.value)])}
                  />
                </div>
              </div>
              <button className="action-button" onClick={fetchTariffData} disabled={loading}>
                {loading ? 'Loading...' : 'Fetch Tariff Data'}
              </button>
            </div>

            <div className="data-visualization">
              {tariffData && (
                <>
                  <h3>Tariff Trends for {selectedCountry.label}</h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={tariffData.trends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis>
                        <Label value="Tariff Value ($)" angle={-90} position="insideLeft" />
                      </YAxis>
                      <Tooltip formatter={(value) => [`$${value}`, 'Tariff Value']} />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="importTariff" 
                        name="Import Tariff" 
                        stroke="#8884d8" 
                        activeDot={{ r: 8 }} 
                      />
                      <Line 
                        type="monotone" 
                        dataKey="exportTariff" 
                        name="Export Tariff" 
                        stroke="#82ca9d" 
                      />
                    </LineChart>
                  </ResponsiveContainer>

                  <h3>Tariff Categories Distribution</h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={tariffData.categories}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="category" />
                      <YAxis>
                        <Label value="Tariff Amount ($)" angle={-90} position="insideLeft" />
                      </YAxis>
                      <Tooltip formatter={(value) => [`$${value}`, 'Amount']} />
                      <Legend />
                      <Bar dataKey="value" fill="#8884d8" name="Tariff Value" />
                    </BarChart>
                  </ResponsiveContainer>
                </>
              )}
            </div>
          </div>
        )}

        {/* Prediction View */}
        {activeTab === 'prediction' && (
          <div className="prediction-container">
            <div className="control-panel">
              <h2>Tariff Predictions</h2>
              <div className="form-group">
                <label>Select Country:</label>
                <Select
                  options={countries}
                  value={selectedCountry}
                  onChange={setSelectedCountry}
                  placeholder="Select a country"
                  isSearchable
                />
              </div>
              <div className="form-group">
                <label>Select Model:</label>
                <select 
                  value={modelType} 
                  onChange={(e) => setModelType(e.target.value)}
                >
                  <option value="linear">Linear Regression</option>
                  <option value="random_forest">Random Forest</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="arima">ARIMA</option>
                </select>
              </div>
              <button className="action-button" onClick={generatePredictions} disabled={loading}>
                {loading ? 'Processing...' : 'Generate Predictions'}
              </button>
            </div>

            <div className="data-visualization">
              {predictions && (
                <>
                  <h3>Tariff Predictions for {selectedCountry.label} using {modelType.replace('_', ' ')} Model</h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={predictions.data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis>
                        <Label value="Tariff Value ($)" angle={-90} position="insideLeft" />
                      </YAxis>
                      <Tooltip formatter={(value) => [`$${value}`, 'Tariff Value']} />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="actual" 
                        name="Historical Data" 
                        stroke="#8884d8" 
                        strokeDasharray="5 5" 
                      />
                      <Line 
                        type="monotone" 
                        dataKey="predicted" 
                        name="Predicted Data" 
                        stroke="#82ca9d" 
                      />
                    </LineChart>
                  </ResponsiveContainer>

                  <div className="prediction-metrics">
                    <h3>Prediction Quality Metrics</h3>
                    <div className="metrics-grid">
                      <div className="metric-card">
                        <h4>MAE</h4>
                        <p>${predictions.metrics.mae.toFixed(2)}</p>
                      </div>
                      <div className="metric-card">
                        <h4>RMSE</h4>
                        <p>${predictions.metrics.rmse.toFixed(2)}</p>
                      </div>
                      <div className="metric-card">
                        <h4>R²</h4>
                        <p>{predictions.metrics.r2.toFixed(3)}</p>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Clustering View */}
        {activeTab === 'clustering' && (
          <div className="clustering-container">
            <h2>Country Clustering by Trade Patterns</h2>
            
            {loading && <p>Loading cluster data...</p>}
            
            {clusterData && (
              <>
                <div className="cluster-visualization">
                  <h3>Country Clusters</h3>
                  <ResponsiveContainer width="100%" height={500}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid />
                      <XAxis 
                        type="number" 
                        dataKey="importValue" 
                        name="Import Value" 
                        unit="$M"
                      >
                        <Label value="Import Value ($M)" offset={0} position="bottom" />
                      </XAxis>
                      <YAxis 
                        type="number" 
                        dataKey="exportValue" 
                        name="Export Value" 
                        unit="$M"
                      >
                        <Label value="Export Value ($M)" angle={-90} position="insideLeft" />
                      </YAxis>
                      <ZAxis 
                        type="number" 
                        dataKey="tariffRatio" 
                        range={[50, 400]} 
                        name="Tariff Ratio" 
                      />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }} 
                        formatter={(value, name, props) => {
                          if (name === 'Import Value' || name === 'Export Value') {
                            return [`$${value}M`, name];
                          }
                          return [value, name];
                        }}
                      />
                      <Legend />
                      {clusterData.clusters.map((entry, index) => (
                        <Scatter 
                          key={`cluster-${index}`} 
                          name={`Cluster ${index + 1}`} 
                          data={entry.countries} 
                          fill={COLORS[index % COLORS.length]} 
                        />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                <div className="cluster-details">
                  <h3>Cluster Distribution</h3>
                  <div className="pie-chart-container">
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={clusterData.clusters.map((cluster, index) => ({
                            name: `Cluster ${index + 1}`,
                            value: cluster.countries.length
                          }))}
                          cx="50%"
                          cy="50%"
                          labelLine={true}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                          label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        >
                          {clusterData.clusters.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => [`${value} countries`, 'Count']} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="cluster-tables">
                    {clusterData.clusters.map((cluster, index) => (
                      <div key={`cluster-table-${index}`} className="cluster-table">
                        <h4>Cluster {index + 1} Countries</h4>
                        <p><strong>Characteristics:</strong> {cluster.description}</p>
                        <div className="country-list">
                          {cluster.countries.map(country => (
                            <div key={country.name} className="country-item">
                              {country.name}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Model Metrics View */}
        {activeTab === 'metrics' && (
          <div className="metrics-container">
            <h2>Model Performance Metrics</h2>
            
            {!metrics ? (
              <p>Loading metrics data...</p>
            ) : (
              <div className="metrics-comparison">
                <h3>Model Comparison</h3>
                <div className="metrics-cards">
                  {metrics.models.map((model, index) => (
                    <div key={`model-card-${index}`} className="model-card">
                      <h4>{model.name}</h4>
                      <div className="model-metrics">
                        <div className="metric">
                          <span className="metric-label">MAE:</span>
                          <span className="metric-value">${model.mae.toFixed(2)}</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">RMSE:</span>
                          <span className="metric-value">${model.rmse.toFixed(2)}</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">R²:</span>
                          <span className="metric-value">{model.r2.toFixed(3)}</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">Explained Variance:</span>
                          <span className="metric-value">{(model.explained_variance * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="metrics-charts">
                  <h3>Performance Comparison</h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart 
                      data={metrics.models.map(model => ({
                        name: model.name,
                        MAE: model.mae,
                        RMSE: model.rmse,
                        R2: model.r2 * 100 // Scale R² to be visible alongside MAE/RMSE
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip formatter={(value, name) => {
                        if (name === 'R2') {
                          return [(value / 100).toFixed(3), 'R²'];
                        }
                        return [`$${value.toFixed(2)}`, name];
                      }} />
                      <Legend />
                      <Bar dataKey="MAE" fill="#8884d8" name="MAE" />
                      <Bar dataKey="RMSE" fill="#82ca9d" name="RMSE" />
                      <Bar dataKey="R2" fill="#ffc658" name="R²" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>US Tariffs Analysis Machine Learning Project</p>
        <p>Developed by Abdallah Salem and Richard Wong</p>
      </footer>
    </div>
  );
}

export default App;