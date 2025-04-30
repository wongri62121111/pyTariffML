// apiService.js - Service for connecting to the Python ML backend

import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// API service object
const apiService = {
  // Fetch available countries
  getCountries: async () => {
    try {
      const response = await api.get('/countries');
      return response.data;
    } catch (error) {
      console.error('Error fetching countries:', error);
      throw error;
    }
  },

  // Fetch tariff data for specific country and year range
  getTariffData: async (country, startYear, endYear) => {
    try {
      const response = await api.get('/tariff-data', {
        params: {
          country,
          startYear,
          endYear
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching tariff data:', error);
      throw error;
    }
  },

  // Generate predictions based on selected model
  generatePredictions: async (country, model, years) => {
    try {
      const response = await api.post('/predict', {
        country,
        model,
        years
      });
      return response.data;
    } catch (error) {
      console.error('Error generating predictions:', error);
      throw error;
    }
  },

  // Fetch cluster data
  getClusterData: async () => {
    try {
      const response = await api.get('/clusters');
      return response.data;
    } catch (error) {
      console.error('Error fetching cluster data:', error);
      throw error;
    }
  },

  // Fetch model metrics
  getModelMetrics: async () => {
    try {
      const response = await api.get('/metrics');
      return response.data;
    } catch (error) {
      console.error('Error fetching model metrics:', error);
      throw error;
    }
  },

  // Fetch a specific model's details
  getModelDetails: async (modelName) => {
    try {
      const response = await api.get(`/model/${modelName}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching ${modelName} model details:`, error);
      throw error;
    }
  }
};

export default apiService;