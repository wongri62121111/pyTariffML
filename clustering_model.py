import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_clustering_data():
    """Prepare data for clustering"""
    df = pd.read_csv('data/processed_tariff_data.csv')
    
    # Get mean tariffs by country
    cluster_data = df.groupby('country_name').agg({
        'import_tariffs': 'mean',
        'export_tariffs': 'mean'
    }).reset_index()
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_data[['import_tariffs', 'export_tariffs']])
    
    return X, cluster_data['country_name'], scaler

def find_optimal_clusters(X):
    """Find optimal number of clusters using elbow method"""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10,6))
    plt.plot(range(1,11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('visualizations/cluster_elbow.png')
    plt.close()

def train_kmeans(n_clusters=3):
    """Train and save KMeans clustering model"""
    X, countries, scaler = prepare_clustering_data()
    
    # Train model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    # Save model and scaler
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/cluster_scaler.pkl')
    
    # Plot clusters
    plt.figure(figsize=(12,8))
    plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='viridis')
    
    # Annotate some points
    for i, country in enumerate(countries):
        if i % 5 == 0:  # Label every 5th country to avoid clutter
            plt.annotate(country, (X[i,0], X[i,1]))
    
    plt.title('Country Clusters by Tariff Patterns')
    plt.xlabel('Scaled Import Tariffs')
    plt.ylabel('Scaled Export Tariffs')
    plt.savefig('visualizations/country_clusters.png')
    plt.close()
    
    return kmeans

if __name__ == "__main__":
    X, _, _ = prepare_clustering_data()
    find_optimal_clusters(X)
    model = train_kmeans(n_clusters=3)