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

# Add to existing KMeans code
def add_cluster_visualizations(kmeans, X, countries):
    """Add advanced cluster visualizations"""
    # 1. Parallel coordinates plot
    import plotly.express as px
    
    cluster_df = pd.DataFrame(X, columns=['import_tariffs', 'export_tariffs'])
    cluster_df['country'] = countries
    cluster_df['cluster'] = kmeans.labels_
    
    fig = px.parallel_coordinates(
        cluster_df, 
        color='cluster',
        dimensions=['import_tariffs', 'export_tariffs'],
        labels={'import_tariffs':'Import Tariffs', 'export_tariffs':'Export Tariffs'},
        title='Cluster Profiles - Parallel Coordinates'
    )
    fig.write_image('visualizations/parallel_coordinates.png')
    
    # 2. 3D plot if we had more dimensions
    # 3. Cluster bar charts showing centroids
    plt.figure(figsize=(10,6))
    pd.DataFrame(kmeans.cluster_centers_, 
                columns=['Import', 'Export']).plot(kind='bar')
    plt.title('Cluster Centroids')
    plt.ylabel('Scaled Tariff Values')
    plt.savefig('visualizations/cluster_centroids.png')
    plt.close()

if __name__ == "__main__":
    X, _, _ = prepare_clustering_data()
    find_optimal_clusters(X)
    model = train_kmeans(n_clusters=3)
    add_cluster_visualizations(model, X, _)
    print("Clustering model trained and visualizations saved.")