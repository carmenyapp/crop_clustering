import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Preprocess the data
def preprocess_data(df, threshold=3):
    # Handle categorical variables
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column])

    # Scale numeric variables
    scaler = StandardScaler()
    numeric_df = df.select_dtypes(include=[float, int])
    scaled_data = scaler.fit_transform(numeric_df)

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    scaled_df_pca = pca.fit_transform(scaled_data)

    # Drop outliers
    z_scores = zscore(scaled_df_pca)
    outliers = (z_scores.abs() > threshold)
    outlier_indices = outliers.any(axis=1)
    cleaned_df = scaled_df_pca[~outlier_indices]

    return cleaned_df

# Clustering evaluation
def evaluate_clustering(labels, data):
    silhouette_avg = silhouette_score(data, labels) if len(set(labels)) > 1 else None
    calinski_harabasz = calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None
    davies_bouldin = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None
    return silhouette_avg, calinski_harabasz, davies_bouldin

# Plot clusters
def plot_clusters(data, labels, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='Cluster Label')
    st.pyplot()

# Main function
def main():
    st.title("Clustering Model Selection and Parameter Tuning")

    # Model selection
    model_options = ["KMeans", "DBSCAN", "AgglomerativeClustering", "HDBSCAN", "SpectralClustering"]
    selected_model = st.selectbox("Choose a Clustering Model", model_options)

    # Parameter scaling
    if selected_model == "KMeans":
        n_clusters_slider = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    elif selected_model == "DBSCAN":
        eps_slider = st.slider("Epsilon (Eps)", min_value=0.1, max_value=2.0, value=1.0)
        min_samples_slider = st.slider("Min Samples", min_value=2, max_value=20, value=5)
    elif selected_model == "AgglomerativeClustering":
        n_clusters_slider = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    elif selected_model == "HDBSCAN":
        min_cluster_size_slider = st.slider("Min Cluster Size", min_value=2, max_value=20, value=5)
        min_samples_slider = st.slider("Min Samples", min_value=2, max_value=20, value=5)
    elif selected_model == "SpectralClustering":
        n_clusters_slider = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        affinity_options = ["rbf", "nearest_neighbors"]
        selected_affinity = st.selectbox("Affinity", affinity_options)

    # Clustering and evaluation
    if st.button("Cluster"):
        preprocessed_df = preprocess_data(df)

        if selected_model == "KMeans":
            model = KMeans(n_clusters=n_clusters_slider, random_state=42)
        elif selected_model == "DBSCAN":
            model = DBSCAN(eps=eps_slider, min_samples=min_samples_slider)
        elif selected_model == "AgglomerativeClustering":
            model = AgglomerativeClustering(n_clusters=n_clusters_slider)
        elif selected_model == "HDBSCAN":
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_slider, min_samples=min_samples_slider)
        elif selected_model == "SpectralClustering":
            model = SpectralClustering(n_clusters=n_clusters_slider, affinity=selected_affinity)

        labels = model.fit_predict(preprocessed_df)

        # Evaluate clustering
        metrics = evaluate_clustering(labels, preprocessed_df)

        st.write(f"**{selected_model} Clustering Results:**")
        st.write(f"Silhouette Score: {metrics[0]}")
        st.write(f"Calinski-Harabasz Score: {metrics[1]}")
        st.write(f"Davies-Bouldin Score: {metrics[2]}")

        # Visualize clusters
        plot_clusters(preprocessed_df, labels, f"{selected_model} Clustering Visualization")

if __name__ == "__main__":
    main()
