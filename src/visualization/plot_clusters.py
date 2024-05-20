from typing import Optional

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from src.configuration import  COLORMAP

def visualize_clusters(data, labels, centroids=None, title: Optional[str] = 'Cluster visualisation'):
    """
    Visualize clusters in a 2D plot.

    :param data: A numpy array representing the original data points.
    :param labels: A list or numpy array containing cluster labels assigned to each data point.
                   For DBSCAN, -1 indicates noise points.
    :param centroids: A numpy array representing the centroids of clusters (optional, default=None).
                      Only required for k-means clustering.
    :return: None
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))

    # Plot data points with cluster labels
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap=COLORMAP, alpha=0.5,
                marker='o', label='Data Points')

    # Plot centroids for k-means clustering (if provided)
    if centroids is not None:
        reduced_centroids = pca.transform(centroids)

        # TODO: https://www.geeksforgeeks.org/matplotlib-pyplot-colorbar-function-in-python/
        plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c='red', marker='x',
                    s=100, label='Centroids')

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.colorbar(label='Cluster')
    plt.legend()
