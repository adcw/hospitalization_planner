import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn_extra.cluster import KMedoids


def remove_medoid_by_index(kmedoids: KMedoids, index_to_remove: int) -> KMedoids:
    """
    Removes a specified medoid from the KMedoids object based on its index in the medoid array
    and adjusts the logic of the object so that future predictions do not consider the removed medoid.

    Args:
        kmedoids (KMedoids): The KMedoids clustering object.
        index_to_remove (int): The index of the medoid to be removed in the medoid array.

    Returns:
        KMedoids: A new KMedoids object with the specified medoid removed.
    """
    if index_to_remove >= len(kmedoids.cluster_centers_):
        raise ValueError("The provided index is out of range for the current medoids array.")

    new_centers = np.delete(kmedoids.cluster_centers_, index_to_remove, axis=0)

    new_kmedoids = KMedoids(n_clusters=len(new_centers), metric=kmedoids.metric, init='k-medoids++',
                            max_iter=kmedoids.max_iter)
    new_kmedoids.fit(new_centers)

    return new_kmedoids


def plot_clusters(X, y, medoids=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.5)
    if medoids is not None:
        for m in medoids:
            plt.scatter(m[0], m[1], c='red', s=200, marker='X', label='Medoids')
    plt.title('Cluster Plot')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


N_CENTERS = 6
MAX_PLOTS = 10
if __name__ == '__main__':
    X, _ = make_blobs(n_samples=300, centers=N_CENTERS, cluster_std=0.60, random_state=0)

    # Inicjalizacja i dopasowanie modelu KMedoids
    kmedoids = KMedoids(n_clusters=N_CENTERS, random_state=0, max_iter=1000)
    kmedoids.fit(X)

    # Wyświetlenie pierwotnego podziału na klastry
    plot_clusters(X, kmedoids.labels_, medoids=kmedoids.cluster_centers_)

    for i in range(min(N_CENTERS, MAX_PLOTS)):
        new_kmedoids = remove_medoid_by_index(kmedoids, index_to_remove=i)
        new_labels = new_kmedoids.predict(X)
        plot_clusters(X, new_labels, medoids=new_kmedoids.cluster_centers_)
