from typing import List, Optional
import matplotlib.pyplot as plt
import dtreeviz
import numpy as np
import pandas as pd
import sklearn_extra
from sklearn.tree import DecisionTreeClassifier
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from src.tools.extract import extract_seq_features
from src.session.utils.save_plots import save_viz, save_plot
from src.visualization.colormap import get_dtreeviz_colors
from src.visualization.plot_clusters import visualize_clusters
from sklearn.metrics import silhouette_score


def learn_clusters(windows: List[pd.DataFrame],
                   n_clusters: int,
                   input_cols: Optional[List[str]] = None,
                   save_plots: bool = True,

                   n_probes: int = 5
                   ):
    features = extract_seq_features(windows, input_cols=input_cols)

    best_score = -1
    best_kmed = None

    p_bar = tqdm(range(n_probes), desc="Testing different clustering options")
    for _ in p_bar:
        kmed = KMedoids(n_clusters=n_clusters, init='k-medoids++')
        score = silhouette_score(features, kmed.fit_predict(features))

        if score > best_score:
            best_score = score
            best_kmed = kmed

            p_bar.set_postfix({"Best silhouette score": best_score})

    if save_plots:
        visualize_clusters(features, best_kmed.labels_, best_kmed.cluster_centers_)
        save_plot("clusters.png")
        plt.show()

    return best_kmed


def visualize_clustering_rules(windows: List[pd.DataFrame], labels: List,
                               tree_depths: Optional[List] = None, input_cols: Optional[List[str]] = None):
    # if tree_depths is None:
    #     tree_depths = [3, 4, 5]

    features = extract_seq_features(windows, input_cols=input_cols)
    num_classes = len(np.unique(labels))
    colors = get_dtreeviz_colors(num_classes)

    # for tree_depth in tqdm(tree_depths, desc="Creating trees"):
    clf = DecisionTreeClassifier(min_samples_split=int(len(features) * (1/num_classes) * 0.8)).fit(features, labels)

    viz_model = dtreeviz.model(clf,
                               X_train=features,
                               feature_names=features.columns,

                               y_train=labels,

                               target_name="Klasy")

    viz = viz_model.view(colors={
        "classes": colors
    })
    # save_viz(f"pattern_tree_d{tree_depth}.svg", viz)

    save_viz(f"pattern_tree.svg", viz)

    return features


def label_sequences(seqs: List[pd.DataFrame], stratify_cols: Optional[List[str]]) -> sklearn_extra.cluster.KMedoids:
    seq_features = extract_seq_features(seqs, input_cols=stratify_cols)
    kmed = KMedoids(n_clusters=min(10, len(seqs)), init='k-medoids++')
    kmed.fit(seq_features)

    return kmed
