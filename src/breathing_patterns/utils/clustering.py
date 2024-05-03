from typing import List, Optional

import dtreeviz
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn_extra.cluster import KMedoids

from src.error_analysis.extract import extract_seq_features
from src.session.utils.save_plots import save_viz, save_plot
from src.visualization.plot_clusters import visualize_clusters


def learn_clusters(windows: List[pd.DataFrame], n_clusters: int, input_cols: Optional[List[str]] = None):
    features = extract_seq_features(windows, input_cols=input_cols)
    kmed = KMedoids(n_clusters=n_clusters, init='k-medoids++')
    kmed.fit(features)

    visualize_clusters(features, kmed.labels_, kmed.cluster_centers_)
    save_plot("clusters.png")
    plt.show()

    return kmed


def visualize_clustering_rules(windows: List[pd.DataFrame], labels: List,
                               max_tree_depth: int = 4, input_cols: Optional[List[str]] = None):
    features = extract_seq_features(windows, input_cols=input_cols)
    clf = DecisionTreeClassifier(max_depth=max_tree_depth).fit(features, labels)

    viz_model = dtreeviz.model(clf,
                               X_train=features,
                               feature_names=features.columns,

                               y_train=labels,

                               target_name="Klasy")

    viz = viz_model.view()
    save_viz("pattern_tree.svg", viz)

    return features
