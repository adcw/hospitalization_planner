import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

import data.colnames_original as c
from src.error_analysis.core.tree_utils import top_rules
from src.error_analysis.extract import extract_seq_features
from src.session.utils.save_plots import save_txt, save_viz
import dtreeviz

FEAT_COLS = [
    c.FIO2,
    c.PO2,
    c.PTL,
    c.RTG_RDS,
    c.RTG_PDA,

    c.GENERAL_SURFACTANT,
    c.GENERAL_PDA_CLOSED,

    c.ADRENALINA,
    c.PENICELINA1,
    c.KARBAPENEM,
    c.AMINOGLIKOZYD,
    c.AMINA_PRESYJNA,
    c.STERYD,
    c.ANTYBIOTYK,
    c.RESPIRATION
]


def perform_error_analysis(sequences: List[pd.DataFrame], losses: List[float], accept_percentile: float = 75):
    assert len(sequences) == len(
        losses), f"Number of sequences should match number of losses, {len(sequences)=}, {len(losses)=}"

    threshold = np.percentile(losses, accept_percentile)
    low_loss_indices = [i for i, value in enumerate(losses) if value < threshold]

    ys = [0 for i in range(len(losses))]

    for i in low_loss_indices:
        ys[i] = 1

    features = extract_seq_features(sequences, input_cols=FEAT_COLS)

    clf = DecisionTreeClassifier(max_depth=4).fit(features, ys)

    viz_model = dtreeviz.model(clf,
                               X_train=features, y_train=pd.Series(ys),
                               feature_names=features.columns,
                               target_name="Skuteczność modelu", class_names=["niska", "wysoka"])

    viz = viz_model.view()
    # viz.show()

    save_viz("tree.svg", viz)

    # rules = top_rules(tree=clf, dataframe=features, ys=ys)
    # save_txt("top_rules.txt", rules)

    pass
