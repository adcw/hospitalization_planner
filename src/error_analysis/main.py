import random

from sklearn.tree import DecisionTreeClassifier, export_text

import data.colnames_original as c
from src.error_analysis.core.tree_utils import print_top_rules
from src.error_analysis.extract import extract_seq_features
from src.session.model_manager import _get_sequences

if __name__ == '__main__':
    seqs, _ = _get_sequences(path="../../data/input.csv")

    feats = extract_seq_features(seqs, input_cols=[c.RESPIRATION])

    dummy_y = [round(random.random()) for row in feats.itertuples(index=False)]

    clf = DecisionTreeClassifier()

    clf = clf.fit(feats, dummy_y)

    print(export_text(clf))

    print_top_rules(tree=clf, dataframe=feats)
