import pandas as pd
from sklearn.tree import _tree
from sklearn.metrics import classification_report, f1_score
import numpy as np


def get_decision_paths(tree, feature_names):
    """
    Generate decision rules from a decision tree.

    Parameters:
        tree : DecisionTreeClassifier
            Trained decision tree.
        feature_names : list
            List of feature names.

    Returns:
        list: List of decision rule strings.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, current_path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Go to the left child
            left_child = tree_.children_left[node]
            current_path_left = current_path + [f"{name} <= {threshold}"]
            recurse(left_child, current_path_left)

            # Go to the right child
            right_child = tree_.children_right[node]
            current_path_right = current_path + [f"{name} > {threshold}"]
            recurse(right_child, current_path_right)
        else:
            # Reached a leaf - end the rule
            current_class = np.argmax(tree_.value[node][0])
            current_path.append(f"class: {current_class}")
            rules.append(" and ".join(current_path))

    rules = []
    recurse(0, [])
    return rules


def calculate_support(dataframe: pd.DataFrame, rule: str, real_ys):
    """
    Calculate the support and F1-score for a decision rule in a DataFrame.

    Parameters:
        dataframe : pd.DataFrame
            Input DataFrame.
        rule : str
            Decision rule string.
        real_ys : array-like
            True labels.

    Returns:
        tuple: Support as a percentage and F1-score.
    """
    try:
        rule, est_class = rule.split(" and class: ")
        est_class = int(est_class)

        mask = dataframe.eval(rule)
        covered_ys = [y for v, y in zip(mask.values, real_ys) if v]

        num_cases = mask.sum()
        support = num_cases / len(dataframe) * 100

        precision = covered_ys.count(est_class) / num_cases * 100

        return support, precision
    except Exception as e:
        print(f"Warning: Rule '{rule}' could not be evaluated. Skipping...")
        return None, None


def top_rules(tree, dataframe, ys, n=5):
    """
    Print the top decision rules with their support and F1-score.

    Parameters:
        tree : DecisionTreeClassifier
            Trained decision tree.
        dataframe : pd.DataFrame
            Input dataframe.
        ys : array-like
            True labels.
        n : int, optional
            Number of top rules to print, by default 5.
    """
    decision_paths = get_decision_paths(tree, dataframe.columns)

    supports_and_f1 = {rule: calculate_support(dataframe, rule, ys) for rule in decision_paths}

    # Remove empty rules
    supports_and_f1 = {rule: (s, p) for rule, (s, p) in supports_and_f1.items() if s is not None}

    sorted_rules = sorted(supports_and_f1.items(), key=lambda x: x[1][0], reverse=True)
    top_rules = sorted_rules[:n] if len(sorted_rules) >= n else sorted_rules

    text = ""

    for i, (rule, (support, f1)) in enumerate(top_rules, start=1):
        text += f"Top {i} Rule: {rule}, Support: {support:.2f}%, Precision: {f1:.2f}\n"

    text = text.replace(" and", "\nand").replace("\nTop", "\n\nTop").replace("and class", "\tand class")

    return text
