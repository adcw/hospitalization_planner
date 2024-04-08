from sklearn.tree import _tree
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


def calculate_support(dataframe, rule):
    """
    Calculate the support for a decision rule in a dataframe.

    Parameters:
        dataframe : pd.DataFrame
            Input dataframe.
        rule : str
            Decision rule string.

    Returns:
        float: Support as a percentage.
    """
    try:
        rule = rule.split(" and class")[0]
        mask = dataframe.eval(rule)
        num_cases = mask.sum()
        support = num_cases / len(dataframe) * 100
        return support
    except Exception as e:
        print(f"Warning: Rule '{rule}' could not be evaluated. Skipping...")
        return None


def print_top_rules(tree, dataframe, n=5):
    """
    Print the top decision rules with their support.

    Parameters:
        tree : DecisionTreeClassifier
            Trained decision tree.
        dataframe : pd.DataFrame
            Input dataframe.
        n : int, optional
            Number of top rules to print, by default 5.
    """
    decision_paths = get_decision_paths(tree, dataframe.columns)

    supports = {rule: calculate_support(dataframe, rule) for rule in decision_paths}

    # Remove empty rules
    supports = {rule: support for rule, support in supports.items() if support is not None}

    sorted_rules = sorted(supports.items(), key=lambda x: x[1], reverse=True)
    top_rules = sorted_rules[:n] if len(sorted_rules) >= n else sorted_rules

    for i, (rule, support) in enumerate(top_rules, start=1):
        print(f"Top {i} Rule: {rule}, Support: {support:.2f}%")
