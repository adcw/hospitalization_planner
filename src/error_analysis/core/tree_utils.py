from sklearn.tree import _tree
import numpy as np


def get_decision_paths(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, current_path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Idź do lewego dziecka
            left_child = tree_.children_left[node]
            current_path_left = current_path + [f"{name} <= {threshold}"]
            recurse(left_child, current_path_left)

            # Idź do prawego dziecka
            right_child = tree_.children_right[node]
            current_path_right = current_path + [f"{name} > {threshold}"]
            recurse(right_child, current_path_right)
        else:
            # Osiągnięto liść - zakończ regułę
            current_class = np.argmax(tree_.value[node][0])
            current_path.append(f"class: {current_class}")
            rules.append(" and ".join(current_path))

    rules = []
    recurse(0, [])
    return rules

def calculate_support(dataframe, rule):
    # Oblicz liczbę przypadków, które spełniają regułę
    rule = rule.split(" and class")[0]
    mask = dataframe.eval(rule)
    num_cases = mask.sum()

    # Oblicz wsparcie jako procent przypadków spełniających regułę
    support = num_cases / len(dataframe) * 100
    return support


def print_top_rules(tree, feature_names, dataframe, n=5):
    # Pobierz reguły decyzyjne
    decision_paths = get_decision_paths(tree, feature_names)

    # Oblicz wsparcie dla każdej reguły
    supports = {rule: calculate_support(dataframe, rule) for rule in decision_paths}

    # Posortuj reguły według wsparcia
    sorted_rules = sorted(supports.items(), key=lambda x: x[1], reverse=True)

    # Wyświetl topowe reguły
    top_rules = sorted_rules[:n] if len(sorted_rules) >= n else sorted_rules
    for i, (rule, support) in enumerate(top_rules, start=1):
        print(f"Top {i} Rule: {rule}, Support: {support:.2f}%")
