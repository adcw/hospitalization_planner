from typing import Dict, List, Any, Optional

import src.colnames_original as c

import pandas as pd
from sklearn.impute import KNNImputer

from src.preprocessing.encoding.rankencoder import RankEncoder

"""
for impute_target, impute_values in impute_dict.items():
if impute_target not in impute_values:
    impute_values.append(impute_target)

    sub_df = processed_df.copy()
    sub_df_slice = sub_df[impute_values]
    target_loc = sub_df_slice.columns.get_loc(impute_target)
    
    # imputer = KNNImputer(n_neighbors=10)
    imputer = SimpleImputer()
    imputed_values = imputer.fit_transform(sub_df_slice)
    sub_df[impute_target] = imputed_values[:, target_loc]
    
    processed_df = pd.DataFrame(sub_df, columns=processed_df.columns)
"""


def _choose_indices(n, start, end):
    indices = [start]

    interval = (end - start) / (n - 1)

    for i in range(1, n - 1):
        index = round(start + i * interval)
        indices.append(index)

    indices.append(end - 1)
    return indices


class SequenceImputer:
    def __init__(self,
                 impute_dict: Dict[str, List[str]],
                 n_points: int = 2,
                 imputer_class: Any = KNNImputer,
                 imputer_kwargs: Optional[Dict[str, Any]] = None
                 ):
        assert n_points >= 2, "Number of imputation points should be higher or equal to 2"

        self.n_points: int = n_points
        self.impute_dict = impute_dict

        self.imputer_class = imputer_class
        self.imputer_kwargs = imputer_kwargs if imputer_class is not None else {'n_neighbors': 3}
        self.imputers_dict = {}

    def transform(self, sequences: List[pd.DataFrame]):

        for target_col, input_cols in self.impute_dict.items():
            target_col_index = sequences[0].columns.get_loc(target_col)
            if target_col not in input_cols:
                input_cols.append(target_col)

            # Get whole and missing sequences
            dfs_with_nans = []
            dfs_whole = []
            for s in sequences:
                na_sum = s[target_col].isna().sum()
                if na_sum == 0:
                    dfs_whole.append(s)
                else:
                    dfs_with_nans.append(s)

            # Concatenate together and get target cols
            input_vals = pd.concat(dfs_whole, copy=True)[input_cols]
            input_vals_len = input_vals.shape[0]

            # Get points to impute
            impute_points = []
            indexes_of_dfs = []
            for df in dfs_with_nans:
                df_len = df.shape[0]
                indexes = _choose_indices(self.n_points, 0, df_len)
                indexes_of_dfs.append(indexes)
                impute_points.append(df[input_cols].iloc[indexes].copy().reset_index(drop=True))
            impute_points_df = pd.concat(impute_points)

            # Perfrom imputation
            imputation_input = pd.concat([input_vals, impute_points_df], ignore_index=True)
            imputer = self.imputer_class(
                **self.imputer_kwargs) if self.imputer_kwargs is not None else self.imputer_class()
            imputation_result = imputer.fit_transform(imputation_input)[input_vals_len:]

            # Fill imputed points and interpolate them
            pointer = 0
            for seq_id, indexes in enumerate(indexes_of_dfs):
                for i in indexes:
                    dfs_with_nans[seq_id].iloc[i, target_col_index] = imputation_result[pointer][-1]
                    pointer += 1
                dfs_with_nans[seq_id][target_col].interpolate(inplace=True)
                pass

            sequences = dfs_whole + dfs_with_nans

            pass
        # Get dfs to impute
        # df_with_missing
        return sequences


# def plot_columns(colname: str):
#     tjs = []
#     for _, g in grouper:
#         tjs.append(g[colname])
#
#     plot_trajectories(tjs)

# pat_count = pd.DataFrame()
# cols = [c.CREATININE, c.PTL, c.TOTAL_BILIRUBIN]
#
# for col in cols:
#     n = 0
#     for _, g in grouper:
#         r = g.isna()[col].sum()
#         if r != 0:
#             n += 1
#     pat_count.at[col, 0] = n
# pat_count.astype(int)
if __name__ == '__main__':
    df = pd.read_csv("../../../data/input.csv")
    # df.dropna(inplace=True)

    encoder = RankEncoder({c.RESPIRATION: ['WLASNY', 'CPAP', 'MAP1', 'MAP2', 'MAP3']})
    df = encoder.transform(df)

    grouper = df.groupby(c.PATIENTID)

    sequences = [s for _, s in grouper]

    impute_dict = {c.CREATININE: [c.RESPIRATION, c.FIO2]}
    imputer = SequenceImputer(impute_dict=impute_dict, n_points=3)

    sequences = imputer.transform(sequences)

    # plot_columns(c.TOTAL_BILIRUBIN)
    # example_data = {
    #     "A": [1, 2, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4],
    #     "B": [0, 1, 0, 1, 0, 1, 0, 1, None, None, None, None],
    #     "C": [2, 3, 3, 3, 2, 2, 2, 3, 2, 3, 3, 3],
    #     "D": [5, 4, 3, 3, 5, 3, 3, 2, 5, 4, 3, 3]
    # }


    # impute_dict = {"B": ["C", "D"]}
    #
    # df = pd.DataFrame(example_data)
    #
    # imputer = SequenceImputer(impute_dict, imputer_class=KNNImputer, imputer_kwargs={'n_neighbors': 1})
    # # imputer = SequenceImputer(impute_dict, imputer_class=SimpleImputer)
    # imp = imputer.fit_transform(df)

    pass
