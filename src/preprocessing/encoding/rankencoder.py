import pandas as pd
import numpy as np


class RankEncoder:
    def __init__(self, ranking: dict):
        self.ranking = ranking

    def transform(self, df: pd.DataFrame):
        processed_df = df.copy()

        for column, values in self.ranking.items():
            n_ranks = len(values)
            ranks = {k: v for v, k in enumerate(values)}
            step = 1 / (n_ranks - 1)

            processed_df[column] = processed_df[column].map(
                lambda x: ranks[x] * step, na_action='ignore')

        return processed_df

    def inverse_transform(self, df: pd.DataFrame):
        reversed_df = df.copy()

        for column, values in self.ranking.items():
            n_ranks = len(values)
            step = 1 / (n_ranks - 1)
            rank_pos = np.array([i * step for i in range(n_ranks)])

            reversed_df[column] = reversed_df[column].map(
                lambda x: values[np.argmin(np.abs(rank_pos - x))], na_action='ignore'
            )
        return reversed_df


"""
Test the encoder
"""
if __name__ == '__main__':
    _data = {
        'letters': ['A', 'B', 'C', 'D', 'B', np.nan, 'C', 'C', 'D'],
        'letters_orig': ['A', 'B', 'C', 'D', 'B', np.nan, 'C', 'C', 'D'],
        'numbers': ['2', '6', '3', '6', '3', '2', '1', '5', '3'],
        'numbers_orig': ['2', '6', '3', '6', '3', '2', '1', '5', '3'],
    }

    _ranking = {
        'letters': ['A', 'B', 'C', 'D'],
        'numbers': ['9', '8', '7', '6', '5', '4', '3', '2', '1']
    }

    _df = pd.DataFrame(_data)

    encoder = RankEncoder(_ranking)
    _transform_result = encoder.transform(_df)
    _inverse_transform_result = encoder.inverse_transform(_transform_result)

    assert _df.equals(_inverse_transform_result), "Inversion does not match original dataframe."

    _random_data = {k: np.random.choice(v, 20) for k, v in _ranking.items()}
    _random_df = pd.DataFrame(_random_data)

    _random_transform_result = encoder.transform(_random_df)
    _random_inv_transform_result = encoder.inverse_transform(_random_transform_result)

    print(f"_random_df:\n{_random_df}")
    print(f"_random_inv_transform_result:\n{_random_inv_transform_result}")

    pass
