from typing import Optional

import pandas as pd


class OneHotEncoder:
    def __init__(self):
        """
        Create encoder class.\n
        **Note**: It can be used to encode and decode any dataframe.
        The name of columns and their categorical value
        should be the same in both fit_transform and inverse_transform calls.
        """
        # learnable fields
        self.columns: Optional[list[str]] = None
        self.encoded_columns: Optional[list[str]] = []
        self.mapping: dict = {}
        self.original_column_order: Optional[list[str]] = None

    def fit_transform(self, dataframe: pd.DataFrame, columns: list[str]):
        """
        One hot encode chosen columns of dataframe. The rest remains unchanged.
        :param dataframe: Dataframe to be transformed
        :param columns: Columns to be one-hot encoded.
        :return: Result of transformation
        """
        self.columns = columns

        transformed_df = dataframe.copy()

        self.original_column_order = list(dataframe.columns)

        for column in self.columns:
            encoded = pd.get_dummies(dataframe[column], prefix=column, prefix_sep='__', dtype=float)
            self.encoded_columns.extend(encoded.columns)
            self.mapping[column] = encoded.columns
            transformed_df = pd.concat([transformed_df, encoded], axis=1)
            transformed_df = transformed_df.drop(column, axis=1)

        return transformed_df

    def inverse_transform(self, dataframe: pd.DataFrame):
        """
        Inverse transform on-hot encoded dataframe.\n
        The format will be the same as the input dataframe for transform operation.
        :param dataframe: One-hot encoded dataframe
        :return: Regular dataframe
        """
        original_df = dataframe.copy()
        inverse_mapping: dict = {}

        for column in self.encoded_columns:
            original_column = column.split('__')[0]

            if original_column not in inverse_mapping:
                inverse_mapping[original_column] = [col for col in dataframe.columns if
                                                    col.startswith(original_column + '__')]

                original_df[original_column] = original_df[inverse_mapping[original_column]].idxmax(axis=1).apply(
                    lambda x: x.split("__")[1])

                original_df = original_df.drop(inverse_mapping[original_column], axis=1)

        original_df = original_df[self.original_column_order]

        return original_df
