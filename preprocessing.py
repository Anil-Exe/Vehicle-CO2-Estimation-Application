"""Align a prediction row with the feature columns saved during training."""

import pandas as pd


def build_model_input(input_df, categorical_columns, feature_columns):
    # A single row has only one observed category. Dropping its first category
    # would erase every categorical selection. Training's omitted baseline
    # is instead handled by reindexing to its saved feature columns.
    encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=False)
    return encoded.reindex(columns=list(feature_columns), fill_value=0).fillna(0)
