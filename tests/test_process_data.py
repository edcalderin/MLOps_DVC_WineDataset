import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

from src.processing import getX_Y, split_train_test, process_data

from pandas.testing import assert_frame_equal, assert_series_equal
import pytest


@pytest.fixture(scope="session")
def dataframe_dictionaries()->Tuple[Dict, np.ndarray]:
    SIZE = 10
    features_df = {
        "feature1": np.random.random(SIZE),
        "feature2": np.random.choice(["a", "b", "c"], SIZE),
    }
    labels = np.random.choice([0, 1], SIZE)
    return features_df, labels


def test_getX_Y(dataframe_dictionaries):
    features_df = dataframe_dictionaries[0]
    labels = dataframe_dictionaries[1]
    df = pd.DataFrame({**features_df, "label": labels})

    X, y = getX_Y(df, "label")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert_frame_equal(pd.DataFrame(features_df), X)
    assert_series_equal(pd.Series(labels), y, check_names=False)


def test_split_train_test(dataframe_dictionaries):
    features_df = dataframe_dictionaries[0]
    labels = dataframe_dictionaries[1]

    splitted_dataset = split_train_test(
        pd.DataFrame(features_df), pd.Series(labels), 42
    )
    assert all(
        [key in ["X_train", "y_train", "X_test", "y_test"] for key in splitted_dataset]
    )
    assert len(splitted_dataset["X_train"]) == len(splitted_dataset["y_train"])
    assert len(splitted_dataset["X_test"]) == len(splitted_dataset["y_test"])
    assert len(splitted_dataset["X_train"]) > len(splitted_dataset["X_test"])
