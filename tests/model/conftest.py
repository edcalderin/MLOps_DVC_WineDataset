import dvc.api
from mlem.api import load
import pandas as pd
from pathlib import Path
from typing import Dict
from src.helper import load_data

from deepchecks.tabular import Dataset
import pytest

TARGET = 'target'

@pytest.fixture(scope='session')
def params()->Dict:
    return dvc.api.params_show()

@pytest.fixture(scope='session')
def model(params):
    path = Path(params['model_path'])
    return load(path)

@pytest.fixture(scope='session')
def train_data(params)->Dataset:
    X_train = load_data(f"{params['data']['intermediate']}/X_train.pkl")
    y_train = load_data(f"{params['data']['intermediate']}/y_train.pkl")
    df = pd.concat([X_train, y_train], axis=1)
    return Dataset(df, label=params['process']['target'], cat_features=[])

@pytest.fixture(scope='session')
def test_data(params)->Dataset:
    X_test = load_data(f"{params['data']['intermediate']}/X_test.pkl")
    y_test = load_data(f"{params['data']['intermediate']}/y_test.pkl")
    df = pd.concat([X_test, y_test], axis=1)
    return Dataset(df, label=params['process']['target'], cat_features=[])
