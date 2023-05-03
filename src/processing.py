from helper import load_data, save_data
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
import dvc.api


def getX_Y(data: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    return data.drop(target_name, axis=1), data[target_name]


def split_train_test(X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed
    )

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def process_data():
    df = load_data()
    X, y = getX_Y(df, 'target')

    params = dvc.api.params_show()

    splitted_dataset = split_train_test(X, y, params["process"]["random_state"])

    for name, data in splitted_dataset.items():
        save_data(data, params["data"]["intermediate"], name)

if __name__ == "__main__":
    process_data()
