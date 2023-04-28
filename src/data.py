from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return
    X_train, X_test, y_train, y_test
    """
    data = load_wine()

    df = pd.DataFrame(data=data["data"], columns=data["feature_names"])

    X, y = df, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
