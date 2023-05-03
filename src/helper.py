from sklearn.datasets import load_wine
import pandas as pd
from pathlib import Path


def load_data(path: str = None) -> pd.DataFrame:
    if path is None:
        data = load_wine()

        df = pd.DataFrame(data=data["data"], columns=data["feature_names"])

        df["target"] = data["target"]

    else:
        file_path = Path(path)
        if file_path.suffix == ".pkl":
            df = pd.read_pickle(file_path)
        else:
            raise ValueError("File format not supported. Please use a PKL file")

    return df


def save_data(df: pd.DataFrame, path: str, name: str) -> None:
    """
    Method to save data

    Parameters:

    df: Dataframe
    path: Path name
    name: Name of the data

    Return
    None
    """
    path = f"{path}/{name}.pkl"
    Path(path).parent.mkdir(exist_ok=True)
    df.to_pickle(path)