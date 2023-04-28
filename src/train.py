from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.pipeline import Pipeline
from data import load_data
from pathlib import Path
from mlem.api import save
import dvc.api

# DVC
from dvclive import Live


def create_pipeline():
    return Pipeline(
        [
            ("Drop correlated features", DropCorrelatedFeatures(threshold=0.7)),
            ("StandardScaler", StandardScaler()),
            ("DecisionTree", DecisionTreeClassifier()),
        ]
    )


def save_model(model, path: str, X_train: pd.DataFrame) -> None:
    Path(path).parent.mkdir(exist_ok=True)
    save(model, path, sample_data=X_train)


def train() -> None:
    pipeline = create_pipeline()

    data = load_data()
    parameters = dvc.api.params_show()

    with Live(save_dvc_exp=True) as _:
        pipeline.fit(data.get('X_train'), data.get('y_train'))
        save_model(pipeline, parameters["model_path"], data.get('X_train'))


if __name__ == "__main__":
    train()
