import dvc.api
from dvclive import Live
from helper import load_data
from mlem.api import load
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def evaluate() -> None:
    params = dvc.api.params_show()
    model: DecisionTreeClassifier = load(params["model_path"])
    X_test = load_data(f"{params['data']['intermediate']}/X_test.pkl")
    y_test = load_data(f"{params['data']['intermediate']}/y_test.pkl")
    with Live(save_dvc_exp=True, resume=True) as live:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{accuracy=}")
        live.log_metric(name="accuracy score", val=accuracy)


if __name__ == "__main__":
    evaluate()
