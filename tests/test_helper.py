import pandas as pd
from src.helper import load_data, save_data

from pandas.testing import assert_frame_equal
from pathlib import Path
import pytest


class TestLoadData:
    def test_correctly(self):
        result: pd.DataFrame = load_data()
        assert isinstance(result, pd.DataFrame)
        assert "target" in result.columns

    @pytest.mark.xfail(raises=ValueError)
    def test_raises(self):
        load_data("any-path.csv")


class TestSaveData:
    def test_save_data(self, fs):
        data: pd.DataFrame = pd.DataFrame({"a": [0, 1]})
        path = "new"
        name = "data"

        fs.create_dir(path)

        save_data(data, path, name)

        path = Path(f"{path}/{name}.pkl")

        assert_frame_equal(pd.read_pickle(path), data)
