from src.train import create_pipeline

import numpy as np


class TestCreatePipeline:
    def test_create_pipeline(self):
        pipeline = create_pipeline()
        N = 5
        X_train = np.random.random((N, 3))
        y_train = np.random.choice([1, 2], N)

        X_test = np.random.random((N, 3))

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        assert len(y_pred) == len(X_test)
        assert all(y_pred >= 1) and all(y_pred <= 2)
