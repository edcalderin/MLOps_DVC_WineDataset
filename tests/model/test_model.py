from deepchecks.tabular.checks import (
    ModelInferenceTime,
    PredictionDrift,
    SimpleModelComparison,
    TrainTestPerformance,
)

class TestModel:
    def test_model_inference_time(self, test_data, model):
        check = ModelInferenceTime()
        print('Tamanio', len(test_data))
        result = check.run(test_data, model)
        assert result.passed_conditions()

    def test_prediction_drift(self, train_data, test_data, model):
        check = PredictionDrift()
        result = check.run(train_data, test_data, model)
        assert result.passed_conditions()

    def test_simple_model_comparision(self, train_data, test_data, model):
        check = SimpleModelComparison()
        result = check.run(train_data, test_data, model)
        assert result.passed_conditions()

    def test_train_test_performance(self, train_data, test_data, model):
        check = TrainTestPerformance(
            scorers=["neg_mean_absolute_error", "neg_mean_squared_error", "r2"]
        )
        result = check.run(train_data, test_data, model)
        assert result.passed_conditions()
