import numpy as np

from ..Model import Model

class MeanPredictor(Model):
    def __init__(self, past_predictions_to_consider=1):
        super().__init__()

        self.__past_predictions_to_consider = past_predictions_to_consider

    def train(self):
        raise NotImplementedError

    def predict(self, series, horizon=10):
        """
        series: time series containing (delta_progress, deltas)
        curvatures: all curavetures from the trajectory (beggining at 0) + future
        """

        mean = series[-self.__past_predictions_to_consider].reshape(-1, 2).mean(axis=0)

        repeated_mean = np.repeat(mean, horizon, axis=0).reshape(2, -1).T

        return repeated_mean

    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError
