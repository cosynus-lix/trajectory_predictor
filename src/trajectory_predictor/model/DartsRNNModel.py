import os

import numpy as np
import pandas as pd
import pickle

from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler

from .Model import Model

class DartsRNNModel(Model):
    def __init__(self, n_layers=2, input_chunk_length=200, training_length=500):
        super().__init__()

        self._rnn = RNNModel(input_chunk_length=input_chunk_length, 
                    training_length=training_length, 
                    n_rnn_layers=n_layers)

        self._trf = None

    def train(self, dataset, epochs=1):
        dataset_np = dataset.to_np()

        covariates_np = dataset_np[:, 2]
        series_np = dataset_np[:, :-1]

        covariates = TimeSeries.from_values(covariates_np)
        series = TimeSeries.from_values(series_np)

        self._trf = Scaler()
        series = self._trf.fit_transform(series)

        self._rnn.fit(series, 
                    future_covariates=covariates, 
                    epochs=epochs, 
                    verbose=True)

    def predict(self, series, curvatures, horizon=10):
        """
        series: time series containing (delta_progress, deltas)
        curvatures: all curavetures from the trajectory (beggining at 0) + future
        """
        series = TimeSeries.from_values(series)
        future_covariates = TimeSeries.from_values(curvatures)
        series = self._trf.transform(series)
        prediction = self._rnn.predict(horizon,
                              series=series,
                              past_covariates=None,
                              future_covariates=future_covariates)
        prediction = self._trf.inverse_transform(prediction)
        return prediction.values()

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = 'darts_rnn_model.pth.tar'
        self._rnn.save_model(f'{path}/{model_name}')

        with open(f'{path}/transformer.pickle', 'wb') as handle:
            pickle.dump(self._trf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError
        model_name = 'darts_rnn_model.pth.tar'
        self._rnn = RNNModel.load_model(f'{path}/{model_name}')

        with open(f'{path}/transformer.pickle', 'rb') as handle:
            self._trf = pickle.load(handle)