import os

import numpy as np
import pandas as pd
import pickle

from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler

from ..trajectory.Trajectory import Trajectory
from .PastPredictor.MeanPredictor import MeanPredictor
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

    def predict(self, trajectory: Trajectory, horizon=10):
        """
        trajectory: trajectory containing (delta_progress, deltas) 
        """
        predictor = MeanPredictor()
        series = trajectory.as_dt()
        print(trajectory.as_dt())
        past_curvatures = trajectory.curvatures_dt()
        future_curvatures = trajectory.get_future_curvatures(predictor, horizon)
        curvatures = np.concatenate((past_curvatures, future_curvatures))

        series = TimeSeries.from_values(series)
        future_covariates = TimeSeries.from_values(curvatures)
        series = self._trf.transform(series)
        prediction = self._rnn.predict(horizon,
                              series=series,
                              past_covariates=None,
                              future_covariates=future_covariates)
        prediction = self._trf.inverse_transform(prediction)
    
        return Trajectory.from_dt(prediction.values(), trajectory.get_optim(), trajectory.get_dt(), trajectory.get_final_progress())

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