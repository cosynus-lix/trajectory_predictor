import os

import numpy as np
import pandas as pd
import pickle

from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler

from ..trajectory.Trajectory import Trajectory
from ..dataset.Dataset import Dataset
from ..dataset.SimpleDataset import SimpleDataset
from .PastPredictor.MeanPredictor import MeanPredictor
from .Model import Model

class DartsRNNModel(Model):
    def __init__(self, n_layers=2, input_chunk_length=200, training_length=500):
        super().__init__()

        self._rnn = RNNModel(input_chunk_length=input_chunk_length, 
                    training_length=training_length, 
                    n_rnn_layers=n_layers)

        self._trf = None
        self._dt = None

    def dataset_to_series_and_curvatures_timeseries_list(self, dataset: Dataset):
        trajectories = dataset.get_trajectories()
        series_timeseries_list = []
        curvatures_timeseries_list = []
        self._dt = trajectories[0].get_dt()
        for trajectory in trajectories:
            if trajectory.get_dt() != self._dt:
                raise ValueError('All trajectories must have the same dt')
            series_timeseries_list.append(TimeSeries.from_values(trajectory.as_dt()))
            curvatures_timeseries_list.append(TimeSeries.from_values(trajectory.curvatures_dt()))
        return series_timeseries_list, curvatures_timeseries_list

    def train(self, dataset: Dataset, epochs=1, TfClass=Scaler):
        # TODO: change defalut tf class to none
        # TODO: add max splits per time series
        # if TfClass 
        series, covariates = self.dataset_to_series_and_curvatures_timeseries_list(dataset)

        if TfClass is not None:
            self._trf = TfClass()
            series = self._trf.fit_transform(series)
        else:
            self._trf = None

        self._rnn.fit(series, 
                    future_covariates=covariates, 
                    epochs=epochs, 
                    verbose=True)

    def predict(self, trajectory: Trajectory, horizon=10):
        """
        trajectory: trajectory containing (delta_progress, deltas) 
        """
        trajectory_dt = trajectory.get_dt()
        assert trajectory_dt == self._dt, f'Trajectory dt ({trajectory_dt}) must match the model ({self._dt})'

        predictor = MeanPredictor()
        series = trajectory.as_dt()
        past_curvatures = trajectory.curvatures_dt()
        future_curvatures = trajectory.get_future_curvatures(predictor, horizon)
        curvatures = np.concatenate((past_curvatures, future_curvatures))

        series = TimeSeries.from_values(series)
        future_covariates = TimeSeries.from_values(curvatures)
        if self._trf is not None:
            series = self._trf.transform(series)
        prediction = self._rnn.predict(horizon,
                              series=series,
                              past_covariates=None,
                              future_covariates=future_covariates)
        if self._trf is not None:
            prediction = self._trf.inverse_transform(prediction)
    
        return Trajectory.from_dt(prediction.values(), trajectory.get_optim(), trajectory.get_dt(), trajectory.get_final_progress())

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        # TODO: update pytorch-lightning when https://github.com/unit8co/darts/issues/1116 is solved
        model_name = 'darts_rnn_model.pth.tar'
        self._rnn.save_model(f'{path}/{model_name}')

        with open(f'{path}/rnn_helpers.pickle', 'wb') as handle:
            pickle.dump((self._trf, self._dt), handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError
        model_name = 'darts_rnn_model.pth.tar'
        self._rnn = RNNModel.load_model(f'{path}/{model_name}')

        with open(f'{path}/rnn_helpers.pickle', 'rb') as handle:
            self._trf, self._dt = pickle.load(handle)
