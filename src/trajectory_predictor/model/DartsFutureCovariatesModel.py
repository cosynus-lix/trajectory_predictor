import os

import numpy as np
import pandas as pd
import pickle

from darts import TimeSeries
from sklearn.preprocessing import MinMaxScaler

from ..trajectory.Trajectory import Trajectory
from ..dataset.Dataset import Dataset
from .PastPredictor.MeanPredictor import MeanPredictor
from .Model import Model

class DartsFutureCovariatesModel(Model):
    def __init__(self, darts_model, darts_model_class):
        super().__init__()

        self._model = darts_model
        self._model_class = darts_model_class

        self._trf = None
        self._dt = None
        self._mins_maxs = None

    def _get_mins_maxs(self, series_np):
        mins_maxs = np.vstack([np.min(series_np, axis=0), np.max(series_np, axis=0)])
        return mins_maxs

    def _simplify_mins_maxs(self, mins_maxs1, mins_maxs2):
        mins_maxs_stack = np.vstack([mins_maxs1, mins_maxs2])
        return self._get_mins_maxs(mins_maxs_stack)

    def dataset_to_series_and_curvatures_timeseries_list(self, dataset: Dataset):
        trajectories = dataset.get_trajectories()
        series_timeseries_list = []
        curvatures_timeseries_list = []
        self._dt = trajectories[0].get_dt()
        mins_maxs = None
        for trajectory in trajectories:
            if trajectory.get_dt() != self._dt:
                raise ValueError('All trajectories must have the same dt')
            series = trajectory.as_dt()
            mins_maxs = self._get_mins_maxs(series) if mins_maxs is None \
                else self._simplify_mins_maxs(mins_maxs, self._get_mins_maxs(series))
        self._trf = MinMaxScaler(feature_range=(-1, 1))
        self._trf.fit(mins_maxs)
        for trajectory in trajectories:
            series = trajectory.as_dt()
            transformed_series = self._trf.transform(series)
            series_timeseries_list.append(TimeSeries.from_values(transformed_series))
            curvatures_timeseries_list.append(TimeSeries.from_values(trajectory.curvatures_dt()))
        return series_timeseries_list, curvatures_timeseries_list

    def train(self, dataset: Dataset, **kwargs):
        # TODO: add max splits per time series
        # if TfClass 
        series, covariates = self.dataset_to_series_and_curvatures_timeseries_list(dataset)

        self._model.fit(series, 
                    past_covariates=covariates,  
                    verbose=True, 
                    **kwargs)

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

        transformed_series = self._trf.transform(series)
        timeseries = TimeSeries.from_values(transformed_series)

        future_covariates = TimeSeries.from_values(curvatures)
        prediction = self._model.predict(horizon,
                              series=timeseries,
                            #   past_covariates=None,
                              past_covariates=future_covariates)
        prediction = self._trf.inverse_transform(prediction.values())
    
        return Trajectory.from_dt(prediction, trajectory.get_optim(), trajectory.get_dt(), trajectory.get_final_progress())

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        # TODO: update pytorch-lightning when https://github.com/unit8co/darts/issues/1116 is solved
        model_name = 'darts_model.pth.tar'
        self._model.save_model(f'{path}/{model_name}')

        with open(f'{path}/helpers.pickle', 'wb') as handle:
            pickle.dump((self._trf, self._dt), handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError
        model_name = 'darts_model.pth.tar'
        self._model = self._model_class.load_model(f'{path}/{model_name}')

        with open(f'{path}/helpers.pickle', 'rb') as handle:
            self._trf, self._dt = pickle.load(handle)
