import numpy as np

from .PastPredictor import PastPredictor
from ...trajectory.Trajectory import Trajectory

class MeanPredictor(PastPredictor):
    def __init__(self, past_predictions_to_consider=1):
        self.__past_predictions_to_consider = past_predictions_to_consider

    def train(self):
        raise NotImplementedError

    def predict(self, trajectory: Trajectory, horizon=10):
        """
        trajectory: trajectory containing (delta_progress, deltas)
        horizon: number of timesteps to predict
        """

        traj_np = trajectory.as_dt()
        mean = traj_np[-self.__past_predictions_to_consider].reshape(-1, 2).mean(axis=0)

        repeated_mean = np.repeat(mean, horizon+1, axis=0).reshape(2, -1).T
        
        return Trajectory.from_dt(repeated_mean, trajectory.get_optim(), trajectory.get_dt(), trajectory.get_final_progress())

    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError
