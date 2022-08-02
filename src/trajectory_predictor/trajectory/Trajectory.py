import numpy as np

from ..model.PastPredictor.PastPredictor import PastPredictor

class Trajectory:
    def __init__(self, history, optimizer, timestep):
        self.__history = history
        self.__optimizer = optimizer
        self.__data_np = self.history_to_series()
        self.__timestep = timestep

    def from_dt(series, optimizer, timestep, initial_progress):
        """
        Returns the trajectory object from series of [delta_progress, delta]

        series: trajectory in the format in rows of [delta_progress, delta]
        initial_progress: initial progress of the trajectory
        optimizer: spline optimzer

        returns: trajectory object
        """

        # Converting delta progress to progress
        history = series.copy()
        history[:, 0] = np.cumsum(history[:, 0]) + initial_progress
        return Trajectory(history, optimizer, timestep)

    def history_to_series(self):
        progresses = self.__history[:, 0]
        deltas = self.__history[:, 1]
        curvatures = [self.__optimizer.k(progress) for progress in progresses]

        delta_progress = np.diff(progresses)

        return np.array([delta_progress, deltas[:-1], curvatures[:-1]]).T

    def as_dt(self):
        """
        Returns the trajectory alongside in fixed timesteps

        returns: trajectory in the format in rows of [delta_progress, delta]
        """
        return self.__data_np[:, :-1]

    def curvatures_dt(self):
        """
        Returns the curvatures for fixed timesteps

        returns: curvatures
        """

        return self.__data_np[:, 2]

    def get_dt(self):
        """
        Returns the timestep

        returns: timestep in miliseconds
        """
        return self.__timestep

    def get_optim(self):
        """
        Returns the spline optimizer
        
        returns: spline optimizer
        """
        return self.__optimizer

    def get_final_progress(self):
        """
        Returns the final progress of the trajectory

        returns: final progress
        """
        return self.__history[-1, 0]

    def get_history(self):
        """
        Returns the history of the trajectory

        returns: history in the format in rows of [progress, delta]
        """
        return self.__history

    def get_future_curvatures(self, predictor: PastPredictor, horizon: int):
        if not isinstance(predictor, PastPredictor):
            raise Exception("Only past predictor can be used to get future ccurvatures")
        
        future_trajectory = predictor.predict(self, horizon)
        return future_trajectory.curvatures_dt()
        