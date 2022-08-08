import numpy as np

from ..model.PastPredictor.PastPredictor import PastPredictor

class Trajectory:
    def __init__(self, history, optimizer, timestep):
        """
        Given a history of the trajectory, creates a trajectory object

        :param history: history in the format in rows of [s, delta]
        """
        self.__history = history
        self.__optimizer = optimizer
        self.__data_np = self.history_to_series()
        self.__timestep = timestep

    def from_dt(series, optimizer, timestep, initial_progress):
        """
        Returns the trajectory object from series of [delts_s, delta]

        :param series: trajectory in the format in rows of [delts_s, delta]
        :param initial_progress: initial progress of the trajectory
        :param optimizer: spline optimzer
        :return: trajectory object
        """

        # Converting delta progress to progress
        history = series.copy()
        history[:, 0] = np.cumsum(history[:, 0]) + initial_progress
        return Trajectory(history, optimizer, timestep)

    def history_to_series(self):
        s = self.__history[:, 0]
        delta_s = np.diff(s)
        progresses = s / self.__optimizer.get_track_length()
        deltas = self.__history[:, 1]
        curvatures = [self.__optimizer.k(progress) for progress in progresses]

        return np.array([delta_s, deltas[:-1], curvatures[:-1]]).T

    def as_dt(self):
        """
        Returns the trajectory alongside in fixed timesteps

        :return: trajectory in the format in rows of [delta_s, delta]
        """
        return self.__data_np[:, :-1]

    def curvatures_dt(self):
        """
        Returns the curvatures for fixed timesteps

        :return: curvatures
        """

        return self.__data_np[:, 2]

    def get_dt(self):
        """
        Returns the timestep

        :return: timestep in miliseconds
        """
        return self.__timestep

    def get_optim(self):
        """
        Returns the spline optimizer
        
        :return: spline optimizer
        """
        return self.__optimizer

    def get_final_progress(self):
        """
        Returns the final progress of the trajectory

        :return: final progress
        """
        return self.__history[-1, 0]

    def get_history(self):
        """
        Returns the history of the trajectory

        :return: history in the format in rows of [progress, delta]
        """
        return self.__history

    def get_normalized_history(self):
        """
        Returns the normalized history of the trajectory

        :return: normalized history in the format in rows of [progress, delta]
        """
        progresses = self.__history[:, 0] / self.__optimizer.get_track_length()
        return np.array([progresses, self.__history[:, 1]]).T
    
    def slice_time(self, init=0, end=1):
        """
        Returns the trajectory sliced between init and end, in a scaled between 0 and 1 one in time units

        :param init: initial time scaled between 0 and 1
        :param end: end time scaled between 0 and 1 (bigger than init)
        """
        def time_progress_to_index(progress):
            return int(progress*len(self.__history))
        slice_history = self.__history[time_progress_to_index(init):time_progress_to_index(end)]
        return Trajectory(slice_history, self.__optimizer, self.__timestep)

    def get_future_curvatures(self, predictor: PastPredictor, horizon: int):
        if not isinstance(predictor, PastPredictor):
            raise Exception("Only past predictor can be used to get future ccurvatures")
        
        future_trajectory = predictor.predict(self, horizon)
        return future_trajectory.curvatures_dt()
 