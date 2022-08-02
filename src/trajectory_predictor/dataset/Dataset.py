import numpy as np

from ..utils.SplineOptimizer import SplineOptimizer
from ..trajectory.Trajectory import Trajectory

class Dataset:
    def __init__(self):
        self.data_np = None
        self.optimizer = None
        self.optimizer2 = None

    def history_to_series(self, history):
        # TODO: Maybe rermove?
        if self.optimizer is None:
            raise Exception('SplineOptimizer is not initialized')
        progresses = history[:, 0]
        deltas = history[:, 1]
        curvatures = [self.optimizer.k(progress) for progress in progresses]

        delta_progress = np.diff(progresses)

        return np.array([delta_progress, deltas[:-1], curvatures[:-1]]).T
    
    def load_data(self, centerline_path, spline_path, history_path, timestep=0.01):
        centerline = np.loadtxt(centerline_path, delimiter=',')
        self.optimizer = SplineOptimizer(centerline)
        self.optimizer.load_spline(filename=spline_path)
        history = np.load(history_path)
        self.__trajectory = Trajectory(history, self.optimizer, timestep)
        self.data_np = self.__trajectory.as_dt()

    def add_data(self, centerline_path, spline_path,history_path):
        centerline = np.loadtxt(centerline_path, delimiter=',')
        self.optimizer2 = SplineOptimizer(centerline)
        self.optimizer2.load_spline(filename=spline_path)
        trajectory = np.load(history_path)

        self.data_np = np.vstack((self.data_np,self.history_to_series(trajectory)))

    def dump(self, filename):
        np.save(filename, self.data_np)

    def get_delta_progress(self):
        return self.data_np[:, 0]

    def get_progress(self):
        return np.cumsum(self.data_np[:, 0])

    def get_delta(self):
        return self.data_np[:, 1]
    
    def get_curvature(self):
        return self.data_np[:, 2]

    def to_np(self):
        return self.data_np

    def get_trajectory(self, start=0, end=1):
        def progress_to_index(progress):
            return int(progress*len(self.data_np))
        start, end = progress_to_index(start), progress_to_index(end)
        history = self.__trajectory.get_history()
        return Trajectory(history[start:end], self.optimizer, self.__trajectory.get_dt())
