import numpy as np
from .Trajectory import Trajectory

class TrajectoryDs():
    def __init__(self, history, optimizer, ds, initial_progress):
        self.__history = history
        self.__optimizer = optimizer
        self.__initial_progress = initial_progress
        self.__ds = ds
        self.__data_np = self.history_to_series()

    # TODO: add history to series

    def from_trajectory_dt(trajectory: Trajectory, ds, initial_time):
        history_dt = trajectory.get_history()
        dt = trajectory.get_dt()
        series_ds = TrajectoryDs.history_dt_to_series_ds(history_dt, dt, ds)
        # TODO initial progress
        return TrajectoryDs.from_ds(series_ds, trajectory.get_optim(), ds, history_dt[0, 0], initial_time)

    def to_trajectory(self, dt):
        series_dt = TrajectoryDs.history_ds_to_series_dt(self.__history, dt, self.__ds)
        # series_dt[:, 0] = series_dt[:, 0] / self.__optimizer.get_track_length()
        print(len(series_dt))
        exit()
        return Trajectory.from_dt(series_dt, self.__optimizer, dt, self.__initial_progress)

    def from_ds(series, optimizer, ds, initial_progress, initial_time):
        # Converting delta progress to progress
        history = series.copy()
        history[:, 0] = np.cumsum(history[:, 0]) + initial_time
        traj = TrajectoryDs(history, optimizer, ds, initial_progress)
        return traj

    def get_s_space(self):
         return np.arange(0, len(self.__history))*self.__ds+self.__initial_progress

    def history_dt_to_series_ds(history, dt, ds=0.05):
        current_s = history[0, 0]+ds
        cumm_time = 0
        history_s = []
        for point in history:
            point_s = point[0]
            point_delta = point[1]
            if point_s > current_s:
                history_s.append([cumm_time, point_delta])
                current_s += ds
            cumm_time += dt
        history_s = np.vstack(history_s)
        return history_s

    def history_ds_to_series_dt(history, dt, ds):
        current_t = history[0, 0]+dt
        cumm_s = 0
        history_t = []
        for point in history:
            point_t = point[0]
            point_delta = point[1]
            if point_t > current_t:
                history_t.append([cumm_s, point_delta])
                current_t += dt
            cumm_s += ds
        history_t = np.vstack(history_t)
        return history_t

    def history_to_series(self):
        t = self.__history[:, 0]
        delta_t = np.diff(t)
        s = np.arange(0, len(delta_t)+1)*self.__ds+self.__initial_progress
        progresses = s / self.__optimizer.get_track_length()
        deltas = self.__history[:, 1]
        curvatures = [self.__optimizer.k(progress) for progress in progresses]
        return np.array([delta_t, deltas[:-1], curvatures[:-1]]).T

    def as_ds(self):
        return self.__data_np[:, :-1]

    def get_history(self):
        return self.__history
    
    def curvatures_ds(self):
        return self.__data_np[:, 2]

    def get_optim(self):
        return self.__optimizer

    def get_future_curvatures(self, horizon: int):
        final_progress = (len(self.__history)-1)*self.__ds+self.__initial_progress
        s = np.arange(0, horizon)*self.__ds+final_progress
        progresses = s / self.__optimizer.get_track_length()
        curvatures = [self.__optimizer.k(progress) for progress in progresses]
        return curvatures
    def get_final_progress(self):
        return (len(self.__history)-1)*self.__ds+self.__initial_progress