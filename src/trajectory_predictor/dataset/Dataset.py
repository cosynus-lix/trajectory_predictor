import os

import pickle

class Dataset:
    def __init__(self):
        self.spline_to_trajectories = {}

    def add_data(self, trajectories):
        """
        Adds list of trajectories to the datset based on the same map

        centerline_path: path to the centerline
        spline_path: path to the save where the centerline spline was saved
        trajectories: list of trajectories
        """

        for trajectory in trajectories:
            id = trajectory.get_optim().get_id()
            if id not in self.spline_to_trajectories:
                self.spline_to_trajectories[id] = [trajectory]
            else:
                self.spline_to_trajectories[id].append(trajectory)
    
    def get_trajectories(self):
        """Returns all trajectories in the dataset as list"""

        trajectories = []
        for trajectories_on_spline in self.spline_to_trajectories.values():
            trajectories += trajectories_on_spline

        return trajectories
 
    def dump(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(f'{path}/dataset.pickle', 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(f'{path}/dataset.pickle', 'rb') as f:
            self.__dict__ = pickle.load(f)
