import time
import os

import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as shp
import gym

from .Simulator import Simulator
from ..utils.SplineOptimizer import SplineOptimizer
from ..utils.TrajectoryPrinter import TrajectoryPrinter
from ..trajectory.Trajectory import Trajectory

class F1TenthSoloSimulator(Simulator):
    def __init__(self, map_path, controller, max_track_width, verbose=False, timestep=0.01):
        super().__init__(None)
        self.map_path = map_path
        self.timestep = timestep
        self._verbose = verbose
        self._controller = controller
        self._max_track_width = max_track_width
        track = np.loadtxt(f'{map_path}/centerline.csv', delimiter=',')
        optim = SplineOptimizer(track)
        optim.load_spline(f'{map_path}/spline.pickle')
        self._spline_optimizer = optim

        self._history = []
        self._laptime = 0.0
        self._s = 0.0
        self._speed, self._steer = 0.0, 0.0

    def _step(self, n_laps):
        obs, step_reward, done, _ = self.env.step(np.array([[self._steer, self._speed]]))

        if obs['lap_counts'][0] >= n_laps:
            done = True
        
        # Record history
        # TODO replace with custom function on the spline itself
        track_ring = self._spline_optimizer.get_track_ring()
        current_position = shp.Point(obs['poses_x'][0], obs['poses_y'][0])
        self._s = track_ring.project(current_position)
        projection = track_ring.interpolate(self._s)
        projection_before = track_ring.interpolate(self._s-0.001)
        ccw = shp.LinearRing([projection_before, projection, current_position]).is_ccw
        distance = projection.distance(current_position)
        if ccw:
            distance = -projection.distance(current_position)
        self._history.append([self._s, distance])
        if self._verbose:
            progress = self._s / self._spline_optimizer.get_track_length()
            print(f's: {self._s:.2f}, delta:{distance:.2f}, k:{self._spline_optimizer.k(progress):.2f}, r:{step_reward} {self.map_path}')

        # Car shouldn't step out of bounds
        if abs(distance) > self._max_track_width:
            raise Exception(f'Car stepped out of bounds: {distance}/{self._max_track_width}')

        # Control
        self._steer, self._speed = self._controller.get_control(obs)
        self._laptime += step_reward
        
        return obs, done
    
    def run(self, should_print_trajectory=False, n_laps=1, init_speed=5):
        # Needs to make before each run as some parameters are stored 
        # in class variables
        self.env = gym.make('f110_gym:f110-v0', map=f'{self.map_path}/map', 
            map_ext='.pgm', num_agents=1, timestep=self.timestep)
        printer = TrajectoryPrinter(self.map_path, '.pgm', 
            f'{self.map_path}/centerline.csv', self._max_track_width) if should_print_trajectory else None

        start_time = time.time()
        self._speed, self._steer = init_speed, 0.0
        self._history = []
        self._laptime = 0.0
        self._s = 0.0

        _, _, done, _ = self.env.reset(np.array([[0, 0, np.pi/2]]))

        def print_trajectory():
            if should_print_trajectory:
                printer.plot_trajectory_frame(Trajectory(np.array(self._history), 
                    self._spline_optimizer, self.env.timestep))

        step_i = 0
        while not done:
            obs, done = self._step(n_laps)
            
            if step_i % 100 == 0:
                print_trajectory()

            step_i += 1

        print_trajectory()

        return Trajectory(np.array(self._history), self._spline_optimizer, self.env.timestep)

    def save_history(self, path):
        # Create path if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Save history
        np.save(os.path.join(path, 'history.npy'), np.array(self._history))

        # Dump spline
        self._spline_optimizer.dump_spline_and_points(os.path.join(path, 'spline.pickle'))
