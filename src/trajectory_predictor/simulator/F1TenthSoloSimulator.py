import time
import os

import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as shp
import gym

from .Simulator import Simulator

class F1TenthSoloSimulator(Simulator):
    def __init__(self, map_path, controller, spline_optimizer):
        env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.pgm', num_agents=1)
        self._controller = controller
        self._spline_optimizer = spline_optimizer
        super().__init__(env)

        self._history = []
        self._laptime = 0.0
        self._progress = 0.0
        self._speed, self._steer = 0.0, 0.0

    def _step(self, n_laps):
        obs, step_reward, done, _ = self.env.step(np.array([[self._steer, self._speed]]))

        if obs['lap_counts'][0] >= n_laps:
            done = True
        
        # Record history
        # TODO replace with custom function on the spline itself
        track_ring = self._spline_optimizer.get_track_ring()
        current_position = shp.Point(obs['poses_x'][0], obs['poses_y'][0])
        self._progress = track_ring.project(current_position, normalized=True)
        projection = track_ring.interpolate(self._progress, normalized=True)
        projection_before = track_ring.interpolate(self._progress-0.001, normalized=True)
        ccw = shp.LinearRing([projection_before, projection, current_position]).is_ccw
        distance = projection.distance(current_position)
        if ccw:
            distance = -projection.distance(current_position)
        self._history.append([self._progress, distance])
        print(f's: {self._progress:.2f}, delta:{distance:.2f}, k:{self._spline_optimizer.k(self._progress):.2f}')

        # Control
        self._steer, self._speed = self._controller.get_control(obs)
        self._laptime += step_reward
        
        return obs, done
    
    def run(self, printer=None, n_laps=1, display=True, init_speed=5):
        start_time = time.time()
        self._speed, self._steer = init_speed, 0.0
        self._history = []
        self._laptime = 0.0
        self._progress = 0.0

        _, _, done, _ = self.env.reset(np.array([[0, 0, np.pi/2]]))

        step_i = 0
        while not done:
            obs, done = self._step(n_laps)
            
            if printer is not None and step_i % 100 == 0:
                current_trajectory = np.array(self._history)
                
                printer.plot_trajectory_frame(current_trajectory)

            step_i += 1

        printer.plot_trajectory_frame(current_trajectory)
        #plt.waitforbuttonpress()

    def save_history(self, path):
        # Create path if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Save history
        np.save(os.path.join(path, 'history.npy'), np.array(self._history))

        # Dump spline
        self._spline_optimizer.dump_spline_and_points(os.path.join(path, 'spline.npy'))
