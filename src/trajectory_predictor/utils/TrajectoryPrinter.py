import yaml

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shapely.geometry as shp

from .SplineOptimizer import SplineOptimizer
from .CoordinateTransform import CoordinateTransform

LATEX_OUTPUT = False

if LATEX_OUTPUT:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

class TrajectoryPrinter:
    def __init__(self, map_path, map_ext, centerline_path, track_width, track_origin, track_scale):
        # Read configuration yaml file for the map
        map_config_path = f'{map_path}.yaml'
        with open(map_config_path, 'r') as f:
            self.map_config = yaml.safe_load(f)

        # Read centerline
        self.centerline = np.loadtxt(centerline_path, delimiter=',')
        self.track_ring = shp.LinearRing(self.centerline)
        track_poly = shp.Polygon(self.centerline)
        self.track_origin = track_origin
        self.track_scale = track_scale
        self.half_width = track_width / 2
        track_xy_offset_in = track_poly.buffer(self.half_width)
        track_xy_offset_out = track_poly.buffer(-self.half_width)
        self.inner = np.array(track_xy_offset_in.exterior)
        self.outer = np.array(track_xy_offset_out.exterior)

    def _load_trajectory(self, trajectory_path):
        self.trajectory = np.load(trajectory_path)
        self.progress = self.trajectory[:, 0]
        self.deltas = self.trajectory[:, 1]

    def _add_to_centerline(self, progresses, deltas):
        points_over_centerline = []
        for p in progresses:
            points_over_centerline.append(self.track_ring.interpolate(p, normalized=True))

        displaced_points = []
        for point, p, d in zip(points_over_centerline, progresses, deltas):
            pos = np.array([point.x, point.y])
            point_before = self.track_ring.interpolate(p-0.001, normalized=True)
            pos_before = np.array([point_before.x, point_before.y])
            ds = pos - pos_before
            ds_rotated = np.array([ds[1], -ds[0]])
            ds_rot_normalized = ds_rotated / np.linalg.norm(ds_rotated) * d
            pos += ds_rot_normalized
            displaced_points.append(pos)
        displaced_points = np.vstack(displaced_points)
        
        return displaced_points

    def _print_map_matplotlib(self, ax, displaced_points, title='Map with trajectory (magenta)', show_centerline=False, color='m'):
        ax.plot(self.inner[:,0], self.inner[:,1], 'k')
        ax.plot(self.outer[:,0], self.outer[:,1], 'k')

        ax.plot(displaced_points[:,0], displaced_points[:,1], color)

        if show_centerline:
            progresses = np.linspace(0, 1-0.0001, 1000)
            points_over_centerline = self._add_to_centerline(progresses, np.zeros(len(progresses)))
            ax.plot(points_over_centerline[:,0], points_over_centerline[:,1], 'r-.', alpha=0.5)

        # Plot config
        ax.set_title(title)
        ax.set_aspect('equal')
        # Remove x and y axis
        ax.set_xticks([])
        ax.set_yticks([])
    
    def plot_curvature_with_delta(self, trajectory_path, tolerance=0.5, verbose=False, optimize=True):
        """
        Produces a plot which shows the evolution of the curvature over the trajectory
        
        trajectory_path: path to the trajectory file
        tolerance: tolerance for the spline interpolation
        verbose: if True, prints the optimization steps
        optimize: if True, optimizes the spline to have fewer points
        """

        self._load_trajectory(trajectory_path)
        spline_optim = SplineOptimizer(self.centerline)
        spline_optim.sample_spline_by_tolerance(tolerance, optimize=optimize, verbose=verbose)

        progress_space = np.linspace(0, 1-0.0001, 1000)
        curvatures = np.array([spline_optim.k(s) for s in progress_space])
        curvatures = curvatures/np.max(curvatures)*self.half_width
        displaced_curvature_points = self._add_to_centerline(progress_space, curvatures)
        spline_points = spline_optim.interp(progress_space)
        control_spline_points = spline_optim.get_spline_and_points()[1]
        spline_errors = np.array([self.track_ring.distance(shp.Point(p[0], p[1])) for p in spline_points])
        progress_control_points = np.array([self.track_ring.project(shp.Point(p[0], p[1]), normalized=True) 
            for p in control_spline_points])

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(212)

        self._print_map_matplotlib(ax1, displaced_curvature_points, show_centerline=True, title='Curvature along map', color='g')

        self._print_map_matplotlib(ax2, spline_points, show_centerline=True, title='Error along map', color='b')
        ax2.plot(control_spline_points[:, 0], control_spline_points[:, 1], 'm.')

        ax3.set_title(f'Curvature and spline error by path progress with tolerance {tolerance}')
        ax3.plot(progress_space, curvatures, color='g', label='Curvature')
        ax3.plot(progress_space, spline_errors, color='b', label='Spline error')
        ax3.legend()
        ax3.axhline(y=self.half_width, color='k', linestyle='-')
        ax3.axhline(y=-self.half_width, color='k', linestyle='-')
        for p in progress_control_points:
            ax3.axvline(p, c='m', ls='-.')
        if LATEX_OUTPUT:
            plt.savefig('./trajectory_evaluation.pgf')
        else:
            plt.savefig('./trajectory_evaluation.png')
        plt.show()

    def plot_trajectory(self, trajectory_path):
        """
        Produces a plot which shows the trajectory
        
        trajectory_path: path to the trajectory file
        matplotlib: if True, uses matplotlib to plot the trajectory else uses a OpenCV
        """
        self._load_trajectory(trajectory_path)

        displaced_points = self._add_to_centerline(self.progress, self.deltas)

        _, ax = plt.subplots()
        self._print_map_matplotlib(ax, displaced_points)
        plt.savefig('./trajectory.png')

    def plot_trajectory_with_prediction(self, trajectory, prediction, color='b'):
        progress_trajectory = trajectory[:, 0]
        deltas_trajectory = trajectory[:, 1]
        progress_prediction = prediction[:, 0]
        deltas_prediction = prediction[:, 1]

        displaced_points_trajectory = self._add_to_centerline(progress_trajectory, deltas_trajectory)
        displaced_points_prediction = self._add_to_centerline(progress_prediction, deltas_prediction)

        _, ax = plt.subplots()
        self._print_map_matplotlib(ax, displaced_points_trajectory)
        self._print_map_matplotlib(ax, displaced_points_prediction, color='b')
        plt.savefig('./trajectory.png')

    def plot_trajectory_frame(self, trajectory, pause_delay=0.05):
        progress = trajectory[:, 0]
        deltas = trajectory[:, 1]

        displaced_points = self._add_to_centerline(progress, deltas)
        _, ax = plt.subplots()
        self._print_map_matplotlib(ax, displaced_points)
        plt.pause(pause_delay)


if __name__ == '__main__':
    map_path = '../track_generator/maps/map0'
    centerline_path = '../track_generator/centerline/map0.csv'
    trajectory = './history.npy'

    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458, np.array([-78.21853769831466,-44.37590462453829]), 0.0625)
    # trajectory_printer.plot_trajectory('history.npy', matplotlib=True) 
    trajectory_printer.plot_curvature_with_delta('history.npy', tolerance=0.7, verbose=True, optimize=True) 
