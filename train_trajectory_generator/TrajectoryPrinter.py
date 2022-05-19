import yaml

import numpy as np
import matplotlib.pyplot as plt
import cv2
import shapely.geometry as shp

from SplineOptimizer import SplineOptimizer
from CoordinateTransform import CoordinateTransform

class TrajectoryPrinter:
    def __init__(self, map_path, map_ext, centerline_path):
        # Read black and white image
        map_image_path = f'{map_path}{map_ext}'
        self.map = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)

        # Read configuration yaml file for the map
        map_config_path = f'{map_path}.yaml'
        with open(map_config_path, 'r') as f:
            self.map_config = yaml.safe_load(f)

        # Read centerline
        self.centerline = np.loadtxt('../track_generator/centerline/map0.csv', delimiter=',')
        self.track_ring = shp.LinearRing(self.centerline)
        track_poly = shp.Polygon(self.centerline)
        # TODO: remove hardcode width
        WIDTH = 3.243796630159458/2
        self.width = WIDTH
        track_xy_offset_in = track_poly.buffer(WIDTH)
        track_xy_offset_out = track_poly.buffer(-WIDTH)
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

    def _print_map_matplotlib(self, ax, displaced_points, title='Map with trajectory (magenta)', show_centerline=False):
        ax.plot(self.inner[:,0], self.inner[:,1], 'k')
        ax.plot(self.outer[:,0], self.outer[:,1], 'k')

        ax.plot(displaced_points[:,0], displaced_points[:,1], 'm')

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

    def _print_over_map_image(self, displaced_points, origin, resolution):
        # Covert map to RGB
        map_rgb = cv2.cvtColor(self.map, cv2.COLOR_GRAY2RGB)
        output_shape = (map_rgb.shape[0], map_rgb.shape[1])

        for point in displaced_points:
            # Radius of circles which compose the trajectory
            radius = 1
            
            # Blue color in BGR (magenta)
            color = (255, 0, 255)
            thickness = 1

            map_rgb = cv2.circle(map_rgb, 
                                CoordinateTransform.metric_to_image(point, origin, output_shape, resolution),
                                radius, color, thickness)


        cv2.imshow('Map with trajectory (magenta)', map_rgb)
        cv2.waitKey(0)
    
    def print_trajectory_delta_by_s(self, trajectory_path):
        # TODO: add information about the track margins
        # TODO: refactor
        # TODO: publish changes
        # TODO: plot segmented centerline, curvature along the line and displacement on the curvatuer plot
        # TODO: numbe the points and show them as vertical dotted lines in the curvature plot
        # TODO: show the error in the title as well
        # TODO: do all that in the function below and then put all these filters somewere else
        from scipy.signal import savgol_filter, butter, filtfilt

        self._load_trajectory(trajectory_path)

        # plt.plot(self.progress, self.deltas)
        # print(self.track_ring.length)
        # exit()
        space = np.linspace(0, 1-0.0001, 300)
        points = []
        for s in space:
            p = self.track_ring.interpolate(s, normalized=True)
            p = np.array([p.x, p.y])
            points.append(p)
        points = np.vstack(points)
        print(points)

        cf = CurvatureFinder(points)
        curvatures = [cf.k(s) for s in space]
        from scipy.fft import fft, fftfreq
        
        # yf = fft(curvatures)
        # xf = fftfreq(400, 1/400)
        # plt.plot(xf, yf)
        b, a = butter(5, 1/1.2, 'low')
        output = filtfilt(b, a, curvatures)
        res = np.concatenate((output, np.array([0, 0])))
        res2 = np.concatenate((np.array([0, 0]), output))
        ds = (1-0.0001)/500
        delta = (res-res2)/ds
        # plt.plot(np.concatenate((space, np.array([0]))), delta[:-1])
        # plt.plot(space, curvatures, label='filtered')
        plt.plot(space, savgol_filter(output, 5, 3))
        # plt.plot(space, output)

        plt.show()
 
    def plot_new_curvature(self, trajectory_path):
        # TODO: refactor and change position
        self._load_trajectory(trajectory_path)
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_aspect('equal')
        ax1.plot(self.inner[:,0], self.inner[:,1], 'k')
        ax1.plot(self.outer[:,0], self.outer[:,1], 'k')
        space = np.linspace(0, 1-0.0001, 10)
        points = []
        for s in space:
            p = self.track_ring.interpolate(s, normalized=True)
            p = np.array([p.x, p.y])
            points.append(p)
        points = np.vstack(points)

        cf = SplineOptimizer(self.centerline)
        cf.sample_spline_by_tolerance(0.5, optimize=False, verbose=True)
        spline, points = cf.get_spline_and_points()
        new_space = np.linspace(0, 1-0.0001, 500)
        curvatures = [cf.k(s) for s in new_space]
        values = cf.interp(new_space)
        # print(values)

        ax1.plot(values[:, 0], values[:, 1], 'm')
        ax1.plot(points[:, 0], points[:, 1], '.m')
        ax2.plot(new_space, curvatures)
        # plt.plot(self.outer)
        # plt.plot(self.progress, self.deltas)
        plt.savefig('./test.png')
    
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
        # curvatures = curvatures/np.max(curvatures)
        displaced_points = self._add_to_centerline(progress_space, curvatures*self.width)


        _, ax = plt.subplots(2)
        self._print_map_matplotlib(ax[0], displaced_points, show_centerline=True)

        ax[1].plot(progress_space, curvatures)
        # plt.savefig('./trajectory.png')
        plt.show()

    def plot_trajectory(self, trajectory_path, matplotlib=False):
        """
        Produces a plot which shows the trajectory
        
        trajectory_path: path to the trajectory file
        matplotlib: if True, uses matplotlib to plot the trajectory else uses a OpenCV
        """
        self._load_trajectory(trajectory_path)

        displaced_points = self._add_to_centerline(self.progress, self.deltas)

        if matplotlib:
            _, ax = plt.subplots()
            self._print_map_matplotlib(ax, displaced_points)
            plt.savefig('./trajectory.png')
        else:
            args = (np.array([-78.21853769831466,-44.37590462453829]), 0.0625)
            self._print_over_map_image(displaced_points, *args)



if __name__ == '__main__':
    map_path = '../track_generator/maps/map0'
    centerline_path = '../track_generator/centerline/map0.csv'
    trajectory = './history.npy'

    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path)
    # trajectory_printer.plot_trajectory('history.npy', matplotlib=True) 
    trajectory_printer.plot_curvature_with_delta('history.npy', verbose=True, optimize=False) 
    # trajectory_printer.print_trajectory_delta_by_s(trajectory)
    # trajectory_printer.plot_new_curvature(trajectory)
