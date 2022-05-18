import yaml

import numpy as np
import matplotlib.pyplot as plt
import cv2
import shapely.geometry as shp

from CurvatureFinder import CurvatureFinder

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
        track_xy_offset_in = track_poly.buffer(WIDTH)
        track_xy_offset_out = track_poly.buffer(-WIDTH)
        self.inner = np.array(track_xy_offset_in.exterior)
        self.outer = np.array(track_xy_offset_out.exterior)

    def load_trajectory(self, trajectory_path):
        self.trajectory = np.load(trajectory_path)
        self.progress = self.trajectory[:, 0]
        self.deltas = self.trajectory[:, 1]

    def plot_new_curvature(self, trajectory_path):
        self.load_trajectory(trajectory_path)
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

        cf = CurvatureFinder(self.centerline)
        # spline, points = cf.optimize_points(50)
        spline, points = cf.optimize_k(optimize=True)
        new_space = np.linspace(0, 1-0.0001, 500)
        curvatures = [cf.k(s) for s in new_space]
        values = spline(new_space)
        # print(values)

        ax1.plot(values[:, 0], values[:, 1], 'm')
        ax1.plot(points[:, 0], points[:, 1], '.m')
        ax2.plot(new_space, curvatures)
        # plt.plot(self.outer)
        # plt.plot(self.progress, self.deltas)
        plt.savefig('./test.png')

    def print_map_with_trajectory(self, trajectory_path):
        # TODO: refactor and make the code more readable and add initial  point as arguemnt
        self.load_trajectory(trajectory_path)
        # Covert map to RGB
        map_rgb = cv2.cvtColor(self.map, cv2.COLOR_GRAY2RGB)

        trajectory_projection_xy = []
        for p in self.progress:
            trajectory_projection_xy.append(self.track_ring.interpolate(p, normalized=True))
        
        # Convert from metric to pixel
        for point, p, d in zip(trajectory_projection_xy, self.progress, self.deltas):
            # Draw circle in 
            point_before = self.track_ring.interpolate(p-0.001, normalized=True)

            print(point)
            pos = np.array([-point.x, point.y])
            pos_before = np.array([-point_before.x, point_before.y])
            delta = pos - pos_before
            # swap x and y
            delta = np.array([-delta[1], delta[0]])
            # normalize
            delta = delta / np.linalg.norm(delta) * d
            pos += np.array([-77.86034420414003,44.3286213256227])+delta
            coords = np.array([0, 1600])-np.array(pos)*(1/0.0625)
            print(coords, [point.x, point.y])
            coords = coords.astype(int)
            center_coordinates = coords
 
            # Radius of circle
            radius = int(1)
            
            # Blue color in BGR
            color = (255, 0, 255)
            
            # Line thickness of 2 px
            thickness = 5
            
            # Using cv2.circle() method
            # Draw a circle with blue line borders of thickness of 2 px
            map_rgb = cv2.circle(map_rgb, center_coordinates, radius, color, thickness)
            # print()


        cv2.imshow('Map with trajectory (green)', map_rgb)
        cv2.waitKey(0)
    
    def print_trajectory_delta_by_s(self, trajectory_path):
        # TODO: add information about the track margins
        # TODO: add better curvature in curvature finder
        # TODO interactive plot
        from scipy.signal import savgol_filter, butter, filtfilt

        self.load_trajectory(trajectory_path)

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

if __name__ == '__main__':
    map_path = '../track_generator/maps/map0'
    centerline_path = '../track_generator/centerline/map0.csv'
    trajectory = './history.npy'

    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path)
    # trajectory_printer.print_map_with_trajectory('history.npy') 
    # trajectory_printer.print_trajectory_delta_by_s(trajectory)
    trajectory_printer.plot_new_curvature(trajectory)
