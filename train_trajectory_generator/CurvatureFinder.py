from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shp
import scipy.optimize as optimize


class CurvatureFinder:
    def __init__(self, track):
        self.track = track

        self.initialize()

    def initialize(self):
        self.track_ring = shp.LinearRing(self.track)
        progress = []
        for p in self.track:
            progress.append(self.track_ring.project(shp.Point(p[0], p[1]), normalized=True))
        progress.append(1)
        progress = np.array(progress)

        self.cs = CubicSpline(progress, np.vstack((self.track, self.track[0, :])), bc_type='periodic')
        self.der = self.cs.derivative()
        self.der2 = self.der.derivative()
    
    def generate_spline_and_points(self, n_points, should_optimize=True):
        """
        Returns optimized spline for n_points
        """

        s_values = np.linspace(0, 1-1/n_points, n_points)

        def spline_from_s_values(s_values):
            points = []
            for s in s_values:
                points.append(self.track_ring.interpolate(s, normalized=True))
            points = np.vstack(points+[points[0]])
            return CubicSpline(np.hstack((s_values, np.array([1]))), points, bc_type='periodic'), points

        def error_spline_true_value(spline):
            # TODO add the mean dist was well
            points = np.linspace(0, 1, 1000)
            spline_points = spline(points)
            
            cumm_dist = 0
            max_dist = 0
            for p in spline_points:
                distance = self.track_ring.distance(shp.Point(p[0], p[1]))
                cumm_dist += distance
                max_dist = max(max_dist, distance)
            return max_dist

        def s_values_to_error(s_values):
            if not np.all(np.diff(s_values) > 0):
                return np.inf
            if s_values[-1] == 1:
                s_values[-1] = 1-1e-6
            spline, _ = spline_from_s_values(s_values)
            return error_spline_true_value(spline)

        non_optimized_spline = spline_from_s_values(s_values)[0]
        non_optimized_max_dist = error_spline_true_value(non_optimized_spline)
        if not should_optimize:
            return spline_from_s_values(s_values), non_optimized_max_dist, non_optimized_max_dist
        
        bounds = np.array([[0, 1]]*n_points)
        new_s_values = optimize.minimize(s_values_to_error, s_values, bounds=bounds, method='SLSQP')
        new_s_values = new_s_values.x
        optimized_spline = spline_from_s_values(new_s_values)[0]
        self.der = optimized_spline.derivative()
        self.der2 = self.der.derivative()
        optimized_max_dist = error_spline_true_value(optimized_spline)

        return spline_from_s_values(new_s_values), non_optimized_max_dist, optimized_max_dist

    def optimize_k(self, tolerance=0.1, optimize=True):
        """
        Returns spline that best approximates the road
        """

        nb_points = 2
        spline, max_dist_before_optim, curr_max_dist = self.generate_spline_and_points(nb_points, should_optimize=optimize)
        old_dist = curr_max_dist
        old_delta = 1 # any value > 0
        while old_delta > 0:
            delta = 1
            curr_max_dist = old_dist
            while curr_max_dist > tolerance:
                delta += delta
                old_dist = curr_max_dist
                old_delta = delta // 4
                spline, max_dist_before_optim, curr_max_dist = self.generate_spline_and_points(nb_points+delta//2, should_optimize=optimize)
                print(f'Computing for : {nb_points+delta//2}, old_value: {max_dist_before_optim}, new_value: {curr_max_dist}')
            nb_points += old_delta
        return spline

    def k(self, s):
        """
        Returns the curavature for s in [0, 1]
        """
        assert 0 <= s <= 1
    

        num = self.der(s)[0]*self.der2(s)[1] - self.der(s)[1]*self.der2(s)[0]
        den = np.linalg.norm(self.der(s))**3
        return num/den

    def interp(self, s):
        """
        Returns the point for s in [0, 1] (accepts array)
        """
        return self.cs(s)

    def k_alternative(self):
        """
        Returns the curavature for s in [0, 1]
        """
        # assert 0 <= s <= 1
        # points = []
        # track_ring = shp.LinearRing(self.track)
        # for s in np.linspace(0, 1, 100):
        #     point = self.track_ring.interpolate(s, normalized=True)
        #     points.append([point.x, point.y])
        # points = np.vstack(points)
        #     print(s, self.k(s))
        # num = self.der(s)[0]*self.der2(s)[1] - self.der(s)[1]*self.der2(s)[0]
        # den = np.linalg.norm(self.der(s))**3
        # return num/den
