from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shapely.geometry as shp
import scipy.optimize as optimize

N_POINTS_SPLINE_DISCRETIZATION = 1000

class SplineOptimizer:
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
        Returns optimized spline for n_points and statistics about the approximation
        before and after optimization
        """

        s_values = np.linspace(0, 1-1/n_points, n_points)

        def spline_and_points_from_s_values(s_values):
            points = []
            for s in s_values:
                points.append(self.track_ring.interpolate(s, normalized=True))
            points = np.vstack(points+[points[0]])
            return CubicSpline(np.hstack((s_values, np.array([1]))), points, bc_type='periodic'), points

        def error_spline_true_value(spline):
            points = np.linspace(0, 1, N_POINTS_SPLINE_DISCRETIZATION)
            spline_points = spline(points)
            
            cumm_dist = 0
            max_dist = 0
            for p in spline_points:
                distance = self.track_ring.distance(shp.Point(p[0], p[1]))
                cumm_dist += distance
                max_dist = max(max_dist, distance)
            mean_dist = cumm_dist / len(spline_points)
            return max_dist, mean_dist

        def s_values_to_error(s_values):
            """
            Objective function to minimize

            Returns the maximum distance between the spline and the points
            which define the centerline. For that, the spline is first discretized
            """
            if not np.all(np.diff(s_values) > 0):
                return np.inf
            if s_values[-1] == 1:
                s_values[-1] = 1-1e-6
            spline, _ = spline_and_points_from_s_values(s_values)
            return error_spline_true_value(spline)[0]

        spline_and_points = spline_and_points_from_s_values(s_values)
        non_optimized_max_dist, non_optimized_mean_dist = error_spline_true_value(spline_and_points[0])
        stats_bef = {'mean_dist': non_optimized_mean_dist, 'max_dist': non_optimized_max_dist}
        
        bounds = np.array([[0, 1]]*n_points)
        stats_aft = None
        if should_optimize:
            new_s_values = optimize.minimize(s_values_to_error, s_values, bounds=bounds, method='SLSQP')
            new_s_values = new_s_values.x
            spline_and_points = spline_and_points_from_s_values(new_s_values)
            optimized_max_dist, optimized_mean_dist = error_spline_true_value(spline_and_points[0])
            stats_aft = {'mean_dist': optimized_mean_dist, 'max_dist': optimized_max_dist}

        return spline_and_points, stats_bef, stats_aft 

    def sample_spline_by_tolerance(self, tolerance=0.1, optimize=True, verbose=False):
        """
        Returns spline that best approximates the road
        """

        def get_curr_max_dist(stats_aft, stats_bef):
            if not stats_aft:
                return stats_bef['max_dist']
            return stats_aft['max_dist']

        nb_points = 2
        spline_and_points, stats_bef, stats_aft = self.generate_spline_and_points(nb_points, should_optimize=optimize)
        curr_max_dist = get_curr_max_dist(stats_aft, stats_bef)
        old_dist = curr_max_dist
        old_delta = 1 # any value > 0
        while old_delta > 0:
            delta = 1
            curr_max_dist = old_dist
            while curr_max_dist > tolerance:
                delta += delta
                old_dist = curr_max_dist
                old_delta = delta // 4
                next_nb_points = nb_points + delta//2
                if verbose:
                    print(f'Stats before optimization : {stats_bef}')
                    print(f'Stats after optimization  : {stats_aft}')
                    print(f'Computing for : {next_nb_points}')
                spline_and_points, stats_bef, stats_aft = self.generate_spline_and_points(next_nb_points, should_optimize=optimize)
                curr_max_dist = get_curr_max_dist(stats_aft, stats_bef)
            nb_points += old_delta
        self.set_spline_and_points(spline_and_points)

    def set_spline_and_points(self, spline_and_points):
        self.cs, self.spline_points = spline_and_points
        self.der = self.cs.derivative()
        self.der2 = self.der.derivative()

        self.spline_s_discretization = np.linspace(0, 1, N_POINTS_SPLINE_DISCRETIZATION)
        spline_points = self.cs(self.spline_s_discretization)
        spline_points_ring = shp.LinearRing(spline_points)
        self.spline_progress_discretization = []
        for p in spline_points:
            self.spline_progress_discretization.append(spline_points_ring.project(shp.Point(p[0], p[1]), normalized=True))
        self.spline_progress_discretization = np.array(self.spline_progress_discretization)

    def dump_spline_and_points(self):
        pickle.dump((self.cs, self.spline_points), open('spline.pickle', 'wb'))

    def load_spline(self, filename='spline.pickle'):
        spline_and_points = pickle.load(open(filename, 'rb'))
        self.set_spline_and_points(spline_and_points)
    
    def get_spline_and_points(self):
        return self.cs, self.spline_points

    def get_track_ring(self):
        return self.track_ring

    def map_progress_to_s(self, progress):
        # TODO: maybe use dicotomy to invert this
        return self.spline_s_discretization[np.searchsorted(self.spline_progress_discretization, progress)-1]

    def k(self, progress):
        """
        Returns the curavature for progress in [0, 1]
        """
        assert 0 <= progress <= 1
    
        s = self.map_progress_to_s(progress)

        num = self.der(s)[0]*self.der2(s)[1] - self.der(s)[1]*self.der2(s)[0]
        den = np.linalg.norm(self.der(s))**3
        return num/den

    def interp(self, progress):
        """
        Maps a progress in [0, 1] on the road to a point on the spline
        """
        return self.cs(self.map_progress_to_s(progress))
