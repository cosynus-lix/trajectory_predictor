from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shp


class CurvatureFinder:
    def __init__(self, track):
        self.track = track

        self.initialize()

    def initialize(self):
        track_ring = shp.LinearRing(self.track)
        progress = []
        for p in self.track:
            progress.append(track_ring.project(shp.Point(p[0], p[1]), normalized=True))
        progress.append(1)
        progress = np.array(progress)
        print(np.vstack((self.track, self.track[0, :])).shape, self.track.shape)

        self.cs = CubicSpline(progress, np.vstack((self.track, self.track[0, :])), bc_type='periodic')
        self.der = self.cs.derivative()
        self.der2 = self.der.derivative()

    def k(self, s):
        """
        Returns the curavature for s in [0, 1]
        """
        assert 0 <= s <= 1
    

        num = self.der(s)[0]*self.der2(s)[1] - self.der(s)[1]*self.der2(s)[0]
        den = np.linalg.norm(self.der(s))**3
        return num/den

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
