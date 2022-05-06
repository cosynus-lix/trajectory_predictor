import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from math_helpers import *

def track_line_segments(track_pts):
    """Converts a list of track points into pairs of points in the correct sequence"""
    segments = []
    for i in range(len(track_pts)):
        if i == len(track_pts) - 1:
            segments.append(np.array([track_pts[i], track_pts[0]]))
        else:
            segments.append(np.array([track_pts[i], track_pts[i+1]]))

    return segments

def adjecent_segments(segment_index, n_segments):
    """Returns the indices of the two adjacent segments"""
    adjacent_segments = []
    prev = (segment_index - 1 + n_segments) % n_segments
    next = (segment_index + 1) % n_segments
    adjacent_segments.append(prev)
    adjacent_segments.append(next)
    return adjacent_segments

def feasible_track(segments, threshold):
    """Checks if track segments are separated enough between each other"""

    n_segments = len(segments)

    for i in range(n_segments):
        for j in range(n_segments):
            # Don't compare with neighboring segments
            if j == i or j in adjecent_segments(i, n_segments):
                continue
            if dist_ling_segs(segments[i][0], segments[i][1], segments[j][0], segments[j][1]) < threshold:
                return False
    return True

def distance_point_segments(segment_starts, segment_ends, point):
    """Returns the distance from a point to the closest line segment"""
    dists = lineseg_dists(point.reshape(-1, 2), segment_starts, segment_ends) 
    return min(dists)

def generate_random_track():
    ## HARDCODED PARAMETERS         
    image_size = (500, 500)
    threshold = 50
    n_points = 30

    valid_track = False
    while not valid_track:
        rng = np.random.default_rng()
        points = rng.random((n_points, 2))
        points[:, 0] *= image_size[0]-threshold
        points[:, 1] *= image_size[1]-threshold
        points[:, 0] += threshold/2
        points[:, 1] += threshold/2
        hull = ConvexHull(points)
        track_points = points[hull.vertices]
        track_segments = track_line_segments(track_points)
        valid_track = feasible_track(track_segments, threshold)

    image = np.zeros(image_size, dtype=np.uint8)
    segment_starts = np.stack([track_segments[i][0] for i in range(len(track_segments))])
    segment_ends = np.stack([track_segments[i][1] for i in range(len(track_segments))])
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            dist = distance_point_segments(segment_starts, segment_ends, np.array([i, j]))
            if dist < threshold/2:
                image[i, j] = 255
    return image
# image = generate_random_track()
# cv2.imshow('image', image)
# cv2.waitKey(0)
# print(track_points)

# plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
# plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
# plt.show()
# print(hull.vertices)

# print(lineseg_dists(np.array([[0,0]]), np.array([[3,0], [6,0]]), np.array([[0,4], [0,8]])))
# print(dist_ling_segs(points[0], points[1], points[2], points[3]))
# print(feasible_track(track_segments, 0.1))