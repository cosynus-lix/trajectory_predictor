import numpy as np

def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)

def dist_ling_segs(a1, b1, a2, b2):
    """Cartesian distance from line segment to line segment"""

    a1 = a1.reshape(-1, 2)
    b1 = b1.reshape(-1, 2)
    a2 = a2.reshape(-1, 2)
    b2 = b2.reshape(-1, 2)

    d1 = lineseg_dists(a1, a2, b2)
    d2 = lineseg_dists(b1, a2, b2)
    d3 = lineseg_dists(a2, a1, b1)
    d4 = lineseg_dists(b2, a1, b1)

    return min(d1, d2, d3, d4)