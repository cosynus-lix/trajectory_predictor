import os

import numpy as np
import argparse

from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer

################################################
#  Generates the centerline given a map path   #
################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates the centerline given a map path')
    parser.add_argument('--map_path',type=str, required=True, help='Path to the map')
    path = parser.parse_args().map_path

    # Initialization
    track = np.loadtxt(f'{path}/centerline.csv', delimiter=',')
    optim = SplineOptimizer(track)
    optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)

    optim.dump_spline_and_points(os.path.join(path, 'spline.pickle'))
