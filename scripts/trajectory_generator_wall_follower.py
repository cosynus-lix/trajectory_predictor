import numpy as np

from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer
from trajectory_predictor.controller.WallFollowerController import WallFollowerController
from trajectory_predictor.simulator.F1TenthSoloSimulator import F1TenthSoloSimulator
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
    ###############################################
    # Generates a trajectory using wall following #
    ###############################################

    # Initialization
    controller = WallFollowerController()
    track = np.loadtxt('../centerline/map0.csv', delimiter=',')
    optim = SplineOptimizer(track)
    optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)
    simulator = F1TenthSoloSimulator('../maps/map0', controller, optim)

    # Running simulation
    map_path = '../maps/map0'
    centerline_path = '../centerline/map0.csv'

    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458)
    simulator.run(printer=trajectory_printer)
    
    simulator.save_history('../runs/run0')
