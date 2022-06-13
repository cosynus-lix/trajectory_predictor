import numpy as np

from trajectory_predictor.trajectory_generator.SplineOptimizer import SplineOptimizer
from trajectory_predictor.controller.WallFollowerController import WallFollowerController
from trajectory_predictor.simulator.F1TenthSoloSimulator import F1TenthSoloSimulator

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
    simulator.run()
    
    # TODO: Save history
