import numpy as np
import argparse

from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer
from trajectory_predictor.controller.WallFollowerController import WallFollowerController
from trajectory_predictor.controller.MedialAxisFollowerController import MedialAxisFollowerController
from trajectory_predictor.simulator.F1TenthSoloSimulator import F1TenthSoloSimulator
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument('--map_index',type=int,default=0,help='Map index (should be in the maps/ directory)')
parser.add_argument('--speed',type=float,default=5,help='Speed of the car')
args = parser.parse_args()
map_index = args.map_index
vehicle_speed = args.speed

if __name__ == "__main__":
    ###############################################
    # Generates a trajectory using wall following #
    ###############################################

    # Initialization
    controller = WallFollowerController(speed=vehicle_speed)
    track = np.loadtxt(f'../centerline/map{map_index}.csv', delimiter=',')
    optim = SplineOptimizer(track)
    optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)
    simulator = F1TenthSoloSimulator(f'../maps/map{map_index}', controller, optim)
    vitesse=str(vehicle_speed)
    # Running simulation
    map_path = f'../maps/map{map_index}'
    centerline_path = f'../centerline/map{map_index}.csv'

    trajectory_printer = TrajectoryPrinter(map_path, '.pgm', centerline_path, 3.243796630159458)
    simulator.run(printer=trajectory_printer)
    
    simulator.save_history(f'../runs/run{map_index}speed{vitesse}')
