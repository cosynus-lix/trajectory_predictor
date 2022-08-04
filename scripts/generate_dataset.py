import numpy as np
from tqdm import tqdm

from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.trajectory.Trajectory import Trajectory
from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer
from trajectory_predictor.controller.WallFollowerController import WallFollowerController
from trajectory_predictor.simulator.F1TenthSoloSimulator import F1TenthSoloSimulator


def main():
    dataset = Dataset()
    map_path = '/trajectory_predictor/maps/map0'
    centerline_path = f'{map_path}/centerline.csv'
    spline_path = f'{map_path}/spline.pickle'
    timestep = 0.03

    track = np.loadtxt(centerline_path, delimiter=',')
    optim = SplineOptimizer(track)
    trajectory_list = []

    print("Generating trajectories...")
    optim = SplineOptimizer(track)
    optim.load_spline(spline_path)
    for speed in tqdm(np.linspace(2.5, 3, 3)):
        # Initialization
        controller = WallFollowerController(speed=speed)
        simulator = F1TenthSoloSimulator(map_path, controller, optim, False, timestep)

        # Running simulation
        trajectory = simulator.run()
        trajectory_list.append(trajectory)

    print("Saving dataset...")
    dataset.add_data(trajectory_list)
    dataset.dump(f'/trajectory_predictor/datasets/test_datset')

if __name__ == "__main__":
    main()
