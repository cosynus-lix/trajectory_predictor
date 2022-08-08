from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter
from trajectory_predictor.dataset.Dataset import Dataset

map_path = '../maps/map0'
centerline_path = '../maps/map0/centerline.csv'

trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458)
dataset = Dataset()
dataset.load(f'/trajectory_predictor/datasets/test_datset')
trajectory = dataset.get_trajectories()[0]

print("Length: ", trajectory.get_optim().get_track_length(), trajectory.get_optim().spline_discretization_ring.length)
trajectory_printer.plot_curvature_with_delta(trajectory, tolerance=0.1, verbose=True, optimize=False) 
