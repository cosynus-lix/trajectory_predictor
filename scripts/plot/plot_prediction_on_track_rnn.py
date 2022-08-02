from inspect import ismethoddescriptor
import numpy as np

from trajectory_predictor.model.DartsRNNModel import DartsRNNModel
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
    # Load model
    model = DartsRNNModel()
    model.load('../../experiments/model0')

    # Load dataset
    dataset = Dataset()
    dataset.load_data('../../centerline/map0.csv', '../../runs/run0/spline.npy', '../../runs/run0/history.npy')

    predict_progress = 0.4
    full_trajectory = dataset.get_trajectory(end=1)
    past_trajectory = dataset.get_trajectory(end=predict_progress)
    prediction = model.predict(past_trajectory, 800)

    # Setting up the printer
    map_path = '../../maps/map0'
    centerline_path = '../../centerline/map0.csv'
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458)

    full_trajectory_history = full_trajectory.get_history()
    init_index = int(len(full_trajectory_history) * predict_progress)

    trajectory_printer.plot_trajectory_with_prediction(init_index, full_trajectory.get_history(), prediction.get_history())
