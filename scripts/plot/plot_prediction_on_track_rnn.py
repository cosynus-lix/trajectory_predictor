from inspect import ismethoddescriptor
import numpy as np

from trajectory_predictor.model.DartsRNNModel import DartsRNNModel
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
    ###############################################
    # Generates a trajectory using wall following #
    ###############################################

        # Load model
    model = DartsRNNModel()
    model.load('../../experiments/model0')

    # Load dataset
    dataset = Dataset()
    dataset.load_data('../../centerline/map0.csv', '../../runs/run0/spline.npy', '../../runs/run0/history.npy')

    # Get series to predict
    data_np = dataset.to_np()
    point = 1500
    series = data_np[:point, :-1]
    curvatures = data_np[:, 2]
    prediction = model.predict(series, curvatures, 800)

    # Setting up the printer
    map_path = '../maps/map0'
    centerline_path = '../centerline/map0.csv'
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458)
    trajectory = np.array([dataset.get_progress(), dataset.get_delta()]).T

    # Converting delta progress to progress in predicion
    prediction[:, 0] = np.cumsum(prediction[:, 0]) + trajectory[point, 0]

    # prediction[:, 0] = trajectory[point:point+800, 0]

    trajectory_printer.plot_trajectory_with_prediction(trajectory, prediction)
