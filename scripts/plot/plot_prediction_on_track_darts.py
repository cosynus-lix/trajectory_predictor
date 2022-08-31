import numpy as np

from trajectory_predictor.model.PastPredictor import MeanPredictor
from trajectory_predictor.dataset.SimpleDataset import SimpleDataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter
from trajectory_predictor.model.DartsFutureCovariatesModel import DartsFutureCovariatesModel
from trajectory_predictor.dataset.Dataset import Dataset

from darts.models import RNNModel

if __name__ == "__main__":
    # Load model
    model_class = RNNModel
    darts_model = RNNModel(input_chunk_length=20, 
                    training_length=50, 
                    n_rnn_layers=2)
    model = DartsFutureCovariatesModel(darts_model, model_class)
    model.load('../../experiments/model0')

    dataset = Dataset()
    dataset.load(f'/trajectory_predictor/datasets/test_dataset')
    full_trajectory = dataset.get_trajectories()[0]

    predict_progress = 0.35
    # predict_progress = 0.6
    past_trajectory = full_trajectory.slice_time(end=predict_progress)
    prediction = model.predict(past_trajectory, 400)

    # Setting up the printer
    map_path = '../../maps/map0'
    centerline_path = '../../maps/map0/centerline.csv'
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458)

    full_trajectory_history = full_trajectory.get_history()
    init_index = int(len(full_trajectory_history) * predict_progress)

    trajectory_printer.plot_trajectory_with_prediction(init_index, full_trajectory, prediction)
