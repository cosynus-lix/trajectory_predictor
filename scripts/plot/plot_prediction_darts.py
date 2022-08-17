from logging.handlers import MemoryHandler
import matplotlib.pyplot as plt
import numpy as np

from darts.models import TFTModel
from darts.models import RNNModel

from trajectory_predictor.model.DartsFutureCovariatesModel import DartsFutureCovariatesModel
from trajectory_predictor.model.PastPredictor.MeanPredictor import MeanPredictor
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.evaluation.TrajectoryEvaluator import TrajectoryEvaluator

def main():
    # model_class = TFTModel
    model_class = RNNModel
    darts_model = RNNModel(input_chunk_length=200, 
                    training_length=500, 
                    n_rnn_layers=2)
    # darts_model = model_class(input_chunk_length=20, 
    #                             output_chunk_length=50)
    model = DartsFutureCovariatesModel(darts_model, model_class)
    model.load('../../experiments/model0')
    model = MeanPredictor(100)

    dataset = Dataset()
    dataset.load(f'/trajectory_predictor/datasets/test_datset')
    full_trajectory = dataset.get_trajectories()[3]

    predict_progress = .25
    horizon = 400
    trajectory = full_trajectory.slice_time(end=predict_progress)
    prediction = model.predict(trajectory, horizon)

    ax = plt.gca()
    ax.plot(full_trajectory.get_history()[:, 0], full_trajectory.get_history()[:, 1])
    ax.plot(prediction.get_history()[:, 0], prediction.get_history()[:, 1])

    TrajectoryEvaluator.evaluate(full_trajectory, model, [10, 20, 50, 100, 400])

    plt.savefig('./prediction.png')

if __name__ == "__main__":
    main()
