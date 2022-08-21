from logging.handlers import MemoryHandler
import matplotlib.pyplot as plt
import numpy as np

from darts.models import TFTModel
from darts.models import RNNModel
from trajectory_predictor.trajectory.TrajectoryDs import TrajectoryDs

from trajectory_predictor.model.DartsFutureCovariatesModel import DartsFutureCovariatesModel
from trajectory_predictor.model.PastPredictor.MeanPredictor import MeanPredictor
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.evaluation.TrajectoryEvaluator import TrajectoryEvaluator

def main():
    # model_class = TFTModel
    model_class = RNNModel
    darts_model = RNNModel(input_chunk_length=20, 
                    training_length=50, 
                    n_rnn_layers=2)
    # darts_model = model_class(input_chunk_length=20, 
    #                             output_chunk_length=50)
    model = DartsFutureCovariatesModel(darts_model, model_class)
    model.load('../../experiments/model0')
    # model = MeanPredictor(100)

    dataset = Dataset()
    dataset.load(f'/trajectory_predictor/datasets/test_dataset')
    full_trajectory = dataset.get_trajectories()[0]

    predict_progress = .25
    horizon = 300
    trajectory = full_trajectory.slice_time(end=predict_progress)
    prediction = model.predict(trajectory, horizon)

    trajds = TrajectoryDs.from_trajectory_dt(full_trajectory, 0.1, 0)
    # print(trajds.as_ds()[:, 0].sum(), trajds.as_ds()[:, 0].sum()/len(full_trajectory.get_history()))
    # exit()
    ax = plt.gca()
    # ax.plot(full_trajectory.get_history()[:, 0], full_trajectory.get_history()[:, 1])
    ax.plot(trajds.get_s_space(), trajds.get_history()[:, 0])
    ax.plot(prediction.get_s_space(), prediction.get_history()[:, 0])

    print(TrajectoryEvaluator.evaluate(full_trajectory, model, [8, 15, 38, 75, 300]))

    plt.savefig('./prediction.png')

if __name__ == "__main__":
    main()
