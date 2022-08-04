import matplotlib.pyplot as plt
import numpy as np

from trajectory_predictor.model.DartsRNNModel import DartsRNNModel
from trajectory_predictor.dataset.Dataset import Dataset

def main():
    model = DartsRNNModel()
    model.load('../../experiments/model0')

    dataset = Dataset()
    dataset.load(f'/trajectory_predictor/datasets/test_datset')
    full_trajectory = dataset.get_trajectories()[0]

    predict_progress = 0.3
    horizon = 800
    trajectory = full_trajectory.slice_time(end=predict_progress)
    prediction = model.predict(trajectory, horizon)

    ax = plt.gca()
    ax.plot(full_trajectory.get_history()[:, 0], full_trajectory.get_history()[:, 1])
    ax.plot(prediction.get_history()[:, 0], prediction.get_history()[:, 1])

    plt.savefig('./prediction.png')

if __name__ == "__main__":
    main()