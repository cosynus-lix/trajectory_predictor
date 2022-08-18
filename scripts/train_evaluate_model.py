import time
import glob

import matplotlib.pyplot as plt
import numpy as np

from darts.models import TFTModel
from darts.models import RNNModel
from darts.models import NBEATSModel
from darts.models import TCNModel
from darts.models import NHiTSModel
from darts.utils.likelihood_models import QuantileRegression

from trajectory_predictor.model.DartsFutureCovariatesModel import DartsFutureCovariatesModel
from trajectory_predictor.model.PastPredictor.MeanPredictor import MeanPredictor
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.evaluation.TrajectoryEvaluator import TrajectoryEvaluator

def main():
    model_class = NBEATSModel
    model_params = {'input_chunk_length': 20, 'output_chunk_length': 50}
    darts_model = model_class(**model_params)

    train_dataset = Dataset()
    dataset_name = 'train_medium'
    train_dataset.load(f'/trajectory_predictor/datasets/{dataset_name}')

    model = DartsFutureCovariatesModel(darts_model, model_class)
    train_args = {'epochs': 3}
    t0 = time.time()
    model.train(train_dataset, **train_args)
    model.save(f'/trajectory_predictor/experiments/model0')
    t1 = time.time() - t0

    model.load(f'/trajectory_predictor/experiments/model0')

    evaluation_dataset = Dataset()
    evaluation_dataset_name = 'test_dataset'
    evaluation_dataset.load(f'/trajectory_predictor/datasets/{evaluation_dataset_name}')

    full_trajectory = evaluation_dataset.get_trajectories()[0]
    table = TrajectoryEvaluator.evaluate(full_trajectory, model, [10, 20, 50, 100, 400])

    predict_progress = .25
    horizon = 400
    trajectory = full_trajectory.slice_time(end=predict_progress)
    prediction = model.predict(trajectory, horizon)

    ax = plt.gca()
    ax.plot(full_trajectory.get_history()[:, 0], full_trajectory.get_history()[:, 1])
    ax.plot(prediction.get_history()[:, 0], prediction.get_history()[:, 1])
    model_class_name = model_class.__name__

    # Dump everything to file
    filename = f'{model_class_name}_{dataset_name}_{evaluation_dataset_name}'
    files = glob.glob(f'/trajectory_predictor/experiments/{filename}_*')
    try:
        largest_number = max([int(file.split('_')[-1].split('.')[0]) for file in files])
        index = largest_number + 1
    except ValueError:
        index = 0
    
    print(index)
    with open(f'/trajectory_predictor/experiments/{filename}_{index}.txt', 'w') as f:
        f.write(f'{model_class_name}\n')
        f.write(f'Train dataset: {dataset_name}\n')
        f.write(f'Evaluation dataset: {evaluation_dataset_name}\n')
        f.write(f'Train args: {train_args}\n')
        f.write(f'Train time: {t1}\n')
        f.write(f'Model parameters: {model_params}\n')
        f.write(f'{table}\n')
    plt.savefig(f'/trajectory_predictor/experiments/prediction_{filename}_{index}.png')



if __name__ == "__main__":
    main()