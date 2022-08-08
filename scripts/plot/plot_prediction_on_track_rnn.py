from trajectory_predictor.model.DartsRNNModel import DartsRNNModel
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
    model = DartsRNNModel()
    model.load('../../experiments/model0')

    dataset = Dataset()
    dataset.load(f'/trajectory_predictor/datasets/test_datset')
    full_trajectory = dataset.get_trajectories()[0]

    predict_progress = 0.4
    past_trajectory = full_trajectory.slice_time(end=predict_progress)
    prediction = model.predict(past_trajectory, 800)

    # Setting up the printer
    map_path = '../../maps/map0'
    centerline_path = '../../maps/map0/centerline.csv'
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458)

    full_trajectory_history = full_trajectory.get_history()
    init_index = int(len(full_trajectory_history) * predict_progress)

    trajectory_printer.plot_trajectory_with_prediction(init_index, full_trajectory, prediction)
