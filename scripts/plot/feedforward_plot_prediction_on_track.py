from inspect import ismethoddescriptor
import numpy as np
from trajectory_predictor.model.Basic_feedforward.BasicfeedforwardModel import BasicfeedforwardModel
from trajectory_predictor.model.DartsRNNModel import DartsRNNModel
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
    ###############################################
    # Generates a trajectory using wall following #
    ###############################################

        # Load model
    past = 30
    horizon = 4
    epochs = 10
    dataset = Dataset()
    dataset.load_data('../../centerline/map0.csv', '../../runs/run0/spline.npy', '../../runs/run0/history.npy')
    model = BasicfeedforwardModel(past,64,horizon)
    model.load('../../experiments/Feedforward_model/model1/Feedforward_model.pt')


    # Get series to predict
    data_np = dataset.to_np()
    point = 1500
    curvatures = data_np[:, 2]
    prediction,y = model.predict(dataset,1500,1500,plot=True)
    print(prediction.shape)

    # Setting up the printer
    map_path = '../../maps/map0'
    centerline_path = '../../centerline/map0.csv'
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458, np.array([-78.21853769831466,-44.37590462453829]), 0.0625)
    trajectory = np.array([dataset.get_progress(), dataset.get_delta()]).T

    # Converting delta progress to progress in predicion
    prediction[:, 0] = np.cumsum(prediction[:, 0]) + trajectory[point, 0]

    # prediction[:, 0] = trajectory[point:point+800, 0]

    trajectory_printer.plot_trajectory_with_prediction(trajectory, prediction)
