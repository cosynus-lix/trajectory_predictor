import numpy as np
import matplotlib.pyplot as plt
from trajectory_predictor.model.Baseline.Baselinemodel import BaselineModel
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
    ###############################################
    # Generates a trajectory using wall following #
    ###############################################

        # Load model
    past = 300
    horizon = 1000
    epochs = 1
    dataset = Dataset()
    dataset.load_data('../../centerline/map0.csv', '../../runs/run0/spline.npy', '../../runs/run0/history.npy')
    model = BaselineModel(past,64,horizon)
    #model.train(dataset,epochs,plot = True)    
    #model.save('../../experiments/Feedforward_model/model1')
    model.load('../../experiments/Feedforward_model/model1/Feedforward_model_scenario2.pt')

    value_to_test = 2700
    # Get series to predict
    dataset2=Dataset()
    dataset2.load_data('../../centerline/map7.csv', '../../runs/run7/spline.npy', '../../runs/run7/history.npy')
    dat_np = dataset.to_np()
    data_np = dataset2.to_np()
    point = value_to_test + model.past 
    curvatures = data_np[:, 2]
    prediction,y = model.predict(dataset2,1,value_to_test,plot=False)

    # Setting up the printer
    map_path = '../../maps/map7'
    centerline_path = '../../centerline/map7.csv'
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458, np.array([-78.21853769831466,-44.37590462453829]), 0.0625)
    trajectory = np.array([dataset2.get_progress(), dataset2.get_delta()]).T

    # Converting delta progress to progress in predicion
    prediction[:, 0] = np.cumsum(prediction[:, 0]) + trajectory[point, 0]

    # prediction[:, 0] = trajectory[point:point+800, 0]

    trajectory_printer.plot_trajectory_with_prediction(past+value_to_test,trajectory, prediction,name='Scenario1_model1')

