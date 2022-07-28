import numpy as np
import matplotlib.pyplot as plt
from trajectory_predictor.model.Baseline.Baselinemodel import BaselineModel
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
     ###############################################
    # Print Model 1 prediction #
    ###############################################


        #Generate trajectory with different speed

        #Generating model - Scenario 4

    past = 300
    horizon = 1000
    epochs = 30
    dataset = Dataset()
    dataset.load_data('../../centerline/map6.csv', '../../runs/run6/spline.npy', '../../runs/run6/history.npy')
    dataset.add_data('../../centerline/map6.csv', '../../runs/run6speed2/spline.npy', '../../runs/run6speed2/history.npy')
    dataset.add_data('../../centerline/map6.csv', '../../runs/run6speed4/spline.npy', '../../runs/run6speed4/history.npy')
    dataset.add_data('../../centerline/map6.csv', '../../runs/run6speed5/spline.npy', '../../runs/run6speed5/history.npy')
    dataset.add_data('../../centerline/map6.csv', '../../runs/run6speed6/spline.npy', '../../runs/run6speed6/history.npy')
    model = BaselineModel(past,64,horizon)
    #model.train(dataset,epochs,plot = True)    
    #model.save('../../experiments/Baseline_model/Baseline','Baseline_model_scenario4')

    #Load data

    model.load('../../experiments/Baseline_model/Baseline/Baseline_model_scenario4.pt')
    value_to_test = 2700

    # Get series to predict

    dataset2=Dataset()
    dataset2.load_data('../../centerline/map6.csv', '../../runs/run6speed1/spline.npy', '../../runs/run6speed1/history.npy')
    dat_np = dataset.to_np()
    data_np = dataset2.to_np()
    point = value_to_test + model.past 
    curvatures = data_np[:, 2]
    prediction,y = model.predict(dataset2,1,value_to_test,plot=False)

    # Setting up the printer
    map_path = '../../maps/map6'
    centerline_path = '../../centerline/map6.csv'
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458)
    trajectory = np.array([dataset2.get_progress(), dataset2.get_delta()]).T

    # Converting delta progress to progress in predicion
    prediction[:, 0] = np.cumsum(prediction[:, 0]) + trajectory[point, 0]

    # final print

    trajectory_printer.plot_trajectory_with_prediction(past+value_to_test,trajectory, prediction,name='Scenario 4 - Baseline Model')

