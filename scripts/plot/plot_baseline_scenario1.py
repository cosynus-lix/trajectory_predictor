import imp
import numpy as np
import matplotlib.pyplot as plt
from trajectory_predictor.model.Baseline.Baselinemodel import BaselineModel
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
    ###############################################
    # Print Model 1 prediction - scenario 1 #
    ###############################################

        
        #Generating model - Scenario 1

    past = 300
    horizon = 10
    epochs = 50


    dataset = Dataset()
    dataset.load_data('../../centerline/map0.csv', '../../runs/run0/spline.npy', '../../runs/run0/history.npy')
    dataset.add_data('../../centerline/map1.csv', '../../runs/run1/spline.npy', '../../runs/run1/history.npy')
    dataset.add_data('../../centerline/map2.csv', '../../runs/run2/spline.npy', '../../runs/run2/history.npy')
    dataset.add_data('../../centerline/map3.csv', '../../runs/run3/spline.npy', '../../runs/run3/history.npy')
    dataset.add_data('../../centerline/map4.csv', '../../runs/run4/spline.npy', '../../runs/run4/history.npy')
    dataset.add_data('../../centerline/map5.csv', '../../runs/run6/spline.npy', '../../runs/run6/history.npy')
    dataset.add_data('../../centerline/map6.csv', '../../runs/run6/spline.npy', '../../runs/run6/history.npy')
    model = BaselineModel(past,64,horizon)
    #model.train(dataset,epochs,plot = True)    
    #model.save('../../experiments/Baseline_model/Baseline','Baseline_model_scenario1')


        #load Model

    model.load('../../experiments/Baseline_model/Baseline/Baseline_model_scenario1.pt')
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
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458)

    trajectory = np.array([dataset2.get_progress(), dataset2.get_delta()]).T

    # Converting delta progress to progress in predicion
    prediction[:, 0] = np.cumsum(prediction[:, 0]) + trajectory[point, 0]
    print(prediction)
    for k in range(10):
        print(trajectory[value_to_test + 300 + k])

    #Final Print

    trajectory_printer.plot_trajectory_with_prediction(past+value_to_test,trajectory, prediction,name='Scenario 2 - Baseline Model')

