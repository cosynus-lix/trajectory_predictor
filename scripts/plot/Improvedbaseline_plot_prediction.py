import numpy as np
import matplotlib.pyplot as plt
from trajectory_predictor.model.Improved_baseline.Improved_baseline_preprocessing import Improved_Baselinepreprocessing
from trajectory_predictor.model.Improved_baseline.Improved_baseline_model import Improved_BaselineModel

from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

if __name__ == "__main__":
    ###############################################
    # Generates a trajectory using wall following #
    ###############################################

        # Load model
    past = 30
    epochs = 100
    dataset = Dataset()
    #dataset.load_data('../../centerline/map0.csv', '../../runs/run0/spline.npy', '../../runs/run0/history.npy')
    model = Improved_BaselineModel(past,64)
    
    #model.train(dataset,epochs,plot = True)    
    #model.save('../../experiments/Feedforward_model/modeltimeseries')
    model.load('../../experiments/Baseline_model/Improved_baseline/Time_series_model.pt')

    init = 2970
    len = 3001
    # Get series to predict
    dataset2=Dataset()
    dataset2.load_data('../../centerline/map7.csv', '../../runs/run7/spline.npy', '../../runs/run7/history.npy')
    track = np.loadtxt('../../centerline/map7.csv', delimiter=',')
    optim = SplineOptimizer(track)
    optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)
    dat_np = dataset.to_np()
    data_np = dataset2.to_np()
    point = init + model.past 
    curvatures = data_np[:, 2]
    prediction = model.predict(dataset2,point,len,optim)
    #print(prediction)

    # Setting up the printer
    map_path = '../../maps/map7'
    centerline_path = '../../centerline/map7.csv'
    trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458, np.array([-78.21853769831466,-44.37590462453829]), 0.0625)
    trajectory = np.array([dataset2.get_progress(), dataset2.get_delta()]).T

    # Converting delta progress to progress in predicion
    prediction[:, 0] = np.cumsum(prediction[:, 0]) + trajectory[point, 0]

    # prediction[:, 0] = trajectory[point:point+800, 0]

    trajectory_printer.plot_trajectory_with_prediction(init+past,trajectory, prediction)
    print(prediction.shape)
    print(trajectory.shape)
    print(f'predicted total progress = {prediction[len-1,0]}\ntrue total progress = {trajectory[init+len+30,0]}')
    print(f'begin predict progress = {prediction[0,0]}\nbegin true progress = {trajectory[init+30,0]}')


