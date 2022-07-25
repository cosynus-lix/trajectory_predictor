#%%
import numpy as np
import matplotlib.pyplot as plt
from trajectory_predictor.model.Timeseries_feedforward.Timeseriespreprocessing import Timeseriespreprocessing
from trajectory_predictor.model.Timeseries_feedforward.TimeseriesfeedforwardModel import TimeseriesfeedforwardModel

from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter

#%%
model = TimeseriesfeedforwardModel(30,64)
model.load('../../experiments/Feedforward_model/modeltimeseries/Time_series_model.pt')
Dataset = Dataset()
Dataset.load_data('../../centerline/map7.csv', '../../runs/run7/spline.npy', '../../runs/run7/history.npy')
track = np.loadtxt('../../centerline/map7.csv', delimiter=',')
optim = SplineOptimizer(track)
optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)
data_np = Dataset.to_np()
#%%

init = 3000
len = 1000
point = init + model.past 
curvatures = data_np[:, 2]
prediction = model.predict(Dataset,init,len,optim)
#%%
# Setting up the printer
map_path = '../../maps/map7'
centerline_path = '../../centerline/map7.csv'
trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458, np.array([-78.21853769831466,-44.37590462453829]), 0.0625)
trajectory = np.array([Dataset.get_progress(), Dataset.get_delta()]).T
equiv_trajectory = trajectory[init +30:]
equiv_trajectory = equiv_trajectory[:len]
# Converting delta progress to progress in predicion
prediction[:, 0] = np.cumsum(prediction[:, 0]) + trajectory[point, 0]


#print(prediction.shape)
#print(f'prediction[0] = {prediction[0]}')
#print(equiv_trajectory.shape)
#print(f'equiv_trajectory[0]={equiv_trajectory[0]}')

max_delta_error = 0
deltachange=0
max_Deltap_error = 0
Deltapchange = 0
for k in range(10):
    #print(f'prediction progress = {prediction[k][0]}, progress = {equiv_trajectory[k][0]}')
    if (abs(prediction[k][0]-equiv_trajectory[k][0])>max_Deltap_error):
        max_Deltap_error = abs(prediction[k][0]-equiv_trajectory[k][0])
        Deltapchange = k
    if (abs(prediction[k][1]-equiv_trajectory[k][1])>max_delta_error):
        max_delta_error = abs(prediction[k][1]-equiv_trajectory[k][1])
        deltachange = k
print(f'delta error 1 = {max_delta_error}, delta change 1 = {deltachange}\n Delta p error 1 = {max_Deltap_error}, Delta p change 1 = {Deltapchange}' )

max_delta_error = 0
deltachange=0
max_Deltap_error = 0
Deltapchange = 0
for k in range(1000):
    #print(f'prediction progress = {prediction[k][0]}, progress = {equiv_trajectory[k][0]}')
    if (abs(prediction[k][0]-equiv_trajectory[k][0])>max_Deltap_error):
        max_Deltap_error = abs(prediction[k][0]-equiv_trajectory[k][0])
        Deltapchange = k
    if (abs(prediction[k][1]-equiv_trajectory[k][1])>max_delta_error):
        max_delta_error = abs(prediction[k][1]-equiv_trajectory[k][1])
        deltachange = k
print(f'delta error 2 = {max_delta_error}, delta change 2 = {deltachange}\n Delta p error 2 = {max_Deltap_error}, Delta p change 2 = {Deltapchange}' )

print(prediction[:,0].shape)
plt.plot(abs(prediction[:,0]-equiv_trajectory[:,0]))
plt.show()
