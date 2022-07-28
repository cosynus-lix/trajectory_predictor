
import numpy as np
import matplotlib.pyplot as plt
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter
from trajectory_predictor.model.Baseline.Baselinemodel import BaselineModel
from trajectory_predictor.model.Baseline.Baseline_preprocessing import Baselinepreprocessing
from trajectory_predictor.model.Improved_baseline.Improved_baseline_preprocessing import Improved_Baselinepreprocessing
from trajectory_predictor.model.Improved_baseline.Improved_baseline_model import Improved_BaselineModel
from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer


# Load model1 for scenario 1 and 2

past = 300
horizon = 1000
horizon1 =10
epochs = 1
dataset = Dataset()
model = BaselineModel(past,64,horizon)
model.load('../../experiments/Baseline_model/Baseline/Feedforward_model_scenario2.pt')
model1 = BaselineModel(past,64,horizon1)
model1.load('../../experiments/Baseline_model/Baseline/Feedforward_model_scenario1.pt')


init = 3000
# Get series to predict
dataset=Dataset()
dataset.load_data('../../centerline/map7.csv', '../../runs/run7/spline.npy', '../../runs/run7/history.npy')
data_np = dataset.to_np()
point = init + model.past
curvatures = data_np[:, 2]
prediction,y = model.predict(dataset,1,init,plot=False)
prediction1,y1 = model.predict(dataset,1,init,plot=False)


# Setting up the printer
map_path = '../../maps/map7'
centerline_path = '../../centerline/map7.csv'
trajectory_printer = TrajectoryPrinter(map_path, '.png', centerline_path, 3.243796630159458, np.array([-78.21853769831466,-44.37590462453829]), 0.0625)
trajectory = np.array([dataset.get_progress(), dataset.get_delta()]).T
equiv_trajectory = trajectory[init:]
equiv_trajectory = equiv_trajectory[:1000]

# Converting delta progress to progress in predicion
prediction[:, 0] = np.cumsum(prediction[:, 0]) + trajectory[point, 0]
prediction1[:, 0] = np.cumsum(prediction1[:, 0]) + trajectory[point, 0]

max_delta_error = 0
deltachange=0
max_Deltap_error = 0
Deltapchange = 0
for k in range(10):
    #print(f'prediction progress = {prediction[k][0]}, progress = {equiv_trajectory[k][0]}')
    if (abs(prediction1[k][0]-equiv_trajectory[k][0])>max_Deltap_error):
        max_Deltap_error = abs(prediction1[k][0]-equiv_trajectory[k][0])
        Deltapchange = k
    if (abs(prediction1[k][1]-equiv_trajectory[k][1])>max_delta_error):
        max_delta_error = abs(prediction1[k][1]-equiv_trajectory[k][1])
        deltachange = k
print(f'delta error 1.1= {max_delta_error}, delta change 1.1 = {deltachange}\n Delta p error 1.1 = {max_Deltap_error}, Delta p change 1.1 = {Deltapchange}')


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
print(f'delta error 1.2 = {max_delta_error}, delta change 1.2= {deltachange}\n Delta p error 1.2 = {max_Deltap_error}, Delta p change 1.2 = {Deltapchange}' )


## model 2
model2 = Improved_BaselineModel(30,64)
model2.load('../../experiments/Baseline_model/Improved_baseline/Time_series_model.pt')
Dataset = Dataset()
Dataset.load_data('../../centerline/map7.csv', '../../runs/run7/spline.npy', '../../runs/run7/history.npy')
track = np.loadtxt('../../centerline/map7.csv', delimiter=',')
optim = SplineOptimizer(track)
optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)
data_np = Dataset.to_np()

init2 = init+300
len = 1000
curvatures = data_np[:, 2]
prediction2 = model2.predict(Dataset,init2,len,optim)
prediction2[:, 0] = np.cumsum(prediction2[:, 0]) + trajectory[point, 0]

max_delta_error = 0
deltachange=0
max_Deltap_error = 0
Deltapchange = 0
for k in range(10):
    #print(f'prediction progress = {prediction[k][0]}, progress = {equiv_trajectory[k][0]}')
    if (abs(prediction2[k][0]-equiv_trajectory[k][0])>max_Deltap_error):
        max_Deltap_error = abs(prediction2[k][0]-equiv_trajectory[k][0])
        Deltapchange = k
    if (abs(prediction2[k][1]-equiv_trajectory[k][1])>max_delta_error):
        max_delta_error = abs(prediction2[k][1]-equiv_trajectory[k][1])
        deltachange = k
print(f'delta error 2.1 = {max_delta_error}, delta change 2.1 = {deltachange}\n Delta p error 2.1 = {max_Deltap_error}, Delta p change 2.1 = {Deltapchange}' )

max_delta_error = 0
deltachange=0
max_Deltap_error = 0
Deltapchange = 0
for k in range(1000):
    #print(f'prediction progress = {prediction[k][0]}, progress = {equiv_trajectory[k][0]}')
    if (abs(prediction2[k][0]-equiv_trajectory[k][0])>max_Deltap_error):
        max_Deltap_error = abs(prediction2[k][0]-equiv_trajectory[k][0])
        Deltapchange = k
    if (abs(prediction2[k][1]-equiv_trajectory[k][1])>max_delta_error):
        max_delta_error = abs(prediction2[k][1]-equiv_trajectory[k][1])
        deltachange = k
print(f'delta error 2.2 = {max_delta_error}, delta change 2.2 = {deltachange}\n Delta p error 2.2 = {max_Deltap_error}, Delta p change 2.2 = {Deltapchange}' )

plt.figure()
plt.title('Progres error')
plt.plot(abs(prediction[:,0]-equiv_trajectory[:,0]),label = 'model1')
plt.plot(abs(prediction2[:,0]-equiv_trajectory[:,0]),label = 'model2')
plt.legend()


plt.figure()
plt.title('Cumulated progress error')
plt.subplot(1,1,1)
plt.plot(np.cumsum(abs(prediction[:,0]-equiv_trajectory[:,0])),label = 'model1')
plt.plot(np.cumsum(abs(prediction2[:,0]-equiv_trajectory[:,0])),label = 'model2')
plt.legend()


trajectory_printer.plot_trajectory_with_prediction(past+init,trajectory, prediction2,name='Scenario2_model2')
trajectory_printer.plot_trajectory_with_prediction(past+init,trajectory, prediction,name='Scenario2_model1')