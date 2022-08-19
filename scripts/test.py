import numpy as np
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.trajectory.Trajectory import Trajectory
from trajectory_predictor.trajectory.TrajectoryDs import TrajectoryDs
import matplotlib.pyplot as plt

dataset = Dataset()
evaluation_dataset_name = 'test_dataset'
dataset.load(f'/trajectory_predictor/datasets/{evaluation_dataset_name}')

traj_full = dataset.get_trajectories()[0]
traj = traj_full.slice_time(end=0.5)
print(traj.as_dt()[:, 0].mean()*400)
ds = 0.1
trajds = TrajectoryDs.from_trajectory_dt(traj, ds)
# new_traj = trajds.to_trajectory(1000)
# # # print(len(trajds.as_ds())*ds, traj.get_final_progress(), new_traj.get_final_progress())
# # exit()
ax = plt.gca()
ax.plot(traj_full.get_history()[:, 0], traj_full.get_history()[:, 1])
ax.plot(traj_full.get_history()[:-1, 0], traj_full.curvatures_dt())
# ax.plot(trajds.get_s_space(), trajds.get_history()[:, 1])
horizon = 300
fp = trajds.get_s_space()[-1]
p = np.arange(0, horizon)*ds+fp
ax.plot(p, trajds.get_future_curvatures(horizon))
# ax.plot(new_traj.get_history()[:, 0], new_traj.get_history()[:, 1])
print(np.cumsum([1,2,3])) 
plt.savefig('./prediction.png')
# def history_to_series_ds(history, ds, dt=0.03):
#     current_s = history[0, 0]+ds
#     cumm_time = 0
#     history_s = []
#     for point in history:
#         point_s = point[0]
#         point_delta = point[1]
#         if point_s > current_s:
#             history_s.append([cumm_time, point_delta])
#             current_s += ds
#         cumm_time += dt
#     history_s = np.vstack(history_s)
#     print(history_s)

# history_to_series_ds(traj.get_history(), 10)