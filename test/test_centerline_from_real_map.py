import cv2
import numpy as np
import os
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter
from trajectory_predictor.utils.mapping import image_to_centerline, save_centerline_points_metric, filter_track_on_image

map_path = '../maps/real_map/better_map'
filtered_map_path = '../maps/real_map/better_map_filtered'
centerline_path = f'{filtered_map_path}.csv'

print("Filtering map...")
img = filter_track_on_image(map_path, '.pgm')
cv2.imwrite(f"{filtered_map_path}.pgm", img)

# There needs to be a corresponding yaml file for the filtered map
os.system(f"cp {map_path}.yaml {filtered_map_path}.yaml")

print("Getting map centerline...")
centerline = image_to_centerline(filtered_map_path)

save_centerline_points_metric(centerline, filtered_map_path, '.pgm')

trajectory_printer = TrajectoryPrinter(filtered_map_path, '.pgm', centerline_path, 3.243796630159458)

progress = np.linspace(0, 1, 1000)
deltas = np.zeros(1000)
history = np.column_stack((progress, deltas))
np.save('./history.npy', history)
trajectory_printer.plot_trajectory('history.npy', matplotlib=False) 
