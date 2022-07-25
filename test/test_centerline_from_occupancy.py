import cv2
import numpy as np
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter
from trajectory_predictor.utils.mapping import image_to_centerline, save_centerline_points_metric

map_path = '../maps/gmapping/new_map'
centerline_path = '../maps/gmapping/new_map.csv'
centerline = image_to_centerline(map_path)

save_centerline_points_metric(centerline, map_path, '.pgm')

trajectory_printer = TrajectoryPrinter(map_path, '.pgm', centerline_path, 3.243796630159458)

progress = np.linspace(0, 1, 1000)
deltas = np.zeros(1000)
history = np.column_stack((progress, deltas))
np.save('./history.npy', history)
trajectory_printer.plot_trajectory('history.npy', matplotlib=False) 
