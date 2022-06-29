from math import fabs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.model.Basic_feedforward.BasicfeedforwardModel import BasicfeedforwardModel
from trajectory_predictor.model.Basic_feedforward.Feedforwardpreprocessing import Feedforwardpreprocessing


def main():
    past = 30
    horizon = 2
    epochs = 100
    dataset = Dataset()
    dataset.load_data('../centerline/map0.csv', '../runs/run0/spline.npy', '../runs/run0/history.npy')
    model = BasicfeedforwardModel(past,64,horizon)
    #model.train(dataset,epochs,plot = False)    
    #model.save('../experiments/Feedforward_model/model1')
    model.load('../experiments/Feedforward_model/model1/Feedforward_model.pt')
    model.predict(dataset,3200,0,plot=True)

if __name__ == "__main__":
    main()
