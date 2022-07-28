from ast import Del
from asyncio.proactor_events import _ProactorDuplexPipeTransport
from turtle import pd
from matplotlib.ft2font import HORIZONTAL
import numpy as np
import matplotlib.pyplot as plt
from trajectory_predictor.dataset.Dataset import Dataset as ds
from trajectory_predictor.utils.TrajectoryPrinter import TrajectoryPrinter
from trajectory_predictor.model.Baseline.Baselinemodel import BaselineModel
from trajectory_predictor.model.Baseline.Baseline_preprocessing import Baselinepreprocessing
from trajectory_predictor.model.Improved_baseline.Improved_baseline_preprocessing import Improved_Baselinepreprocessing
from trajectory_predictor.model.Improved_baseline.Improved_baseline_model import Improved_BaselineModel
from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer
class metrics():
    def __init__(self,models =1,scenario =1 ,begin = 1000):

        #load model

        super().__init__()
       
        if (scenario==1)or(scenario==3):
            horizon = 10
        elif (scenario ==2)or(scenario ==4):
            horizon = 1000
        

        if (models ==1):
            model = BaselineModel(300,64,horizon)
            model.load('../../experiments/Baseline_model/Baseline/Baseline_model_scenario'+"%s"%scenario+'.pt')
            
        elif (models ==2):
            model = Improved_BaselineModel(30,64)
            model.load('../../experiments/Baseline_model/Improved_baseline/Improved_Baseline_model_scenario'+"%s"%scenario+'.pt')
        else:
            print('Model must be 1 or 2')
            raise ValueError 

        #def trajectory

        Dataset = ds()

        if (scenario==1)or(scenario==2):
            Dataset.load_data('../../centerline/map7.csv', '../../runs/run7/spline.npy', '../../runs/run7/history.npy')
        else:
            Dataset.load_data('../../centerline/map6.csv', '../../runs/run6speed1/spline.npy', '../../runs/run6speed1/history.npy')
        raw_trajectory = np.array([Dataset.get_progress(), Dataset.get_delta()]).T
        trajectory = raw_trajectory[begin:]
        trajectory = trajectory[:horizon]
        
        #Get prediction

        if(models == 1):
            prediction,y = model.predict(Dataset,1,begin-300,plot=False)
            prediction[:, 0] = np.cumsum(prediction[:, 0]) + raw_trajectory[begin-2, 0]

        elif(models ==2):
            if (scenario ==1)or(scenario ==2):
                track = np.loadtxt('../../centerline/map7.csv', delimiter=',')
            else:
                track = np.loadtxt('../../centerline/map6.csv', delimiter=',')
            optim = SplineOptimizer(track)
            optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)
            
            prediction= model.predict(Dataset,begin,horizon,optim)
            prediction[:, 0] = np.cumsum(prediction[:, 0]) + raw_trajectory[begin-2, 0]
        
        self.models = models
        self.horizon = horizon
        self.scenario = scenario
        self.model = model
        self.begin = begin
        self.dataset = Dataset
        self.raw_trajectory = raw_trajectory
        self.trajectory = trajectory
        self.prediction = prediction


    def DeltapMSE(self):
        ##Computing mean squared error
        Deltap_pred=np.reshape(self.prediction[:,0],(self.horizon))
        Deltap_true = np.reshape(self.trajectory[:,0],(self.horizon))
        summation = 0  #variable to store the summation of differences
        n = len(Deltap_pred)#finding total number of items in list
        for i in range (n):  #looping through each element of the list
            difference = Deltap_pred[i] - Deltap_true[i]  #finding the difference between observed and predicted value
            squared_difference = difference**2  #taking square of the differene 
            summation = summation + squared_difference  #taking a sum of all the differences
        MSE = summation/n  #dividing summation by total values to obtain average
        print(f'Delta progress MSE scenario {self.scenario} model {self.models}: {MSE}')
        return MSE

    def deltaMSE(self):
        ##Computing mean squared error
        
        Deltap_pred=np.reshape(self.prediction[:,1],(self.horizon))
        Deltap_true = np.reshape(self.trajectory[:,1],(self.horizon))
        summation = 0  #variable to store the summation of differences
        n = len(Deltap_pred)#finding total number of items in list
        for i in range (n):  #looping through each element of the list
            difference = Deltap_pred[i] - Deltap_true[i]  #finding the difference between observed and predicted value
            squared_difference = difference**2  #taking square of the differene 
            summation = summation + squared_difference  #taking a sum of all the differences
        MSE = summation/n  #dividing summation by total values to obtain average
        print(f'delta MSE scenario {self.scenario} model {self.models}: {MSE}')
        return MSE

    def standardMSE(self):
        #means and standard
        Deltapmean = np.mean(self.trajectory[:,0])
        Deltaps = np.std(self.trajectory[:,0])
        deltamean = np.mean(self.trajectory[:,1])
        deltas = np.std(self.trajectory[:,1])

        #true trajectory
        Deltap = np.reshape(((self.trajectory[:,0]-Deltapmean)/Deltaps),(self.horizon))
        delta = np.reshape(((self.trajectory[:,1]-deltamean)/deltas),(self.horizon))
        strajectory = np.column_stack((Deltap,delta))

        #predicted trajectory
        pDelta = np.reshape(((self.prediction[:,0]-Deltapmean)/Deltaps),(self.horizon))
        pdelta = np.reshape(((self.prediction[:,1]-deltamean)/deltas),(self.horizon))
        sprediction = np.column_stack((pDelta,pdelta))
        
        standardtrajectory = np.reshape(strajectory,(2*self.horizon))
        standardpred = np.reshape(sprediction,(2*self.horizon))
        summation = 0
        n = len(standardpred)#finding total number of items in list
        for i in range (n):  #looping through each element of the list
            difference = standardpred[i] - standardtrajectory[i]  #finding the difference between observed and predicted value
            squared_difference = difference**2  #taking square of the differene 
            summation = summation + squared_difference  #taking a sum of all the differences
        MSE = summation/n  #dividing summation by total values to obtain average
        print(f'MSE scenario {self.scenario} model {self.models}: {MSE}')
        return MSE


    def plot(self):
        plt.figure()
        plt.title('delta error')
        plt.plot((self.prediction[:,1]),label = 'Prediction')
        plt.plot((self.trajectory[:,1]),label = 'Trajectory')
        plt.legend()
        plt.show()
    