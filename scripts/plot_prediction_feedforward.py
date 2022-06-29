import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.model.Basic_feedforward.BasicfeedforwardModel import BasicfeedforwardModel
from trajectory_predictor.model.Basic_feedforward.Feedforwardpreprocessing import Feedforwardpreprocessing
from tqdm import tqdm

def main():
    # Load model
    model = BasicfeedforwardModel(30,64,4)
    model.load('../experiments/Feedforward_model/model1/Feedforward_model.pt')


    # Load dataset
    
    dataset = Dataset()
    dataset.load_data('../centerline/map0.csv', '../runs/run0/spline.npy', '../runs/run0/history.npy')
    prediction,truevalues = model.predict(dataset,3318,0)
    delta = truevalues[:,1].T
    delta = np.delete(delta,slice(model.past))
    Deltas = truevalues[:,0].T
    Deltas = np.delete(Deltas,slice(model.past))
    print(prediction.shape)
    print(truevalues.shape)
    print(prediction[1][:])
    pred_Deltas = np.zeros(1)
    pred_delta = np.zeros(1)

    for k in range(1000):
            pred_Deltas = np.append(pred_Deltas,prediction[model.horizon*k][0])
            pred_delta = np.append(pred_delta,prediction[model.horizon*k][1])
    pred_Deltas = np.delete(pred_Deltas,0,0)
    pred_delta = np.delete(pred_delta,0,0)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Delta progress, prediction and reality')
    plt.plot(Deltas, color = 'red',label = 'true Delta progress')
    plt.plot(pred_Deltas, color = 'green',label = 'predicted Delta progress')  
    plt.subplot(2,1,2)
    plt.plot(delta, color = 'red',label = 'true delta')
    plt.plot(pred_delta, color = 'green',label = 'predicted delta')
    plt.legend()
    plt.title('dela, prediction and reality')
    plt.show()
    #print(f'y = {y}\n prediction = {prediction}')
        #series = np.append(series,prediction[0,1])
        #reality = np.append(reality,y[1])
        #print(f'series = {series}\n reality = {reality}')
    


if __name__ == "__main__":
    main()