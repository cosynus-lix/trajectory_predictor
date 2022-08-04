from re import T
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from trajectory_predictor.dataset.SimpleDataset import SimpleDataset
from trajectory_predictor.model.Baseline.Baselinemodel import BaselineModel
from trajectory_predictor.model.Baseline.Baseline_preprocessing import Baselinepreprocessing
from trajectory_predictor.model.Improved_baseline.Improved_baseline_preprocessing import Improved_Baselinepreprocessing
from trajectory_predictor.model.Improved_baseline.Improved_baseline_model import Improved_BaselineModel


def main():
    past = 300
    horizon = 10
    epochs = 100
    dataset = SimpleDataset()
    dataset.load_data('../../centerline/map0.csv', '../../runs/run0/spline.npy', '../../runs/run0/history.npy')
    dataset.add_data('../../centerline/map1.csv', '../../runs/run1/spline.npy', '../../runs/run1/history.npy')
    dataset.add_data('../../centerline/map2.csv', '../../runs/run2/spline.npy', '../../runs/run2/history.npy')
    dataset.add_data('../../centerline/map3.csv', '../../runs/run3/spline.npy', '../../runs/run3/history.npy')
    dataset.add_data('../../centerline/map4.csv', '../../runs/run4/spline.npy', '../../runs/run4/history.npy')
    dataset.add_data('../../centerline/map5.csv', '../../runs/run6/spline.npy', '../../runs/run6/history.npy')
    dataset.add_data('../../centerline/map6.csv', '../../runs/run6/spline.npy', '../../runs/run6/history.npy')
    #print(dataset.to_np().shape)

    # Baseline model
    model = BaselineModel(past,64,horizon)
    model.train(dataset,epochs,plot = True)    
    model.save('../../experiments/Feedforward_model/model1')
    model.load('../../experiments/Feedforward_model/model1/Feedforward_model.pt')
    model.predict(dataset,3200,0,plot=True)

    # Improved baseline model )
    # model = Improved_BaselineModel(past,64)
    # model.train(dataset,epochs,True)
    # model.save('../../experiments/Feedforward_model/modeltimeseries')
    # model.load('../../experiments/Feedforward_model/modeltimeseries/Time_series_model.pt')
    #print(model.__sizeof__())
    #model.predict(dataset,30,10)

    ###Plot maps

    #map0 = mpimg.imread('../../maps/map0.png')
    #map1 = mpimg.imread('../../maps/map1.png')
    #map2 = mpimg.imread('../../maps/map2.png')
    #map3 = mpimg.imread('../../maps/map3.png')
    #map4 = mpimg.imread('../../maps/map4.png')
    #map5 = mpimg.imread('../../maps/map5.png')
    #map6 = mpimg.imread('../../maps/map6.png')
    #map7 = mpimg.imread('../../maps/map7.png')

    #plt.figure()
    #plt.title('Maps used in the experience')
    #plt.subplot(4,2,1)
    #plt.imshow(map0,cmap='Greys_r')
    #plt.title('Training map 1')
    #plt.axis('off')
    #plt.subplot(4,2,2)
    #plt.axis('off')
    #plt.imshow(map1,cmap='Greys_r')
    #plt.title('Training map 2')
    #plt.subplot(4,2,3)
    #plt.imshow(map2,cmap='Greys_r')
    #plt.axis('off')
    #plt.title('Training map 3')
    #plt.subplot(4,2,4)
    #plt.imshow(map3,cmap='Greys_r')
    #plt.axis('off')
    #plt.title('Training map 4')
    #plt.subplot(4,2,5)
    #plt.imshow(map4,cmap='Greys_r')
    #plt.axis('off')
    #plt.title('Training map 5')
    #plt.subplot(4,2,6)
    #plt.imshow(map5,cmap='Greys_r')
    #plt.axis('off')
    #plt.title('Training map 6')
    #plt.subplot(4,2,7)
    #plt.imshow(map6,cmap='Greys_r')
    ##plt.axis('off')
    #plt.title('Training map 7')
    #plt.subplot(4,2,8)
    #plt.imshow(map7,cmap='Greys_r')
    #plt.axis('off')
    #plt.title('Testing map')
    #plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    main()
