import matplotlib.pyplot as plt
import numpy as np

from trajectory_predictor.model.DartsRNNModel import DartsRNNModel
from trajectory_predictor.dataset.Dataset import Dataset

def main():
    # Load model
    model = DartsRNNModel()
    model.load('../experiments/model0')

    # Load dataset
    dataset = Dataset()
    dataset.load_data('../centerline/map0.csv', '../runs/run0/spline.npy', '../runs/run0/history.npy')

    # Get series to predict
    data_np = dataset.to_np()
    point = 1500
    series = data_np[:point, :-1]
    curvatures = data_np[:, 2]
    prediction = model.predict(series, curvatures, 800)

    plt.plot(data_np[:, 1])
    plt.plot(np.arange(point, point+len(prediction)), prediction[:, 1])

    # Plot to image
    plt.savefig('./prediction.png')
    print(prediction)

if __name__ == "__main__":
    main()