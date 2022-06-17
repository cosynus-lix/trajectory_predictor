from trajectory_predictor.model.DartsRNNModel import DartsRNNModel
from trajectory_predictor.dataset.Dataset import Dataset

def main():
    dataset = Dataset()
    dataset.load_data('../centerline/map0.csv', '../runs/run0/spline.npy', '../runs/run0/history.npy')
    model = DartsRNNModel()
    model.train(dataset)
    model.save('../experiments/model0')

if __name__ == "__main__":
    main()
