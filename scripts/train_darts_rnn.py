from trajectory_predictor.model.DartsRNNModel import DartsRNNModel
from trajectory_predictor.dataset.Dataset import Dataset

def main():
    dataset = Dataset()
    dataset.load(f'/trajectory_predictor/datasets/test_datset')
    model = DartsRNNModel()
    model.train(dataset, 10)
    model.save('../experiments/model0')

if __name__ == "__main__":
    main()
