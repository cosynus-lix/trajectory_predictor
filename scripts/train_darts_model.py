from trajectory_predictor.model.DartsFutureCovariatesModel import DartsFutureCovariatesModel
from darts.models import RNNModel
from darts.models import TFTModel
from darts.models import TransformerModel
from trajectory_predictor.dataset.Dataset import Dataset

def main():
    dataset = Dataset()
    dataset.load(f'/trajectory_predictor/datasets/test_dataset')
    # model_class = TFTModel
    model_class = RNNModel
    darts_model = RNNModel(input_chunk_length=20, 
                    training_length=50, 
                    n_rnn_layers=2)
    # darts_model = model_class(input_chunk_length=20, 
    #                             output_chunk_length=50)
    model = DartsFutureCovariatesModel(darts_model, model_class)
    train_args = {'epochs': 1}
    model.train(dataset, **train_args)
    model.save('../experiments/model0')

if __name__ == "__main__":
    main()
