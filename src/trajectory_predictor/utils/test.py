import numpy as np

out = np.load('output.npy')

covariates_np = out[:, 2]
series_np = out[:, :-1]

from darts import TimeSeries

covariates = TimeSeries.from_values(covariates_np)
series = TimeSeries.from_values(series_np)

from darts.models import RNNModel

rnn_rain = RNNModel(input_chunk_length=200, 
                    training_length=500, 
                    n_rnn_layers=2)

rnn_rain.fit(series, 
             future_covariates=covariates, 
             epochs=1, 
             verbose=True)

from darts.metrics import rmse

# We first set aside the first 80% as training series:
flow = series
flow_train, _ = flow.split_before(0.8)

def eval_model(model, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests
    
    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    prediction = model.predict(820,
                              series=flow.split_before(0.5)[0],
                              past_covariates=past_covariates,
                              future_covariates=future_covariates)
    
    flow.plot()
    prediction.plot(label='backtest (n=1000)')
    print('Backtest RMSE = {}'.format(rmse(flow, prediction)))
eval_model(rnn_rain, 
           future_covariates=covariates)
# Save matplotlib figure as png
import matplotlib.pyplot as plt
plt.savefig('predtion.png')
print(series)