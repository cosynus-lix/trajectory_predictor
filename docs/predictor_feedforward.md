# Feedforward Predictor

Pierre PEAUCELLE

Two models are available : baseline model, and improved baseline. 
This model are used to be implemented 4 scenarios (long or short prediction, map known or not)
 

## Plot model's prediction

To generate and plot a a model, for instance the Baseline model

```
python plot_baseline_scenario1.py
```
If you have already train and save the model once, you can comment the lines starting by model.train and model.save

If you want to plot scenario 3 or 4, whatever the model, you need do create a run for map 6 with a speed of 1 previously, that you will name run6speed1

## Plot model's metrics

To compare the results of two models

```
python baseline_metrics.py
```
