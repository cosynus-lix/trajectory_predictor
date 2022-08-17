import numpy as np
import prettytable

class TrajectoryEvaluator:
    def __init__(self):
        pass

    def evaluate(reference_trajectory, model, horizons=[10, 20, 50, 100, 400]):
        predict_progresses = np.linspace(.25, 0.5, num=10)
        
        mae_average_list = None
        me_average_list = None
        for progress in predict_progresses:
            mae_list, me_list = TrajectoryEvaluator.evaluate_for_progress(reference_trajectory, 
                model, horizons, progress)
            if mae_average_list is None:
                mae_average_list = mae_list
                me_average_list = me_list
            else:
                for i in range(len(mae_list)):
                    mae_average_list[i] = np.vstack((mae_average_list[i], mae_list[i]))
                    me_average_list[i] = np.vstack((me_average_list[i], me_list[i]))
        for i in range(len(mae_average_list)):
            mae_average_list[i] = np.mean(mae_average_list[i], axis=0)
            me_average_list[i] = np.mean(me_average_list[i], axis=0)
        table = prettytable.PrettyTable(['Horizon', 'MAE (s)', 'ME (s)', 'MAE (delta)', 'ME (delta)'])
        for i in range(len(mae_average_list)):
            table.add_row([horizons[i]*reference_trajectory.get_dt(), mae_average_list[i][0], me_average_list[i][0], 
                mae_average_list[i][1], me_average_list[i][1]])
        return str(table)


    def evaluate_for_progress(reference_trajectory, model, horizons, predict_progress):
        trajectory = reference_trajectory.slice_time(end=predict_progress)
        trajectory_len = len(trajectory.get_history())
        max_horizon = max(horizons)
        prediction = model.predict(trajectory, max_horizon)

        mae_list = []
        me_list = []
        for horizon in horizons:
            history_prediction = prediction.get_history()[0: horizon]
            history_reference = reference_trajectory.get_history()[trajectory_len:trajectory_len+horizon]
            mae = np.mean(np.abs(history_prediction - history_reference), axis=0)
            me = np.max(np.abs(history_prediction - history_reference), axis=0)
            mae_list.append(mae)
            me_list.append(me)
        return mae_list, me_list
        