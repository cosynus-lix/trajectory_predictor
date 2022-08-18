import numpy as np
from tqdm import tqdm

from trajectory_predictor.dataset.Dataset import Dataset
from trajectory_predictor.controller.WallFollowerController import WallFollowerController
from trajectory_predictor.simulator.F1TenthSoloSimulator import F1TenthSoloSimulator
from trajectory_predictor.utils.generators import multiparameter_generator

def main():
    dataset = Dataset()
    base_map_path = '/trajectory_predictor/maps/map'
    n_maps = 100
    timestep = 0.03

    maps = [i for i in range(1, n_maps+8)]
    maps.remove(19)
    maps.remove(30)
    maps.remove(39)
    maps.remove(69)
    maps.remove(85)
    maps.remove(91)
    maps.remove(99)
    # remove 30

    controller_parameters = {
        'speed': np.linspace(2.5, 3, 20),
        'kp': [0.0001],
    }
    controllers = [controller for controller in multiparameter_generator(WallFollowerController, controller_parameters)]
    simulator_parameters = {
        'map_path': [f'{base_map_path}{i}' for i in maps],
        'controller': controllers,
        'timestep': [timestep],
        'max_track_width': [3.243796630159458],
        'verbose': [False],
    }
    simulators = [simulator for simulator in multiparameter_generator(F1TenthSoloSimulator, simulator_parameters)]

    print("Generating trajectories...")
    trajectory_list = []

    for simulator in tqdm(simulators):
        trajectory = simulator.run()
        final_progress = trajectory.get_final_progress()
        if final_progress < 0.99:
            print(f"Warning: trajectory did not reach goal and stopped at {final_progress}")
            continue
        trajectory_list.append(trajectory)

    print("Saving dataset...")
    dataset.add_data(trajectory_list)
    dataset_name = 'train_big'
    dataset.dump(f'/trajectory_predictor/datasets/{dataset_name}')
    # Add dataset description
    with open(f'/trajectory_predictor/datasets/{dataset_name}/description.txt', 'w') as f:
        f.write(f'{len(trajectory_list)} trajectories generated with {len(simulators)} simulators and {len(controllers)} controllers')
        f.write(f'\nTimestep: {timestep}')
        f.write(f'\nController parameters: {controller_parameters}')
        f.write(f'\nSimulator parameters: {simulator_parameters}')
        f.write(f'\nN maps: {n_maps}')

if __name__ == "__main__":
    main()
