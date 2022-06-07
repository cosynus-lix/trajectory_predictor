import numpy as np
import matplotlib.pyplot as plt

from SplineOptimizer import SplineOptimizer

def main():
    centerline = np.loadtxt('../track_generator/centerline/map0.csv', delimiter=',')
    optimizer = SplineOptimizer(centerline)
    optimizer.load_spline()

    trajectory = np.load('./history.npy')

    progresses = trajectory[:, 0]
    deltas = trajectory[:, 1]
    curvatures = [optimizer.k(progress) for progress in progresses]

    delta_progress = np.diff(progresses)

    # print(len(delta_progress), len(deltas))

    # exit()
    output = np.array([delta_progress, deltas[:-1], curvatures[:-1]]).T

    # Save output to file
    np.save('output.npy', output)
    print(output.shape)
    # exit()
    plt.plot(progresses, trajectory[:, 1], '-')
    # plt.plot(progresses[:-1], delta_progress)
    plt.plot(progresses, curvatures, '-')
    
    plt.show()
    print(trajectory)

if __name__ == "__main__":
    main()