from locale import normalize
import time

import gym
import numpy as np
import shapely.geometry as shp

from pyglet.gl import GL_POINTS

from CurvatureFinder import CurvatureFinder

###############################################
# Generates a trajectory using wall following #
###############################################

def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        # track = np.loadtxt('../track_generator/centerline/map0.csv', delimiter=',')
        # for i in range(track.shape[0]):
        #     b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [track[i, 0]*50, track[i, 1]*50, 0.]),
        #                         ('c3B/stream', [183, 193, 222]))
                # self.drawn_waypoints.append(b)
            # else:
            #     pass
                # self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

def main():
    N_LAPS = 1
    DISPLAY = True
    env = gym.make('f110_gym:f110-v0', map='../track_generator/maps/map0', map_ext='.pgm', num_agents=1)
    env.add_render_callback(render_callback)
    # Rad csv file with numpy
    track = np.loadtxt('../track_generator/centerline/map0.csv', delimiter=',')
    track_ring = shp.LinearRing(track)
    cf = CurvatureFinder(track)
    
    obs, step_reward, done, info = env.reset(np.array([[0, 0, 1.37]]))
    if DISPLAY:
        env.render()

    laptime = 0.0
    start = time.time()
    speed, steer = 5,0
    progress = 0
    history = []

    while not done:
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        if obs['lap_counts'][0] >= N_LAPS:
            done = True
        
        # Record history
        current_position = shp.Point(obs['poses_x'][0], obs['poses_y'][0])
        progress = track_ring.project(current_position, normalized=True)
        projection = track_ring.interpolate(progress, normalized=True)
        projection_before = track_ring.interpolate(progress-0.001, normalized=True)
        ccw = shp.LinearRing([projection_before, projection, current_position]).is_ccw
        if ccw:
            distance = -projection.distance(current_position)
        else:
            distance = projection.distance(current_position)
        history.append([progress, distance])
        print(f's: {progress:.2f}, delta:{distance:.2f}, k:{cf.k(progress):.2f}')

        # Control
        ranges = np.array([x if x < 100 else 0 for x in obs['scans'][0]])
        delta = np.sum(ranges[:len(ranges)//2]-ranges[len(ranges)//2:])
        control = delta*0.0001
        angle = max(min(control, np.pi/2), -np.pi/2)
        steer = -angle

        laptime += step_reward
        if DISPLAY:
            env.render(mode='human')
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
    np.save('history.npy', np.array(history))

if __name__ == '__main__':
    main()