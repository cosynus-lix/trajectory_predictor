import numpy as np
from .Controller import Controller

class WallFollowerController(Controller):
    def __init__(self, speed=5, kp=0.0001):
        self.kp = kp
        self.speed = speed

    def get_control(self, obs):
        ranges = np.array([x if x < 100 else 0 for x in obs['scans'][0]])
        delta = np.sum(ranges[:len(ranges)//2]-ranges[len(ranges)//2:])
        control = delta*self.kp
        angle = max(min(control, np.pi/2), -np.pi/2)
        steer = -angle
        return steer, self.speed