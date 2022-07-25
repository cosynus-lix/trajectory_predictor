import numpy as np
from .Controller import Controller

class MedialAxisFollowerController(Controller):
    def __init__(self, speed=0.3, kp=3):
        self.kp = kp
        self.speed = speed

    def get_control(self, obs):
        ranges = obs['scans'][0]
        len_scans = len(ranges)
        delta = np.min(ranges[:len_scans//4])-np.min(ranges[-len_scans//4:])
        control = delta*self.kp
        angle = max(min(control, np.pi/2), -np.pi/2)
        steer = angle
        return steer, self.speed