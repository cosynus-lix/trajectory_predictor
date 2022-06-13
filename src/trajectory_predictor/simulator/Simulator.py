import abc

class Simulator(metaclass=abc.ABCMeta):
    def __init__(self, env):
        self.env = env

    @abc.abstractmethod
    def _step(self):
        pass
    