import abc

class Controller(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def get_control(self, obs):
      """
      Given an observation from the environment returns the control value

      obs: observation
      returns: control value (steering, speed)
      """
      return