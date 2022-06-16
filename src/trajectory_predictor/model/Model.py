import abc

class Model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self):
        """
        Train the model using 
        """
        return

    @abc.abstractmethod
    def save(self, path):
        """
        Saves the model to a given path
        """
        return

    @abc.abstractmethod
    def load(self):
        """
        Loads model from a given path
        """
        return

    @abc.abstractmethod
    def predict(self):
        """
        Predict the next deltas and variations in progresses
        """
        return