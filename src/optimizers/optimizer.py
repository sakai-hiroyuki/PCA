from time import time
from abc import ABCMeta, abstractmethod


class Optimizer(object, metaclass=ABCMeta):
    def __init__(self):
        pass
    
    @abstractmethod
    def optimize(self):
        pass

    def logging(self, v: float):
        if hasattr(self, 'history'):
            toc = time()
            self.history.append([v, toc - self.tic])
        else:
            self.tic = time()
            self.history = [[v, 0.]]
