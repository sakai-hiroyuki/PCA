import numpy as np
import numpy.linalg as la
from math import sqrt


class Stiefel(object):
    def __init__(self, p: int, n: int):
        self._p = p
        self._n = n

    def __str__(self):
        return 'Stiefel Manifold St({0},{1})'.format(self._p, self._n)

    @property
    def dim(self):
        return self._n * self._p - self._p * (self._p + 1) / 2
    
    def retraction(self, x: np.ndarray, xi: np.ndarray) -> np.ndarray:
        q = self._qf(x + xi)
        return q

    def projection(self, x: np.ndarray, xi: np.ndarray) -> np.ndarray:
        return xi - np.dot(x, np.dot(x.T, xi))

    def _sym(self, M: np.ndarray) -> np.ndarray:
        return (M + M.T) / 2

    def _qf(self, M: np.ndarray) -> np.ndarray:
        return la.qr(M)[0]
