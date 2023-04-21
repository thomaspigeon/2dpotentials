import numpy as np
from potentials.General2DPotential import General2DPotential

class MullerBrown(General2DPotential):
    """"Muller Brown potential from https://doi.org/10.1007/BF00547608"""
    def __init__(self):
        super().__init__(np.array([[-0.55822364, 1.44172584]]),
                         0.1,
                         np.array([[0.6234994, 0.02803776]]),
                         0.1,
                         0.0001,
                         [np.array([[-0.82505,  0.62072]]), np.array([[0.2120, 0.2927]])],
                         [-1.6, 1.2],
                         [-0.35, 2],
                         [-150, 0],
                         100,
                         100)
        self.minimum_energy_paths = self.computeMEPs()

    def V(self, x):
        """Compute potential energy of an arbitrary number of points

        :param x: np.array, ndim==2, shape==[any, 2]
        :return V(x): np.array, ndim==2, shape==[any, 1]"""
        a = -200 * np.exp(-1 * (x[:, 0] - 1) ** 2 + 0 * (x[:, 0] - 1) * (x[:, 1] - 0) - 10 * (x[:, 1] - 0) ** 2)
        b = -100 * np.exp(-1 * (x[:, 0] - 0) ** 2 + 0 * (x[:, 0] - 0) * (x[:, 1] - 0.5) - 10 * (x[:, 1] - 0.5) ** 2)
        c = -170 * np.exp(
            -6.5 * (x[:, 0] + 0.5) ** 2 + 11 * (x[:, 0] + 0.5) * (x[:, 1] - 1.5) - 6.5 * (x[:, 1] - 1.5) ** 2)
        d = 15 * np.exp(0.7 * (x[:, 0] + 1) ** 2 + 0.6 * (x[:, 0] + 1) * (x[:, 1] - 1) + 0.7 * (x[:, 1] - 1) ** 2)
        return a + b + c + d

    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate
        :return: dVx: float, derivative of the potential with respect to x
        """
        a = -200 * (2 * (-1) * (x - 1) + 0 * (y - 0)) * np.exp(
            -1 * (x - 1) ** 2 + 0 * (x - 1) * (y - 0) - 10 * (y - 0) ** 2)
        b = -100 * (2 * (-1) * (x - 0) + 0 * (y - 0.5)) * np.exp(
            -1 * (x - 0) ** 2 + 0 * (x - 0) * (y - 0.5) - 10 * (y - 0.5) ** 2)
        c = -170 * (2 * (-6.5) * (x + 0.5) + 11 * (y - 1.5)) * np.exp(
            -6.5 * (x + 0.5) ** 2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5) ** 2)
        d = 15 * (2 * (0.7) * (x + 1) + 0.6 * (y - 1)) * np.exp(
            0.7 * (x + 1) ** 2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1) ** 2)
        dVx = a + b + c + d
        return dVx

    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate
        :return: dVy: float, derivative of the potential with respect to y
        """
        a = -200 * (2 * (-10) * (y - 0) + 0 * (x - 1)) * np.exp(
            -1 * (x - 1) ** 2 + 0 * (x - 1) * (y - 0) - 10 * (y - 0) ** 2)
        b = -100 * (2 * (-10) * (y - 0.5) + 0 * (x - 0)) * np.exp(
            -1 * (x - 0) ** 2 + 0 * (x - 0) * (y - 0.5) - 10 * (y - 0.5) ** 2)
        c = -170 * (2 * (-6.5) * (y - 1.5) + 11 * (x + 0.5)) * np.exp(
            -6.5 * (x + 0.5) ** 2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5) ** 2)
        d = 15 * (2 * (0.7) * (y - 1) + 0.6 * (x + 1)) * np.exp(
            0.7 * (x + 1) ** 2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1) ** 2)
        dVy = a + b + c + d
        return dVy

    def nabla_V(self, X):
        """Gradient of the potential energy fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: grad(X): np.array, array of gradients with respect to position vector (x,y), ndim = 2, shape = (,2)
        """
        assert (type(X) == np.ndarray)
        assert (X.ndim == 2)
        assert (X.shape[1] == 2)
        return np.column_stack((self.dV_x(X[:, 0], X[:, 1]), self.dV_y(X[:, 0], X[:, 1])))



