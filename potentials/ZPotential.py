import numpy as np
from potentials.General2DPotential import General2DPotential

class ZPotential(General2DPotential):
    """"Tripple well potential with two deep well and one less deep. The main path from one deep well to another
    for langevin dynamics switches with the temperature"""
    def __init__(self):
        super().__init__(np.array([[-7.19886062, -5.10043032]]),
                         0.5,
                         np.array([[7.19886062, 5.10043032]]),
                         0.5,
                         0.03,
                         [np.array([[-1.125, -1.01]]), np.array([[1.125, 1.01]]),
                          np.array([[7.20, -5.71]]), np.array([[-7.20, 5.71]])],
                         [-2.5, 2.5],
                         [-1.5, 2.5],
                         [-4,3],
                         100,
                         100)
        self.minimum_energy_paths = self.computeMEPs(n=5000)

    def V(self, X):
        """Potential fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: V: np.array, array of potential energy values
        """
        assert (type(X) == np.ndarray)
        assert (X.ndim == 2)
        assert (X.shape[1] == 2)
        a = - 3 * np.exp(- 0.01 * (X[:, 0] + 5) ** 2 - 0.2 * (X[:, 1] + 5) ** 2)
        b = - 3 * np.exp(- 0.01 * (X[:, 0] - 5) ** 2 - 0.2 * (X[:, 1] - 5) ** 2)
        c = + 5 * np.exp(- 0.20 * (X[:, 0] + 3 * (X[:, 1] - 3)) ** 2) / (1 + np.exp(- X[:, 0] - 3))
        d = + 5 * np.exp(- 0.20 * (X[:, 0] + 3 * (X[:, 1] + 3)) ** 2) / (1 + np.exp(+ X[:, 0] - 3))
        e = + 3 * np.exp(- 0.01 * (X[:, 0] ** 2 + X[:, 1] ** 2))
        f = (X[:, 0] ** 4 + X[:, 1] ** 4) / 20480
        V = a + b + c + d + e + f
        return V

    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVx: float, derivative of the potential with respect to x
        """
        a = + 0.06 * (x + 5) * np.exp(- 0.01 * (x + 5) ** 2 - 0.2 * (y + 5) ** 2)
        b = + 0.06 * (x - 5) * np.exp(- 0.01 * (x - 5) ** 2 - 0.2 * (y - 5) ** 2)
        d = + (5 / (1 + np.exp(- x - 3)) ** 2) * (
                    + np.exp(- x - 3) * np.exp(- 0.2 * (x + 3 * (y - 3)) ** 2) - 0.4 * (x + 3 * (y - 3)) * np.exp(
                -0.2 * (x + 3 * (y - 3)) ** 2) * (1 + np.exp(- x - 3)))
        c = + (5 / (1 + np.exp(+ x - 3)) ** 2) * (
                    - np.exp(+ x - 3) * np.exp(- 0.2 * (x + 3 * (y + 3)) ** 2) - 0.4 * (x + 3 * (y + 3)) * np.exp(
                -0.2 * (x + 3 * (y + 3)) ** 2) * (1 + np.exp(+ x - 3)))
        e = - 0.06 * x * np.exp(- 0.01 * (x ** 2 + y ** 2))
        f = (4 * x ** 3) / 20480
        dVx = a + b + c + d + e + f
        return dVx

    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVy: float, derivative of the potential with respect to y
        """
        a = + 1.2 * (y + 5) * np.exp(- 0.01 * (x + 5) ** 2 - 0.2 * (y + 5) ** 2)
        b = + 1.2 * (y - 5) * np.exp(- 0.01 * (x - 5) ** 2 - 0.2 * (y - 5) ** 2)
        c = - (5 / (1 + np.exp(- x - 3))) * 1.2 * (x + 3 * (y - 3)) * np.exp(- 0.2 * (x + 3 * (y - 3)) ** 2)
        d = - (5 / (1 + np.exp(+ x - 3))) * 1.2 * (x + 3 * (y + 3)) * np.exp(- 0.2 * (x + 3 * (y + 3)) ** 2)
        e = -  0.06 * y * np.exp(- 0.01 * (x ** 2 + y ** 2))
        f = (4 * y ** 3) / 20480
        dVy = a + b + c + d + e + f
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