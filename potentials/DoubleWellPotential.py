import numpy as np
from potentials.General2DPotential import General2DPotential

class DoubleWellPotential(General2DPotential):
    """"Double well potential separated in the x direction. A parameter epsilon allows to change the spread in the
    y direction. The smaller epsilon, the more it is spread."""
    def __init__(self, epsilon):
        super().__init__(np.array([[-1., 0.]]),
                         0.1,
                         np.array([[1., 0.]]),
                         0.1,
                         0.01,
                         [np.array([[0, 0]])],
                         [-2.5, 2.5],
                         [-1.5, 1.5],
                         [0, 5],
                         100,
                         100)
        self.eps = epsilon
        self.y_domain = [self.y_domain[0]/self.eps, self.y_domain[1]/self.eps]
        self.minimum_energy_paths = self.computeMEPs()

    def V(self, X):
        """Potential fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: V: np.array, array of potential energy values
        """
        assert (type(X) == np.ndarray)
        assert (X.ndim == 2)
        assert (X.shape[1] == 2)
        a = (X[:, 0] ** 2 - 1) ** 2
        b = (self.eps * X[:, 1]) ** 4
        V = a + b
        return V

    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVx: float, derivative of the potential with respect to x
        """
        a = 4 * x * (x ** 2 - 1)
        b = 0
        dVx = a + b
        return dVx

    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVy: float, derivative of the potential with respect to y
        """
        a = 0
        b = self.eps ** 4 * 4 * y ** 3

        dVy = a + b
        return dVy

    def nabla_V(self, X):
        """Gradient of the potential energy fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: grad(X): np.array, array of gradients with respect to position vector (x,y), ndim = 2, shape = (,2)
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)
        return np.column_stack((self.dV_x(X[:, 0], X[:, 1]), self.dV_y(X[:, 0], X[:, 1])))