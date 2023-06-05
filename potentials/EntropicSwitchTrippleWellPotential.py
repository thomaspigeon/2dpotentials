import numpy as np
from potentials.General2DPotential import General2DPotential

def g(a):
    """Gaussian function

    :param a: np.array
    :return: float, g(a)
    """
    return np.exp(- a ** 2)
class EntropicSwitchTrippleWellPotential(General2DPotential):
    """"Tripple well potential with two deep well and one less deep. The main path from one deep well to another
    for langevin dynamics switches with the temperature"""
    def __init__(self):
        super().__init__(np.array([[-1.04805499, -0.04209367]]),
                         0.1,
                         np.array([[1.04805499, -0.04209367]]),
                         0.1,
                         0.001,
                         [np.array([[0., -0.315]]), np.array([[-0.6115, 1.0985]]), np.array([[0.6115, 1.0985]])],
                         [-2.5, 2.5],
                         [-1.5, 2.5],
                         [-4,3],
                         100,
                         100)
        self.minimum_energy_paths = self.computeMEPs()

    def V(self, X):
        """Compute potential energy of an arbitrary number of points

        :param x: np.array, ndim==2, shape==[any, 2]
        :return V(x): np.array, ndim==2, shape==[any, 1]"""
        assert (type(X) == np.ndarray)
        assert (X.ndim == 2)
        assert (X.shape[1] == 2)
        x = X[:, 0]
        y = X[:, 1]
        u = g(x) * (g(y - 1 / 3) - g(y - 5 / 3))
        v = g(y) * (g(x - 1) + g(x + 1))
        V = 3 * u - 5 * v + 0.2 * (x ** 4) + 0.2 * ((y - 1 / 3) ** 4)
        return V

    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVx: float, derivative of the potential with respect to x
        """
        u = g(x) * (g(y - 1 / 3) - g(y - 5 / 3))
        a = g(y) * ((x - 1) * g(x - 1) + (x + 1) * g(x + 1))
        dVx = -6 * x * u + 10 * a + 0.8 * (x ** 3)
        return dVx

    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVy: float, derivative of the potential with respect to y
        """
        u = g(x) * ((y - 1 / 3) * g(y - 1 / 3) - (y - 5 / 3) * g(y - 5 / 3))
        b = g(y) * (g(x - 1) + g(x + 1))
        dVy = -6 * u + 10 * y * b + 0.8 * ((y - 1 / 3) ** 3)
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