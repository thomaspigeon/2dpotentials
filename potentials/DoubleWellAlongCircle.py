import numpy as np
from potentials.General2DPotential import General2DPotential

class DoubleWellAlongCircle(General2DPotential):
    """"Double well potential on a circle such that two paths symmetric with respect to y = 0 link the two metastable
    states. A parameter epsilon allows to change the spread around the circle."""
    def __init__(self, epsilon):
        super().__init__(np.array([[-1., 0.]]),
                         0.1,
                         np.array([[1., 0.]]),
                         0.1,
                         0.01,
                         [np.array([[0, 0.95]]), np.array([[0, -0.95]])],
                         [-1.5, 1.5],
                         [-1.5, 1.5],
                         [0, 5],
                         100,
                         100)
        self.eps = epsilon
        self.minimum_energy_paths = self.computeMEPs()

    def V(self, x):
        """Compute potential energy of an arbitrary number of points

        :param x: np.array, ndim==2, shape==[any, 2]
        :return V(x): np.array, ndim==2, shape==[any, 1]"""
        return 2.0 * x[:, 1] ** 2 + 1.0 / self.eps * (x[:, 0] ** 2 + x[:, 1] ** 2 - 1) ** 2

    def nabla_V(self, x):
        """Gradient of the potential energy fuction

        :param x: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return grad(X): np.array, array of gradients with respect to position vector (x,y), ndim = 2, shape = (,2)"""
        return np.column_stack((4.0 * x[:, 0] / self.eps * (x[:, 0] ** 2 + x[:, 1] ** 2 - 1),
                                4.0 * x[:, 1] + 4.0 * x[:, 1] / self.eps * (x[:, 0] ** 2 + x[:, 1] ** 2 - 1)))
