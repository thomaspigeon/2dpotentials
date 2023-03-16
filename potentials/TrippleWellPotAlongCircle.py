import numpy as np
from potentials.General2DPotential import General2DPotential

class DoubleWellAlongCircle(General2DPotential):
    """"Tripple well potential on a circle. A parameter epsilon allows to change the spread around the circle."""
    def __init__(self, epsilon):
        super().__init__(np.array([[-0.50736758, -0.87878643]]),
                         0.1,
                         np.array([[-0.50736758, 0.87878643]]),
                         0.1,
                         0.001,
                         [np.array([[0.505 ,  0.875]]), np.array([[0.505, -0.875]])],
                         [-2, 2],
                         [-2, 2],
                         [0, 5],
                         100,
                         100)
        self.eps = epsilon
        self.minimum_energy_paths = self.computeMEPs()

    def V(self, x):
        """Compute potential energy of an arbitrary number of points

        :param x: np.array, ndim==2, shape==[any, 2]
        :return V(x): np.array, ndim==2, shape==[any, 1]"""
        # compute angle in [-pi, pi]
        theta = np.arctan2(x[:, 1], x[:, 0])
        # compute radius
        r = np.sqrt(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1])

        v_vec = np.zeros(len(x))
        for idx in range(len(x)):
            # potential V_1
            if theta[idx] > np.pi / 3:
                v_vec[idx] = (1 - (theta[idx] * 3 / np.pi - 1.0) ** 2) ** 2
            if theta[idx] < - np.pi / 3:
                v_vec[idx] = (1 - (theta[idx] * 3 / np.pi + 1.0) ** 2) ** 2
            if theta[idx] > - np.pi / 3 and theta[idx] < np.pi / 3:
                v_vec[idx] = 3.0 / 5.0 - 2.0 / 5.0 * np.cos(3 * theta[idx])
            # potential V_2
        v_vec = v_vec * 1.0 + (r - 1) ** 2 * 1.0 / self.eps + 5.0 * np.exp(-5.0 * r ** 2)
        return v_vec

    def nabla_V(self, x):
        """Gradient of the potential energy fuction

        :param x: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return grad(X): np.array, array of gradients with respect to position vector (x,y), ndim = 2, shape = (,2)"""
        # angle
        theta = np.arctan2(x[:, 1], x[:, 0])
        # radius
        r = np.sqrt(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1])
        if any(np.fabs(r) < 1e-8):
            print("warning: radius is too small! r=%.4e" % r)
        dv1_dangle = np.zeros(len(x))
        # derivative of V_1 w.r.t. angle
        for idx in range(len(x)):
            if theta[idx] > np.pi / 3:
                dv1_dangle[idx] = 12 / np.pi * (theta[idx] * 3 / np.pi - 1) * (
                            (theta[idx] * 3 / np.pi - 1.0) ** 2 - 1)
            if theta[idx] < - np.pi / 3:
                dv1_dangle[idx] = 12 / np.pi * (theta[idx] * 3 / np.pi + 1) * (
                            (theta[idx] * 3 / np.pi + 1.0) ** 2 - 1)
            if theta[idx] > -np.pi / 3 and theta[idx] < np.pi / 3:
                dv1_dangle[idx] = 1.2 * np.sin(3 * theta[idx])
        # derivative of V_2 w.r.t. angle
        dv2_dangle = np.zeros(len(x))
        # derivative of V_2 w.r.t. radius
        dv2_dr = 2.0 * (r - 1.0) / self.eps - 50.0 * r * np.exp(-r ** 2 / 0.2)
        return np.column_stack((-(dv1_dangle + dv2_dangle) * x[:, 1] / (r * r) + dv2_dr * x[:, 0] / r,
                                (dv1_dangle + dv2_dangle) * x[:, 0] / (r * r) + dv2_dr * x[:, 1] / r))
