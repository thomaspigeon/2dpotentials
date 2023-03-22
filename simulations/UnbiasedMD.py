import numpy as np


class Simulation:
    """A general class for unbiased trajectories"""
    def __init__(self, pot, beta, dt=None, seed=None):
        """

        :param pot:     A potential object from 2dpotentials/potentials
        :param beta:    float, inverse thermal energy 1/(k_B*T)
        :param dt:      float, time step should be taken according to the potential used, if not given, the default dx
                        of potential is used
        :param seed:    int, random number generator seed
        """
        self.pot = pot
        self.beta = beta
        if dt is None:
            self.dt = pot.dx
        self.seed = seed
        self.r = np.random.RandomState(self.seed)

    def set_seed(self, seed):
        """Allows to reset the seed.

        :param seed:
        """
        self.seed = seed
        self.r = np.random.RandomState(self.seed)

    def set_dt(self, dt):
        """Allows to reset the time step dt.

        :param seed:
        """
        self.dt = dt


class OverdampedLangevin(Simulation):
    """Class for generating unbiased Overdamped Langevin trajectories"""
    def __init__(self, pot, beta, dt=None, seed=None):
        """

        :param pot:     A potential object from 2dpotentials/potentials
        :param beta:    float, inverse thermal energy 1/(k_B*T)
        :param dt:      float, time step should be taken according to the potential used, if not given, the default dx
                        of potential is used
        :param seed:    int, random number generator seed
        """
        super().__init__(pot, beta, dt=dt, seed=seed)

    def step(self, x):
        """One step of integration of the overdamped langevin dynamics discretized with the Euler-Maruyama scheme

        :param x:       np.array, ndim==2, shape==[1, 2], position
        :return: x:     np.array, ndim==2, shape==[1, 2], next position
        :return: grad:  np.array, ndim==2, shape==[1, 2], forces acting on x
        :return: gauss: np.array, ndim==2, shape==[1, 2], gaussian drawn from r
        """
        grad = self.pot.nabla_V(x)
        gauss = self.r.normal(size=(x.shape[1]))
        x = x - grad * self.dt + np.sqrt(2 * self.dt / self.beta) * gauss
        return x, grad, gauss

    def run(self, x_0, n_time_steps, save_grad=False, save_gauss=False):
        """ Runs and unbiased dynamics for

        :param x_0:             np.array, ndim==2, shape==[1, 2], initial position of the dynamics
        :param n_time_steps:    int, number of time steps.
        :param save_grad:       boolean, whether the forces should be saved
        :param save_gauss:      boolean, whether the gaussian should be saved
        :return: trajectory     dict, trajectory["x_traj"] is the trajectory of the positions, trajectory["grad_traj"]
                                is the trajectory of the forces if required and trajectory["gauss_traj"] is the
                                trajectory of the gaussians if required
        """
        trajectory = {}
        x_traj = []
        x = x_0
        if not save_grad and not save_gauss:
            for i in range(n_time_steps):
                x, _, _ = self.step(x)
                x_traj.append(x)
            trajectory["x_traj"] = np.array(x_traj)
            return trajectory
        if save_grad and not save_gauss:
            grad_traj = []
            for i in range(n_time_steps):
                x, grad, _ = self.step(x)
                x_traj.append(x)
                grad_traj.append(grad)
            trajectory["x_traj"] = np.array(x_traj)
            trajectory["grad_traj"] = np.array(grad_traj)
            return trajectory
        if not save_grad and save_gauss:
            gauss_traj = []
            for i in range(n_time_steps):
                x, _, gauss = self.step(x)
                x_traj.append(x)
                gauss_traj.append(gauss)
            trajectory["x_traj"] = np.array(x_traj)
            trajectory["gauss_traj"] = np.array(gauss_traj)
            return trajectory
        if save_grad and save_gauss:
            grad_traj = []
            gauss_traj = []
            for i in range(n_time_steps):
                x, grad, gauss = self.step(x)
                x_traj.append(x)
                grad_traj.append(grad)
                gauss_traj.append(gauss)
            trajectory["x_traj"] = np.array(x_traj)
            trajectory["grad_traj"] = np.array(grad_traj)
            trajectory["gauss_traj"] = np.array(gauss_traj)
            return trajectory

class Langevin(Simulation):
    """Class for generating unbiased Overdamped Langevin trajectories"""
    def __init__(self, pot, beta, M, gamma, dt=None, seed=None):
        """

        :param pot:     A potential object from 2dpotentials/potentials
        :param beta:    float, inverse thermal energy 1/(k_B*T)
        :param M:       np.array, ndim==2, shape==[1, 2], mass
        :param gamma:   np.array, ndim==2, shape==[1, 2], friction
        :param dt:      float, time step should be taken according to the potential used, if not given, the default dx
                        of potential is used
        :param seed:    int, random number generator seed
        """
        super().__init__(pot, beta, dt=dt, seed=seed)
        self.M = M
        self.gamma = gamma

    def step(self, x, p):
        """

        :param x:       np.array, ndim==2, shape==[1, 2], position
        :param p:       np.array, ndim==2, shape==[1, 2], momentum
        :param r:       np.random.RandomState
        :return: x:     np.array, ndim==2, shape==[1, 2], next position
        :return: grad:  np.array, ndim==2, shape==[1, 2], forces acting on x
        :return: gauss: np.array, ndim==2, shape==[1, 2], gaussian drawn from r
        """
        gauss = self.r.normal(size=(x.shape[1]))
        p = p + (self.dt / 2) * self.pot.nabla_V(x)
        x = x + (self.dt / 2) * p / self.M
        p = np.exp(- self.gamma * self.dt / self.M) * p + \
            np.sqrt((1 - np.exp(- 2 * self.gamma * self.dt / self.M)) / self.beta) * gauss
        x = x + (self.dt / 2) * p / self.M
        grad = self.pot.nabla_V(x)
        p = p + (self.dt / 2) * grad
        return x, p, grad, gauss

    def run(self, x_0, p_0, n_time_steps, save_grad=False, save_gauss=False):
        """

        :param x_0:             np.array, ndim==2, shape==[1, 2], initial position of the dynamics
        :param p_0:             np.array, ndim==2, shape==[1, 2], initial momentum of the dynamics
        :param n_time_steps:    int, number of time steps.
        :param save_grad:       boolean, whether the forces should be saved
        :param save_gauss:      boolean, whether the gaussian should be saved
        :return: trajectory     dict, trajectory["x_traj"] is the trajectory of the positions, trajectory["p_traj"] is
                                the trajectory of the momenta, trajectory["grad_traj"] is the trajectory of the forces
                                if required and trajectory["gauss_traj"] is the trajectory of the gaussians if required
        """
        trajectory = {}
        x_traj = []
        p_traj = []
        x = x_0
        p = p_0
        if not save_grad and not save_gauss:
            for i in range(n_time_steps):
                x, p, _, _ = self.step(x, p)
                x_traj.append(x)
                p_traj.append(p)
            trajectory["x_traj"] = np.array(x_traj)
            trajectory["p_traj"] = np.array(p_traj)
            return trajectory
        if save_grad and  not save_gauss:
            grad_traj = []
            for i in range(n_time_steps):
                x, p, grad, _ = self.step(x, p)
                x_traj.append(x)
                p_traj.append(p)
                grad_traj.append(grad)
            trajectory["x_traj"] = np.array(x_traj)
            trajectory["p_traj"] = np.array(p_traj)
            trajectory["grad_traj"] = np.array(grad_traj)
            return trajectory
        if not save_grad and save_gauss:
            gauss_traj = []
            for i in range(n_time_steps):
                x, p, _, gauss = self.step(x, p)
                x_traj.append(x)
                p_traj.append(p)
                gauss_traj.append(gauss)
            trajectory["x_traj"] = np.array(x_traj)
            trajectory["p_traj"] = np.array(p_traj)
            trajectory["gauss_traj"] = np.array(gauss_traj)
            return trajectory
        if save_grad and save_gauss:
            grad_traj = []
            gauss_traj = []
            for i in range(n_time_steps):
                x, p, grad, gauss = self.step(x, p)
                x_traj.append(x)
                p_traj.append(p)
                grad_traj.append(grad)
                gauss_traj.append(gauss)
            trajectory["x_traj"] = np.array(x_traj)
            trajectory["p_traj"] = np.array(p_traj)
            trajectory["grad_traj"] = np.array(grad_traj)
            trajectory["gauss_traj"] = np.array(gauss_traj)
            return trajectory

