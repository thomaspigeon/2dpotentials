import numpy as np 
import math 

class GeneralPotential:
    """Class containing usefull functions for all 2D potentials"""
    def __init__(self, beta):
        """Initialise 

        Parameters
        ----------
        beta: np.float, 1/(k_B * T) where T is the temperature)"""
        self.beta = beta 
        self.MEPs = None

    def overdamped_trajectory(X_0, T=1000, save=1, seed=None, save_grad=False,  save_gauss=False, save_ener=False):    
        """Simulates an overdamped langevin trajectory with a Euler-Maruyama numerical scheme 
        
        Parameters
        ----------
        pot: potential object, must have methods for energy gradient and energy evaluation
        X_0: Initial position, must be a 2D vector
        T: Number of points in the trajectory (the total simulation time is therefore T * delta_t)
        save: Integer giving the frequency (counted in number of steps) at which the trajectory is saved
        seed: Integer, random number generator seed
        save_grad: Boolean parameter to save forces along the trajectory
        save_gauss: Boolean parameter to save the drawn gaussians along the trajectory
        save_ener: Boolean parameter to save energy along the trajectory

        Returns
        -------
        traj: np.array with ndim = 2 and shape = (T // save, 2)
        grad_traj: np.array with ndim = 2 and shape = (T // save, 2)
        gauss_traj: np.array with ndim = 2 and shape = (T // save, 2)
        ener_traj: np.array with ndim = 2 and shape = (T // save, 1)
        """
        r = np.random.RandomState(seed)
        X = X_0.reshape(1,2)
        dim = X.shape[1]
        traj = []
        if save_grad:
            grad_traj= []
        if save_gauss:
            gauss_traj = []
        if save_ever:
            ener_traj = []
        for i in range(T):
            b = r.normal(size=(dim,))
            X = X - self.nabla_V(X.reshape(1,2)) * self.delta_t + np.sqrt(2 * self.delta_t/self.beta) * b
            if i % save == 0:
                traj.append(X[0,:])
                if save_grad:
                    grad_traj.append(self.nabla_V(X.reshape(1,2)))
                if save_gauss:
                    gauss_traj.append(b)
                if save_ever:
                    ener_traj.append(self.V(X.reshape(1,2))
        if save_grad and not save_gauss and not save_ener:
            return np.array(traj), np.array(grad_traj)
        elif save_gauss and not save_grad and not save_ener:
            return np.array(traj), np.array(gauss_traj)
        elif not save_grad and not save_gauss and save_ener:
            return np.array(traj), np.array(ener_traj)
        elif save_grad and not save_gauss and save_ener:
            return np.array(traj), np.array(grad_traj), np.array(ener_traj)
        elif save_gauss and not save_grad and save_ener:
            return np.array(traj), np.array(gauss_traj), np.array(ener_traj)
        elif save_grad and save_gauss and not save_ener:
            return np.array(traj), np.array(grad_traj), np.array(gauss_traj)
        elif not save_gauss and not save_grad and not save_ener:
            return np.array(traj)
        else:
            return np.array(traj), np.array(grad_traj), np.array(gauss_traj), np.array(ener_traj)

    def computeMEPs(self, n=1000):
        """Compute the various minimum energy paths by gradient descent starting close to the 
        various saddle points of the potential.

        Returns
        -------
        MEPs: List of np.array with ndim = 2. each array is a MEP betwwen two local minima
        """
        MEPs = []
        for j in range(len(saddle_points)):
            X_0 = self.saddle_points[j]
            dX_0 = np.array([[self.dx, self.dx]])
            X_plus = [X_0, X_0 + dX_0]
            X_minus = [X_0, X_0 - dX_0]
            for i in range(n):
                X_plus.append(X_plus[-1] - self.nabla_V(X_plus[-1]) * self.dx)
                X_minus.append(X_minus[-1] - self.nabla_V(X_minus[-1]) * self.dx)
            X_plus = np.array(X_plus)
            X_minus = np.array(X_minus)
            MEPs.append(np.append(np.flip(X_minus, axis=0), X_plus, axis=0))
        return MEPs

    @property
    def set_MEPs(self):
        """Set and store the MEPs if not already done"""
        if not isinstance(self.MEPs, list):
            self.MEPs = self.computeMEPs()
            return self.MEPs
        
                

