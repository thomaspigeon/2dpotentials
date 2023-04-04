import numpy as np


class General2DPotential:
    """Class containing useful functions for all 2D potentials.

    It contains the definition of a state R and a state P. They are defined as discs of radius R_radius (resp. P_radius)
    and center minR (resp. minP). A surface enclosing R and P states is also defined through a function above_SigmaR
    (resp. above_SigmaP) that measures if the distance to minR (resp. minP) is greater than SigmaR_position * R_radius
    (resp. SigmaP_position * P_radius)."""
    def __init__(self, minR, R_radius, minP, P_radius, dx, saddle_points, x_domain, y_domain, V_domain, n_bins_x, n_bins_y):
        """Initialise the class

        :param minR:             np.array with ndim==2, shape==[1,2], center of the disc defining the R state
        :param R_radius:         float, radius of the disc defining the R state
        :param minP:             np.array with ndim==2, shape==[1,2], center of the disc defining the P state
        :param P_radius:         float, radius of the disc defining the P state
        :param dx:               step for the gradient descent to compute minimum energy path starting from first order
                                 saddle point
        :param saddle_points:    list, len==any, list of np.array with ndim==2, shape==[1,2], list of first order saddle
                                 points
        :param x_domain:         list of two int, min and max value of x for the plots
        :param y_domain:         list of two int, min and max value of y for the plots
        :param V_domain:         list of two int, min and max value of the potential for the heat maps plot
        :param n_bins_x:         int, number of bins in the x direction for the 2D plots
        :param n_bins_y:         int, number of bins in the y direction for the 2D plots"""
        self.minR = minR
        self.R_radius = R_radius
        self.minP = minP
        self.P_radius = P_radius
        self.dx = dx
        self.minimum_energy_paths = None
        self.saddle_points = saddle_points
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.V_domain = V_domain
        self.n_bins_x = None
        self.n_bins_y = None
        self.x_plot = None
        self.y_plot = None
        self.x2d = None
        self.set_2D_plot_grid_precision(n_bins_x, n_bins_y)
        self.SigmaR_position = 1.1
        self.SigmaP_position = 1.1

    def set_minR(self, minR):
        """Allows to reset the position of the center of state R

        :param minR: np.array, ndim==2, shape==[1,2]"""
        self.minR = minR

    def set_R_radius(self, R_radius):
        """Allows to reset the radius of state R
        :param R_radius: float, radius of the disc defining the R state"""
        self.R_radius = R_radius

    def set_SigmaR_position(self, SigmaR_position):
        """Allows to reset the position of SigmaP

        :param SigmaR_position:  float, the factor of R_radius defining the position of SigmaR, it should be >= 1"""
        self.SigmaR_position = SigmaR_position

    def set_minP(self, minP):
        """Allows to reset the position of the center of state R

        :param minR: np.array, ndim==2, shape==[1,2]"""
        self.minP = minP

    def set_P_radius(self, P_radius):
        """Allows to reset the radius of state P

        :param P_radius: float, radius of the disc defining the P state"""
        self.P_radius = P_radius

    def set_SigmaP_position(self, SigmaP_position):
        """Allows to reset the position of SigmaP

        :param SigmaP_position:  float, the factor of P_radius defining the position of SigmaP, it should be >= 1
        """
        self.SigmaP_position = SigmaP_position
    def in_R(self, x):
        """Definition of the Reactant state R

        :param x: np.array, ndim==2, shape==[any, 2]
        :returns bool (if x.ndim == 1) or array of bool (if x.ndim == 2) Whether the argument x is inside state R
        """
        if x.ndim == 1:
            return np.sqrt(np.sum((x - self.minR) ** 2)) < self.R_radius
        elif x.ndim == 2:
            return np.sqrt(np.sum((x - self.minR) ** 2, axis=1)) < self.R_radius

    def above_SigmaR(self, x):
        """Test if a point is above the surface Sigma_R enclosing the state R

        :param x: np.array, ndim==2, shape==[any, 2]
        :returns bool (if x.ndim == 1) or array of bool (if x.ndim == 2) Whether the argument x is inside above SigmaR
        """
        if x.ndim == 1:
            return np.sqrt(np.sum((x - self.minR)**2)) >= self.SigmaR_position * self.R_radius
        elif x.ndim == 2:
            return np.sqrt(np.sum((x - self.minR)**2, axis=1)) >= self.SigmaR_position * self.R_radius

    def in_P(self, x):
        """Definition of the Reactant state P

        :param x: np.array, ndim==2, shape==[any, 2]
        :returns bool (if x.ndim == 1) or array of bool (if x.ndim == 2) Whether the argument x is inside state P
        """
        if x.ndim == 1:
            return np.sqrt(np.sum((x - self.minP) ** 2)) < self.P_radius
        elif x.ndim == 2:
            return np.sqrt(np.sum((x - self.minP) ** 2, axis=1)) < self.P_radius

    def above_SigmaP(self, x):
        """Test if a point is above the surface Sigma_R enclosing the state R

        :param x: np.array, ndim==2, shape==[any, 2]
        :returns bool (if x.ndim == 1) or array of bool (if x.ndim == 2) Whether the argument x is inside above SigmaP
        """
        if x.ndim == 1:
            return np.sqrt(np.sum((x - self.minP)**2)) >= self.SigmaP_position * self.P_radius
        elif x.ndim == 2:
            return np.sqrt(np.sum((x - self.minP)**2, axis=1)) >= self.SigmaP_position * self.P_radius

    def computeMEPs(self, n=1000):
        """Compute the various minimum energy paths by gradient descent starting close to the 
        various saddle points of the potential.

        :returns minimum_energy_paths: List of np.array with ndim = 2. each array is a MEP between two local minima
        """
        MEPs = []
        for j in range(len(self.saddle_points)):
            X_0 = self.saddle_points[j]
            dX_0 = np.array([[self.dx, self.dx]])
            X_plus = [X_0, X_0 + dX_0]
            X_minus = [X_0, X_0 - dX_0]
            for i in range(n):
                X_plus.append(X_plus[-1] - self.nabla_V(X_plus[-1]) * self.dx)
                X_minus.append(X_minus[-1] - self.nabla_V(X_minus[-1]) * self.dx)
            X_plus = np.array(X_plus)
            X_minus = np.array(X_minus)
            MEPs.append(np.append(np.flip(X_minus, axis=0), X_plus, axis=0).sum(axis=1))
        return MEPs

    def set_2D_plot_grid_precision(self, n_bins_x, n_bins_y):
        """Set the number of bins in the x and y directions

        :param n_bins_x: int,
        :param n_bins_y: int,
        """
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        gridx = np.linspace(self.x_domain[0], self.x_domain[1], self.n_bins_x)
        gridy = np.linspace(self.y_domain[0], self.y_domain[1], self.n_bins_y)
        self.x_plot = np.outer(gridx, np.ones(self.n_bins_y))
        self.y_plot = np.outer(gridy, np.ones(self.n_bins_x)).T
        self.x2d = np.concatenate((self.x_plot.reshape(self.n_bins_x * self.n_bins_y, 1),
                                   self.y_plot.reshape(self.n_bins_x * self.n_bins_y, 1)),
                                  axis=1)

    def plot_potential_heat_map(self, ax, set_lim=True):
        """Plot the potential heat map to the given ax

        :param ax: Instance of matplotlib.axes.Axes"""
        pot_on_grid = self.V(self.x2d).reshape(self.n_bins_x, self.n_bins_x)
        if set_lim:
            ax.set_ylim(self.y_domain[0], self.y_domain[1])
            ax.set_xlim(self.x_domain[0], self.x_domain[1])
        ax.pcolormesh(self.x_plot,
                      self.y_plot,
                      pot_on_grid,
                      cmap='coolwarm',
                      shading='auto',
                      vmin=self.V_domain[0],
                      vmax=self.V_domain[1])

    def plot_function_iso_levels(self, ax, function, n_lines, set_lim=True):
        """Plot the iso-lines of a given function to the given ax

        :param ax:         Instance of matplotlib.axes.Axes
        :param function:   a function taking as argument np.array with ndim==2, shape==[any, 2]
        :param n_lines:    int, number of iso-lines to plot
        :param set_lim:    boolean, whether the limits of the x and y axes should be set."""
        function_on_grid = function(self.x2d).reshape(self.n_bins_x, self.n_bins_y)
        if set_lim:
            ax.set_ylim(self.y_domain[0], self.y_domain[1])
            ax.set_xlim(self.x_domain[0], self.x_domain[1])
        ax.contour(self.x_plot, self.y_plot, function_on_grid, n_lines, cmap='viridis')
