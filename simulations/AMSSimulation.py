import numpy as np
from simulations.UnbiasedMD import OverdampedLangevin, Langevin


class AMSOverdampedLangevin(OverdampedLangevin):
    """Class to do AMS simulations"""
    def __init__(self, pot, xi, beta, forward, dt=None, seed=None, threshold=10**(-5)):
        """

        :param pot:         A potential object from 2dpotentials/potentials
        :param xi:          function that takes np.array with ndim==2 and shape==[any, 2] (x) as argument and return a
                            array of float of shape=[any] (z). Reaction coordinate to index the progress of replicas
                            (or walker) during the algorythm.
        :param beta:        float, inverse thermal energy 1/(k_B*T)
        :param forward:     boolean, whether the simulation if a forward (R -> P) ams or a backward one (P -> R)
        :param dt:          float, time step should be taken according to the potential used, if not given, the default
                            dx of potential is used
        :param seed:        int, random number generator seed
        :param threshold:   float, threshold value to consider if there is a difference between two reaction coordinate
                            value.

        """
        super().__init__(pot, beta, dt=dt, seed=seed)
        self.r = np.random.RandomState(self.seed)
        self.forward = forward
        self.xi = xi
        self.threshold = threshold
        if forward:
            self.in_R = self.pot.in_R
            self.above_Sigma = self.pot.above_SigmaR
            self.in_P = self.pot.in_P
        else:
            self.in_R = self.pot.in_P
            self.above_Sigma = self.pot.above_SigmaP
            self.in_P = self.pot.in_R
        self.ini_x = None

    def set_forward(self, forward):
        """Method to reset whether the simulation if a forward (R -> P) ams or a backward one (P -> R)

        :param forward boolean, whether the simulation if a forward (R -> P) ams or a backward one (P -> R)
        """
        self.forward = forward
        self.ini_x = None
        if forward:
            self.in_R = self.pot.in_R
            self.above_Sigma = self.pot.above_SigmaR
            self.in_P = self.pot.in_P
        else:
            self.in_R = self.pot.in_P
            self.above_Sigma = self.pot.above_SigmaP
            self.in_P = self.pot.in_R

    def set_xi(self, xi):
        """Method to reset the reaction coordinate xi

        :param xi:  function that takes np.array with ndim==2 and shape==[any, 2] (x) as argument and return a array of
                    float of shape=[any] (z). Reaction coordinate to index the progress of replicas (or walker) during
                    the algorythm.
        """
        self.xi = xi

    def set_threshold(self, threshold):
        """Method to reset the threshold for differences in xi to consider that two xi values are equal

        :param threshold:  positive real value
        """
        self.threshold = threshold

    def sample_initial_conditions(self, n_conditions, x_0=None, save_grad=False, save_gauss=False):
        """Simple md to sample initial conditions, if it reaches the product state, the next position is one of the
        previously seen configuration during the .

        :param n_conditions:    int, number of initial conditions to sample
        :param x_0:             np.array, ndim==2, shape==[1, 2], initial position (if not given, the center of R
                                (if self.forward==True) or P (if self.forward==False) is taken
        :param save_grad:       boolean, whether the forces should be saved
        :param save_gauss:      boolean, whether the gaussian should be saved
        :return: ini_traj:      dict, ini_traj["x_traj"] is the trajectory of the positions, ini_traj["grad_traj"]
                                is the trajectory of the forces if required and ini_traj["gauss_traj"] is the
                                trajectory of the gaussians if required
        :return initials:       dict, initials["x"] is initial positions,
        :return t_loop:         np.array of int, len<=n_conditions, number of time steps it took to go from R to SigmaR and
                                then back to R. It can be smaller than n _conditions as if P is reached, the trajectory
                                "jumps" back to a random previous time and branch at this point. Then the corresponding
                                t_loop would not be physical
        :return md_steps        int, number of md steps done
        """
        # Set the initial position of the dynamics

        if self.ini_x is None:
            if self.forward:
                self.ini_x = self.pot.minR
            else:
                self.ini_x = self.pot.minP
        md_steps = 0
        # First make sure the trajectory stats in R by running dynamics until R is reached
        while not self.in_R(self.ini_x):
            self.ini_x, _, _ = self.step(self.ini_x)
            md_steps += 1
        initial_x = []
        x_traj = []
        t_loop = []
        ini_traj = {}
        initials = {}
        # Four different cases depending on the values to store in memory
        if not save_grad and not save_gauss:
            for i in range(n_conditions):               # run until the required number of initial condition is sampled
                t = 0
                while not self.above_Sigma(self.ini_x):          # run util the dynamics reaches SigmaR
                    self.ini_x, _, _ = self.step(self.ini_x)
                    x_traj.append(self.ini_x)
                    md_steps += 1
                    t += 1
                initial_x.append(self.ini_x)
                while not self.in_R(self.ini_x):                 # run until the dynamics goes back to R
                    self.ini_x, _, _ = self.step(self.ini_x)
                    x_traj.append(self.ini_x)
                    md_steps += 1
                    t += 1
                    if self.in_P(self.ini_x):                    # if the dynamics reaches P, branch at a random previous time
                        self.ini_x = x_traj[np.random.choice(len(x_traj)-1)]
                        t -= 10**6                      # The value is arbitrary, to ensure it is negative
                if t > 0:
                    t_loop.append(t)
        elif save_grad and not save_gauss:
            grad_traj = []
            initial_grad = []
            for i in range(n_conditions):
                t = 0
                while not self.above_Sigma(self.ini_x):
                    self.ini_x, grad, _ = self.step(self.ini_x)
                    x_traj.append(self.ini_x)
                    grad_traj.append(grad)
                    md_steps += 1
                    t += 1
                initial_x.append(self.ini_x)
                initial_grad.append(grad)
                while not self.in_R(x):
                    self.ini_x, grad, _ = self.step(self.ini_x)
                    x_traj.append(self.ini_x)
                    grad_traj.append(grad)
                    md_steps += 1
                    t += 1
                    if self.in_P(self.ini_x):
                        self.ini_x = x_traj[np.random.choice(len(x_traj)-1)]
                        t -= 10**6
                if t > 0:
                    t_loop.append(t)
            ini_traj["grad_traj"] = np.array(grad_traj).sum(axis=1)
            initials["grad"] = np.array(initial_grad).sum(axis=1)
        elif not save_grad and save_gauss:
            gauss_traj = []
            initial_gauss = []
            for i in range(n_conditions):
                t = 0
                while not self.above_Sigma(self.ini_x):
                    self.ini_x, _, gauss = self.step(self.ini_x)
                    x_traj.append(self.ini_x)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    t += 1
                initial_x.append(self.ini_x)
                initial_gauss.append(gauss)
                while not self.in_R(self.ini_x):
                    self.ini_x, _, gauss = self.step(self.ini_x)
                    x_traj.append(self.ini_x)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    t += 1
                    if self.in_P(self.ini_x):
                        self.ini_x = x_traj[np.random.choice(len(x_traj)-1)]
                        t -= 10**6
                if t > 0:
                    t_loop.append(t)
            ini_traj["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
            initials["gauss"] = np.array(initial_gauss).sum(axis=1)
        else:
            grad_traj = []
            initial_grad = []
            gauss_traj = []
            initial_gauss = []
            for i in range(n_conditions):
                t = 0
                while not self.above_Sigma(self.ini_x):
                    self.ini_x, grad, gauss = self.step(self.ini_x)
                    x_traj.append(self.ini_x)
                    grad_traj.append(grad)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    t += 1
                initial_x.append(self.ini_x)
                initial_grad.append(grad)
                initial_gauss.append(gauss)
                while not self.in_R(self.ini_x):
                    self.ini_x, grad, gauss = self.step(self.ini_x)
                    x_traj.append(self.ini_x)
                    grad_traj.append(grad)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    t += 1
                    if self.in_P(self.ini_x):
                        self.ini_x = x_traj[np.random.choice(len(x_traj)-1)]
                        t -= 10 ** 6
                if t > 0:
                    t_loop.append(t)
            ini_traj["grad_traj"] = np.array(grad_traj).sum(axis=1)
            initials["grad"] = np.array(initial_grad).sum(axis=1)
            ini_traj["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
            initials["gauss"] = np.array(initial_gauss).sum(axis=1)
        ini_traj["x_traj"] = np.array(x_traj).sum(axis=1)
        initials["x"] = np.array(initial_x).sum(axis=1)
        return ini_traj, initials, np.array(t_loop), md_steps

    def ams_initialization(self, n_rep, initials, save_grad=False, save_gauss=False):
        """Initialize an AMS run from given initial conditions.

        :param n_rep:       int, number of replicas (or walkers) of the AMS
        :param initials:    dict, initials["x"] is the array of initial positions
        :param save_grad:   boolean, whether the forces should be saved
        :param save_gauss:  boolean, whether the gaussian should be saved
        :return reps:       list of dict, reps[i]["x_traj"] is the trajectory of the positions, reps[i]["grad_traj"] is
                            the trajectory of the forces if required, reps[i]["grauss_traj"] is the trajectory of the
                            gaussians if required, reps[i]["zmax"] is the maximum value of the reaction coordinate of
                            replica i, reps[i]["in_P"] is a boolean to indicate whether the replica i finishes in the
                            state P and reps[i]["in_P"] is the statistical weight of the replicas.
        :return md_steps    int, number of md steps done
        """
        reps = []
        md_steps = 0
        # Four different cases depending on the values to store in memory
        if not save_grad and not save_gauss:
            for i in range(n_rep):                              # run the n_rep trajectories
                reps.append({})
                reps[i]["weight"] = [1 / n_rep]
                x_traj = []
                x = initials["x"][i: i+1]
                x_traj.append(x)
                z_max = self.xi(x)
                while not self.in_R(x) and not self.in_P(x):     # until they reach either R or P
                    x, _, _ = self.step(x)
                    x_traj.append(x)
                    md_steps += 1
                    if self.xi(x) > z_max:
                        z_max = self.xi(x)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["z_max"] = z_max
        elif save_grad and not save_gauss:
            for i in range(n_rep):  # run the n_rep trajectories
                reps.append({})
                reps[i]["weight"] = [1 / n_rep]
                x_traj = []
                grad_traj = []
                x = initials["x"][i: i + 1]
                x_traj.append(x)
                grad_traj.append(initials["grad"][i: i + 1])
                z_max = self.xi(x)
                while not self.in_R(x) and not self.in_P(x):  # until they reach either R or P
                    x, grad, _ = self.step(x)
                    x_traj.append(x)
                    grad_traj.append(grad)
                    md_steps += 1
                    if self.xi(x) > z_max:
                        z_max = self.xi(x)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["grad_traj"] = np.array(grad_traj).sum(axis=1)
                reps[i]["z_max"] = z_max
        elif not save_grad and save_gauss:
            for i in range(n_rep):  # run the n_rep trajectories
                reps.append({})
                reps[i]["weight"] = [1 / n_rep]
                x_traj = []
                gauss_traj = []
                x = initials["x"][i: i + 1]
                x_traj.append(x)
                gauss_traj.append(initials["gauss"][i: i + 1])
                z_max = self.xi(x)
                while not self.in_R(x) and not self.in_P(x):  # until they reach either R or P
                    x, _, gauss = self.step(x)
                    x_traj.append(x)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    if self.xi(x) > z_max:
                        z_max = self.xi(x)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
                reps[i]["z_max"] = z_max
        else:
            for i in range(n_rep):  # run the n_rep trajectories
                reps.append({})
                reps[i]["weight"] = [1 / n_rep]
                x_traj = []
                grad_traj = []
                gauss_traj = []
                x = initials["x"][i: i + 1]
                x_traj.append(x)
                grad_traj.append(initials["grad"][i: i + 1])
                gauss_traj.append(initials["gauss"][i: i + 1])
                z_max = self.xi(x)
                while not self.in_R(x) and not self.in_P(x):  # until they reach either R or P
                    x, grad, gauss = self.step(x)
                    x_traj.append(x)
                    grad_traj.append(grad)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    if self.xi(x) > z_max:
                        z_max = self.xi(x)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["grad_traj"] = np.array(grad_traj).sum(axis=1)
                reps[i]["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
                reps[i]["z_max"] = z_max
        return reps, md_steps

    def ams_iteration(self, reps, k_min, save_grad=False, save_gauss=False):
        """

        :param reps:        list of dict, reps[i]["x_traj"] is the trajectory of the positions, reps[i]["grad_traj"] is
                            the trajectory of the forces if required, reps[i]["grauss_traj"] is the trajectory of the
                            gaussians if required, reps[i]["zmax"] is the maximum value of the reaction coordinate of
                            replica i, reps[i]["in_P"] is a boolean to indicate whether the replica i finishes in the
                            state P and reps[i]["in_P"] is the statistical weight of the replicas.
        :param k_min:       int, minimum number of replicas to kill at each iteration
        :param save_grad:   boolean, whether the forces should be saved
        :param save_gauss:  boolean, whether the gaussian should be saved
        :return reps:       list of dict, reps[i]["x_traj"] is the trajectory of the positions, reps[i]["grad_traj"] is
                            the trajectory of the forces if required, reps[i]["grauss_traj"] is the trajectory of the
                            gaussians if required, reps[i]["zmax"] is the maximum value of the reaction coordinate of
                            replica i, reps[i]["in_P"] is a boolean to indicate whether the replica i finishes in the
                            state P and reps[i]["in_P"] is the statistical weight of the replicas.
        :return killed:     list of int, the list of indices corresponding to the replicas killed during the iteration
        :return z_kill:     float, maximum value of the reaction coordinate of the k_min'th trajectory where
                            trajectories are sorted according to their z_max
        :return md_steps    int, number of md steps done
        """
        md_steps = 0
        z_maxs = np.array([reps[i]["z_max"] for i in range(len(reps))])
        z_kill = np.sort(z_maxs, axis=0)[k_min - 1]
        killed_trajs = np.where(z_maxs - z_kill <= self.threshold)[0].tolist()
        killed = killed_trajs.copy()
        alive = np.setdiff1d(range(len(reps)), killed)
        if len(alive) == 0:                                             # Extinction case
            return reps, killed_trajs, z_kill, md_steps
        if not save_grad and not save_gauss:
            while len(killed) > 0:
                i = killed.pop()
                j = np.random.choice(alive)
                x_traj = []
                k = 0
                reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                while self.xi(reps[j]["x_traj"][k:k + 1]) < z_kill:
                    x_traj.append(reps[j]["x_traj"][k:k + 1])
                    if self.xi(reps[j]["x_traj"][k:k + 1]) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1],)
                    k = k + 1
                x = reps[j]["x_traj"][k:k + 1]
                x_traj.append(reps[j]["x_traj"][k:k + 1])
                if self.xi(reps[j]["x_traj"][k:k + 1]) >= reps[i]["z_max"]:
                    reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                while not self.in_R(x) and not self.in_P(x):
                    x, _, _ = self.step(x)
                    x_traj.append(x)
                    md_steps += 1
                    if self.xi(x) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(x)
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
        elif save_grad and not save_gauss:
            while len(killed) > 0:
                i = killed.pop()
                j = np.random.choice(alive)
                x_traj = []
                grad_traj = []
                k = 0
                reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                while self.xi(reps[j]["x_traj"][k:k + 1]) < z_kill:
                    x_traj.append(reps[j]["x_traj"][k:k + 1])
                    grad_traj.append(reps[j]["grad_traj"][k:k + 1])
                    if self.xi(reps[j]["x_traj"][k:k + 1]) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                    k = k + 1
                x = reps[j]["x_traj"][k:k + 1]
                x_traj.append(reps[j]["x_traj"][k:k + 1])
                grad_traj.append(reps[j]["grad_traj"][k:k + 1])
                if self.xi(reps[j]["x_traj"][k:k + 1]) >= reps[i]["z_max"]:
                    reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                while not self.in_R(x) and not self.in_P(x):
                    x, grad, _ = self.step(x)
                    x_traj.append(x)
                    grad_traj.append(grad)
                    md_steps += 1
                    if self.xi(x) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(x)
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["grad_traj"] = np.array(grad_traj).sum(axis=1)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
        elif not save_grad and save_gauss:
            while len(killed) > 0:
                i = killed.pop()
                j = np.random.choice(alive)
                x_traj = []
                gauss_traj = []
                k = 0
                reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                while self.xi(reps[j]["x_traj"][k:k + 1]) < z_kill:
                    x_traj.append(reps[j]["x_traj"][k:k + 1])
                    gauss_traj.append(reps[j]["gauss_traj"][k:k + 1])
                    if self.xi(reps[j]["x_traj"][k:k + 1]) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                    k = k + 1
                x = reps[j]["x_traj"][k:k + 1]
                x_traj.append(reps[j]["x_traj"][k:k + 1])
                gauss_traj.append(reps[j]["gauss_traj"][k:k + 1])
                if self.xi(reps[j]["x_traj"][k:k + 1]) >= reps[i]["z_max"]:
                    reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                while not self.in_R(x) and not self.in_P(x):
                    x, _, gauss = self.step(x)
                    x_traj.append(x)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    if self.xi(x) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(x)
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
        else:
            while len(killed) > 0:
                i = killed.pop()
                j = np.random.choice(alive)
                x_traj = []
                grad_traj = []
                gauss_traj = []
                k = 0
                reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                while self.xi(reps[j]["x_traj"][k:k + 1]) < z_kill:
                    x_traj.append(reps[j]["x_traj"][k:k + 1])
                    grad_traj.append(reps[j]["grad_traj"][k:k + 1])
                    gauss_traj.append(reps[j]["gauss_traj"][k:k + 1])
                    if self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                    k = k + 1
                x = reps[j]["x_traj"][k:k + 1]
                p = reps[j]["p_traj"][k:k + 1]
                x_traj.append(reps[j]["x_traj"][k:k + 1])
                grad_traj.append(reps[j]["grad_traj"][k:k + 1])
                gauss_traj.append(reps[j]["gauss_traj"][k:k + 1])
                if self.xi(reps[j]["x_traj"][k:k + 1]) >= reps[i]["z_max"]:
                    reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1])
                while not self.in_R(x) and not self.in_P(x):
                    x, grad, gauss = self.step(x)
                    x_traj.append(x)
                    grad_traj.append(grad)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    if self.xi(x, p) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(x, p)
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["grad_traj"] = np.array(grad_traj).sum(axis=1)
                reps[i]["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
        return reps, killed_trajs, z_kill, md_steps

    def ams_run(self, initials, n_rep, k_min, return_all=False, save_grad=False, save_gauss=False):
        """

        :param initials:    np.array, ndim==3, shape==[2, n_rep, 2], the initial conditions.
        :param n_rep:       int, number of replicas
        :param k_min:       int, minimum number of replicas to kill at each iteration
        :param return_all:  boolean, whether all the trajectories at each iteration should be returned, in practice only
                            the trajectories that differ from the previous iteration. If false, only the reactive
                            trajectories are returned
        :param save_grad:   boolean, whether the forces should be saved
        :param save_gauss:  boolean, whether the gaussian should be saved
        :return p:          float, estimated probability of reaching P before R starting from the initial conditions
        :return z_kills:    list of float, all the various values of z_kill
        :return replicas:   lists of dicts, len== number of ams iteration + 1, each dict is a replica containing
                            the key "x_traj", "p_traj", "grad_traj" if save_grad, "gauss_traj" if save_gauss, "z_max",
                            "in_P" and "weight".
        :return md_steps:   int, number of steps of dynamics
        """

        reps, total_md_steps = self.ams_initialization(n_rep, initials, save_grad=save_grad, save_gauss=save_gauss)
        killed = []
        z_kills = []
        replicas = []
        p = 1
        while not np.prod(np.array([reps[i]["in_P"] for i in range(n_rep)])) and len(killed) != n_rep:
            reps, killed, z_kill, md_steps = self.ams_iteration(reps, k_min, save_grad=save_grad, save_gauss=save_gauss)
            z_kills.append(z_kill)
            total_md_steps += md_steps
            p = p * (1 - len(killed) / n_rep)
            for i in range(n_rep):
                if i in killed:
                    reps[i]["weight"] = [p / n_rep]
                    if return_all:
                        replicas.append(reps[i].copy())
                else:
                    reps[i]["weight"].append(p / n_rep)
        if len(killed) != n_rep:
            replicas += reps
        return p, z_kills, replicas, total_md_steps


class AMSLangevin(Langevin):
    """Class to do AMS simulations"""
    def __init__(self, pot, xi,  M, gamma,  beta, forward, dt=None, seed=None, threshold=10**(-5)):
        """

        :param pot:         A potential object from 2dpotentials/potentials
        :param xi:          function that takes two np.array with ndim==2 and shape==[any, 2] (x, p) as argument and
                            returns an array of float of shape=[any] (z). Reaction coordinate to index the progress of
                            replicas (or walker) during the algorythm.
        :param M:           np.array, ndim==2, shape==[1, 2], mass
        :param gamma:       np.array, ndim==2, shape==[1, 2], friction
        :param beta:        float, inverse thermal energy 1/(k_B*T)
        :param forward:     boolean, whether the simulation if a forward (R -> P) ams or a backward one (P -> R)
        :param dt:          float, time step should be taken according to the potential used, if not given, the default
                            dx of potential is used
        :param seed:        int, random number generator seed
        :param threshold:   float, threshold value to consider if there is a difference between two reaction coordinate
                            value.

        """
        super().__init__(pot, beta,  M, gamma, dt=dt, seed=seed)
        self.r = np.random.RandomState(self.seed)
        self.forward = forward
        self.xi = xi
        self.threshold = threshold
        if forward:
            self.in_R = self.pot.in_R
            self.above_Sigma = self.pot.above_SigmaR
            self.in_P = self.pot.in_P
        else:
            self.in_R = self.pot.in_P
            self.above_Sigma = self.pot.above_SigmaP
            self.in_P = self.pot.in_R
        self.ini_x = None
        self.ini_p = None

    def set_forward(self, forward):
        """Method to reset whether the simulation if a forward (R -> P) ams or a backward one (P -> R)

        :param forward boolean, whether the simulation if a forward (R -> P) ams or a backward one (P -> R)
        """
        self.forward = forward
        self.ini_x = None
        self.ini_p = None
        if forward:
            self.in_R = self.pot.in_R
            self.above_Sigma = self.pot.above_SigmaR
            self.in_P = self.pot.in_P
        else:
            self.in_R = self.pot.in_P
            self.above_Sigma = self.pot.above_SigmaP
            self.in_P = self.pot.in_R

    def set_xi(self, xi):
        """Method to reset the reaction coordinate xi

        :param xi:  function that takes two np.array with ndim==2 and shape==[any, 2] (x, p) as argument and return an
                    array of float of shape=[any] (z). Reaction coordinate to index the progress of replicas (or walker)
                    during the algorythm.
        """
        self.xi = xi

    def sample_initial_conditions(self, n_conditions, save_grad=False, save_gauss=False):
        """Simple md to sample initial conditions, if it reaches the product state, the next position is one of the
        previously seen configuration during the .

        :param n_conditions:    int, number of initial conditions to sample
        :param save_grad:       boolean, whether the forces should be saved
        :param save_gauss:      boolean, whether the gaussian should be saved
        :return: ini_traj:      dict, ini_traj["x_traj"] is the trajectory of the positions, ini_traj["p_traj"] is the
                                trajectory of the momenta, ini_traj["grad_traj"] is the trajectory of the forces if
                                required and ini_traj["gauss_traj"] is the trajectory of the gaussians if required
        :return initials:       dict, initials["x"] is the array of initial positions, initials["p"] is the array of
                                initial momenta,
        :return t_loop:         np.array of int, len<=n_conditions, number of time steps it took to go from R to SigmaR and
                                then back to R. It can be smaller than n _conditions as if P is reached, the trajectory
                                "jumps" back to a random previous time and branch at this point. Then the corresponding
                                t_loop would not be physical
        :return md_steps        int, number of md steps done
        """
        # Set the initial position of the dynamics
        if self.ini_x is None:
            if self.forward:
                self.ini_x = self.pot.minR
            else:
                self.ini_x = self.pot.minP
        if self.ini_p is None:
            self.ini_p = np.sqrt(self.M / self.beta) * self.r.normal(size=(self.ini_x.shape[1]))
        md_steps = 0

        # First make sure the trajectory stats in R by running dynamics until R is reached
        while not self.in_R(self.ini_x):
            self.ini_x, self.ini_p, _, _ = self.step(self.ini_x, self.ini_p)
            md_steps += 1
        t_loop = []
        initial_x = []
        initial_p = []
        x_traj = []
        p_traj = []
        ini_traj = {}
        initials = {}
        # Four different cases depending on the values to store in memory
        if not save_grad and not save_gauss:
            for i in range(n_conditions):               # run until the required number of initial condition is sampled
                t = 0
                while not self.above_Sigma(self.ini_x):          # run util the dynamics reaches SigmaR
                    self.ini_x, self.ini_p, _, _ = self.step(self.ini_x, self.ini_p)
                    x_traj.append(self.ini_x)
                    p_traj.append(self.ini_p)
                    md_steps += 1
                    t += 1
                initial_x.append(self.ini_x)
                initial_p.append(self.ini_p)
                while not self.in_R(self.ini_x):                 # run until the dynamics goes back to R
                    self.ini_x, self.ini_p, _, _ = self.step(self.ini_x, self.ini_p)
                    x_traj.append(self.ini_x)
                    p_traj.append(self.ini_p)
                    md_steps += 1
                    t += 1
                    if self.in_P(self.ini_x):         # if the dynamics reaches P, branch at a random previous time
                        index = np.random.choice(len(x_traj))
                        self.ini_x = x_traj[index]
                        self.ini_p = p_traj[index]
                        t -= 10**6                      # The value is arbitrary, to ensure it is negative
                if t > 0:
                    t_loop.append(t)
        elif save_grad and not save_gauss:
            grad_traj = []
            initial_grad = []
            for i in range(n_conditions):
                t = 0
                while not self.above_Sigma(self.ini_x):
                    self.ini_x, self.ini_p, grad, _ = self.step(self.ini_x, self.ini_p)
                    x_traj.append(self.ini_x)
                    p_traj.append(self.ini_p)
                    grad_traj.append(grad)
                    md_steps += 1
                    t += 1
                initial_x.append(self.ini_x)
                initial_p.append(self.ini_p)
                initial_grad.append(grad)
                while not self.in_R(self.ini_p):
                    self.ini_x, self.ini_p, grad, _ = self.step(self.ini_x, self.ini_p)
                    x_traj.append(self.ini_x)
                    p_traj.append(self.ini_p)
                    grad_traj.append(grad)
                    md_steps += 1
                    t += 1
                    if self.in_P(self.ini_x):
                        index = np.random.choice(len(x_traj))
                        self.ini_x = x_traj[index]
                        self.ini_p = p_traj[index]
                        t -= 10**6
                if t > 0:
                    t_loop.append(t)
            ini_traj["grad_traj"] = np.array(grad_traj).sum(axis=1)
            initials["grad"] = np.array(initial_grad).sum(axis=1)
        elif not save_grad and save_gauss:
            gauss_traj = []
            initial_gauss = []
            for i in range(n_conditions):
                t = 0
                while not self.above_Sigma(self.ini_x):
                    self.ini_x, self.ini_p, _, gauss = self.step(self.ini_x, self.ini_p)
                    x_traj.append(self.ini_x)
                    p_traj.append(self.ini_p)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    t += 1
                initial_x.append(self.ini_x)
                initial_p.append(self.ini_p)
                initial_gauss.append(gauss)
                while not self.in_R(self.ini_x):
                    self.ini_x, self.ini_p, _, gauss = self.step(self.ini_x, self.ini_p)
                    x_traj.append(self.ini_x)
                    p_traj.append(self.ini_p)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    t += 1
                    if self.in_P(self.ini_x):
                        index = np.random.choice(len(x_traj))
                        self.ini_x = x_traj[index]
                        self.ini_p = p_traj[index]
                        t -= 10**6
                if t > 0:
                    t_loop.append(t)
            ini_traj["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
            initials["gauss"] = np.array(initial_gauss).sum(axis=1)
        else:
            grad_traj = []
            gauss_traj = []
            initial_grad = []
            initial_gauss = []
            for i in range(n_conditions):
                t = 0
                while not self.above_Sigma(self.ini_x, self.ini_p):
                    self.ini_x, self.ini_p, grad, gauss = self.step(self.ini_x, self.ini_p)
                    x_traj.append(self.ini_x)
                    p_traj.append(self.ini_p)
                    grad_traj.append(grad)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    t += 1
                initial_x.append(self.ini_x)
                initial_p.append(self.ini_p)
                initial_grad.append(grad)
                initial_gauss.append(gauss)
                while not self.in_R(self.ini_x):
                    self.ini_x, self.ini_p, grad, gauss = self.step(self.ini_x, self.ini_p)
                    x_traj.append(self.ini_x)
                    p_traj.append(self.ini_p)
                    grad_traj.append(grad)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    t += 1
                    if self.in_P(self.ini_x):
                        index = np.random.choice(len(x_traj))
                        self.ini_x = x_traj[index]
                        self.ini_p = p_traj[index]
                        t -= 10 ** 6
                if t > 0:
                    t_loop.append(t)
            ini_traj["grad_traj"] = np.array(grad_traj).sum(axis=1)
            ini_traj["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
            initials["grad"] = np.array(initial_grad).sum(axis=1)
            initials["gauss"] = np.array(initial_gauss).sum(axis=1)
        ini_traj["x_traj"] = np.array(x_traj).sum(axis=1)
        ini_traj["p_traj"] = np.array(p_traj).sum(axis=1)
        initials["x"] = np.array(initial_x).sum(axis=1)
        initials["p"] = np.array(initial_p).sum(axis=1)
        return ini_traj, initials, np.array(t_loop), md_steps

    def ams_initialization(self, n_rep, initials, save_grad=False, save_gauss=False):
        """Initialize an AMS run from given initial conditions.

        :param n_rep:       int, number of replicas (or walkers) of the AMS
        :param initials:    dict, initials["x"] is the array of initial positions, initials["p"] is the array of
                            initial momenta,
        :param save_grad:   boolean, whether the forces should be saved
        :param save_gauss:  boolean, whether the gaussian should be saved
        :return reps:       list of dict, reps[i]["x_traj"] is the trajectory of the positions, reps["p_traj"] is the
                            trajectory of the momenta, reps[i]["grad_traj"] is the trajectory of the forces if required,
                            reps[i]["grauss_traj"] is the trajectory of the gaussians if required, reps[i]["zmax"] is
                            the maximum value of the reaction coordinate of replica i, reps[i]["in_P"] is a boolean to
                            indicate whether the replica i finishes in the state P and reps[i]["in_P"] is the
                            statistical weight of the replicas.
        :return md_steps    int, number of md steps done
        """
        reps = []
        md_steps = 0
        # Four different cases depending on the values to store in memory
        if not save_grad and not save_gauss:
            for i in range(n_rep):                              # run the n_rep trajectories
                reps.append({})
                reps[i]["weight"] = [1 / n_rep]
                x_traj = []
                p_traj = []
                x = initials["x"][i: i+1]
                p = initials["p"][i: i+1]
                x_traj.append(x)
                p_traj.append(p)
                z_max = self.xi(x, p)
                while not self.in_R(x) and not self.in_P(x):     # until they reach either R or P
                    x, p, _, _ = self.step(x, p)
                    x_traj.append(x)
                    p_traj.append(p)
                    md_steps += 1
                    if self.xi(x, p) > z_max:
                        z_max = self.xi(x, p)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["p_traj"] = np.array(p_traj).sum(axis=1)
                reps[i]["z_max"] = z_max
        elif save_grad and not save_gauss:
            for i in range(n_rep):  # run the n_rep trajectories
                reps.append({})
                reps[i]["weight"] = [1 / n_rep]
                x_traj = []
                p_traj = []
                grad_traj = []
                x = initials["x"][i: i + 1]
                p = initials["p"][i: i + 1]
                x_traj.append(x)
                p_traj.append(p)
                grad_traj.append(initials["grad"][i: i + 1])
                z_max = self.xi(x, p)
                while not self.in_R(x) and not self.in_P(x):  # until they reach either R or P
                    x, p, grad, _ = self.step(x, p)
                    x_traj.append(x)
                    p_traj.append(p)
                    grad_traj.append(grad)
                    md_steps += 1
                    if self.xi(x, p) > z_max:
                        z_max = self.xi(x, p)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["p_traj"] = np.array(p_traj).sum(axis=1)
                reps[i]["grad_traj"] = np.array(grad_traj).sum(axis=1)
                reps[i]["z_max"] = z_max
        elif not save_grad and save_gauss:
            for i in range(n_rep):  # run the n_rep trajectories
                reps.append({})
                reps[i]["weight"] = [1 / n_rep]
                x_traj = []
                p_traj = []
                gauss_traj = []
                x = initials["x"][i: i + 1]
                p = initials["p"][i: i + 1]
                x_traj.append(x)
                p_traj.append(p)
                gauss_traj.append(initials["gauss"][i: i + 1])
                z_max = self.xi(x, p)
                while not self.in_R(x) and not self.in_P(x):  # until they reach either R or P
                    x, p, _, gauss = self.step(x, p)
                    x_traj.append(x)
                    p_traj.append(p)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    if self.xi(x, p) > z_max:
                        z_max = self.xi(x, p)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["p_traj"] = np.array(p_traj).sum(axis=1)
                reps[i]["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
                reps[i]["z_max"] = z_max
        else:
            for i in range(n_rep):  # run the n_rep trajectories
                reps.append({})
                reps[i]["weight"] = [1 / n_rep]
                x_traj = []
                p_traj = []
                grad_traj = []
                gauss_traj = []
                x = initials["x"][i: i + 1]
                p = initials["p"][i: i + 1]
                x_traj.append(x)
                p_traj.append(p)
                grad_traj.append(initials["grad"][i: i + 1])
                gauss_traj.append(initials["gauss"][i: i + 1])
                z_max = self.xi(x, p)
                while not self.in_R(x) and not self.in_P(x):  # until they reach either R or P
                    x, p, grad, gauss = self.step(x, p)
                    x_traj.append(x)
                    p_traj.append(p)
                    grad_traj.append(grad)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    if self.xi(x, p) > z_max:
                        z_max = self.xi(x, p)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["p_traj"] = np.array(p_traj).sum(axis=1)
                reps[i]["grad_traj"] = np.array(grad_traj).sum(axis=1)
                reps[i]["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
                reps[i]["z_max"] = z_max
        return reps, md_steps

    def ams_iteration(self, reps, k_min, save_grad=False, save_gauss=False):
        """

        :param reps:        list of dict, reps[i]["x_traj"] is the trajectory of the positions, reps["p_traj"] is the
                            trajectory of the momenta, reps[i]["grad_traj"] is the trajectory of the forces if required,
                            reps[i]["grauss_traj"] is the trajectory of the gaussians if required, reps[i]["zmax"] is
                            the maximum value of the reaction coordinate of replica i, reps[i]["in_P"] is a boolean to
                            indicate whether the replica i finishes in the state P and reps[i]["in_P"] is the
                            statistical weight of the replicas.
        :param k_min:       int, minimum number of replicas to kill at each iteration
        :param save_grad:   boolean, whether the forces should be saved
        :param save_gauss:  boolean, whether the gaussian should be saved
        :return reps:       list of dict, reps[i]["x_traj"] is the trajectory of the positions, reps["p_traj"] is the
                            trajectory of the momenta, reps[i]["grad_traj"] is the trajectory of the forces if required,
                            reps[i]["grauss_traj"] is the trajectory of the gaussians if required, reps[i]["zmax"] is
                            the maximum value of the reaction coordinate of replica i, reps[i]["in_P"] is a boolean to
                            indicate whether the replica i finishes in the state P and reps[i]["in_P"] is the
                            statistical weight of the replicas.
        :return killed:     list of int, the list of indices corresponding to the replicas killed during the iteration
        :return z_kill:     float, maximum value of the reaction coordinate of the k_min'th trajectory where
                            trajectories are sorted according to their z_max
        :return md_steps    int, number of md steps done
        """
        md_steps = 0
        z_maxs = np.array([reps[i]["z_max"] for i in range(len(reps))])
        z_kill = np.sort(z_maxs, axis=0)[k_min - 1]
        killed_trajs = np.where(z_maxs - z_kill <= self.threshold)[0].tolist()
        killed = killed_trajs.copy()
        alive = np.setdiff1d(range(len(reps)), killed)
        if len(alive) == 0:                                             # Extinction case
            return reps, killed_trajs, z_kill, md_steps
        if not save_grad and not save_gauss:
            while len(killed) > 0:
                i = killed.pop()
                j = np.random.choice(alive)
                x_traj = []
                p_traj = []
                k = 0
                reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k+1], reps[j]["p_traj"][k:k+1])
                while self.xi(reps[j]["x_traj"][k:k+1], reps[j]["p_traj"][k:k+1]) < z_kill:
                    x_traj.append(reps[j]["x_traj"][k:k+1])
                    p_traj.append(reps[j]["p_traj"][k:k+1])
                    if self.xi(reps[j]["x_traj"][k:k+1], reps[j]["p_traj"][k:k+1]) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k+1], reps[j]["p_traj"][k:k+1])
                    k = k + 1
                x = reps[j]["x_traj"][k:k+1]
                p = reps[j]["p_traj"][k:k+1]
                x_traj.append(reps[j]["x_traj"][k:k+1])
                p_traj.append(reps[j]["p_traj"][k:k+1])
                if self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) >= reps[i]["z_max"]:
                    reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                while not self.in_R(x) and not self.in_P(x):
                    x, p, _, _ = self.step(x, p)
                    x_traj.append(x)
                    p_traj.append(p)
                    md_steps += 1
                    if self.xi(x,p) > z_maxs[i]:
                        z_maxs[i] = self.xi(x,p)
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["p_traj"] = np.array(p_traj).sum(axis=1)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
        elif save_grad and not save_gauss:
            while len(killed) > 0:
                i = killed.pop()
                j = np.random.choice(alive)
                x_traj = []
                p_traj = []
                grad_traj = []
                k = 0
                reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                while self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) < z_kill:
                    x_traj.append(reps[j]["x_traj"][k:k + 1])
                    p_traj.append(reps[j]["p_traj"][k:k + 1])
                    grad_traj.append(reps[j]["grad_traj"][k:k + 1])
                    if self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                    k = k + 1
                x = reps[j]["x_traj"][k:k + 1]
                p = reps[j]["p_traj"][k:k + 1]
                x_traj.append(reps[j]["x_traj"][k:k + 1])
                p_traj.append(reps[j]["p_traj"][k:k + 1])
                grad_traj.append(reps[j]["grad_traj"][k:k + 1])
                if self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) >= reps[i]["z_max"]:
                    reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                while not self.in_R(x) and not self.in_P(x):
                    x, p, grad, _ = self.step(x, p)
                    x_traj.append(x)
                    p_traj.append(p)
                    grad_traj.append(grad)
                    md_steps += 1
                    if self.xi(x, p) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(x, p)
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["p_traj"] = np.array(p_traj).sum(axis=1)
                reps[i]["grad_traj"] = np.array(grad_traj).sum(axis=1)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
        elif not save_grad and save_gauss:
            while len(killed) > 0:
                i = killed.pop()
                j = np.random.choice(alive)
                x_traj = []
                p_traj = []
                gauss_traj = []
                k = 0
                reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                while self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) < z_kill:
                    x_traj.append(reps[j]["x_traj"][k:k + 1])
                    p_traj.append(reps[j]["p_traj"][k:k + 1])
                    gauss_traj.append(reps[j]["gauss_traj"][k:k + 1])
                    if self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                    k = k + 1
                x = reps[j]["x_traj"][k:k + 1]
                p = reps[j]["p_traj"][k:k + 1]
                x_traj.append(reps[j]["x_traj"][k:k + 1])
                p_traj.append(reps[j]["p_traj"][k:k + 1])
                gauss_traj.append(reps[j]["gauss_traj"][k:k + 1])
                if self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) >= reps[i]["z_max"]:
                    reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                while not self.in_R(x) and not self.in_P(x):
                    x, p, _, gauss = self.step(x, p)
                    x_traj.append(x)
                    p_traj.append(p)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    if self.xi(x, p) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(x, p)
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["p_traj"] = np.array(p_traj).sum(axis=1)
                reps[i]["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
        else:
            while len(killed) > 0:
                i = killed.pop()
                j = np.random.choice(alive)
                x_traj = []
                p_traj = []
                grad_traj = []
                gauss_traj = []
                k = 0
                reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                while self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) < z_kill:
                    x_traj.append(reps[j]["x_traj"][k:k + 1])
                    p_traj.append(reps[j]["p_traj"][k:k + 1])
                    grad_traj.append(reps[j]["grad_traj"][k:k + 1])
                    gauss_traj.append(reps[j]["gauss_traj"][k:k + 1])
                    if self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                    k = k + 1
                x = reps[j]["x_traj"][k:k + 1]
                p = reps[j]["p_traj"][k:k + 1]
                x_traj.append(reps[j]["x_traj"][k:k + 1])
                p_traj.append(reps[j]["p_traj"][k:k + 1])
                grad_traj.append(reps[j]["grad_traj"][k:k + 1])
                gauss_traj.append(reps[j]["gauss_traj"][k:k + 1])
                if self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1]) >= reps[i]["z_max"]:
                    reps[i]["z_max"] = self.xi(reps[j]["x_traj"][k:k + 1], reps[j]["p_traj"][k:k + 1])
                while not self.in_R(x) and not self.in_P(x):
                    x, p, grad, gauss = self.step(x, p)
                    x_traj.append(x)
                    p_traj.append(p)
                    grad_traj.append(grad)
                    gauss_traj.append(gauss)
                    md_steps += 1
                    if self.xi(x, p) >= reps[i]["z_max"]:
                        reps[i]["z_max"] = self.xi(x, p)
                reps[i]["x_traj"] = np.array(x_traj).sum(axis=1)
                reps[i]["p_traj"] = np.array(p_traj).sum(axis=1)
                reps[i]["grad_traj"] = np.array(grad_traj).sum(axis=1)
                reps[i]["gauss_traj"] = np.array(gauss_traj).sum(axis=1)
                if self.in_P(x):
                    reps[i]["in_P"] = True
                else:
                    reps[i]["in_P"] = False
        return reps, killed_trajs, z_kill, md_steps

    def ams_run(self, initials, n_rep, k_min, return_all=False, save_grad=False, save_gauss=False):
        """

        :param initials:    np.array, ndim==3, shape==[2, n_rep, 2], the initial conditions.
        :param n_rep:       int, number of replicas
        :param k_min:       int, minimum number of replicas to kill at each iteration
        :param return_all:  boolean, whether all the trajectories at each iteration should be returned, in practice only
                            the trajectories that differ from the previous iteration. If false, only the reactive
                            trajectories are returned
        :param save_grad:   boolean, whether the forces should be saved
        :param save_gauss:  boolean, whether the gaussian should be saved
        :return p:          float, estimated probability of reaching P before R starting from the initial conditions
        :return z_kills:    list of float, all the various values of z_kill
        :return replicas:   lists of dicts, len== number of ams iteration + 1, each dict is a replica containing
                            the key "x_traj", "ptraj", "grad_traj" if save_grad, "gauss_traj" if save_gauss, "z_max",
                            "in_P" and "weight".
        :return md_steps:   int, number of steps of dynamics
        """

        reps, total_md_steps = self.ams_initialization(n_rep, initials, save_grad=save_grad, save_gauss=save_gauss)
        killed = []
        z_kills = []
        replicas = []
        p = 1
        while not np.prod(np.array([reps[i]["in_P"] for i in range(n_rep)])) and len(killed) != n_rep:
            reps, killed, z_kill, md_steps = self.ams_iteration(reps, k_min, save_grad=save_grad, save_gauss=save_gauss)
            z_kills.append(z_kill)
            total_md_steps += md_steps
            p = p * (1 - len(killed) / n_rep)
            for i in range(n_rep):
                if i in killed:
                    reps[i]["weight"] = [p / n_rep]
                    if return_all:
                        replicas.append(reps[i].copy())
                else:
                    reps[i]["weight"].append(p / n_rep)
        if len(killed) != n_rep:
            replicas += reps
        return p, z_kills, replicas, total_md_steps
