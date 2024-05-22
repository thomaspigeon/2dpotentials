import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.model_selection import KFold


class TrainCommittor:
    """Class to train committor function models. It contains a decoder model to build a mean path from reactive
    trajectories or sampled Boltzmann distribution

    The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
    last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
    other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
    to generate the validation and test data.
    """

    def __init__(self, committor_model, pot, dataset, eps, penalization_points=None):
        """

        :param committor_model:     AE model from committor.neural_net_models with 2D input.
        :param pot:                 two-dimensional potential object from potentials
        :param dataset:             dict,
                                    dataset["dt"], float: integration time step
                                    dataset["beta"], float: the inverse temperature energy 1/ k_B*T

                                    dataset["single_trajs_pos"], np.array with ndim==3, shape==[n_ini_pos, len(trajs), 2],
                                    positions along the single trajs
                                    dataset["single_trajs_mom"], np.array with ndim==3, shape==[n_ini_pos, len(trajs), 2],
                                    momenta along the single trajs
                                    dataset["single_trajs_gauss"], np.array with ndim==3, shape==[n_ini_pos, len(trajs)-1, 2],
                                    gaussions along the single trajs
                                    dataset["single_trajs_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["boltz_pos"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array of points on the 2D potentials distributed according ot the boltzmann gibbs
                                    measure
                                    dataset["boltz_pos_lagged"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array of points on the 2D potentials distributed according ot the boltzmann gibbs
                                    measure with a time lag compared to the set of points dataset["boltz_pos"]
                                    dataset["boltz_mom"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["boltz_mom_lagged"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["boltz_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["multiple_trajs_pos"], np.array with ndim==4, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                    positions along the multiple trajs
                                    dataset["multiple_trajs_mom"], np.array with ndim==3, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                    momenta along the multiple trajs
                                    dataset["multiple_trajs_gauss"], np.array with ndim==3, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                    gaussions along the multiple trajs
                                    dataset["multiple_trajs_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["react_pos"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array  of points
                                    on the 2D potentials distributed according ot the probability measure of reactive
                                    trajectories
                                    dataset["react_mom"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["react_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

        :param penalization_points: np.array, ndim==2, shape=[any, 3(or 5)], penalization_point[:, :2(or 4)] are the
                                    postions (and momenta) on which the encoder is penalized if its values on these
                                    points differ from penalization_point[:, 2 (or 4)]
        """
        self.committor_model = committor_model
        self.pot = pot
        self.dataset = dataset
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpe'
        if penalization_points is None:
            penalization_points = np.append(np.append(pot.minR, np.zeros([1, 1]), axis=1),
                                            np.append(pot.minP, np.ones([1, 1]), axis=1),
                                            axis=0)
            self.penalization_points = torch.tensor(penalization_points.astype('float32')).to(device)
        else:
            self.penalization_points = torch.tensor(penalization_points.astype('float32')).to(device)
        # attributes that have to be set through later defined methods
        self.training_dataset = None
        self.test_dataset = None
        self.Kfold_splits = None
        self.train_data = None
        self.validation_data = None

        self.ito_loss_weight = None
        self.boltz_traj_fixed_point_loss_weight = None
        self.multiple_trajs_fixed_point_loss_weight_1 = None
        self.multiple_trajs_fixed_point_loss_weight_2 = None
        self.mse_boltz_weight = None
        self.mse_react_weight = None
        self.squared_grad_boltz_weight = None
        self.l1_pen_weight = None
        self.l2_pen_weight = None
        self.pen_points_weight = None
        self.n_wait = None

        self.eps = eps

    def set_loss_weight(self, loss_params):
        """Function to set the loss parameters.

        :param loss_params:     dict, containing:
                                loss_params["ito_loss_weight"], float >= 0, prefactor of the ito loss term
                                loss_params["boltz_traj_fixed_point_loss_weight"], float >= 0,
                                loss_params["multiple_trajs_fixed_point_loss_weight"], float >= 0,
                                loss_params["squared_grad_boltz_weight"], float >= 0, prefactor of the
                                squared gradient the encoder on the Bolzmann-Gibbs distribution,
                                loss_params["mse_boltz_weight"] float >= 0, prefactor of the MSE term
                                of the loss on the Bolzmann gibbs distribution,
                                loss_params["mse_react_weight"], float >= 0, prefactor of the MSE term of the loss on
                                the reactive trajectories distribution,
                                loss_params["l1_pen_weight"], float >= 0, prefactor of the L1 weight decay penalization,
                                loss_params["l2_pen_weight"], float >= 0, prefactor of the L2 weight decay penalization,
                                loss_params["pen_points_weight"], float >= 0, prefactor of the penalization so that
                                certain points have a certain encoded value.
                                loss_params["n_wait"], int >= 1, early stopping patience parameter. The model kept in
                                self.committor_model is the one corresponding to the minimal test loss
        """
        if "ito_loss_weight" not in loss_params.keys():
            self.ito_loss_weight = 0.
            print("""ito_loss_weight value not provided, set to default value of: """, 0.)
        elif type(loss_params["ito_loss_weight"]) != float or loss_params["ito_loss_weight"] < 0.:
            raise ValueError("""loss_params["ito_loss_weight"] must be a float >= 0.""")
        else:
            self.ito_loss_weight = loss_params["ito_loss_weight"]

        if "log_ito_loss_weight" not in loss_params.keys():
            self.log_ito_loss_weight = 0.
            print("""log_ito_loss_weight value not provided, set to default value of: """, 0.)
        elif type(loss_params["log_ito_loss_weight"]) != float or loss_params["log_ito_loss_weight"] < 0.:
            raise ValueError("""loss_params["log_ito_loss_weight"] must be a float >= 0.""")
        else:
            self.log_ito_loss_weight = loss_params["log_ito_loss_weight"]

        if "boltz_traj_fixed_point_loss_weight_1" not in loss_params.keys():
            self.boltz_traj_fixed_point_loss_weight_1 = 0.
            print("""boltz_traj_fixed_point_loss_weight_1 value not provided, set to default value of: """, 0.)
        elif type(loss_params["boltz_traj_fixed_point_loss_weight_1"]) != float or loss_params["boltz_traj_fixed_point_loss_weight_1"] < 0.:
            raise ValueError("""loss_params["boltz_traj_fixed_point_loss_weight"] must be a float >= 0.""")
        else:
            self.boltz_traj_fixed_point_loss_weight_1 = loss_params["boltz_traj_fixed_point_loss_weight_1"]

        if "boltz_traj_fixed_point_loss_weight_2" not in loss_params.keys():
            self.boltz_traj_fixed_point_loss_weight_2 = 0.
            print("""boltz_traj_fixed_point_loss_weight_2 value not provided, set to default value of: """, 0.)
        elif type(loss_params["boltz_traj_fixed_point_loss_weight_2"]) != float or loss_params["boltz_traj_fixed_point_loss_weight_2"] < 0.:
            raise ValueError("""loss_params["boltz_traj_fixed_point_loss_weight_2"] must be a float >= 0.""")
        else:
            self.boltz_traj_fixed_point_loss_weight_2 = loss_params["boltz_traj_fixed_point_loss_weight_2"]

        if "multiple_trajs_fixed_point_loss_weight_1" not in loss_params.keys():
            self.multiple_trajs_fixed_point_loss_weight_1 = 0.
            print("""multiple_trajs_fixed_point_loss_weight_1 value not provided, set to default value of: """, 0.)
        elif type(loss_params["multiple_trajs_fixed_point_loss_weight_1"]) != float or loss_params["multiple_trajs_fixed_point_loss_weight_1"] < 0.:
            raise ValueError("""loss_params["multiple_trajs_fixed_point_loss_weight_1"] must be a float >= 0.""")
        else:
            self.multiple_trajs_fixed_point_loss_weight_1 = loss_params["multiple_trajs_fixed_point_loss_weight_1"]

        if "multiple_trajs_fixed_point_loss_weight_2" not in loss_params.keys():
            self.multiple_trajs_fixed_point_loss_weight_2 = 0
            print("""multiple_trajs_fixed_point_loss_weight_2 value not provided, set to default value of: """, 0.)
        elif type(loss_params["multiple_trajs_fixed_point_loss_weight_2"]) != float or loss_params["multiple_trajs_fixed_point_loss_weight_2"] < 0.:
            raise ValueError("""loss_params["multiple_trajs_fixed_point_loss_weight_2"] must be a float >= 0.""")
        else:
            self.multiple_trajs_fixed_point_loss_weight_2 = loss_params["multiple_trajs_fixed_point_loss_weight_2"]

        if "squared_grad_boltz_weight" not in loss_params.keys():
            self.squared_grad_boltz_weight = 0
            print("""squared_grad_boltz_weight value not provided, set to default value of: """, 0.)
        elif type(loss_params["squared_grad_boltz_weight"]) != float or loss_params["squared_grad_boltz_weight"] < 0.:
            raise ValueError("""loss_params["squared_grad_boltz_weight"] must be a float >= 0.""")
        else:
            self.squared_grad_boltz_weight = loss_params["squared_grad_boltz_weight"]

        if "mse_boltz_weight" not in loss_params.keys():
            self.mse_boltz_weight = 0.
            print("""mse_boltz_weight value not provided, set to default value of: """, self.mse_boltz_weight)
        elif type(loss_params["mse_boltz_weight"]) != float or loss_params["mse_boltz_weight"] < 0.:
            raise ValueError("""loss_params["mse_boltz_weight"] must be set as a float >= 0.""")
        else:
            self.mse_boltz_weight = loss_params["mse_boltz_weight"]

        if "mse_react_weight" not in loss_params.keys():
            self.mse_react_weight = 0.
            print("""mse_react_weight value not provided, set to default value of: """, self.mse_react_weight)
        elif type(loss_params["mse_react_weight"]) != float or loss_params["mse_react_weight"] < 0.:
            raise ValueError("""loss_params["mse_react_weight"] must be a float >= 0.""")
        else:
            self.mse_react_weight = loss_params["mse_react_weight"]

        if "l1_pen_weight" not in loss_params.keys():
            self.l1_pen_weight = 0
            print("""l1_pen_weight value not provided, set to default value of: """, self.l1_pen_weight)
        elif type(loss_params["l1_pen_weight"]) != float or loss_params["l1_pen_weight"] < 0.:
            raise ValueError("""loss_params["l1_pen_weight"] must be a float >= 0.""")
        else:
            self.l1_pen_weight = loss_params["l1_pen_weight"]

        if "l2_pen_weight" not in loss_params.keys():
            self.l2_pen_weight = 0
            print("""l2_pen_weight value not provided, set to default value of: """, self.l2_pen_weight)
        elif type(loss_params["l2_pen_weight"]) != float or loss_params["l2_pen_weight"] < 0.:
            raise ValueError("""loss_params["l2_pen_weight"] must be a float >= 0.""")
        else:
            self.l2_pen_weight = loss_params["l2_pen_weight"]

        if "pen_points_weight" not in loss_params.keys():
            self.pen_points_weight = 0.
            print("""pen_points_weight value not provided, set to default value of: """, self.pen_points_weight)
        elif type(loss_params["pen_points_weight"]) != float or loss_params["pen_points_weight"] < 0.:
            raise ValueError("""loss_params["pen_points_weight"] must be a float >= 0.""")
        else:
            self.pen_points_weight = loss_params["pen_points_weight"]

        if "n_wait" not in loss_params.keys():
            self.n_wait = 10
            print("""n_wait value not provided, set to default value of: """, self.n_wait)
        elif type(loss_params["n_wait"]) != int or loss_params["n_wait"] < 1:
            raise ValueError("""loss_params["n_wait"] must be a int >= 1""")
        else:
            self.n_wait = loss_params["n_wait"]

    def set_dataset(self, dataset):
        """Method to reset dataset

        :param dataset:         dict,
                                dataset["dt"], float: integration time step
                                dataset["beta"], float: the inverse temperature energy 1/ k_B*T

                                dataset["single_trajs_pos"], np.array with ndim==3, shape==[n_ini_pos, len(trajs), 2],
                                positions along the single trajs
                                dataset["single_trajs_mom"], np.array with ndim==3, shape==[n_ini_pos, len(trajs), 2],
                                momenta along the single trajs
                                dataset["single_trajs_gauss"], np.array with ndim==3, shape==[n_ini_pos, len(trajs)-1, 2],
                                gaussions along the single trajs
                                dataset["single_trajs_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                set to 1 if not provided

                                dataset["boltz_pos"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                an array of points on the 2D potentials distributed according ot the boltzmann gibbs
                                measure
                                dataset["boltz_pos_lagged"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                an array of points on the 2D potentials distributed according ot the boltzmann gibbs
                                measure with a time lag compared to the set of points dataset["boltz_pos"]
                                dataset["boltz_mom"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                dataset["boltz_mom_lagged"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                dataset["boltz_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                set to 1 if not provided

                                dataset["multiple_trajs_pos"], np.array with ndim==4, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                positions along the single trajs
                                dataset["multiple_trajs_mom"], np.array with ndim==3, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                momenta along the single trajs
                                dataset["multiple_trajs_weight"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                set to 1 if not provided

                                dataset["react_pos"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                an array  of points
                                on the 2D potentials distributed according ot the probability measure of reactive
                                trajectories
                                dataset["react_mom"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                dataset["react_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                set to 1 if not provided
        """
        self.dataset = dataset
        self.training_dataset = None
        self.test_dataset = None
        self.Kfold_splits = None
        self.train_data = None
        self.validation_data = None

    def set_pot(self, pot):
        """

        :param pot: two-dimensional potential object from potentials
        """
        self.pot = pot

    def set_penalization_points(self, penalization_points):
        """Method to reset penalization points

        :param penalization_points: np.array, ndim==2, shape=[any, 3(or 5)], penalization_point[:, :2(or 4)] are the
                                    postions (and momenta) on which the encoder is penalized if its values on these points differ from
                                    penalization_point[:, 2 (or 4)]
        """
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpe'
        self.penalization_points = torch.tensor(penalization_points.astype('float32')).to(device)

    def train_test_split(self, train_size=None, test_size=None, seed=None):
        """Method to separate the dataset into training and test dataset.

        :param train_size:  float or int, if float represents the proportion to include in the train split. If int, it
                            corresponds to the exact number of train samples. If None, it is set to be the complement of
                            the test_size. If both are None, it is set to 0.75
        :param test_size:   float or int, if float represents the proportion to include in the test split. If int, it
                            corresponds to the exact number of test samples. If None, it is set to be the complement of
                            the train_size. If both are None, it is set to 0.25
                            corresponds to the exact number of train samples
        :param seed:        int, random state for the splitting
        """
        dset = []
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        beta = self.dataset.pop('beta')
        dt = self.dataset.pop('dt')
        for i in range(len(self.dataset[next(iter(self.dataset.keys()))])):
            dset.append({key: torch.tensor(self.dataset[key][i].astype("float32")).to(device) for key in self.dataset.keys()})
        self.dataset["dt"] = dt
        self.dataset["beta"] = beta
        self.training_dataset, self.test_dataset = ttsplit(dset, test_size=test_size, train_size=train_size, random_state=seed)

    def split_training_dataset_K_folds(self, n_splits, seed=None):
        """ Allows to split the training dataset into multiple groups to optimize eventual hyperparameter

        :param n_splits: int, number of splits
        :param seed:     int, random state
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        self.Kfold_splits = []
        for i, fold in kf.split(self.training_dataset):
            self.Kfold_splits.append(fold)

    def set_train_val_data(self, split_index):
        """Set the training and validation set

        :param split_index:    int, the split of the training data_set, should be such that 0 <= split_index <= n_splits
        """
        if split_index < 0:
            raise ValueError("The split index must be between 0 and the number of splits - 1")
        validation = []
        for i in self.Kfold_splits[split_index]:
            validation.append(self.training_dataset[i])
        indices = np.setdiff1d(range(len(self.Kfold_splits)), split_index)
        train = []
        for i in self.Kfold_splits[indices[0]]:
            train.append(self.training_dataset[i])
        if len(self.Kfold_splits) > 2:
            for i in range(1, len(indices)):
                for j in self.Kfold_splits[indices[i]]:
                    train.append(self.training_dataset[j])
        self.train_data = train
        self.validation_data = validation

    @staticmethod
    def l1_penalization(model):
        """
        :param model:       committor_model model
        :return l1_pen:     torch float
        """
        return sum(p.abs().sum() for p in model.parameters()) / sum(torch.numel(p) for p in model.parameters())

    @staticmethod
    def l2_penalization(model):
        """
        :param model:       committor_model model
        :return l1_pen:     torch float
        """
        return sum(p.pow(2.0).sum() for p in model.parameters()) / sum(torch.numel(p) for p in model.parameters())

    def penalization_on_points(self):
        """

        :return:
        """
        if self.penalization_points.shape[-1] == 3:
            return torch.mean(
            (self.committor_model.encoder(self.penalization_points[:, :2]) - self.penalization_points[:, 2:]) ** 2)
        elif self.penalization_points.shape[-1] == 5:
            return torch.mean(
            (self.committor_model.encoder(self.penalization_points[:, :4]) - self.penalization_points[:, 4:]) ** 2)

    def plot_committor_iso_levels(self, ax, n_lines, set_lim=False):
        """Plot the iso-lines of a given function to the given ax

        :param ax:         Instance of matplotlib.axes.Axes
        :param n_lines:    int, number of iso-lines to plot
        :param set_lim:    boolean, whether the limits of the x and y axes should be set."""
        x = self.pot.x2d
        committor_on_grid = self.committor_model.xi_forward(x).reshape(self.pot.n_bins_x, self.pot.n_bins_y)
        if set_lim:
            ax.set_ylim(self.pot.y_domain[0], self.pot.y_domain[1])
            ax.set_xlim(self.pot.x_domain[0], self.pot.x_domain[1])
        ax.contour(self.pot.x_plot, self.pot.y_plot, committor_on_grid, n_lines, cmap='viridis')

    def squared_grad_encoder_penalization_boltz(self, inp):
        """Squared gradient of the encoder evaluated on the points distributed according to Boltzmann-Gibbs measure.

        :param inp:         torch.tensor, ndim==2, a chunk of self.training_data or self.test_data
        :param enc:         torch.tensor, ndim==2, shape==[any, 1], output of the encoder corresponding to part of the
                            inp distributed according to Boltzmann-Gibbs measure
        :return grad_enc:   torch float, squared gradient of the encoder ie: | \nabla q |²
        """
        if "boltz_mom" in self.dataset.keys():
            X = torch.concat((inp["boltz_pos"], inp["boltz_mom"]), dim=1)
            return (inp["boltz_weights"] * ((torch.autograd.grad(outputs=self.committor_model.committor(X).sum(),
                                                             inputs=X,
                                                             retain_graph=True,
                                                             create_graph=True)[0][:, :2]) ** 2).sum(dim=1)).mean()
        else:
            X = inp["boltz_pos"]
            return (inp["boltz_weights"] * ((torch.autograd.grad(outputs=self.committor_model.committor(X).sum(),
                                                             inputs=X,
                                                             retain_graph=True,
                                                             create_graph=True)[0][:, :2]) ** 2).sum(dim=1)).mean()

    def ito_loss_term(self, inp):
        """
        :param inp:         batch dict with at the keys: "single_trajs_pos","single_trajs_mom",
                            "single_trajs_gauss" and "single_trajs_weights"
        :return: ito_loss:  torch tensor
                """
        if "single_trajs_mom" in inp.keys():
            X = torch.concat((inp["single_trajs_pos"], inp["single_trajs_mom"]), dim=2)
            comm_of_x = self.committor_model.committor(X)
            grad_xi_dot_gauss = torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * torch.sum(
                torch.autograd.grad(outputs=comm_of_x[:, :-1, :].sum(),
                                    inputs=X,
                                    retain_graph=True,
                                    create_graph=True)[0][:, :-1, 2:4] * inp["single_trajs_gauss"][:, :, :], dim=(1, 2))
        else:
            X = inp["single_trajs_pos"]
            comm_of_x = self.committor_model.committor(X)
            grad_xi_dot_gauss = torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * torch.sum(
                torch.autograd.grad(outputs=comm_of_x[:, :-1 :].sum(),
                                    inputs=X,
                                    retain_graph=True,
                                    create_graph=True)[0][:, :-1, :] * inp["single_trajs_gauss"][:, :, :], dim=(1, 2))

        return torch.mean(inp["single_trajs_weights"] * ((comm_of_x[:, -1, 0] - comm_of_x[:, 0, 0])/1 - grad_xi_dot_gauss)**2)

    def log_ito_loss_term(self, inp):
        """
        :param inp:         batch dict with at the keys: "single_trajs_pos","single_trajs_mom",
                            "single_trajs_gauss" and "single_trajs_weights"
        :return: ito_loss:  torch tensor
        """
        if "single_trajs_mom" in inp.keys():
            X = torch.concat((inp["single_trajs_pos"], inp["single_trajs_mom"]), dim=2)
            log_comm_of_x = torch.log(self.committor_model.committor(X) + self.eps)
            grad_log_xi = torch.autograd.grad(outputs=log_comm_of_x[:, :-1, :].sum(),
                                          inputs=X,
                                          retain_graph=True,
                                          create_graph=True
                                          )[0][:, :-1, 2:4]
            integral_1 = torch.sum(grad_log_xi * (
                    (- torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * grad_log_xi + \
                     torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * inp["single_trajs_gauss"][:, :, :]),
                                 dim=(1, 2))
            log_1_m_comm_of_x = torch.log(1 - self.committor_model.committor(X) + self.eps)
            grad_log_1_m_xi = torch.autograd.grad(outputs=log_1_m_comm_of_x[:, :-1, :].sum(),
                                              inputs=X,
                                              retain_graph=True,
                                              create_graph=True
                                              )[0][:, :-1, 2:4]
            integral_2 = torch.sum(grad_log_1_m_xi * (
                    (- torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * grad_log_1_m_xi + \
                    torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * inp["single_trajs_gauss"][:, :, :]),
                                 dim=(1, 2))
        else:
            X = inp["single_trajs_pos"]
            log_comm_of_x = torch.log(self.committor_model.committor(X) + self.eps)
            grad_log_xi = torch.autograd.grad(outputs=log_comm_of_x.sum(),
                                              inputs=X,
                                              retain_graph=True,
                                              create_graph=True
                                              )[0][:, :-1, :2]
            integral_1 = torch.sum(grad_log_xi * (
                    (- torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * grad_log_xi + \
                    torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * inp["single_trajs_gauss"][:, :, :]),
                                 dim=(1, 2))
            log_1_m_comm_of_x = torch.log(1 - self.committor_model.committor(X) + self.eps)
            grad_log_1_m_xi = torch.autograd.grad(outputs=log_1_m_comm_of_x[:, :-1, :].sum(),
                                                  inputs=X,
                                                  retain_graph=True,
                                                  create_graph=True
                                                  )[0][:, :-1, :2]
            integral_2 = torch.sum(grad_log_1_m_xi * (
                    (- torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * grad_log_1_m_xi + \
                    torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * inp["single_trajs_gauss"][:, :, :]),
                                 dim=(1, 2))
        #print(torch.mean(inp["single_trajs_weights"] * (log_comm_of_x[:, -1, 0]- log_comm_of_x[:, 0, 0] - integral_1)**2))
        loss = torch.mean(inp["single_trajs_weights"] * \
                            ((log_comm_of_x[:, -1, 0] - log_comm_of_x[:, 0, 0] - integral_1)**2 + \
                             (log_1_m_comm_of_x[:, -1, 0] - log_1_m_comm_of_x[:, 0, 0] - integral_2)**2))
        return loss

    def multiple_traj_loss_term_1(self, inp):
        """
        :param inp:                 batch dict with the keys: "multiple_trajs_pos","multiple_trajs_mom",
                                    "multiple_trajs_gauss" and "multiple_trajs_weight"
        :return: multi_traj_loss:   torch tensor
        """
        if "multiple_trajs_mom" in inp.keys():
            X = torch.concat((inp["multiple_trajs_pos"], inp["multiple_trajs_mom"]), dim=3)
            comm_of_x = self.committor_model.committor(X)
            grad_xi_dot_gauss = torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * torch.sum(
                torch.autograd.grad(outputs=comm_of_x[:, :, :-1, :].sum(),
                                    inputs=X,
                                    retain_graph=True,
                                    create_graph=True)[0][:, :, :-1, 2:4] * inp["multiple_trajs_gauss"], dim=(2, 3))
        else:
            X = inp["multiple_trajs_pos"]
            comm_of_x = self.committor_model.committor(X)
            grad_xi_dot_gauss = torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * torch.sum(
                torch.autograd.grad(outputs=comm_of_x[:, :, :-1, :].sum(),
                                    inputs=X,
                                    retain_graph=True,
                                    create_graph=True)[0][:, :, :-1, :2] * inp["multiple_trajs_gauss"], dim=(2, 3))

        return torch.mean(inp["multiple_trajs_weights"] * (torch.mean((comm_of_x[:, :, -1, 0] - comm_of_x[:, :, 0, 0] - grad_xi_dot_gauss), dim=1)**2))

    def multiple_traj_loss_term_2(self, inp):
        """
        :param inp:                 batch dict with the keys: "multiple_trajs_pos","multiple_trajs_mom",
                                    "multiple_trajs_gauss" and "multiple_trajs_weight"
        :return: multi_traj_loss:   torch tensor
        """
        if "multiple_trajs_mom" in inp.keys():
            X = torch.concat((inp["multiple_trajs_pos"], inp["multiple_trajs_mom"]), dim=3)
            comm_of_x = self.committor_model.committor(X)
            grad_xi_dot_gauss = torch.sqrt(2 * self.dataset['dt'] / self.dataset['beta']) * torch.sum(
                torch.autograd.grad(outputs=comm_of_x[:, :, :-1, :].sum(),
                                    inputs=X,
                                    retain_graph=True,
                                    create_graph=True)[0][:, :, :-1, 2:4] * inp["multiple_trajs_gauss"], dim=(2, 3))
        else:
            X = inp["multiple_trajs_pos"]
            comm_of_x = self.committor_model.committor(X)
            grad_xi_dot_gauss = torch.sqrt(2 * self.dataset['dt'] / self.dataset['beta']) * torch.sum(
                torch.autograd.grad(outputs=comm_of_x[:, :, :-1, :].sum(),
                                    inputs=X,
                                    retain_graph=True,
                                    create_graph=True)[0][:, :, :-1, :] * inp["multiple_trajs_gauss"], dim=(2, 3))

        return torch.mean(inp["multiple_trajs_weights"] * torch.mean((comm_of_x[:, :, -1, 0] - comm_of_x[:, :, 0, 0] - grad_xi_dot_gauss)**2, dim=1))

    def fixed_point_ergo_traj_1(self, inp):
        """
        :param inp:                 batch dict with the keys: "boltz_pos","boltz_pos_lagged",, "boltz_mom",
                                    "boltz_pos_lagged" and "boltz_weights"
        :return: multi_traj_loss:   torch tensor
        """
        if "boltz_mom" in inp.keys():
            X = torch.concat((inp["boltz_pos"], inp["boltz_mom"]), dim=1)
            X_tau = torch.concat((inp["boltz_pos_lagged"], inp["boltz_mom_lagged"]), dim=1)
        else:
            X = inp["boltz_pos"]
            X_tau = inp["boltz_pos_lagged"]
        return torch.mean(self.committor_model.committor(X) * (self.committor_model.committor(X) - self.committor_model.committor(X_tau)))

    def fixed_point_ergo_traj_2(self, inp):
        """
        :param inp:                 batch dict with the keys: "boltz_pos","boltz_pos_lagged",, "boltz_mom",
                                    "boltz_pos_lagged" and "boltz_weights"
        :return: multi_traj_loss:   torch tensor
        """
        if "boltz_mom" in inp.keys():
            X = torch.concat((inp["boltz_pos"], inp["boltz_mom"]), dim=1)
            X_tau = torch.concat((inp["boltz_pos_lagged"], inp["boltz_mom_lagged"]), dim=1)
        else:
            X = inp["boltz_pos"]
            X_tau = inp["boltz_pos_lagged"]
        return torch.mean((self.committor_model.committor(X_tau) - self.committor_model.committor(X))**2)


class TainCommittorOneDecoder(TrainCommittor):
    """Class to train committor function models with one decoder

    The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
    last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
    other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
    to generate the validation and test data.
    """

    def __init__(self,  committor_model, pot, dataset, penalization_points=None, eps=1. * 10**(-8)):
        """

        :param committor_model:     AE model from committor.neural_net_models.CommittorOneDecoder with 2D input.
        :param pot:                 two-dimensional potential object from potentials
        :param dataset:             dict,
                                    dataset["dt"], float: integration time step
                                    dataset["beta"], float: the inverse temperature energy 1/ k_B*T

                                    dataset["single_trajs_pos"], np.array with ndim==3, shape==[n_ini_pos, len(trajs), 2],
                                    positions along the single trajs
                                    dataset["single_trajs_mom"], np.array with ndim==3, shape==[n_ini_pos, len(trajs), 2],
                                    momenta along the single trajs
                                    dataset["single_trajs_gauss"], np.array with ndim==3, shape==[n_ini_pos, len(trajs)-1, 2],
                                    gaussions along the single trajs
                                    dataset["single_trajs_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["boltz_pos"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array of points on the 2D potentials distributed according ot the boltzmann gibbs
                                    measure
                                    dataset["boltz_pos_lagged"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array of points on the 2D potentials distributed according ot the boltzmann gibbs
                                    measure with a time lag compared to the set of points dataset["boltz_pos"]
                                    dataset["boltz_mom"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["boltz_mom_lagged"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["boltz_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["multiple_trajs_pos"], np.array with ndim==4, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                    positions along the single trajs
                                    dataset["multiple_trajs_mom"], np.array with ndim==3, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                    momenta along the single trajs
                                    dataset["multiple_trajs_weight"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["react_pos"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array  of points
                                    on the 2D potentials distributed according ot the probability measure of reactive
                                    trajectories
                                    dataset["react_mom"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["react_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided
        :param penalization_points: np.array, ndim==2, shape=[any, 3], penalization_point[:, :2] are the points on which
                                    the encoder is penalized if its values on these points differ from
                                    penalization_point[:, 2]
        :param standardize:         boolean, whether the points should be rescaled so that the average in every
                                    direction is zero and has variance equal to 1
        :param zca_whiten:          boolean, whether the data should be whitened using mahalanobis whitening
        """
        super().__init__(committor_model,
                         pot,
                         dataset,
                         penalization_points=penalization_points,
                         eps=eps
                        )
        self.optimizer = None
    def set_optimizer(self, opt, learning_rate, parameters_to_train='all'):
        """
        :param opt:                 str, either 'SGD' or 'Adam' to use the corresponding pytorch optimizer.
        :param learning_rate:       float, value of the learning rate, typically 10**(-3) or smaller gives good results
                                    on the tested potentials
        :param parameters_to_train: str, either 'encoder', 'decoder' or 'all' to set what are the trained parameters
        """
        if opt == 'Adam' and parameters_to_train == 'all':
            self.optimizer = torch.optim.Adam([{'params': self.committor_model.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'all':
            self.optimizer = torch.optim.SGD([{'params': self.committor_model.parameters()}], lr=learning_rate)
        elif opt == 'Adam' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.Adam([{'params': self.committor_model.encoder.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.SGD([{'params': self.committor_model.encoder.parameters()}], lr=learning_rate)
        elif opt == 'Adam' and parameters_to_train == 'decoder':
            self.optimizer = torch.optim.Adam([{'params': self.committor_model.decoder.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'decoder':
            self.optimizer = torch.optim.SGD([{'params': self.committor_model.decoder.parameters()}], lr=learning_rate)
        else:
            raise ValueError("""The parameters opt and parameters_to_train must be specific str, see docstring""")

    def mse_loss_boltz(self, inp):
        """MSE term on points distributed according to Boltzmann-Gibbs measure.

        :param inp:     batch dict with the keys: "boltz_pos" and "boltz_weights"
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure.
        """
        if "boltz_pos" in self.dataset.keys():
            return torch.mean(inp["boltz_weights"] * torch.sum(
                (inp["boltz_pos"] - self.committor_model.decoder(self.committor_model.encoder(inp["boltz_pos"]))) ** 2,
                dim=1))
        else:
            raise ValueError("""Cannot compute this term if there are no points distributed according to the Bolzmann-
                                Gibbs measure in the dataset""")

    def mse_loss_react(self, inp):
        """MSE term on points distributed according to reactive trajectories measure.

        :param inp:     batch dict with the keys: "react_pos" and "react_weights"
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure.
        """
        if "react_pos" in self.dataset.keys():
            return torch.mean(inp["react_weights"]* torch.sum(
                (inp["react_pos"] - self.committor_model.decoder(self.committor_model.encoder(inp["react_pos"]))) ** 2,
                dim=1))
        else:
            raise ValueError("""Cannot compute this term if there are no points distributed according to the reactive
                                trajectory measure in the dataset""")

    def train(self, batch_size, max_epochs):
        """ Do the training of the model self.committor_model

        :param batch_size:      int >= 1, batch size for the mini-batching
        :param max_epochs:      int >= 1, maximal number of epoch of training
        :return loss_dict:      dict, contains the average loss for each epoch and its various components.
        """
        if self.optimizer is None:
            print("""The optimizer has not been set, see set_optimizer method. It is set to use 'Adam' optimizer \n 
                     with a 0.001 learning rate and optimize all the parameters of the model""")
            self.set_optimizer('Adam', 0.001)
        # prepare the various loss list to store
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.committor_model.to(device)
        loss_dict = {"train_loss": [], "test_loss": []}
        if "single_trajs_pos" in self.dataset.keys():
            loss_dict["train_ito_loss"] = []
            loss_dict["test_ito_loss"] = []
            loss_dict["train_log_ito_loss"] = []
            loss_dict["test_log_ito_loss"] = []
        if "boltz_pos" in self.dataset.keys():
            loss_dict["train_mse_boltz"] = []
            loss_dict["test_mse_boltz"] = []
            loss_dict["train_squared_grad_enc_blotz"] = []
            loss_dict["test_squared_grad_enc_blotz"] = []
        if "boltz_pos_lagged" in self.dataset.keys():
            loss_dict["train_fixed_point_ergodic_traj_1"] = []
            loss_dict["test_fixed_point_ergodic_traj_1"] = []
            loss_dict["train_fixed_point_ergodic_traj_2"] = []
            loss_dict["test_fixed_point_ergodic_traj_2"] = []
        if "react_points" in self.dataset.keys():
            loss_dict["train_mse_react"] = []
            loss_dict["test_mse_react"] = []
        if "multiple_trajs_pos" in self.dataset.keys():
            loss_dict["train_multi_traj_loss_1"] = []
            loss_dict["train_multi_traj_loss_2"] = []
            loss_dict["test_multi_traj_loss_1"] = []
            loss_dict["test_multi_traj_loss_2"] = []
        train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=self.validation_data, batch_size=batch_size, shuffle=True)
        epoch = 0
        model = copy.deepcopy(self.committor_model)
        while epoch < max_epochs:
            loss_dict["train_loss"].append([])
            if "single_trajs_pos" in self.dataset.keys():
                loss_dict["train_ito_loss"].append([])
                loss_dict["train_log_ito_loss"].append([])
            if "boltz_pos" in self.dataset.keys():
                loss_dict["train_mse_boltz"].append([])
                loss_dict["train_squared_grad_enc_blotz"].append([])
            if "boltz_pos_lagged" in self.dataset.keys():
                loss_dict["train_fixed_point_ergodic_traj_1"].append([])
                loss_dict["train_fixed_point_ergodic_traj_2"].append([])
            if "react_points" in self.dataset.keys():
                loss_dict["train_mse_react"].append([])
            if "multiple_trajs_pos" in self.dataset.keys():
                loss_dict["train_multi_traj_loss_1"].append([])
                loss_dict["train_multi_traj_loss_2"].append([])
            # train mode
            #self.committor_model.train()
            for iteration, batch in enumerate(train_loader):
                # Set gradient calculation capabilities
                for key in batch.keys():
                    batch[key].requires_grad_()
                # Set the gradient of with respect to parameters to zero
                self.optimizer.zero_grad()
                l1_pen = self.l1_penalization(self.committor_model)
                l2_pen = self.l2_penalization(self.committor_model)
                loss = self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen + \
                       self.pen_points_weight * self.penalization_on_points()
                if "single_trajs_pos" in self.dataset.keys():
                    if self.log_ito_loss_weight > 0.:
                        log_ito_term = self.log_ito_loss_term(batch)
                        loss_dict["train_log_ito_loss"][epoch].append(log_ito_term.cpu().detach().numpy())
                        loss += self.log_ito_loss_weight * log_ito_term
                    else:
                        loss_dict["train_log_ito_loss"][epoch].append(0.)
                    if self.ito_loss_weight > 0.:
                        ito_term = self.ito_loss_term(batch)
                        loss_dict["train_ito_loss"][epoch].append(ito_term.cpu().detach().numpy())
                        loss += self.ito_loss_weight * ito_term
                    else:
                        loss_dict["train_ito_loss"][epoch].append(0.)
                if "boltz_pos" in self.dataset.keys():
                    mse_blotz = self.mse_loss_boltz(batch)
                    squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(batch)
                    loss_dict["train_mse_boltz"][epoch].append(mse_blotz.cpu().detach().numpy())
                    loss_dict["train_squared_grad_enc_blotz"][epoch].append(squared_grad_enc_boltz.cpu().detach().numpy())
                    loss += self.mse_boltz_weight * mse_blotz + self.squared_grad_boltz_weight * squared_grad_enc_boltz
                if "boltz_pos_lagged" in self.dataset.keys():
                    if self.boltz_traj_fixed_point_loss_weight_1 > 0.:
                        fixed_point_ergodic_traj_1 = self.fixed_point_ergo_traj_1(batch)
                        loss += self.boltz_traj_fixed_point_loss_weight_1 * fixed_point_ergodic_traj_1
                        loss_dict["train_fixed_point_ergodic_traj_1"][epoch].append(fixed_point_ergodic_traj_1.cpu().detach().numpy())
                    if self.boltz_traj_fixed_point_loss_weight_2 > 0.:
                        fixed_point_ergodic_traj_2 = self.fixed_point_ergo_traj_2(batch)
                        loss += self.boltz_traj_fixed_point_loss_weight_2 * fixed_point_ergodic_traj_2
                        loss_dict["train_fixed_point_ergodic_traj_2"][epoch].append(fixed_point_ergodic_traj_2.cpu().detach().numpy())
                if "react_pos" in self.dataset.keys():
                    mse_react = self.mse_loss_react(batch)
                    loss_dict["train_mse_react"][epoch].append(mse_react.cpu().detach().numpy())
                    loss += self.mse_react_weight * mse_react
                if "multiple_trajs_pos" in self.dataset.keys():
                    multi_traj_loss_1 = self.multiple_traj_loss_term_1(batch)
                    multi_traj_loss_2 = self.multiple_traj_loss_term_1(batch)
                    loss_dict["train_multi_traj_loss_1"][epoch].append(multi_traj_loss_1.cpu().detach().numpy())
                    loss_dict["train_multi_traj_loss_2"][epoch].append(multi_traj_loss_2.cpu().detach().numpy())
                    loss += self.multiple_trajs_fixed_point_loss_weight_1 * multi_traj_loss_1 + \
                            self.multiple_trajs_fixed_point_loss_weight_2 * multi_traj_loss_2
                loss_dict["train_loss"][epoch].append(loss.cpu().detach().numpy())
                loss.backward()
                self.optimizer.step()
            loss_dict["train_loss"][epoch] = np.mean(loss_dict["train_loss"][epoch])
            loss_dict["test_loss"].append([])
            if "single_trajs_pos" in self.dataset.keys():
                loss_dict["train_ito_loss"][epoch] = np.mean(loss_dict["train_ito_loss"][epoch])
                loss_dict["test_ito_loss"].append([])
                loss_dict["train_log_ito_loss"][epoch] = np.mean(loss_dict["train_log_ito_loss"][epoch])
                loss_dict["test_log_ito_loss"].append([])
            if "boltz_pos" in self.dataset.keys():
                loss_dict["train_mse_boltz"][epoch] = np.mean(loss_dict["train_mse_boltz"][epoch])
                loss_dict["train_squared_grad_enc_blotz"][epoch] = np.mean(loss_dict["train_squared_grad_enc_blotz"][epoch])
                loss_dict["test_mse_boltz"].append([])
                loss_dict["test_squared_grad_enc_blotz"].append([])
            if "boltz_pos_lagged" in self.dataset.keys():
                loss_dict["train_fixed_point_ergodic_traj_1"][epoch] = np.mean(loss_dict["train_fixed_point_ergodic_traj_1"][epoch])
                loss_dict["train_fixed_point_ergodic_traj_2"][epoch] = np.mean(
                    loss_dict["train_fixed_point_ergodic_traj_2"][epoch])
                loss_dict["test_fixed_point_ergodic_traj_1"].append([])
                loss_dict["test_fixed_point_ergodic_traj_2"].append([])
            if "react_pos" in self.dataset.keys():
                loss_dict["train_mse_react"][epoch] = np.mean(loss_dict["train_mse_react"][epoch])
                loss_dict["test_mse_react"].append([])
            if "multiple_trajs_pos" in self.dataset.keys():
                loss_dict["train_multi_traj_loss_2"][epoch] = np.mean(loss_dict["train_multi_traj_loss_2"][epoch])
                loss_dict["train_multi_traj_loss_1"][epoch] = np.mean(loss_dict["train_multi_traj_loss_1"][epoch])
                loss_dict["test_multi_traj_loss_1"].append([])
                loss_dict["test_multi_traj_loss_2"].append([])

            # test mode
            #self.committor_model.eval()
            for iteration, batch in enumerate(test_loader):
                # Set gradient calculation capabilities
                for key in batch.keys():
                    batch[key].requires_grad_()
                l1_pen = self.l1_penalization(self.committor_model)
                l2_pen = self.l2_penalization(self.committor_model)
                loss = self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen + \
                       self.pen_points_weight * self.penalization_on_points()
                if "single_trajs_pos" in self.dataset.keys():
                    if self.log_ito_loss_weight > 0.:
                        log_ito_term = self.log_ito_loss_term(batch)
                        loss_dict["test_log_ito_loss"][epoch].append(log_ito_term.cpu().detach().numpy())
                        loss += self.log_ito_loss_weight * log_ito_term
                    else:
                        loss_dict["test_log_ito_loss"][epoch].append(0.)
                    if self.ito_loss_weight > 0.:
                        ito_term = self.ito_loss_term(batch)
                        loss_dict["test_ito_loss"][epoch].append(ito_term.cpu().detach().numpy())
                        loss += self.ito_loss_weight * ito_term
                    else:
                        loss_dict["test_ito_loss"][epoch].append(0.)
                if "boltz_pos" in self.dataset.keys():
                    mse_blotz = self.mse_loss_boltz(batch)
                    squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(batch)
                    loss_dict["test_mse_boltz"][epoch].append(mse_blotz.cpu().detach().numpy())
                    loss_dict["test_squared_grad_enc_blotz"][epoch].append(
                        squared_grad_enc_boltz.cpu().detach().numpy())
                    loss += self.mse_boltz_weight * mse_blotz + self.squared_grad_boltz_weight * squared_grad_enc_boltz
                if "boltz_pos_lagged" in self.dataset.keys():
                    if self.boltz_traj_fixed_point_loss_weight_1 > 0.:
                        fixed_point_ergodic_traj_1 = self.fixed_point_ergo_traj_1(batch)
                        loss += self.boltz_traj_fixed_point_loss_weight_1 * fixed_point_ergodic_traj_1
                        loss_dict["test_fixed_point_ergodic_traj_1"][epoch].append(
                            fixed_point_ergodic_traj_1.cpu().detach().numpy())
                    if self.boltz_traj_fixed_point_loss_weight_2 > 0.:
                        fixed_point_ergodic_traj_2 = self.fixed_point_ergo_traj_2(batch)
                        loss += self.boltz_traj_fixed_point_loss_weight_2 * fixed_point_ergodic_traj_2
                        loss_dict["test_fixed_point_ergodic_traj_2"][epoch].append(
                            fixed_point_ergodic_traj_2.cpu().detach().numpy())
                if "react_pos" in self.dataset.keys():
                    mse_react = self.mse_loss_react(batch)
                    loss_dict["test_mse_react"][epoch].append(mse_react.cpu().detach().numpy())
                    loss += self.mse_react_weight * mse_react
                if "multiple_trajs_pos" in self.dataset.keys():
                    multi_traj_loss_1 = self.multiple_traj_loss_term_1(batch)
                    multi_traj_loss_2 = self.multiple_traj_loss_term_1(batch)
                    loss_dict["test_multi_traj_loss_1"][epoch].append(multi_traj_loss_1.cpu().detach().numpy())
                    loss_dict["test_multi_traj_loss_2"][epoch].append(multi_traj_loss_2.cpu().detach().numpy())
                    loss += self.multiple_trajs_fixed_point_loss_weight_1 * multi_traj_loss_1 + \
                            self.multiple_trajs_fixed_point_loss_weight_2 * multi_traj_loss_2
                loss_dict["test_loss"][epoch].append(loss.cpu().detach().numpy())

            loss_dict["test_loss"][epoch] = np.mean(loss_dict["test_loss"][epoch])
            if "single_trajs_pos" in self.dataset.keys():
                loss_dict["test_ito_loss"][epoch] = np.mean(loss_dict["test_ito_loss"][epoch])
                loss_dict["test_log_ito_loss"][epoch] = np.mean(loss_dict["test_log_ito_loss"][epoch])
            if "boltz_pos" in self.dataset.keys():
                loss_dict["test_mse_boltz"][epoch] = np.mean(loss_dict["test_mse_boltz"][epoch])
                loss_dict["test_squared_grad_enc_blotz"][epoch] = np.mean(
                    loss_dict["test_squared_grad_enc_blotz"][epoch])
            if "boltz_pos_lagged" in self.dataset.keys():
                loss_dict["test_fixed_point_ergodic_traj_1"][epoch] = np.mean(
                    loss_dict["test_fixed_point_ergodic_traj_1"][epoch])
                loss_dict["test_fixed_point_ergodic_traj_2"][epoch] = np.mean(
                    loss_dict["test_fixed_point_ergodic_traj_2"][epoch])
            if "react_pos" in self.dataset.keys():
                loss_dict["test_mse_react"][epoch] = np.mean(loss_dict["test_mse_react"][epoch])
            if "multiple_trajs_pos" in self.dataset.keys():
                loss_dict["test_multi_traj_loss_2"][epoch] = np.mean(loss_dict["test_multi_traj_loss_2"][epoch])
                loss_dict["test_multi_traj_loss_1"][epoch] = np.mean(loss_dict["test_multi_traj_loss_1"][epoch])


            # Early stopping
            if loss_dict["test_loss"][epoch] == np.min(loss_dict["test_loss"]):
                model = copy.deepcopy(self.committor_model)
            if epoch >= self.n_wait:
                if np.min(loss_dict["test_loss"]) < np.min(loss_dict["test_loss"][- self.n_wait:]):
                    epoch = max_epochs
                    self.committor_model = model
            epoch += 1
        print("training ends after " + str(len(loss_dict["test_loss"])) + " epochs.\n")
        return loss_dict

    def print_test_loss(self, batch_size=None):
        """Print the test loss and its various components"""
        if batch_size is None:
            batch_size = len(self.test_dataset)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.committor_model.to(device)
        loss_dict = {"test_loss": []}
        if "single_trajs_pos" in self.dataset.keys():
            loss_dict["test_ito_loss"] = []
            loss_dict["test_log_ito_loss"] = []
        if "boltz_pos" in self.dataset.keys():
            loss_dict["test_mse_boltz"] = []
            loss_dict["test_squared_grad_enc_blotz"] = []
        if "boltz_pos_lagged" in self.dataset.keys():
            loss_dict["test_fixed_point_ergodic_traj_1"] = []
            loss_dict["test_fixed_point_ergodic_traj_2"] = []
        if "react_points" in self.dataset.keys():
            loss_dict["test_mse_react"] = []
        if "multiple_trajs_pos" in self.dataset.keys():
            loss_dict["test_multi_traj_loss_1"] = []
            loss_dict["test_multi_traj_loss_2"] = []
        for iteration, batch in enumerate(test_loader):
            for key in batch.keys():
                batch[key].requires_grad_()
            l1_pen = self.l1_penalization(self.committor_model)
            l2_pen = self.l2_penalization(self.committor_model)
            loss = self.l1_pen_weight * l1_pen + \
                   self.l2_pen_weight * l2_pen + \
                   self.pen_points_weight * self.penalization_on_points()
            if "single_trajs_pos" in self.dataset.keys():
                log_ito_term = self.log_ito_loss_term(batch)
                loss_dict["test_log_ito_loss"].append(log_ito_term.cpu().detach().numpy())
                loss += self.log_ito_loss_weight * log_ito_term
                ito_term = self.ito_loss_term(batch)
                loss_dict["test_ito_loss"].append(ito_term.cpu().detach().numpy())
                loss += self.ito_loss_weight * ito_term
            if "boltz_pos" in self.dataset.keys():
                mse_blotz = self.mse_loss_boltz(batch)
                squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(batch)
                loss_dict["test_mse_boltz"].append(mse_blotz.cpu().detach().numpy())
                loss_dict["test_squared_grad_enc_blotz"].append(
                    squared_grad_enc_boltz.cpu().detach().numpy())
                loss += self.mse_boltz_weight * mse_blotz + self.squared_grad_boltz_weight * squared_grad_enc_boltz
            if "boltz_pos_lagged" in self.dataset.keys():

                fixed_point_ergodic_traj_1 = self.fixed_point_ergo_traj_1(batch)
                loss += self.boltz_traj_fixed_point_loss_weight_1 * fixed_point_ergodic_traj_1
                loss_dict["test_fixed_point_ergodic_traj_1"].append(
                        fixed_point_ergodic_traj_1.cpu().detach().numpy())

                fixed_point_ergodic_traj_2 = self.fixed_point_ergo_traj_2(batch)
                loss += self.boltz_traj_fixed_point_loss_weight_2 * fixed_point_ergodic_traj_2
                loss_dict["test_fixed_point_ergodic_traj_2"].append(
                        fixed_point_ergodic_traj_2.cpu().detach().numpy())
            if "react_pos" in self.dataset.keys():
                mse_react = self.mse_loss_react(batch)
                loss_dict["test_mse_react"].append(mse_react.cpu().detach().numpy())
                loss += self.mse_react_weight * mse_react
            if "multiple_trajs_pos" in self.dataset.keys():
                multi_traj_loss_1 = self.multiple_traj_loss_term_1(batch)
                multi_traj_loss_2 = self.multiple_traj_loss_term_1(batch)
                loss_dict["test_multi_traj_loss_1"].append(multi_traj_loss_1.cpu().detach().numpy())
                loss_dict["test_multi_traj_loss_2"].append(multi_traj_loss_2.cpu().detach().numpy())
                loss += self.multiple_trajs_fixed_point_loss_weight_1 * multi_traj_loss_1 + \
                        self.multiple_trajs_fixed_point_loss_weight_2 * multi_traj_loss_2
            loss_dict["test_loss"].append(loss.cpu().detach().numpy())

        print("""Test loss: """, np.mean(loss_dict["test_loss"]))
        if "single_trajs_pos" in self.dataset.keys():
            print("""Test ito loss: """, np.mean(loss_dict["test_ito_loss"]))
        if "boltz_pos" in self.dataset.keys():
            print("""Test MSE boltz loss: """, np.mean(loss_dict["test_mse_boltz"]))
            print("""Test squared grad boltz loss: """, np.mean(loss_dict["test_squared_grad_enc_blotz"]))
        if "boltz_pos_lagged" in self.dataset.keys():
            print("""Test fixed point ergodic traj loss 1: """, np.mean(loss_dict["test_fixed_point_ergodic_traj_1"]))
            print("""Test fixed point ergodic traj loss 2: """, np.mean(loss_dict["test_fixed_point_ergodic_traj_2"]))
        if "react_pos" in self.dataset.keys():
            print("""Test MSE react loss: """, np.mean(loss_dict["test_mse_react"]))
        if "multiple_trajs_pos" in self.dataset.keys():
            print("""Test multi traj loss 1: """, np.mean(loss_dict["test_multi_traj_loss_1"]))
            print("""Test multi traj loss 2: """, np.mean(loss_dict["test_multi_traj_loss_2"]))
        self.committor_model.to('cpu')

    def plot_conditional_averages(self, ax, n_bins, set_lim=False):
        """Plot conditional averages computed on the full dataset of boltzmann distributed points to the given ax

        :param ax:              Instance of matplotlib.axes.Axes
        :param n_bins:          int, number of bins to compute conditional averages
        :param set_lim:         boolean, whether the limits of the x and y axes should be set.
        :return bin_population: list of ints, len==n_bins, population of each bin
        """
        X_given_z = [[] for i in range(n_bins)]
        Esp_X_given_z = []
        f_dec_z = []
        xi_values = self.committor_model.xi_forward(self.dataset["boltz_pos"])[:, 0]
        # equal-width bins
        z_bin = np.linspace(xi_values.min(), xi_values.max(), n_bins)
        # compute index of bin
        inds = np.digitize(xi_values, z_bin)
        # distribute train data to each bin
        bin_population = np.zeros(n_bins)
        for bin_idx in range(n_bins):
            X_given_z[bin_idx] = self.dataset["boltz_points"][inds == bin_idx + 1, :2]
            bin_population[bin_idx] = len(X_given_z[bin_idx])
            if len(X_given_z[bin_idx]) != 0:
                Esp_X_given_z.append(torch.tensor(X_given_z[bin_idx].astype('float32')).mean(dim=0))
                f_dec_z.append(self.committor_model(Esp_X_given_z[-1]).detach().numpy())
                Esp_X_given_z[-1] = Esp_X_given_z[-1].detach().numpy()
        Esp_X_given_z = np.array(Esp_X_given_z)
        f_dec_z = np.array(f_dec_z)
        if set_lim:
            ax.set_ylim(self.pot.y_domain[0], self.pot.y_domain[1])
            ax.set_xlim(self.pot.x_domain[0], self.pot.x_domain[1])
        ax.plot(Esp_X_given_z[:, 0], Esp_X_given_z[:, 1], '-o', color='blue', label='cond. avg. best model')
        ax.plot(f_dec_z[:, 0], f_dec_z[:, 1], '*', color='black', label='decoder best model')


class TainCommittorOverdampedTwoDecoder(TrainCommittorOverdamped):
    """Class to train committor function models with one decoder

    The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
    last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
    other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
    to generate the validation and test data.
    """

    def __init__(self,  committor_model, pot, dataset, penalization_points=None):
        """

        :param committor_model:     AE model from committor.neural_net_models.CommittorTwoDecoder with 2D input.
        :param pot:                 two-dimensional potential object from potentials
        :param dataset:             dict,
                                    dataset["dt"], float: integration time step
                                    dataset["beta"], float: the inverse temperature energy 1/ k_B*T

                                    dataset["single_trajs_pos"], np.array with ndim==3, shape==[n_ini_pos, len(trajs), 2],
                                    positions along the single trajs
                                    dataset["single_trajs_mom"], np.array with ndim==3, shape==[n_ini_pos, len(trajs), 2],
                                    momenta along the single trajs
                                    dataset["single_trajs_gauss"], np.array with ndim==3, shape==[n_ini_pos, len(trajs)-1, 2],
                                    gaussions along the single trajs
                                    dataset["single_trajs_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["boltz_pos"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array of points on the 2D potentials distributed according ot the boltzmann gibbs
                                    measure
                                    dataset["boltz_pos_lagged"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array of points on the 2D potentials distributed according ot the boltzmann gibbs
                                    measure with a time lag compared to the set of points dataset["boltz_pos"]
                                    dataset["boltz_mom"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["boltz_mom_lagged"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["boltz_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["multiple_trajs_pos"], np.array with ndim==4, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                    positions along the single trajs
                                    dataset["multiple_trajs_mom"], np.array with ndim==3, shape==[n_ini_pos, n_trajs, len(trajs), 2],
                                    momenta along the single trajs
                                    dataset["multiple_trajs_weight"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided

                                    dataset["react_pos"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    an array  of points
                                    on the 2D potentials distributed according ot the probability measure of reactive
                                    trajectories
                                    dataset["react_mom"], np.array with ndim==2, shape==[n_ini_pos, 2]
                                    dataset["react_weights"], np.array with ndim==2, shape==[n_ini_pos, 1]
                                    set to 1 if not provided
        :param penalization_points: np.array, ndim==2, shape=[any, 3], penalization_point[:, :2] are the points on which
                                    the encoder is penalized if its values on these points differ from
                                    penalization_point[:, 2]
        :param standardize:         boolean, whether the points should be rescaled so that the average in every
                                    direction is zero and has variance equal to 1
        :param zca_whiten:          boolean, whether the data should be whitened using mahalanobis whitening
        """
        super().__init__(committor_model,
                         pot,
                         dataset,
                         penalization_points=penalization_points,
                         standardize=standardize,
                         zca_whiten=zca_whiten)
        self.optimizer = None

    def set_optimizer(self, opt, learning_rate, parameters_to_train='all'):
        """

        :param opt:                 str, either 'SGD' or 'Adam' to use the corresponding pytorch optimizer.
        :param learning_rate:       float, value of the learning rate, typically 10**(-3) or smaller gives good results
                                    on the tested potentials
        :param parameters_to_train: str, either 'encoder', 'decoders',  or 'all' to set what are the trained parameters
        """
        if opt == 'Adam' and parameters_to_train == 'all':
            self.optimizer = torch.optim.Adam([{'params': self.committor_model.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'all':
            self.optimizer = torch.optim.SGD([{'params': self.committor_model.parameters()}], lr=learning_rate)
        elif opt == 'Adam' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.Adam([{'params': self.committor_model.encoder.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.SGD([{'params': self.committor_model.encoder.parameters()}], lr=learning_rate)
        elif opt == 'Adam' and parameters_to_train == 'decoders':
            self.optimizer = torch.optim.Adam(
                [{'params': self.committor_model.decoder1.parameters()},
                 {'params': self.committor_model.decoder2.parameters()}],
                lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'decoders':
            self.optimizer = torch.optim.SGD(
                [{'params': self.committor_model.decoder1.parameters()},
                 {'params': self.committor_model.decoder2.parameters()}],
                lr=learning_rate)
        else:
            raise ValueError("""The parameters opt and parameters_to_train must be specific str, see docstring""")

    def mse_loss_boltz(self, inp, dec1, dec2):
        """MSE term on points distributed according to Boltzmann-Gibbs measure.

        :param inp:     torch.tensor, ndim==2, a chunk of self.training_data or self.test_data
        :param dec1:    torch.tensor, ndim==2, shape==[any, 2], output of the first decoder corresponding to part of the
                        inp distributed according to Boltzmann-Gibbs measure
        :param dec2:    torch.tensor, ndim==2, shape==[any, 2], output of the second decoder corresponding to part of
                        the inp distributed according to Boltzmann-Gibbs measure
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure.
        """
        if "react_points" in self.dataset.keys() and "boltz_points" in self.dataset.keys():
            return torch.mean(inp[:, 2] *
                              torch.minimum(torch.sum((inp[:, 0:2] - dec1) ** 2, dim=1),
                                            torch.sum((inp[:, 0:2] - dec2) ** 2, dim=1)))
        elif "react_points" not in self.dataset.keys() and "boltz_points" in self.dataset.keys():
            return torch.mean(inp[:, 2] *
                              torch.minimum(torch.sum((inp[:, 0:2] - dec1) ** 2, dim=1),
                                            torch.sum((inp[:, 0:2] - dec2) ** 2, dim=1)))
        else:
            raise ValueError("""Cannot compute this term if there are no points distributed according to the Bolzmann-
                                Gibbs measure in the dataset""")

    def mse_loss_react(self, inp, dec1, dec2):
        """MSE term on points distributed according to reactive trajectories measure.

        :param inp:     torch.tensor, ndim==2, a chunk of self.training_data or self.test_data
        :param out:     torch.tensor, ndim==2, shape==[any, 2], output of the autoencoder corresponding to part of the
                        inp distributed according to reactive trajectories measure
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure.
        """
        if "react_points" in self.dataset.keys() and "boltz_points" in self.dataset.keys():
            return torch.mean(inp[:, 5] *
                              torch.minimum(torch.sum((inp[:, 3:5] - dec1) ** 2, dim=1),
                                            torch.sum((inp[:, 3:5] - dec2) ** 2, dim=1)))
        elif "react_points" in self.dataset.keys() and "boltz_points" not in self.dataset.keys():
            return torch.mean(inp[:, 2] *
                              torch.minimum(torch.sum((inp[:, 0:2] - dec1) ** 2, dim=1),
                                            torch.sum((inp[:, 0:2] - dec2) ** 2, dim=1)))
        else:
            raise ValueError("""Cannot compute this term if there are no points distributed according to the reactive
                                trajectory measure in the dataset""")

    def train(self, batch_size, max_epochs):
        """ Do the training of the model self.committor_model

        :param batch_size:      int >= 1, batch size for the mini-batching
        :param max_epochs:      int >= 1, maximal number of epoch of training
        :return loss_dict:      dict, contains the average loss for each epoch and its various components.
        """
        if self.optimizer is None:
            print("""The optimizer has not been set, see set_optimizer method. It is set to use 'Adam' optimizer \n 
                     with a 0.001 learning rate and optimize all the parameters of the model""")
            self.set_optimizer('Adam', 0.001)
        # prepare the various loss list to store
        loss_dict = {"train_loss": [], "test_loss": [], "train_ito_loss": [], "test_ito_loss": []}
        if "boltz_points" in self.dataset.keys():
            loss_dict["train_mse_boltz"] = []
            loss_dict["test_mse_boltz"] = []
            loss_dict["train_squared_grad_enc_blotz"] = []
            loss_dict["test_squared_grad_enc_blotz"] = []
        if "react_points" in self.dataset.keys():
            loss_dict["train_mse_react"] = []
            loss_dict["test_mse_react"] = []
        train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=self.validation_data, batch_size=batch_size, shuffle=True)
        epoch = 0
        model = copy.deepcopy(self.committor_model)
        while epoch < max_epochs:
            loss_dict["train_loss"].append([])
            loss_dict["train_ito_loss"].append([])
            if "boltz_points" in self.dataset.keys():
                loss_dict["train_mse_boltz"].append([])
                loss_dict["train_squared_grad_enc_blotz"].append([])
            if "react_points" in self.dataset.keys():
                loss_dict["train_mse_react"].append([])
            # train mode
            self.committor_model.train()
            for iteration, X in enumerate(train_loader):
                # Set gradient calculation capabilities
                X.requires_grad_()
                # Set the gradient of with respect to parameters to zero
                self.optimizer.zero_grad()
                l1_pen = self.l1_penalization(self.committor_model)
                l2_pen = self.l2_penalization(self.committor_model)
                loss = self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen + \
                       self.pen_points_weight * self.penalization_on_points()

                if "boltz_points" in self.dataset.keys() and "react_points" in self.dataset.keys():
                    enc = self.committor_model.encoder(X[:, 0:2])
                    dec1 = self.committor_model.decoder1(enc)
                    dec2 = self.committor_model.decoder2(enc)
                    mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
                    squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(X, enc)
                    enc_reac = self.committor_model.encoder(X[:, 3:5])
                    dec_reac1 = self.committor_model.decoder1(enc_reac)
                    dec_reac2 = self.committor_model.decoder2(enc_reac)
                    mse_react = self.mse_loss_react(X, dec_reac1, dec_reac2)
                    ito_term = self.ito_loss_term(X[:, 6:])
                    loss += self.ito_loss_weight * ito_term + \
                            self.mse_boltz_weight * mse_blotz + \
                            self.squared_grad_boltz_weight * squared_grad_enc_boltz + \
                            self.mse_react_weight * mse_react
                    loss_dict["train_mse_boltz"][epoch].append(mse_blotz.detach().numpy())
                    loss_dict["train_squared_grad_enc_blotz"][epoch].append(squared_grad_enc_boltz.detach().numpy())
                    loss_dict["train_mse_react"][epoch].append(mse_react.detach().numpy())

                elif "boltz_points" in self.dataset.keys() and "react_points" not in self.dataset.keys():
                    enc = self.committor_model.encoder(X[:, 0:2])
                    dec1 = self.committor_model.decoder1(enc)
                    dec2 = self.committor_model.decoder2(enc)
                    mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
                    squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(X, enc)
                    ito_term = self.ito_loss_term(X[:, 3:])
                    loss += self.ito_loss_weight * ito_term + \
                            self.mse_boltz_weight * mse_blotz + \
                            self.squared_grad_boltz_weight * squared_grad_enc_boltz
                    loss_dict["train_mse_boltz"][epoch].append(mse_blotz.detach().numpy())
                    loss_dict["train_squared_grad_enc_blotz"][epoch].append(squared_grad_enc_boltz.detach().numpy())

                elif "boltz_points" not in self.dataset.keys() and "react_points" in self.dataset.keys():
                    enc_reac = self.committor_model.encoder(X[:, 0:2])
                    dec_reac1 = self.committor_model.decoder1(enc_reac)
                    dec_reac2 = self.committor_model.decoder2(enc_reac)
                    mse_react = self.mse_loss_react(X, dec_reac1, dec_reac2)
                    ito_term = self.ito_loss_term(X[:, 3:])
                    loss += self.ito_loss_weight * ito_term + \
                            self.mse_react_weight * mse_react
                    loss_dict["train_mse_react"][epoch].append(mse_react.detach().numpy())
                else:
                    ito_term = self.ito_loss_term(X)
                    loss += self.ito_loss_weight * ito_term
                loss_dict["train_ito_loss"][epoch].append(ito_term.detach().numpy())
                loss_dict["train_loss"][epoch].append(loss.detach().numpy())
                loss.backward()
                self.optimizer.step()
            loss_dict["train_loss"][epoch] = np.mean(loss_dict["train_loss"][epoch])
            loss_dict["train_ito_loss"][epoch] = np.mean(loss_dict["train_ito_loss"][epoch])

            loss_dict["test_loss"].append([])
            loss_dict["test_ito_loss"].append([])
            if "boltz_points" in self.dataset.keys():
                loss_dict["train_mse_boltz"][epoch] = np.mean(loss_dict["train_mse_boltz"][epoch])
                loss_dict["train_squared_grad_enc_blotz"][epoch] = np.mean(
                    loss_dict["train_squared_grad_enc_blotz"][epoch])
                loss_dict["test_mse_boltz"].append([])
                loss_dict["test_squared_grad_enc_blotz"].append([])
            if "react_points" in self.dataset.keys():
                loss_dict["train_mse_react"][epoch] = np.mean(loss_dict["train_mse_react"][epoch])
                loss_dict["test_mse_react"].append([])
            # test mode
            self.committor_model.eval()
            for iteration, X in enumerate(test_loader):
                # Set gradient calculation capabilities
                # Set gradient calculation capabilities
                X.requires_grad_()
                l1_pen = self.l1_penalization(self.committor_model)
                l2_pen = self.l2_penalization(self.committor_model)
                loss = self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen + \
                       self.pen_points_weight * self.penalization_on_points()
                if "boltz_points" in self.dataset.keys() and "react_points" in self.dataset.keys():
                    enc = self.committor_model.encoder(X[:, 0:2])
                    dec1 = self.committor_model.decoder1(enc)
                    dec2 = self.committor_model.decoder2(enc)
                    mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
                    squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(X, enc)
                    enc_reac = self.committor_model.encoder(X[:, 3:5])
                    dec_reac1 = self.committor_model.decoder1(enc_reac)
                    dec_reac2 = self.committor_model.decoder2(enc_reac)
                    mse_react = self.mse_loss_react(X, dec_reac1, dec_reac2)
                    ito_term = self.ito_loss_term(X[:, 6:])
                    loss += self.ito_loss_weight * ito_term + \
                            self.mse_boltz_weight * mse_blotz + \
                            self.squared_grad_boltz_weight * squared_grad_enc_boltz + \
                            self.mse_react_weight * mse_react
                    loss_dict["test_mse_boltz"][epoch].append(mse_blotz.detach().numpy())
                    loss_dict["test_squared_grad_enc_blotz"][epoch].append(squared_grad_enc_boltz.detach().numpy())
                    loss_dict["test_mse_react"][epoch].append(mse_react.detach().numpy())
                elif "boltz_points" in self.dataset.keys() and "react_points" not in self.dataset.keys():
                    enc = self.committor_model.encoder(X[:, 0:2])
                    dec1 = self.committor_model.decoder1(enc)
                    dec2 = self.committor_model.decoder2(enc)
                    mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
                    squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(X, enc)
                    ito_term = self.ito_loss_term(X[:, 3:])
                    loss += self.ito_loss_weight * ito_term + \
                            self.mse_boltz_weight * mse_blotz + \
                            self.squared_grad_boltz_weight * squared_grad_enc_boltz
                    loss_dict["test_mse_boltz"][epoch].append(mse_blotz.detach().numpy())
                    loss_dict["test_squared_grad_enc_blotz"][epoch].append(squared_grad_enc_boltz.detach().numpy())
                elif "boltz_points" not in self.dataset.keys() and "react_points" in self.dataset.keys():
                    enc_reac = self.committor_model.encoder(X[:, 0:2])
                    dec_reac1 = self.committor_model.decoder1(enc_reac)
                    dec_reac2 = self.committor_model.decoder2(enc_reac)
                    mse_react = self.mse_loss_react(X, dec_reac1, dec_reac2)
                    ito_term = self.ito_loss_term(X[:, 3:])
                    loss += self.ito_loss_weight * ito_term + \
                            self.mse_react_weight * mse_react
                    loss_dict["test_mse_react"][epoch].append(mse_react.detach().numpy())
                else:
                    ito_term = self.ito_loss_term(X)
                    loss += self.ito_loss_weight * ito_term
                loss_dict["test_ito_loss"][epoch].append(ito_term.detach().numpy())
                loss_dict["test_loss"][epoch].append(loss.detach().numpy())
                loss.backward()
                self.optimizer.step()
            loss_dict["test_loss"][epoch] = np.mean(loss_dict["test_loss"][epoch])
            loss_dict["test_ito_loss"][epoch] = np.mean(loss_dict["test_ito_loss"][epoch])
            if "boltz_points" in self.dataset.keys():
                loss_dict["test_mse_boltz"][epoch] = np.mean(loss_dict["test_mse_boltz"][epoch])
                loss_dict["test_squared_grad_enc_blotz"][epoch] = np.mean(
                    loss_dict["test_squared_grad_enc_blotz"][epoch])
            if "react_points" in self.dataset.keys():
                loss_dict["test_mse_react"][epoch] = np.mean(loss_dict["test_mse_react"][epoch])

            # Early stopping
            if loss_dict["test_loss"][epoch] == np.min(loss_dict["test_loss"]):
                model = copy.deepcopy(self.committor_model)
            if epoch >= self.n_wait:
                if np.min(loss_dict["test_loss"]) < np.min(loss_dict["test_loss"][- self.n_wait:]):
                    epoch = max_epochs
                    self.committor_model = model
            epoch += 1
        print("training ends after " + str(len(loss_dict["test_loss"])) + " epochs.\n")
        return loss_dict

    def print_test_loss(self):
        """Print the test loss and its various components"""
        X = self.test_dataset
        X.requires_grad_()
        l1_pen = self.l1_penalization(self.committor_model)
        l2_pen = self.l2_penalization(self.committor_model)
        loss = self.l1_pen_weight * l1_pen + \
               self.l2_pen_weight * l2_pen + \
               self.pen_points_weight * self.penalization_on_points()

        if "boltz_points" in self.dataset.keys() and "react_points" in self.dataset.keys():
            enc = self.committor_model.encoder(X[:, 0:2])
            dec1 = self.committor_model.decoder1(enc)
            dec2 = self.committor_model.decoder2(enc)
            mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
            squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(X, enc)
            enc_reac = self.committor_model.encoder(X[:, 3:5])
            dec_reac1 = self.committor_model.decoder1(enc_reac)
            dec_reac2 = self.committor_model.decoder2(enc_reac)
            mse_react = self.mse_loss_react(X, dec_reac1, dec_reac2)
            ito_term = self.ito_loss_term(X[:, 6:])
            loss += self.ito_loss_weight * ito_term + \
                    self.mse_boltz_weight * mse_blotz + \
                    self.squared_grad_boltz_weight * squared_grad_enc_boltz + \
                    self.mse_react_weight * mse_react
            print("""Test MSE Boltzmann: """, mse_blotz)
            print("""Test squarred grad encoder boltzmann: """, squared_grad_enc_boltz)
            print("""Test MSE reactive: """, mse_react)
        elif "boltz_points" in self.dataset.keys() and "react_points" not in self.dataset.keys():
            enc = self.committor_model.encoder(X[:, 0:2])
            dec1 = self.committor_model.decoder1(enc)
            dec2 = self.committor_model.decoder2(enc)
            mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
            squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(X, enc)
            ito_term = self.ito_loss_term(X[:, 3:])
            loss += self.ito_loss_weight * ito_term + \
                    self.mse_boltz_weight * mse_blotz + \
                    self.squared_grad_boltz_weight * squared_grad_enc_boltz
            print("""Test MSE Boltzmann: """, mse_blotz)
            print("""Test squarred grad encoder boltzmann: """, squared_grad_enc_boltz)
        elif "boltz_points" not in self.dataset.keys() and "react_points" in self.dataset.keys():
            enc_reac = self.committor_model.encoder(X[:, 0:2])
            dec_reac1 = self.committor_model.decoder1(enc_reac)
            dec_reac2 = self.committor_model.decoder2(enc_reac)
            mse_react = self.mse_loss_react(X, dec_reac1, dec_reac2)
            ito_term = self.ito_loss_term(X[:, 3:])
            loss += self.ito_loss_weight * ito_term + \
                    self.mse_react_weight * mse_react
            print("""Test MSE reative: """, mse_react)
        else:
            ito_term = self.ito_loss_term(X)
            loss += self.ito_loss_weight * ito_term
        print("""Test Ito term loss: """, ito_term)
        print("""Test loss: """, loss)

    def plot_conditional_averages(self, ax, n_bins, set_lim=False):
        """Plot conditional averages computed on the full dataset to the given ax

        :param ax:              Instance of matplotlib.axes.Axes
        :param n_bins:          int, number of bins to compute conditional averages
        :param set_lim:         boolean, whether the limits of the x and y axes should be set.
        :return bin_population: list of ints, len==n_bins, population of each bin
        """
        X_given_z1 = [[] for i in range(n_bins)]
        X_given_z2 = [[] for i in range(n_bins)]
        Esp_X_given_z1 = []
        Esp_X_given_z2 = []
        f_dec_z1 = []
        f_dec_z2 = []
        react_points = torch.tensor(self.dataset["react_points"].astype('float32'))
        react_points_decoded1 = self.committor_model.decoder1(self.committor_model.encoder(react_points))
        react_points_decoded2 = self.committor_model.decoder2(self.committor_model.encoder(react_points))
        xi_values = self.committor_model.xi_forward(self.dataset["react_points"])[:, :2]
        # equal-width bins
        z_bin = np.linspace(xi_values.min(), xi_values.max(), n_bins)
        # compute index of bin
        inds = np.digitize(xi_values, z_bin)
        # distribute train data to each bin
        x1 = torch.sum((react_points - react_points_decoded1) ** 2, dim=1).detach().numpy() < torch.sum(
            (react_points - react_points_decoded2) ** 2,
            dim=1).detach().numpy()
        x2 = torch.sum((react_points - react_points_decoded2) ** 2, dim=1).detach().numpy() < torch.sum(
            (react_points - react_points_decoded1) ** 2,
            dim=1).detach().numpy()
        for bin_idx in range(n_bins):
            X_given_z1[bin_idx] = react_points[x1 * (inds == bin_idx + 1)[:, 0], :2]
            X_given_z2[bin_idx] = react_points[x2 * (inds == bin_idx + 1)[:, 0], :2]
            if len(X_given_z1[bin_idx]) > 0:
                Esp_X_given_z1.append(X_given_z1[bin_idx].mean(dim=0))
                f_dec_z1.append(self.committor_model.decoder1(self.committor_model.encoder(Esp_X_given_z1[-1])).detach().numpy())
                Esp_X_given_z1[-1] = Esp_X_given_z1[-1].detach().numpy()
            if len(X_given_z2[bin_idx]) > 0:
                Esp_X_given_z2.append(X_given_z2[bin_idx].mean(dim=0))
                f_dec_z2.append(self.committor_model.decoder2(self.committor_model.encoder(Esp_X_given_z2[-1])).detach().numpy())
                Esp_X_given_z2[-1] = Esp_X_given_z2[-1].detach().numpy()
        if self.standardize:
            Esp_X_given_z1 = self.scaler.inverse_transform(np.array(Esp_X_given_z1))
            f_dec_z1 = self.scaler.inverse_transform(np.array(f_dec_z1))
            Esp_X_given_z2 = self.scaler.inverse_transform(np.array(Esp_X_given_z2))
            f_dec_z2 = self.scaler.inverse_transform(np.array(f_dec_z2))
        elif self.zca_whiten:
            Esp_X_given_z1 = np.linalg.inv(self.ZCAMatrix).dot(np.array(Esp_X_given_z1).T).T
            f_dec_z1 = np.linalg.inv(self.ZCAMatrix).dot(np.array(f_dec_z1).T).T
            Esp_X_given_z2 = np.linalg.inv(self.ZCAMatrix).dot(np.array(Esp_X_given_z2).T).T
            f_dec_z2 = np.linalg.inv(self.ZCAMatrix).dot(np.array(f_dec_z2).T).T
        else:
            Esp_X_given_z1 = np.array(Esp_X_given_z1)
            f_dec_z1 = np.array(f_dec_z1)
            Esp_X_given_z2 = np.array(Esp_X_given_z2)
            f_dec_z2 = np.array(f_dec_z2)
        if set_lim:
            ax.set_ylim(self.pot.y_domain[0], self.pot.y_domain[1])
            ax.set_xlim(self.pot.x_domain[0], self.pot.x_domain[1])
        ax.plot(Esp_X_given_z1[:, 0], Esp_X_given_z1[:, 1], '-o', color='midnightblue', label='cond. avg. decoder 1')
        ax.plot(Esp_X_given_z2[:, 0], Esp_X_given_z2[:, 1], '-o', color='brown', label='cond. avg. decoder 2')
        ax.scatter(self.dataset["react_points"][x1][:, 0],
                    self.dataset["react_points"][x1][:, 1], color='blue', label='decoder1', s=1,
                    alpha=0.2)
        ax.scatter(self.dataset["react_points"][x2][:, 0],
                    self.dataset["react_points"][x2][:, 1], color='purple', label='decoder2', s=1,
                    alpha=0.2)
        ax.plot(f_dec_z1[:, 0], f_dec_z1[:, 1], '*', color='black', label='decoder 1')
        ax.plot(f_dec_z2[:, 0], f_dec_z2[:, 1], '*', color='pink', label='decoder 2')

    def plot_principal_curve_convergence(self, n_bins):
        """Plot conditional averages computed on the full dataset to the given ax

        :param n_bins:          int, number of bins to compute conditional averages
        """
        grads_enc1 = []
        grads_dec1 = []
        grads_enc2 = []
        grads_dec2 = []
        z_values1 = []
        z_values2 = []
        X_given_z1 = [[] for i in range(n_bins)]
        X_given_z2 = [[] for i in range(n_bins)]
        Esp_X_given_z1 = []
        Esp_X_given_z2 = []
        f_dec_z1 = []
        f_dec_z2 = []
        boltz_points = torch.tensor(self.dataset["boltz_points"].astype('float32'))
        boltz_points_decoded1 = self.committor_model.decoder1(self.committor_model.encoder(boltz_points))
        boltz_points_decoded2 = self.committor_model.decoder2(self.committor_model.encoder(boltz_points))
        xi_values = self.committor_model.xi_forward(self.dataset["boltz_points"])[:, 0]
        # equal-width bins
        z_bin = np.linspace(xi_values.min(), xi_values.max(), n_bins)
        # compute index of bin
        inds = np.digitize(xi_values, z_bin)
        # distribute train data to each bin
        x1 = torch.sum((boltz_points - boltz_points_decoded1) ** 2, dim=1).detach().numpy() < torch.sum(
            (boltz_points - boltz_points_decoded1) ** 2,
            dim=1).detach().numpy()
        x2 = torch.sum((boltz_points - boltz_points_decoded2) ** 2, dim=1).detach().numpy() < torch.sum(
            (boltz_points - boltz_points_decoded2) ** 2,
            dim=1).detach().numpy()
        for bin_idx in range(n_bins):
            X_given_z1[bin_idx] = boltz_points[x1 * (inds == bin_idx + 1), :2]
            X_given_z2[bin_idx] = boltz_points[x2 * (inds == bin_idx + 1), :2]
            if len(X_given_z1[bin_idx]) > 0:
                Esp_X_given_z1.append(torch.tensor(X_given_z1[bin_idx].astype('float32')).mean(dim=0))
                f_dec_z1.append(self.committor_model.decoder1(self.committor_model.encoder(Esp_X_given_z1[-1])).detach().numpy())
                Esp_X_given_z1[-1].requires_grad_()
                z1 = self.committor_model.encoder(Esp_X_given_z1[-1])
                z_values1.append(z1.detach().numpy())
                grad_f_enc1 = torch.autograd.grad(z1, Esp_X_given_z1[-1])[0]
                grads_enc1.append(grad_f_enc1.detach().numpy())
                z1.requires_grad_()
                grad_f_dec1 = torch.autograd.functional.jacobian(self.committor_model.decoder1, z1, create_graph=False).sum(dim=1)
                grads_dec1.append(grad_f_dec1.detach().numpy())
                Esp_X_given_z1[-1] = Esp_X_given_z1[-1].detach().numpy()
            if len(X_given_z2[bin_idx]) > 0:
                Esp_X_given_z2.append(torch.tensor(X_given_z2[bin_idx].astype('float32')).mean(dim=0))
                f_dec_z2.append(self.committor_model.decoder2(self.committor_model.encoder(Esp_X_given_z2[-1])).detach().numpy())
                Esp_X_given_z2[-1].requires_grad_()
                z2 = self.committor_model.encoder(Esp_X_given_z2[-1])
                z_values2.append(z2.detach().numpy())
                grad_f_enc2 = torch.autograd.grad(z2, Esp_X_given_z2[-1])[0]
                grads_enc2.append(grad_f_enc2.detach().numpy())
                z2.requires_grad_()
                grad_f_dec2 = torch.autograd.functional.jacobian(self.committor_model.decoder2, z2, create_graph=False).sum(dim=1)
                grads_dec2.append(grad_f_dec2.detach().numpy())
                Esp_X_given_z2[-1] = Esp_X_given_z2[-1].detach().numpy()
        grads_enc1 = np.array(grads_enc1)
        grads_dec1 = np.array(grads_dec1)
        grads_enc2 = np.array(grads_enc2)
        grads_dec2 = np.array(grads_dec2)
        cos_angles1 = np.sum(grads_enc1 * grads_dec1, axis=1) / np.sqrt(
            (np.sum(grads_enc1 ** 2, axis=1) * np.sum(grads_dec1 ** 2, axis=1)))
        dist_dec_exp1 = np.sum(
            (np.array(Esp_X_given_z1) - np.array(f_dec_z1)) ** 2, axis=1)
        cos_angles2 = np.sum(grads_enc2 * grads_dec2, axis=1) / np.sqrt(
            (np.sum(grads_enc2 ** 2, axis=1) * np.sum(grads_dec2 ** 2, axis=1)))
        dist_dec_exp2 = np.sum(
            (np.array(Esp_X_given_z2) - np.array(f_dec_z2)) ** 2, axis=1)
        plt.figure()
        plt.plot(z_values1, cos_angles1,
                 label="""cosine of angle between the gradient of the encoder at the \n 
                 cdt. avg. 1 and the derivative of the decoder1""")
        plt.plot(z_values2, cos_angles2,
                 label="""cosine of angle between the gradient of the encoder at the \n 
                 cdt. avg. 2 and the derivative of the decoder2""")
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(z_values1, dist_dec_exp1,
                 label='distance between the decoder 1 and the conditional average 1')
        plt.plot(z_values2, dist_dec_exp2,
                 label='distance between the decoder 2 and the conditional average 2')
        plt.legend()
        plt.show()

