import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.model_selection import KFold


class TrainEigenFunctions:
    """Class to train committor function models. It contains a decoder model to build a mean path from reactive
    trajectories or sampled Boltzmann distribution

    The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
    last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
    other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
    to generate the validation and test data.
    """

    def __init__(self, ef_model, pot, dataset, n_func):
        """

        :param ef_model:            NN model from eigenfunctions.neural_net_models with 2D input.
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
        self.ef_model = ef_model
        self.pot = pot
        self.dataset = dataset
        self.n_func = n_func
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpe'

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
        self.n_wait = None
        self.normalisation_weight = None
        self.grad_weight = None

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
        if "normalisation_weight" not in loss_params.keys():
            self.normalisation_weight = 0.
            print("""normalisation_weight value not provided, set to default value of: """, 0.)
        elif type(loss_params["normalisation_weight"]) != float or loss_params["normalisation_weight"] < 0.:
            raise ValueError("""loss_params["normalisation_weight"] must be a float >= 0.""")
        else:
            self.normalisation_weight = loss_params["normalisation_weight"]

        if "grad_weight" not in loss_params.keys():
            self.grad_weight = 1.
            print("""grad_weight value not provided, set to default value of: """, 0.)
        elif type(loss_params["grad_weight"]) != float or loss_params["grad_weight"] < 0.:
            raise ValueError("""loss_params["grad_weight"] must be a float >= 0.""")
        else:
            self.grad_weight = loss_params["grad_weight"]

        if "ito_loss_weight" not in loss_params.keys():
            self.ito_loss_weight = 0.
            print("""ito_loss_weight value not provided, set to default value of: """, 0.)
        elif type(loss_params["ito_loss_weight"]) != float or loss_params["ito_loss_weight"] < 0.:
            raise ValueError("""loss_params["ito_loss_weight"] must be a float >= 0.""")
        else:
            self.ito_loss_weight = loss_params["ito_loss_weight"]

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

    def plot_eigenfunc_iso_levels(self, ax, n_lines, idx=0, set_lim=False):
        """Plot the iso-lines of a given function to the given ax

        :param ax:         Instance of matplotlib.axes.Axes
        :param n_lines:    int, number of iso-lines to plot
        :param set_lim:    boolean, whether the limits of the x and y axes should be set."""
        x = self.pot.x2d
        ef_on_grid = self.ef_model.xi(x)[:, idx].reshape(self.pot.n_bins_x, self.pot.n_bins_y)
        if set_lim:
            ax.set_ylim(self.pot.y_domain[0], self.pot.y_domain[1])
            ax.set_xlim(self.pot.x_domain[0], self.pot.x_domain[1])
        ax.contour(self.pot.x_plot, self.pot.y_plot, ef_on_grid, n_lines, cmap='viridis')

    def squared_grad_encoder_penalization_boltz(self, inp):
        """Squared gradient of the encoder evaluated on the points distributed according to Boltzmann-Gibbs measure.

        :param inp:         torch.tensor, ndim==2, a chunk of self.training_data or self.test_data
        :param enc:         torch.tensor, ndim==2, shape==[any, 1], output of the encoder corresponding to part of the
                            inp distributed according to Boltzmann-Gibbs measure
        :return grad_enc:   torch float, squared gradient of the encoder ie: | \nabla q |²
        """
        if "boltz_mom" in self.dataset.keys():
            X = torch.concat((inp["boltz_pos"], inp["boltz_mom"]), dim=1)
        else:
            X = inp["boltz_pos"]
        return (inp["boltz_weights"] * ((torch.autograd.grad(outputs=self.ef_model.encoder(X).sum(),
                                                             inputs=X,
                                                             retain_graph=True,
                                                             create_graph=True)[0][:, :2]) ** 2).sum(dim=1)).mean()

    def ito_loss_term(self, inp, idx):
        """

        :param inp:         batch dict with at the keys: "single_trajs_pos","single_trajs_mom",
                            "single_trajs_gauss" and "single_trajs_weights"
        :return: ito_loss:  torch tensor
        """
        if "single_trajs_mom" in inp.keys():
            X = torch.concat((inp["single_trajs_pos"], inp["single_trajs_mom"]), dim=2)
            ef_of_x = self.ef_model.encoder(X)[:, idx:idx+1]
            grad_xi_dot_gauss = torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * torch.sum(
                torch.autograd.grad(outputs=ef_of_x[:, :-1, :].sum(),
                                    inputs=X,
                                    retain_graph=True,
                                    create_graph=True)[0][:, :-1, 2:4] * inp["single_trajs_gauss"][:, :, :], dim=(1, 2))
        else:
            X = inp["single_trajs_pos"]
            ef_of_x = self.ef_model.encoder(X)[:, idx:idx+1]
            grad_xi_dot_gauss = torch.sqrt(2 * torch.tensor(self.dataset['dt']) / torch.tensor(self.dataset['beta'])) * torch.sum(
                torch.autograd.grad(outputs=ef_of_x[:, :-1 :].sum(),
                                    inputs=X,
                                    retain_graph=True,
                                    create_graph=True)[0][:, :-1, :] * inp["single_trajs_gauss"][:, :, :], dim=(1, 2))
        l_of_X = (ef_of_x[:, -1, 0] - ef_of_x[:, 0, 0] - grad_xi_dot_gauss) / (self.dataset['dt'] * ef_of_x[:, :-1, :]).sum(dim=(1,2))
        """
        grad_l_of_X = torch.sum((torch.autograd.grad(outputs=ef_of_x[:, :-1 :].sum(),
                                          inputs=X,
                                          retain_graph=True,
                                          create_graph=True
                                          )[0][:, :, :])**2, dim=(1,2))
        """
        print(l_of_X)
        return torch.std(l_of_X) # + self.grad_weight * grad_l_of_X)

    def normalisation_term(self, inp):
        if "single_trajs_mom" in inp.keys():
            X = torch.concat((inp["single_trajs_pos"], inp["single_trajs_mom"]), dim=2)
            ef_of_x = self.ef_model.encoder(X)

        else:
            X = inp["single_trajs_pos"]
            ef_of_x = self.ef_model.encoder(X)
        return (torch.cov(ef_of_x.T) - torch.eye(torch.cov(ef_of_x.T).shape[0]))**2

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

    def set_optimizer(self, opt, learning_rate):
        """

        :param opt:                 str, either 'SGD' or 'Adam' to use the corresponding pytorch optimizer.
        :param learning_rate:       float, value of the learning rate, typically 10**(-3) or smaller gives good results
                                    on the tested potentials
        :param parameters_to_train: str, either 'encoder', 'decoder' or 'all' to set what are the trained parameters
        """
        if opt == 'Adam':
            self.optimizer = torch.optim.Adam([{'params': self.ef_model.parameters()}], lr=learning_rate)
        elif opt == 'SGD':
            self.optimizer = torch.optim.SGD([{'params': self.ef_model.parameters()}], lr=learning_rate)

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
        self.ef_model.to(device)
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
        model = copy.deepcopy(self.ef_model)
        while epoch < max_epochs:
            loss_dict["train_loss"].append([])
            if "single_trajs_pos" in self.dataset.keys():
                loss_dict["train_ito_loss"].append([])
                loss_dict["train_log_ito_loss"].append([])
            if "boltz_pos" in self.dataset.keys():
                loss_dict["train_squared_grad_enc_blotz"].append([])
            if "boltz_pos_lagged" in self.dataset.keys():
                loss_dict["train_fixed_point_ergodic_traj_1"].append([])
                loss_dict["train_fixed_point_ergodic_traj_2"].append([])
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
                l1_pen = self.l1_penalization(self.ef_model)
                l2_pen = self.l2_penalization(self.ef_model)
                loss = self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen
                if "single_trajs_pos" in self.dataset.keys():
                    ito_term = self.ito_loss_term(batch, 0)
                    for idx in range(1, self.n_func):
                        ito_term += self.ito_loss_term(batch, idx)
                    loss_dict["train_ito_loss"][epoch].append(ito_term.cpu().detach().numpy())
                    loss += self.ito_loss_weight * ito_term
                if "boltz_pos" in self.dataset.keys():
                    squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(batch)
                    loss_dict["train_squared_grad_enc_blotz"][epoch].append(squared_grad_enc_boltz.cpu().detach().numpy())
                    loss += self.squared_grad_boltz_weight * squared_grad_enc_boltz
                if "boltz_pos_lagged" in self.dataset.keys():
                    fixed_point_ergodic_traj_1 = self.fixed_point_ergo_traj_1(batch)
                    loss += self.boltz_traj_fixed_point_loss_weight_1 * fixed_point_ergodic_traj_1
                    loss_dict["train_fixed_point_ergodic_traj_1"][epoch].append(fixed_point_ergodic_traj_1.cpu().detach().numpy())
                    fixed_point_ergodic_traj_2 = self.fixed_point_ergo_traj_2(batch)
                    loss += self.boltz_traj_fixed_point_loss_weight_2 * fixed_point_ergodic_traj_2
                    loss_dict["train_fixed_point_ergodic_traj_2"][epoch].append(fixed_point_ergodic_traj_2.cpu().detach().numpy())
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
            if "boltz_pos" in self.dataset.keys():
                loss_dict["train_squared_grad_enc_blotz"][epoch] = np.mean(loss_dict["train_squared_grad_enc_blotz"][epoch])
                loss_dict["test_squared_grad_enc_blotz"].append([])
            if "boltz_pos_lagged" in self.dataset.keys():
                loss_dict["train_fixed_point_ergodic_traj_1"][epoch] = np.mean(loss_dict["train_fixed_point_ergodic_traj_1"][epoch])
                loss_dict["train_fixed_point_ergodic_traj_2"][epoch] = np.mean(
                    loss_dict["train_fixed_point_ergodic_traj_2"][epoch])
                loss_dict["test_fixed_point_ergodic_traj_1"].append([])
                loss_dict["test_fixed_point_ergodic_traj_2"].append([])
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
                l1_pen = self.l1_penalization(self.ef_model)
                l2_pen = self.l2_penalization(self.ef_model)
                loss = self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen
                if "single_trajs_pos" in self.dataset.keys():
                    ito_term = self.ito_loss_term(batch, 0)
                    for idx in range(1, self.n_func):
                        ito_term += self.ito_loss_term(batch, idx)
                    loss_dict["test_ito_loss"][epoch].append(ito_term.cpu().detach().numpy())
                    loss += self.ito_loss_weight * ito_term
                if "boltz_pos" in self.dataset.keys():
                    squared_grad_enc_boltz = self.squared_grad_encoder_penalization_boltz(batch)
                    loss_dict["test_squared_grad_enc_blotz"][epoch].append(
                        squared_grad_enc_boltz.cpu().detach().numpy())
                    loss += self.squared_grad_boltz_weight * squared_grad_enc_boltz
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
            if "boltz_pos" in self.dataset.keys():
                loss_dict["test_squared_grad_enc_blotz"][epoch] = np.mean(
                    loss_dict["test_squared_grad_enc_blotz"][epoch])
            if "boltz_pos_lagged" in self.dataset.keys():
                loss_dict["test_fixed_point_ergodic_traj_1"][epoch] = np.mean(
                    loss_dict["test_fixed_point_ergodic_traj_1"][epoch])
                loss_dict["test_fixed_point_ergodic_traj_2"][epoch] = np.mean(
                    loss_dict["test_fixed_point_ergodic_traj_2"][epoch])
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