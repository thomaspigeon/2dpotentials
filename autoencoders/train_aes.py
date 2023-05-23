import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class TrainAE:
    """Class to train AE models

    The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
    last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
    other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
    to generate the validation and test data.
    """

    def __init__(self, ae, pot, dataset, penalization_points=None, standardize=False, zca_whiten=False):
        """

        :param ae:                  AE model from autoencoders.ae_models.DeepAutoEncoder
        :param pot:                 two-dimensional potential object from potentials
        :param dataset:             dict, with dataset["boltz_points"] a np.array with ndim==2, shape==[any, 2] an array
                                    of points on the 2D potentials distributed according ot the boltzmann gibbs measure.
                                    Optionally,  dataset["boltz_weights"] is a np.array with ndim==2, shape==[any, 1],
                                    if not provided the weights are set to 1. Another option is dataset["react_points"],
                                    np.array with ndim==2, shape==[any, 2] an array  of points on the 2D potentials
                                    distributed according ot the probability measure of reactive trajectories.
                                    dataset["react_weights"] can be set as well, set to 1 if not provided
        :param penalization_points: np.array, ndim==2, shape=[any, 3], penalization_point[:, :2] are the points on which
                                    the encoder is penalized if its values on these points differ from
                                    penalization_point[:, 2]
        :param standardize:         boolean, whether the points should be rescaled so that the average in every
                                    direction is zero and has variance equal to 1  for the "boltz_points" WARNING the
                                    weights are currently not considered in this operation
        :param zca_whiten:          boolean, whether the data should be whitened using mahalanobis whitening fitted on
                                    the "boltz_points" WARNING the weights are currently not considered in this
                                    operation
        """
        self.ae = ae
        self.pot = pot
        self.dataset = dataset
        if penalization_points is None:
            penalization_points = np.append(np.append(pot.minR, np.zeros([1, 1]), axis=1),
                                            np.append(pot.minR, np.ones([1, 1]), axis=1),
                                            axis=0)
            self.penalization_point = torch.tensor(penalization_points.astype('float32'))
        else:
            self.penalization_point = torch.tensor(penalization_points.astype('float32'))
        self.standadize = standardize
        if standardize:
            self.scaler = StandardScaler()
            self.dataset["boltz_points"] = self.scaler.fit_transform(dataset["boltz_points"])
            penalization_points = self.scaler.transform(penalization_points)
            self.penalization_point = torch.tensor(penalization_points.astype('float32'))
            if "react_points" in dataset.keys():
                self.dataset["react_points"] = self.scaler.transform(dataset["react_points"])
        self.zca_whiten = zca_whiten
        if zca_whiten:
            cov_matrix = np.cov(dataset["boltz_points"], rowvar=False)  # Compute covariance matrix
            U, D, V = np.linalg.svd(cov_matrix)  # Single value decompostion
            epsilon = 1e-12  # Small value to prevent division by 0
            self.ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(D + epsilon)), U.T))
            self.dataset["boltz_points"] = self.ZCAMatrix.dot(dataset["boltz_points"].T).T
            penalization_points = self.ZCAMatrix.dot(penalization_points.T).T
            self.penalization_point = torch.tensor(penalization_points.astype('float32'))
            if "react_points" in dataset.keys():
                self.dataset["react_points"] = self.ZCAMatrix.dot(dataset["react_points"].T).T
        self.training_dataset = None
        self.test_dataset = None
        self.Kfold_splits = None
        self.train_data = None
        self.validation_data = None
        self.mse_boltz_weight = None
        self.mse_react_weight = None
        self.squared_grad_boltz_weight = None
        self.l1_pen_weight = None
        self.l2_pen_weight = None
        self.pen_points_weight = None
        self.var_enc_weight = None
        self.var_dist_dec_weight = None
        self.n_bins_var_dist_dec = None
        self.n_wait = None

    def set_loss_weight(self, loss_params):
        """Function to set the loss parameters.

        :param loss_params:     dict, containing: loss_params["mse_boltz_weight"] float >= 0, prefactor of the MSE term
                                of the loss on the Bolzmann gibbs distribution, loss_params["mse_react_weight"],
                                float >= 0, prefactor of the MSE term of the loss on the reactive trajectories'
                                distribution, loss_params["squared_grad_boltz_weight"], float >= 0, prefactor of the
                                squared gradient the encoder on the Bolzmann-Gibbs distribution,
                                loss_params["l1_pen_weight"], float >= 0, prefactor of the L1 weight decay penalization,
                                loss_params["l2_pen_weight"], float >= 0, prefactor of the L2 weight decay penalization,
                                loss_params["pen_points_weight"], float >= 0, prefactor of the penalization so that
                                certain points have a certain encoded value. loss_params["var_dist_dec_weight"] int >= 1,
                                prefactor of the penalization term to enforce that equal distances in the lattent space
                                correspond to equal distance in the decoded space. loss_params["n_bins_var_dist_dec"],
                                int >= 2, number of bins to enforce that equal distances in the lattent space correspond
                                to equal distance in the decoded space loss_params["n_wait"], int >= 1, early
                                stopping parameter. If the test loss has not decreased for n_wait epochs, the training
                                is stopped and the model kept in self.ae is the one corresponding to the minimal test
                                loss
        """
        if "mse_boltz_weight" not in loss_params.keys():
            raise ValueError("""loss_params["mse_boltz_weight"] must be set as a float >= 0.""")
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

        if "pen_points_weight" not in loss_params.keys():
            self.pen_points_weight = 0.
            print("""pen_points_weight value not provided, set to default value of: """, self.pen_points_weight)
        elif type(loss_params["pen_points_weight"]) != float or loss_params["pen_points_weight"] < 0.:
            raise ValueError("""loss_params["pen_points_weight"] must be a float >= 0.""")
        else:
            self.pen_points_weight = loss_params["pen_points_weight"]

        if "var_enc_weight" not in loss_params.keys():
            self.var_enc_weight = 0.
            print("""var_enc_weight value not provided, set to default value of: """, self.var_enc_weight)
        elif type(loss_params["var_enc_weight"]) != float or loss_params["var_enc_weight"] < 0.:
            raise ValueError("""loss_params["var_enc_weight"] must be set as a float >= 0.""")
        else:
            self.var_enc_weight = loss_params["var_enc_weight"]

        if "var_dist_dec_weight" not in loss_params.keys():
            self.var_dist_dec_weight = 0.
            print("""var_dist_dec_weight value not provided, set to default value of: """, self.var_dist_dec_weight)
        elif type(loss_params["var_dist_dec_weight"]) != float or loss_params["var_dist_dec_weight"] < 0.:
            raise ValueError("""loss_params["var_dist_dec_weight"] must be a float >= 0.""")
        else:
            self.var_dist_dec_weight = loss_params["var_dist_dec_weight"]

        if "n_bins_var_dist_dec" not in loss_params.keys():
            self.n_bins_var_dist_dec = 20
            print("""n_bins_var_dist_dec value not provided, set to default value of: """, self.n_bins_var_dist_dec)
        elif type(loss_params["n_bins_var_dist_dec"]) != int or loss_params["n_bins_var_dist_dec"] < 1:
            raise ValueError("""loss_params["n_bins_var_dist_dec"] must be a int >= 1""")
        else:
            self.n_bins_var_dist_dec = loss_params["n_bins_var_dist_dec"]

        if "n_wait" not in loss_params.keys():
            self.n_wait = 10
            print("""n_wait value not provided, set to default value of: """, self.n_wait)
        elif type(loss_params["n_wait"]) != int or loss_params["n_wait"] < 1:
            raise ValueError("""loss_params["n_wait"] must be a int >= 1""")
        else:
            self.n_wait = loss_params["n_wait"]

    def set_dataset(self, dataset):
        """Method to reset dataset

        :param dataset:             dict, with dataset["boltz_points"] a np.array with ndim==2, shape==[any, 2] an array
                                    of points on the 2D potentials distributed according ot the boltzmann gibbs measure.
                                    Optionally,  dataset["boltz_weights"] is a np.array with ndim==2, shape==[any, 1],
                                    if not provided the weights are set to 1. Another option is dataset["react_points"],
                                    np.array with ndim==2, shape==[any, 2] an array  of points on the 2D potentials
                                    distributed according ot the probability measure of reactive trajectories.
                                    dataset["react_weights"] can be set as well, set to 1 if not provided
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

        :param penalization_point:  np.array, ndim==2, shape=[any, 3], penalization_point[:, :2] are the points on which
                                    the encoder is penalized if its values on these points differ from
                                    penalization_point[:, 2]
        """
        self.penalization_point = torch.tensor(penalization_points.astype('float32'))

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
        if "boltz_weights" not in self.dataset.keys():
            self.dataset["boltz_weights"] = np.ones([len(self.dataset["boltz_points"]), 1])

        if "react_points" in self.dataset.keys():
            if "react_weights" not in self.dataset.keys():
                self.dataset["react_weights"] = np.ones([len(self.dataset["react_points"]), 1])
            (
                train_b_p,
                test_b_p,
                train_b_w,
                test_b_w,
                train_r_p,
                test_r_p,
                train_r_w,
                test_r_w
            ) = ttsplit(self.dataset["boltz_points"],
                        self.dataset["boltz_weights"],
                        self.dataset["react_points"],
                        self.dataset["react_weights"],
                        test_size=test_size,
                        train_size=train_size,
                        random_state=None)
            self.training_dataset = np.append(train_b_p, train_b_w, axis=1)
            self.training_dataset = np.append(self.training_dataset, train_r_p, axis=1)
            self.training_dataset = np.append(self.training_dataset, train_r_w, axis=1)
            self.test_dataset = np.append(test_b_p, test_b_w, axis=1)
            self.test_dataset = np.append(self.test_dataset, test_r_p, axis=1)
            self.test_dataset = np.append(self.test_dataset, test_r_w, axis=1)
            self.test_dataset = torch.tensor(self.test_dataset.astype('float32'))
        else:
            (
                train_b_p,
                test_b_p,
                train_b_w,
                test_b_w
            ) = ttsplit(self.dataset["boltz_points"],
                        self.dataset["boltz_weights"],
                        test_size=test_size,
                        train_size=train_size,
                        random_state=None)
            self.training_dataset = np.append(train_b_p, train_b_w, axis=1)
            self.test_dataset = np.append(test_b_p, test_b_w, axis=1)
            self.test_dataset = torch.tensor(self.test_dataset.astype('float32'))

    def split_training_dataset_K_folds(self, n_splits, seed=None):
        """ Allows to split the training dataset into multiple groups to optimize eventual hyperparameter

        :param n_splits: int, number of splits, must be int >= 2
        :param seed:     int, random state
        """
        if n_splits < 2:
            raise ValueError("The number of splits must be superior or equal to 2")
        kf = KFold(n_splits=n_splits, random_state=seed)
        self.Kfold_splits = []
        for i, fold in kf.split(self.training_dataset):
            self.Kfold_splits.append(fold)

    def set_train_val_data(self, split_index):
        """Set the training and validation set

        :param split_index:    int, the split of the training data_set, should be such that 0 <= split_index <= n_splits
        """
        if split_index < 0:
            raise ValueError("The split index must be between 0 and the number of splits - 1")
        validation = self.training_dataset[self.Kfold_splits[split_index]]
        indices = np.setdiff1d(range(len(self.Kfold_splits)), split_index)
        train = self.training_dataset[self.Kfold_splits[indices[0]]]
        if len(self.Kfold_splits) > 2:
            for i in range(1, len(indices)):
                train = np.append(train, self.training_dataset[self.Kfold_splits[indices[i]]], axis=0)
        self.train_data = torch.tensor(train.astype('float32'))
        self.validation_data = torch.tensor(validation.astype('float32'))

    @staticmethod
    def l1_penalization(model):
        """

        :param model:       ae model
        :return l1_pen:     torch float
        """
        return sum(p.abs().sum() for p in model.parameters()) / sum(torch.numel(p) for p in model.parameters())

    @staticmethod
    def l2_penalization(model):
        """

        :param model:       ae model
        :return l1_pen:     torch float
        """
        return sum(p.pow(2.0).sum() for p in model.parameters()) / sum(torch.numel(p) for p in model.parameters())

    @staticmethod
    def squared_grad_encoder_penalization(inp, enc):
        """Squared gradient of the encoder evaluated on the points distributed according to Boltzmann-Gibbs measure.

        :param inp:         torch.tensor, ndim==2, shape==[any, 3] or shape==[any, 6], inp[:, :2] is the input of the
                            auto-encoder distributed according to Bolzmann-Gibbs measure and inp[:, 2] are the
                            corresponding weights, inp[:, 2:] are not used
        :param enc:         torch.tensor, ndim==2, shape==[any, 1], output of the encoder corresponding to the described
                            inp
        :return grad_enc:   torch float, squared gradient of the encoder
        """
        return (inp[:, 2] * ((torch.autograd.grad(outputs=enc.sum(),
                                                  inputs=inp,
                                                  retain_graph=True,
                                                  create_graph=True)[0][:, :2]) ** 2).sum(dim=1)).mean()

    def penalization_on_points(self):
        """

        :return:
        """
        return torch.mean((self.ae.encoder(self.penalization_point[:, :2]) - self.penalization_point[:, 2] * torch.ones(
            self.penalization_point[:, :2].shape)) ** 2)

    @staticmethod
    def dist_dec_penalization(dec):
        """Compute the penalization term to ensure the distance between input decoded points is constant

        :param dec:     torch.tensor, ndim==2, shape==[any, 2], output of the auto-encoder corresponding to a linearly
                        spaced grid in the latent space
        :return var:    torch float, the variance or the distance between the successive decoded points.
        """
        return torch.var(torch.sqrt(torch.sum((dec[1:] - dec[:-1]) ** 2, dim=1)))

    def var_encoder(self):
        return (torch.var(self.ae.encoder(self.train_data[:, :2])) - 1)**2

    def plot_encoder_iso_levels(self, ax, n_lines, set_lim=False):
        """Plot the iso-lines of a given function to the given ax

        :param ax:         Instance of matplotlib.axes.Axes
        :param n_lines:    int, number of iso-lines to plot
        :param set_lim:    boolean, whether the limits of the x and y axes should be set."""
        if self.standadize:
            x = self.scaler.transform(self.pot.x2d)
        elif self.zca_whiten:
            x = self.ZCAMatrix.dot(self.pot.x2d.T).T
        else:
            x = self.pot.x2d
        ae_on_grid = self.ae.xi_ae(x).reshape(self.pot.n_bins_x, self.pot.n_bins_y)
        if set_lim:
            ax.set_ylim(self.pot.y_domain[0], self.pot.y_domain[1])
            ax.set_xlim(self.pot.x_domain[0], self.pot.x_domain[1])
        ax.contour(self.pot.x_plot, self.pot.y_plot, ae_on_grid, n_lines, cmap='viridis')


class TainAEOneDecoder(TrainAE):
    """Class to train AE models with one decoder

    The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
    last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
    other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
    to generate the validation and test data.
    """

    def __init__(self, ae, pot, dataset, penalization_points=None, standardize=False, zca_whiten=False):
        """

        :param ae:                  AE model from autoencoders.ae_models.DeepAutoEncoder
        :param pot:                 two-dimensional potential object from potentials
        :param dataset:             dict, with dataset["boltz_points"] a np.array with ndim==2, shape==[any, 2] an array
                                    of points on the 2D potentials distributed according ot the boltzmann gibbs measure.
                                    Optionally,  dataset["boltz_weights"] is a np.array with ndim==2, shape==[any, 1],
                                    if not provided the weights are set to 1. Another option is dataset["react_points"],
                                    np.array with ndim==2, shape==[any, 2] an array  of points on the 2D potentials
                                    distributed according ot the probability measure of reactive trajectories.
                                    dataset["react_weights"] can be set as well, set to 1 if not provided
        :param penalization_points: np.array, ndim==2, shape=[any, 3], penalization_point[:, :2] are the points on which
                                    the encoder is penalized if its values on these points differ from
                                    penalization_point[:, 2]
        :param standardize:         boolean, whether the points should be rescaled so that the average in every
                                    direction is zero and has variance equal to 1
        :param zca_whiten:          boolean, whether the data should be whitened using mahalanobis whitening
        """
        super().__init__(ae,
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
        :param parameters_to_train: str, either 'encoder', 'decoder' or 'all' to set what are the trained parameters
        """
        if opt == 'Adam' and parameters_to_train == 'all':
            self.optimizer = torch.optim.Adam([{'params': self.ae.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'all':
            self.optimizer = torch.optim.SGD([{'params': self.ae.parameters()}], lr=learning_rate)
        elif opt == 'Adam' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.Adam([{'params': self.ae.encoder.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.SGD([{'params': self.ae.encoder.parameters()}], lr=learning_rate)
        elif opt == 'Adam' and parameters_to_train == 'decoder':
            self.optimizer = torch.optim.Adam([{'params': self.ae.decoder.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'decoder':
            self.optimizer = torch.optim.SGD([{'params': self.ae.decoder.parameters()}], lr=learning_rate)
        else:
            raise ValueError("""The parameters opt and parameters_to_train must be specific str, see docstring""")

    @staticmethod
    def mse_loss_boltz(inp, out):
        """MSE term on points distributed according to Boltzmann-Gibbs measure.

        :param inp:     torch.tensor, ndim==2, shape==[any, 3] or shape==[any, 6], inp[:, :2] is the input of the
                        auto-encoder distributed according to Bolzmann-Gibbs measure and inp[:, 2] are the corresponding
                        weights, inp[:, 3:] are not used
        :param out:     torch.tensor, ndim==2, shape==[any, 2], output of the auto-encoder corresponding to the
                        described inp
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure.
        """
        return torch.mean(inp[:, 2] * torch.sum((inp[:, 0:2] - out) ** 2, dim=1))

    @staticmethod
    def mse_loss_react(inp, out):
        """MSE term on points distributed according to reactive trajectories measure.

        :param inp:     torch.tensor, ndim==2, shape==[any, 6], inp[:, 3:5] is the input of the
                        auto-encoder distributed according to reactive trajectories measure and inp[:, 5] are the
                        corresponding weights, inp[:, :3] are not used
        :param out:     torch.tensor, ndim==2, shape==[any, 2], output of the auto-encoder corresponding to the
                        described inp
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure.
        """
        return torch.mean(inp[:, 5] * torch.sum((inp[:, 3:5] - out) ** 2, dim=1))

    def train(self, batch_size, max_epochs):
        """ Do the training of the model self.ae

        :param batch_size:      int >= 1, batch size for the mini-batching
        :param max_epochs:      int >= 1, maximal number of epoch of training
        :return loss_dict:      dict, contains the average loss for each epoch and its various components.
        """
        if self.optimizer is None:
            print("""The optimizer has not been set, see set_optimizer method. It is set to use 'Adam' optimizer \n 
                     with a 0.001 learning rate and optimize all the parameters of the model""")
            self.set_optimizer('Adam', 0.001)
        # prepare the various loss list to store
        loss_dict = {"train_loss": [], "test_loss": [], "train_mse_boltz": [], "test_mse_boltz": [],
                     "train_squared_grad_enc_blotz": [], "test_squared_grad_enc_blotz": []}  
        if "react_points" in self.dataset.keys():
            loss_dict["train_mse_react"] = []
            loss_dict["test_mse_react"] = []
        train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=self.validation_data, batch_size=batch_size, shuffle=True)
        epoch = 0
        model = copy.deepcopy(self.ae)
        while epoch < max_epochs:
            loss_dict["train_loss"].append([])
            loss_dict["train_mse_boltz"].append([])
            loss_dict["train_squared_grad_enc_blotz"].append([])
            if "react_points" in self.dataset.keys():
                loss_dict["train_mse_react"].append([])
            # train mode
            self.ae.train()
            for iteration, X in enumerate(train_loader):
                # Set gradient calculation capabilities
                X.requires_grad_()
                # Set the gradient of with respect to parameters to zero
                self.optimizer.zero_grad()
                # Forward pass for boltzmann gibbs distributed points
                enc = self.ae.encoder(X[:, :2])
                out = self.ae.decoder(enc)
                # Compute the various loss terms
                mse_blotz = self.mse_loss_boltz(X, out)
                squared_grad_enc = self.squared_grad_encoder_penalization(X, enc)
                l1_pen = self.l1_penalization(self.ae)
                l2_pen = self.l2_penalization(self.ae)
                var_enc = self.var_encoder()
                loss = self.mse_boltz_weight * mse_blotz + \
                       self.var_enc_weight * var_enc + \
                       self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen + \
                       self.pen_points_weight * self.penalization_on_points()
                if "react_points" in self.dataset.keys():
                    # Forward pass for reactive trajectories
                    enc_reac = self.ae.encoder(X[:, 3:5])
                    out_reac = self.ae.decoder(enc_reac)
                    mse_react = self.mse_loss_react(X, out_reac)
                    loss += self.mse_react_weight * mse_react
                if self.var_dist_dec_weight > 0.:
                    enc_min = torch.min(self.ae.encoder(self.train_data[:, :2])).detach()
                    enc_max = torch.max(self.ae.encoder(self.train_data[:, :2])).detach()
                    z_grid = torch.linspace(enc_min, enc_max, self.n_bins_var_dist_dec)
                    dec = self.ae.decoder(z_grid)
                    loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec)
                loss.backward()
                self.optimizer.step()
                loss_dict["train_loss"][epoch].append(loss.detach().numpy())
                loss_dict["train_mse_boltz"][epoch].append(mse_blotz.detach().numpy())
                loss_dict["train_squared_grad_enc_blotz"][epoch].append(squared_grad_enc.detach().numpy())
                if "react_points" in self.dataset.keys():
                    loss_dict["train_mse_react"][epoch].append(mse_react.detach().numpy())
            loss_dict["train_loss"][epoch] = np.mean(loss_dict["train_loss"][epoch])
            loss_dict["train_mse_boltz"][epoch] = np.mean(loss_dict["train_mse_boltz"][epoch])
            loss_dict["train_squared_grad_enc_blotz"][epoch] = np.mean(loss_dict["train_squared_grad_enc_blotz"][epoch])
            if "react_points" in self.dataset.keys():
                loss_dict["train_mse_react"][epoch] = np.mean(loss_dict["train_mse_react"][epoch])

            loss_dict["test_loss"].append([])
            loss_dict["test_mse_boltz"].append([])
            loss_dict["test_squared_grad_enc_blotz"].append([])
            if "react_points" in self.dataset.keys():
                loss_dict["test_mse_react"].append([])
            # test mode
            self.ae.eval()
            for iteration, X in enumerate(test_loader):
                # Set gradient calculation capabilities
                X.requires_grad_()
                # Forward pass for boltzmann gibbs distributed points
                enc = self.ae.encoder(X[:, :2])
                out = self.ae.decoder(enc)
                # Compute the various loss terms
                mse_blotz = self.mse_loss_boltz(X, out)
                squared_grad_enc = self.squared_grad_encoder_penalization(X, enc)
                l1_pen = self.l1_penalization(self.ae)
                l2_pen = self.l2_penalization(self.ae)
                var_enc = self.var_encoder()
                loss = self.mse_boltz_weight * mse_blotz + \
                       self.var_enc_weight * var_enc + \
                       self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen + \
                       self.pen_points_weight * self.penalization_on_points()
                if "react_points" in self.dataset.keys():
                    # Forward pass for reactive trajectories
                    enc_reac = self.ae.encoder(X[:, 3:5])
                    out_reac = self.ae.decoder(enc_reac)
                    mse_react = self.mse_loss_react(X, out_reac)
                    loss += self.mse_react_weight * mse_react
                if self.var_dist_dec_weight > 0.:
                    enc_min = torch.min(self.ae.encoder(self.train_data[:, :2])).detach()
                    enc_max = torch.max(self.ae.encoder(self.train_data[:, :2])).detach()
                    z_grid = torch.linspace(enc_min, enc_max, self.n_bins_var_dist_dec)
                    dec = self.ae.decoder(z_grid)
                    loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec)
                loss_dict["test_loss"][epoch].append(loss.detach().numpy())
                loss_dict["test_mse_boltz"][epoch].append(mse_blotz.detach().numpy())
                loss_dict["test_squared_grad_enc_blotz"][epoch].append(squared_grad_enc.detach().numpy())
                if "react_points" in self.dataset.keys():
                    loss_dict["test_mse_react"][epoch].append(mse_react.detach().numpy())
            loss_dict["test_loss"][epoch] = np.mean(loss_dict["test_loss"][epoch])
            loss_dict["test_mse_boltz"][epoch] = np.mean(loss_dict["test_mse_boltz"][epoch])
            loss_dict["test_squared_grad_enc_blotz"][epoch] = np.mean(loss_dict["test_squared_grad_enc_blotz"][epoch])
            if "react_points" in self.dataset.keys():
                loss_dict["test_mse_react"][epoch] = np.mean(loss_dict["test_mse_react"][epoch])
            # Early stopping
            if loss_dict["test_loss"][epoch] == np.min(loss_dict["test_loss"]):
                model = copy.deepcopy(self.ae)
            if epoch >= self.n_wait:
                if np.min(loss_dict["test_loss"]) < np.min(loss_dict["test_loss"][- self.n_wait:]):
                    epoch = max_epochs
                    self.ae = model
            epoch += 1
        print("training ends after " + str(len(loss_dict["test_loss"])) + " epochs.\n")
        return loss_dict

    def print_test_loss(self):
        """Print the test loss and its various components"""
        X = self.test_dataset
        X.requires_grad_()
        enc = self.ae.encoder(X[:, :2])
        out = self.ae.decoder(enc)
        # Compute the various loss terms
        mse_blotz = self.mse_loss_boltz(X, out)
        squared_grad_enc = self.squared_grad_encoder_penalization(X, enc)
        l1_pen = self.l1_penalization(self.ae)
        l2_pen = self.l2_penalization(self.ae)
        var_enc = self.var_encoder()
        loss = self.mse_boltz_weight * mse_blotz + \
               self.var_enc_weight * var_enc + \
               self.l1_pen_weight * l1_pen + \
               self.l2_pen_weight * l2_pen + \
               self.pen_points_weight * self.penalization_on_points()
        if "react_points" in self.dataset.keys():
            # Forward pass for reactive trajectories
            enc_reac = self.ae.encoder(X[:, 3:5])
            out_reac = self.ae.decoder(enc_reac)
            mse_react = self.mse_loss_react(X, out_reac)
            loss += self.mse_react_weight * mse_react
        if self.var_dist_dec_weight > 0.:
            enc_min = torch.min(self.ae.encoder(X[:, :2])).detach()
            enc_max = torch.max(self.ae.encoder(X[:, :2])).detach()
            z_grid = torch.linspace(enc_min, enc_max, self.n_bins_var_dist_dec)
            dec = self.ae.decoder(z_grid)
            loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec)
        print("""Test loss: """, loss)
        print("""Test MSE Boltzmann: """, mse_blotz)
        print("""Test squarred grad encoder: """, squared_grad_enc)
        if "react_points" in self.dataset.keys():
            print("""Test MSE reactive: """, mse_react)

    def plot_conditional_averages(self, ax, n_bins, set_lim=False):
        """Plot conditional averages computed on the full dataset to the given ax

        :param ax:              Instance of matplotlib.axes.Axes
        :param n_bins:          int, number of bins to compute conditional averages
        :param set_lim:         boolean, whether the limits of the x and y axes should be set.
        :return bin_population: list of ints, len==n_bins, population of each bin
        """
        X_given_z = [[] for i in range(n_bins)]
        Esp_X_given_z = []
        f_dec_z = []
        xi_values = self.ae.xi_ae(self.dataset["boltz_points"])[:, 0]
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
                f_dec_z.append(self.ae(Esp_X_given_z[-1]).detach().numpy())
                Esp_X_given_z[-1] = Esp_X_given_z[-1].detach().numpy()
        if self.standadize:
            Esp_X_given_z = self.scaler.inverse_transform(np.array(Esp_X_given_z))
            f_dec_z = self.scaler.inverse_transform(np.array(f_dec_z))
        elif self.zca_whiten:
            Esp_X_given_z = np.linalg.inv(self.ZCAMatrix).dot(np.array(Esp_X_given_z).T).T
            f_dec_z = np.linalg.inv(self.ZCAMatrix).dot(np.array(f_dec_z).T).T
        else:
            Esp_X_given_z = np.array(Esp_X_given_z)
            f_dec_z = np.array(f_dec_z)
        if set_lim:
            ax.set_ylim(self.pot.y_domain[0], self.pot.y_domain[1])
            ax.set_xlim(self.pot.x_domain[0], self.pot.x_domain[1])
        ax.plot(Esp_X_given_z[:, 0], Esp_X_given_z[:, 1], '-o', color='blue', label='cond. avg. best model')
        ax.plot(f_dec_z[:, 0], f_dec_z[:, 1], '*', color='black', label='decoder best model')

    def plot_principal_curve_convergence(self, n_bins):
        """Plot conditional averages computed on the full dataset to the given ax

        :param n_bins:          int, number of bins to compute conditional averages
        """
        grads_enc = []
        grads_dec = []
        z_values = []
        X_given_z = [[] for i in range(n_bins)]
        Esp_X_given_z = []
        f_dec_z = []
        xi_values = self.ae.xi_ae(self.dataset["boltz_points"])[:, 0]
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
                f_dec_z.append(self.ae(Esp_X_given_z[-1]).detach().numpy())
                Esp_X_given_z[-1].requires_grad_()
                z = self.ae.encoder(Esp_X_given_z[-1])
                z_values.append(z.detach().numpy())
                grad_f_enc = torch.autograd.grad(z, Esp_X_given_z[-1])[0]
                grads_enc.append(grad_f_enc.detach().numpy())
                z.requires_grad_()
                grad_f_dec = torch.autograd.functional.jacobian(self.ae.decoder, z, create_graph=False).sum(dim=1)
                grads_dec.append(grad_f_dec.detach().numpy())
                Esp_X_given_z[-1] = Esp_X_given_z[-1].detach().numpy()
        grads_enc = np.array(grads_enc)
        grads_dec = np.array(grads_dec)
        cos_angles = np.sum(grads_enc * grads_dec, axis=1) / np.sqrt(
            (np.sum(grads_enc ** 2, axis=1) * np.sum(grads_dec ** 2, axis=1)))
        dist_fec_exp = (bin_population / np.sum(bin_population)) * np.sum(
            (np.array(Esp_X_given_z) - np.array(f_dec_z)) ** 2, axis=1)
        plt.figure()
        plt.plot(z_values, cos_angles)
        plt.title('cosine of angle between the gradient of the encoder \n at the cdt. avg. and the derivative of the decoder')
        plt.show()
        plt.figure()
        plt.plot(z_values, dist_fec_exp)
        plt.title('distance between the decoder and the conditional average')
        plt.show()


class TainAETwoDecoder(TrainAE):
    """Class to train AE models with one decoder

    The dataset is the described in the __init__. It is first split into a training dataset and a test dataset. This
    last one is not used at all for validation of hyperparameters. It is used to computed conditional averages and
    other quantities to charaterize the convergence of the training. The training dataset has to be split into K  folds
    to generate the validation and test data.
    """

    def __init__(self, ae, pot, dataset, penalization_points=None, standardize=False, zca_whiten=False):
        """

        :param ae:                  AE model from autoencoders.ae_models.DeepAutoEncoder
        :param pot:                 two-dimensional potential object from potentials
        :param dataset:             dict, with dataset["boltz_points"] a np.array with ndim==2, shape==[any, 2] an array
                                    of points on the 2D potentials distributed according ot the boltzmann gibbs measure.
                                    Optionally,  dataset["boltz_weights"] is a np.array with ndim==2, shape==[any, 1],
                                    if not provided the weights are set to 1. Another option is dataset["react_points"],
                                    np.array with ndim==2, shape==[any, 2] an array  of points on the 2D potentials
                                    distributed according ot the probability measure of reactive trajectories.
                                    dataset["react_weights"] can be set as well, set to 1 if not provided
        :param penalization_points: np.array, ndim==2, shape=[any, 3], penalization_point[:, :2] are the points on which
                                    the encoder is penalized if its values on these points differ from
                                    penalization_point[:, 2]
        :param standardize:         boolean, whether the points should be rescaled so that the average in every
                                    direction is zero and has variance equal to 1
        :param zca_whiten:          boolean, whether the data should be whitened using mahalanobis whitening
        """
        super().__init__(ae,
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
            self.optimizer = torch.optim.Adam([{'params': self.ae.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'all':
            self.optimizer = torch.optim.SGD([{'params': self.ae.parameters()}], lr=learning_rate)
        elif opt == 'Adam' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.Adam([{'params': self.ae.encoder.parameters()}], lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'encoder':
            self.optimizer = torch.optim.SGD([{'params': self.ae.encoder.parameters()}], lr=learning_rate)
        elif opt == 'Adam' and parameters_to_train == 'decoders':
            self.optimizer = torch.optim.Adam(
                [{'params': self.ae.decoder1.parameters()}, {'params': self.ae.decoder2.parameters()}],
                lr=learning_rate)
        elif opt == 'SGD' and parameters_to_train == 'decoders':
            self.optimizer = torch.optim.SGD(
                [{'params': self.ae.decoder1.parameters()}, {'params': self.ae.decoder2.parameters()}],
                lr=learning_rate)
        else:
            raise ValueError("""The parameters opt and parameters_to_train must be specific str, see docstring""")

    @staticmethod
    def mse_loss_boltz(inp, dec1, dec2):
        """MSE term on points distributed according to Boltzmann-Gibbs measure.

        :param inp:     torch.tensor, ndim==2, shape==[any, 3] or shape==[any, 6], inp[:, :2] is the input of the
                        auto-encoder distributed according to Bolzmann-Gibbs measure and inp[:, 2] are the corresponding
                        weights, inp[:, 3:] are not used
        :param out:     torch.tensor, ndim==2, shape==[any, 2], output of the auto-encoder corresponding to the
                        described inp
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure.
        """
        return torch.mean(inp[:, 2] *
                          torch.minimum(torch.sum((inp[:, 0:2] - dec1) ** 2, dim=1),
                                        torch.sum((inp[:, 0:2] - dec2) ** 2, dim=1)))

    @staticmethod
    def mse_loss_react(inp, dec1, dec2):
        """MSE term on points distributed according to reactive trajectories measure.

        :param inp:     torch.tensor, ndim==2, shape==[any, 6], inp[:, 3:5] is the input of the
                        auto-encoder distributed according to reactive trajectories measure and inp[:, 5] are the
                        corresponding weights, inp[:, :3] are not used
        :param out:     torch.tensor, ndim==2, shape==[any, 2], output of the auto-encoder corresponding to the
                        described inp
        :return mse:    torch float, mean squared error between input and output for points distributed according to
                        Boltzmann-Gibbs measure.
        """
        return torch.mean(inp[:, 5] *
                          torch.minimum(torch.sum((inp[:, 3:5] - dec1) ** 2, dim=1),
                                        torch.sum((inp[:, 3:5] - dec2) ** 2, dim=1)))

    def train(self, batch_size, max_epochs):
        """ Do the training of the model self.ae

        :param batch_size:      int >= 1, batch size for the mini-batching
        :param max_epochs:      int >= 1, maximal number of epoch of training
        :return loss_dict:      dict, contains the average loss for each epoch and its various components.
        """
        if self.optimizer is None:
            print("""The optimizer has not been set, see set_optimizer method. It is set to use 'Adam' optimizer \n 
                     with a 0.001 learning rate and optimize all the parameters of the model""")
            self.set_optimizer('Adam', 0.001)
        # prepare the various loss list to store
        loss_dict = {"train_loss": [], "test_loss": [], "train_mse_boltz": [], "test_mse_boltz": [],
                     "train_squared_grad_enc_blotz": [], "test_squared_grad_enc_blotz": []}
        if "react_points" in self.dataset.keys():
            loss_dict["train_mse_react"] = []
            loss_dict["test_mse_react"] = []
        train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=self.validation_data, batch_size=batch_size, shuffle=True)
        epoch = 0
        model = copy.deepcopy(self.ae)
        while epoch < max_epochs:
            loss_dict["train_loss"].append([])
            loss_dict["train_mse_boltz"].append([])
            loss_dict["train_squared_grad_enc_blotz"].append([])
            if "react_points" in self.dataset.keys():
                loss_dict["train_mse_react"].append([])
            # train mode
            self.ae.train()
            for iteration, X in enumerate(train_loader):
                # Set gradient calculation capabilities
                X.requires_grad_()
                # Set the gradient of with respect to parameters to zero
                self.optimizer.zero_grad()
                # Forward pass for boltzmann gibbs distributed points
                enc = self.ae.encoder(X[:, :2])
                dec1 = self.ae.decoder1(enc)
                dec2 = self.ae.decoder2(enc)
                # Compute the various loss terms
                mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
                squared_grad_enc = self.squared_grad_encoder_penalization(X, enc)
                l1_pen = self.l1_penalization(self.ae)
                l2_pen = self.l2_penalization(self.ae)
                var_enc = self.var_encoder()
                loss = self.mse_boltz_weight * mse_blotz + \
                       self.var_enc_weight * var_enc + \
                       self.squared_grad_boltz_weight * squared_grad_enc + \
                       self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen + \
                       self.pen_points_weight * self.penalization_on_points()
                if "react_points" in self.dataset.keys():
                    # Forward pass for reactive trajectories
                    enc_reac = self.ae.encoder(X[:, 3:5])
                    dec_reac1 = self.ae.decoder1(enc_reac)
                    dec_reac2 = self.ae.decoder2(enc_reac)
                    mse_react = self.mse_loss_react(X, dec_reac1, dec_reac2)
                    loss += self.mse_react_weight * mse_react
                if self.var_dist_dec_weight > 0.:
                    enc_min = torch.min(self.ae.encoder(self.train_data[:, :2])).detach()
                    enc_max = torch.max(self.ae.encoder(self.train_data[:, :2])).detach()
                    z_grid = torch.linspace(enc_min, enc_max, self.n_bins_var_dist_dec)
                    dec1 = self.ae.decoder1(z_grid)
                    loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec1)
                    dec2 = self.ae.decoder2(z_grid)
                    loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec2)
                loss.backward()
                self.optimizer.step()
                loss_dict["train_loss"][epoch].append(loss.detach().numpy())
                loss_dict["train_mse_boltz"][epoch].append(mse_blotz.detach().numpy())
                loss_dict["train_squared_grad_enc_blotz"][epoch].append(squared_grad_enc.detach().numpy())
                if "react_points" in self.dataset.keys():
                    loss_dict["train_mse_react"][epoch].append(mse_react.detach().numpy())
            loss_dict["train_loss"][epoch] = np.mean(loss_dict["train_loss"][epoch])
            loss_dict["train_mse_boltz"][epoch] = np.mean(loss_dict["train_mse_boltz"][epoch])
            loss_dict["train_squared_grad_enc_blotz"][epoch] = np.mean(loss_dict["train_squared_grad_enc_blotz"][epoch])
            if "react_points" in self.dataset.keys():
                loss_dict["train_mse_react"][epoch] = np.mean(loss_dict["train_mse_react"][epoch])

            loss_dict["test_loss"].append([])
            loss_dict["test_mse_boltz"].append([])
            loss_dict["test_squared_grad_enc_blotz"].append([])
            if "react_points" in self.dataset.keys():
                loss_dict["test_mse_react"].append([])
            # test mode
            self.ae.eval()
            for iteration, X in enumerate(test_loader):
                # Set gradient calculation capabilities
                X.requires_grad_()
                # Forward pass for boltzmann gibbs distributed points
                enc = self.ae.encoder(X[:, :2])
                dec1 = self.ae.decoder1(enc)
                dec2 = self.ae.decoder2(enc)
                # Compute the various loss terms
                mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
                squared_grad_enc = self.squared_grad_encoder_penalization(X, enc)
                l1_pen = self.l1_penalization(self.ae)
                l2_pen = self.l2_penalization(self.ae)
                var_enc = self.var_encoder()
                loss = self.mse_boltz_weight * mse_blotz + \
                       self.var_enc_weight * var_enc + \
                       self.squared_grad_boltz_weight * squared_grad_enc + \
                       self.l1_pen_weight * l1_pen + \
                       self.l2_pen_weight * l2_pen + \
                       self.pen_points_weight * self.penalization_on_points()
                if "react_points" in self.dataset.keys():
                    # Forward pass for reactive trajectories
                    enc_reac = self.ae.encoder(X[:, 3:5])
                    dec_reac1 = self.ae.decoder1(enc_reac)
                    dec_reac2 = self.ae.decoder2(enc_reac)
                    mse_react = self.mse_loss_react(X, dec_reac1, dec_reac2)
                    loss += self.mse_react_weight * mse_react
                if self.var_dist_dec_weight > 0.:
                    enc_min = torch.min(self.ae.encoder(self.train_data[:, :2])).detach()
                    enc_max = torch.max(self.ae.encoder(self.train_data[:, :2])).detach()
                    z_grid = torch.linspace(enc_min, enc_max, self.n_bins_var_dist_dec)
                    dec1 = self.ae.decoder1(z_grid)
                    loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec1)
                    dec2 = self.ae.decoder2(z_grid)
                    loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec2)
                loss_dict["test_loss"][epoch].append(loss.detach().numpy())
                loss_dict["test_mse_boltz"][epoch].append(mse_blotz.detach().numpy())
                loss_dict["test_squared_grad_enc_blotz"][epoch].append(squared_grad_enc.detach().numpy())
                if "react_points" in self.dataset.keys():
                    loss_dict["test_mse_react"][epoch].append(mse_react.detach().numpy())
            loss_dict["test_loss"][epoch] = np.mean(loss_dict["test_loss"][epoch])
            loss_dict["test_mse_boltz"][epoch] = np.mean(loss_dict["test_mse_boltz"][epoch])
            loss_dict["test_squared_grad_enc_blotz"][epoch] = np.mean(loss_dict["test_squared_grad_enc_blotz"][epoch])
            if "react_points" in self.dataset.keys():
                loss_dict["test_mse_react"][epoch] = np.mean(loss_dict["test_mse_react"][epoch])
            # Early stopping
            if loss_dict["test_loss"][epoch] == np.min(loss_dict["test_loss"]):
                model = copy.deepcopy(self.ae)
            if epoch >= self.n_wait:
                if np.min(loss_dict["test_loss"]) < np.min(loss_dict["test_loss"][- self.n_wait:]):
                    epoch = max_epochs
                    self.ae = model
            epoch += 1
        print("training ends after " + str(len(loss_dict["test_loss"])) + " epochs.\n")
        return loss_dict

    def print_test_loss(self):
        """Print the test loss and its various components"""
        X = self.test_dataset
        X.requires_grad_()
        enc = self.ae.encoder(X[:, :2])
        dec1 = self.ae.decoder1(enc)
        dec2 = self.ae.decoder2(enc)
        # Compute the various loss terms
        mse_blotz = self.mse_loss_boltz(X, dec1, dec2)
        squared_grad_enc = self.squared_grad_encoder_penalization(X, enc)
        l1_pen = self.l1_penalization(self.ae)
        l2_pen = self.l2_penalization(self.ae)
        var_enc = self.var_encoder()
        loss = self.mse_boltz_weight * mse_blotz + \
               self.var_enc_weight * var_enc + \
               self.squared_grad_boltz_weight * squared_grad_enc + \
               self.l1_pen_weight * l1_pen + \
               self.l2_pen_weight * l2_pen + \
               self.pen_points_weight * self.penalization_on_points()
        if "react_points" in self.dataset.keys():
            # Forward pass for reactive trajectories
            enc_reac = self.ae.encoder(X[:, 3:5])
            dec_react1 = self.ae.decoder1(enc_reac)
            dec_react2 = self.ae.decoder2(enc_reac)
            mse_react = self.mse_loss_react(X, dec_react1, dec_react2)
            loss += self.mse_react_weight * mse_react
        if self.var_dist_dec_weight > 0.:
            enc_min = torch.min(self.ae.encoder(X[:, :2])).detach()
            enc_max = torch.max(self.ae.encoder(X[:, :2])).detach()
            z_grid = torch.linspace(enc_min, enc_max, self.n_bins_var_dist_dec)
            dec1 = self.ae.decoder1(z_grid)
            loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec1)
            dec2 = self.ae.decoder2(z_grid)
            loss += self.var_dist_dec_weight * self.dist_dec_penalization(dec2)
        print("""Test loss: """, loss)
        print("""Test MSE Boltzmann: """, mse_blotz)
        print("""Test squarred grad encoder: """, squared_grad_enc)
        if "react_points" in self.dataset.keys():
            print("""Test MSE reactive: """, mse_react)

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
        boltz_points = torch.tensor(self.dataset["boltz_points"].astype('float32'))
        boltz_points_decoded1 = self.ae.decoder1(self.ae.encoder(boltz_points))
        boltz_points_decoded2 = self.ae.decoder2(self.ae.encoder(boltz_points))
        xi_values = self.ae.xi_ae(self.dataset["boltz_points"])[:, 0]
        # equal-width bins
        z_bin = np.linspace(xi_values.min(), xi_values.max(), n_bins)
        # compute index of bin
        inds = np.digitize(xi_values, z_bin)
        # distribute train data to each bin
        x1 = torch.sum((boltz_points - boltz_points_decoded1) ** 2, dim=1).detach().numpy() < torch.sum(
            (boltz_points - boltz_points_decoded2) ** 2,
            dim=1).detach().numpy()
        x2 = torch.sum((boltz_points - boltz_points_decoded2) ** 2, dim=1).detach().numpy() < torch.sum(
            (boltz_points - boltz_points_decoded1) ** 2,
            dim=1).detach().numpy()
        for bin_idx in range(n_bins):
            X_given_z1[bin_idx] = self.dataset["boltz_points"][x1 * (inds == bin_idx + 1), :2]
            X_given_z2[bin_idx] = self.dataset["boltz_points"][x2 * (inds == bin_idx + 1), :2]
            if len(X_given_z1[bin_idx]) > 0:
                Esp_X_given_z1.append(torch.tensor(X_given_z1[bin_idx].astype('float32')).mean(dim=0))
                f_dec_z1.append(self.ae.decoder1(self.ae.encoder(Esp_X_given_z1[-1])).detach().numpy())
                Esp_X_given_z1[-1] = Esp_X_given_z1[-1].detach().numpy()
            if len(X_given_z2[bin_idx]) > 0:
                Esp_X_given_z2.append(torch.tensor(X_given_z2[bin_idx].astype('float32')).mean(dim=0))
                f_dec_z2.append(self.ae.decoder2(self.ae.encoder(Esp_X_given_z2[-1])).detach().numpy())
                Esp_X_given_z2[-1] = Esp_X_given_z2[-1].detach().numpy()
        if self.standadize:
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
        ax.plot(Esp_X_given_z1[:, 0], Esp_X_given_z1[:, 1], '-o', color='blue', label='cond. avg. decoder 1')
        ax.plot(Esp_X_given_z2[:, 0], Esp_X_given_z2[:, 1], '-o', color='purple', label='cond. avg. decoder 2')
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
        boltz_points_decoded1 = self.ae.decoder1(self.ae.encoder(boltz_points))
        boltz_points_decoded2 = self.ae.decoder2(self.ae.encoder(boltz_points))
        xi_values = self.ae.xi_ae(self.dataset["boltz_points"])[:, 0]
        # equal-width bins
        z_bin = np.linspace(xi_values.min(), xi_values.max(), n_bins)
        # compute index of bin
        inds = np.digitize(xi_values, z_bin)
        # distribute train data to each bin
        x1 = torch.sum((boltz_points - boltz_points_decoded1) ** 2, dim=1).detach().numpy() < torch.sum(
            (boltz_points - boltz_points_decoded2) ** 2,
            dim=1).detach().numpy()
        x2 = torch.sum((boltz_points - boltz_points_decoded2) ** 2, dim=1).detach().numpy() < torch.sum(
            (boltz_points - boltz_points_decoded1) ** 2,
            dim=1).detach().numpy()
        for bin_idx in range(n_bins):
            X_given_z1[bin_idx] = self.dataset["boltz_points"][x1 * (inds == bin_idx + 1), :2]
            X_given_z2[bin_idx] = self.dataset["boltz_points"][x2 * (inds == bin_idx + 1), :2]
            if len(X_given_z1[bin_idx]) > 0:
                Esp_X_given_z1.append(torch.tensor(X_given_z1[bin_idx].astype('float32')).mean(dim=0))
                f_dec_z1.append(self.ae.decoder1(self.ae.encoder(Esp_X_given_z1[-1])).detach().numpy())
                Esp_X_given_z1[-1].requires_grad_()
                z1 = self.ae.encoder(Esp_X_given_z1[-1])
                z_values1.append(z1.detach().numpy())
                grad_f_enc1 = torch.autograd.grad(z1, Esp_X_given_z1[-1])[0]
                grads_enc1.append(grad_f_enc1.detach().numpy())
                z1.requires_grad_()
                grad_f_dec1 = torch.autograd.functional.jacobian(self.ae.decoder1, z1, create_graph=False).sum(dim=1)
                grads_dec1.append(grad_f_dec1.detach().numpy())
                Esp_X_given_z1[-1] = Esp_X_given_z1[-1].detach().numpy()
            if len(X_given_z2[bin_idx]) > 0:
                Esp_X_given_z2.append(torch.tensor(X_given_z2[bin_idx].astype('float32')).mean(dim=0))
                f_dec_z2.append(self.ae.decoder2(self.ae.encoder(Esp_X_given_z2[-1])).detach().numpy())
                Esp_X_given_z2[-1].requires_grad_()
                z2 = self.ae.encoder(Esp_X_given_z2[-1])
                z_values2.append(z2.detach().numpy())
                grad_f_enc2 = torch.autograd.grad(z2, Esp_X_given_z2[-1])[0]
                grads_enc2.append(grad_f_enc2.detach().numpy())
                z2.requires_grad_()
                grad_f_dec2 = torch.autograd.functional.jacobian(self.ae.decoder2, z2, create_graph=False).sum(dim=1)
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
