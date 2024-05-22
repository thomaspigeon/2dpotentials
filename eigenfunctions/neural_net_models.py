import torch


class EigenFunctions(torch.nn.Module):
    def __init__(self, eigen_func_dims, dropout):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param eigen_func_dims:  list, List of dimensions for encoder, including input/output layers
        :param dropout:         int, value of the dropout probability
        """
        super(EigenFunctions, self).__init__()

        layers = []
        for i in range(len(eigen_func_dims) - 2):
            layers.append(torch.nn.Linear(eigen_func_dims[i], eigen_func_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(eigen_func_dims[-2], eigen_func_dims[-1]))
        layers.append(torch.nn.Sigmoid())
        self.encoder = torch.nn.Sequential(*layers)

    def xi(self,  X):
        """Collective variable defined through an auto encoder model

        :param X: np.array, position (or position and momenta), ndim = 2 (4), shape = [any, 2(4)]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device='cpu'
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float().to(device)
        return self.encoder(X).cpu().detach().numpy()

    def grad_xi(self, X):
        """Gradient of the collective variable defined through an auto encoder model

        :param X: np.array, position (or position and momenta), ndim = 2 (4), shape = [any, 2(4)]
        :return: xi: np.array, ndim = 2, shape = [any, 2]
        """
        self.eval()
        if not torch.is_tensor(X):
            x = torch.from_numpy(X).float()
        X.requires_grad_()
        enc = self.encoder(X)
        if X.shape[1] == 2:
            return torch.autograd.grad(outputs=enc.sum(), inputs=X)[0][:, :2]
        if X.shape[1] == 4:
            return torch.autograd.grad(outputs=enc.sum(), inputs=X)[0][:, :4]
