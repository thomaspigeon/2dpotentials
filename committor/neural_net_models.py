import torch


class CommittorOneDecoder(torch.nn.Module):
    def __init__(self, committor_dims, decoder_dims, dropout, pot):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param committor_dims:  list, List of dimensions for encoder, including input/output layers
        :param decoder_dims:    list, List of dimensions for decoder, including input/output layers
        :param dropout:         int, value of the dropout probability
        :param pot:             General2DPotential, object containing information concerning the potential et the
                                definition of the reactant and product state
        """
        super(CommittorOneDecoder, self).__init__()
        layers = []
        for i in range(len(committor_dims) - 2):
            layers.append(torch.nn.Linear(committor_dims[i], committor_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(committor_dims[-2], committor_dims[-1]))
        layers.append(torch.nn.Sigmoid())
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 2):
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder = torch.nn.Sequential(*layers)

        self.minR = torch.tensor(pot.minR.astype('float32'))
        self.R_radius = pot.R_radius
        self.minP = torch.tensor(pot.minP.astype('float32'))
        self.P_radius = pot.P_radius

        self.HT = torch.nn.Hardtanh()

    def inR(self, inp):
        return 1 - (1 / 2 + (1 / 2) * self.HT(
            (200 * (torch.sqrt(torch.sum((inp - self.minR) ** 2, dim=1)).reshape(
                [len(inp), 1]) - self.R_radius) / self.R_radius) + 1))

    def inP(self, inp):
        return 1 - (1 / 2 + (1 / 2) * self.HT(
            (200 * (torch.sqrt(torch.sum((inp - self.minP) ** 2, dim=1)).reshape(
                [len(inp), 1]) - self.P_radius) / self.P_radius) + 1))

    def decoded(self, inp):
        enc = self.encoder(inp)
        dec = self.decoder(enc)
        return dec

    def committor(self, inp):
        committor = (1 - self.inR(inp)) * ((1 - self.inP(inp)) * self.encoder(inp) + self.inP(inp))
        return committor

    def xi_forward(self,  x):
        """Collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 2]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        return self.committor(x).detach().numpy()

    def xi_backward(self,  x):
        """Collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 2]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        return 1 - self.committor(x).detach().numpy()

    def grad_xi(self, x):
        """Gradient of the collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 1]
        :return: xi: np.array, ndim = 2, shape = [any, 2]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        x.requires_grad_()
        enc = self.encoder(x)
        return torch.autograd.grad(outputs=enc.sum(), inputs=x)[0][:, :2]

class CommittorTwoDecoder(torch.nn.Module):
    def __init__(self, committor_dims, decoder_dims, dropout, pot):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param committor_dims:  list, List of dimensions for encoder, including input/output layers
        :param decoder_dims:    list, List of dimensions for decoder, including input/output layers
        :param dropout:         int, value of the dropout probability
        :param pot:             General2DPotential, object containing information concerning the potential et the
                                definition of the reactant and product state
        """
        super(CommittorOneDecoder, self).__init__()
        layers = []
        for i in range(len(committor_dims) - 2):
            layers.append(torch.nn.Linear(committor_dims[i], committor_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(committor_dims[-2], committor_dims[-1]))
        layers.append(torch.nn.Sigmoid())
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 2):
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder1 = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 2):
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder2 = torch.nn.Sequential(*layers)

        self.minR = torch.tensor(pot.minR.astype('float32'))
        self.R_radius = pot.R_radius
        self.minP = torch.tensor(pot.minP.astype('float32'))
        self.P_radius = pot.P_radius
        self.HT = torch.nn.Hardtanh()

    def inR(self, inp):
        return 1 - (1 / 2 + (1 / 2) * self.HT(
            (200 * (torch.sqrt(torch.sum((inp - self.minR) ** 2, dim=1)).reshape(
                [len(inp), 1]) - self.R_radius) / self.R_radius) + 1))

    def inP(self, inp):
        return 1 - (1 / 2 + (1 / 2) * self.HT(
            (200 * (torch.sqrt(torch.sum((inp - self.minP) ** 2, dim=1)).reshape(
                [len(inp), 1]) - self.P_radius) / self.P_radius) + 1))

    def committor(self, inp):
        committor = (1 - self.inR(inp)) * ((1 - self.inP(inp)) * self.encoder(inp) + self.inP(inp))
        return committor

    def decoded1(self, inp):
        enc = self.encoder(inp)
        dec = self.decoder1(enc)
        return dec

    def decoded2(self, inp):
        enc = self.encoder(inp)
        dec = self.decoder2(enc)
        return dec


    def xi(self,  x):
        """Collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 2]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        return self.committor(x).detach().numpy()

    def grad_xi_ae(self, x):
        """Gradient of the collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 1]
        :return: xi: np.array, ndim = 2, shape = [any, 2]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        x.requires_grad_()
        enc = self.encoder(x)
        return ((torch.autograd.grad(outputs=enc.sum(), inputs=x)[0][:, :2]) ** 2).mean()